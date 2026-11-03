"""Finite synthetic checks; no source data, network or models."""
import copy
import inspect
import io
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import protein_fresh_data as f


def entry(accession="SYNTH", length=300):
    def feature(kind, q):
        return {"type": kind, "location": {"start": {"value": q + 1}, "end": {"value": q + 1}},
                "evidences": [{"source": "PDB", "id": "1ABC", "evidenceCode": "ECO:0007829"}]}
    return {"primaryAccession": accession, "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "sequence": {"value": "A" * length, "length": length},
            "features": [feature("Helix", 7), feature("Beta strand", length - 8)],
            "uniProtKBCrossReferences": [{"database": "PDB", "id": "1ABC", "properties": [
                {"key": "Method", "value": "X-ray"}, {"key": "Chains", "value": f"A=1-{length}"}]}]}


class FreshSourceTests(unittest.TestCase):
    def test_all_raw_invalid_bridge_and_exactseq_bridge(self):
        old = {"components": [{"accessions": ["OLD0"], "pfam": ["PF00001"]},
                               {"accessions": ["OLD1"], "pfam": ["PF00002"]}],
               "selected_pair_groups": [{"origin": "a", "id": "OLD0", "old_component_index": 0},
                                        {"origin": "b", "id": "OLD1", "old_component_index": 1}]}
        seq = {f.digest("OLD_SEQ"): 0}
        # Invalid/no-label B links via exact sequence to A and via Pfam to old0;
        # C links via its Pfam to B and exact sequence to old1's bridge D.
        nodes = [{"id": "A", "sequence": "NEW_SEQ", "pfam": []},
                 {"id": "B", "sequence": "NEW_SEQ", "pfam": ["PF00001", "PF99999"]},
                 {"id": "C", "sequence": "OTHER", "pfam": ["PF99999"]},
                 {"id": "D", "sequence": "OTHER", "pfam": ["PF00002"]},
                 {"id": "E", "sequence": "OLD_SEQ", "pfam": []},
                 {"id": "F", "sequence": "FRESH", "pfam": []}]
        components, join, mapping = f.full_union(old, seq, nodes)
        self.assertEqual(len(join), 6)
        self.assertTrue(all(join[i]["connected_to_old_pool"] for i in range(5)))
        self.assertFalse(join[5]["connected_to_old_pool"])
        self.assertEqual(mapping[0]["current_known_union_group"], mapping[1]["current_known_union_group"])
        baseline = [dict(mapping[0]), dict(mapping[1], current_known_union_group="old-union-000002")]
        self.assertFalse(f.groups_unchanged(baseline, mapping))
        self.assertEqual(join[0]["known_group_id"], join[4]["known_group_id"])
        self.assertEqual(join[5]["component_key"], "acc:F")

    def test_accession_bridge_without_sequence_or_Pfam(self):
        old = {"components": [{"accessions": ["OLD"], "pfam": []}], "selected_pair_groups": []}
        nodes = [{"id": "OLD", "sequence": None, "pfam": []}, {"id": "NEW", "sequence": None, "pfam": []}]
        _, join, _ = f.full_union(old, {}, nodes)
        self.assertTrue(join[0]["connected_to_old_pool"])
        self.assertFalse(join[1]["connected_to_old_pool"])
        invalid = {"primaryAccession": "OLD", "sequence": None, "uniProtKBCrossReferences": None}
        self.assertEqual(f.project(invalid), {"id": "OLD", "sequence": None, "pfam": []})

    def test_source_ID_release_total_and_page_contract(self):
        headers = {"x-uniprot-release": f.RELEASE[0], "x-uniprot-release-date": f.RELEASE[1],
                   "x-total-results": str(f.TOTAL), "content-type": "application/json"}
        ids = [f"SYNTH{i:05d}" for i in range(f.TOTAL)]
        rows = [{"primaryAccession": x} for x in ids[:500]]
        f.check_page(rows, headers, ids, 0)
        tail = [{"primaryAccession": x} for x in ids[13500:]]
        f.check_page(tail, headers, ids, 13500)
        with self.assertRaises(f.SourceFailure):
            f.check_page(rows[::-1], headers, ids, 0)
        with self.assertRaises(f.SourceFailure):
            f.check_page(rows, dict(headers, **{"x-total-results": "13659"}), ids, 0)
        with self.assertRaises(f.SourceFailure):
            f.check_page(rows, dict(headers, **{"x-uniprot-release": "changed"}), ids, 0)
        with self.assertRaises(f.SourceFailure):
            f.check_page(rows[:-1], headers, ids, 0)
        with self.assertRaises(f.SourceFailure):
            f.check_page(rows, dict(headers, **{"x-uniprot-release-date": "changed"}), ids, 0)

    def test_length_only_population_change_and_exact_selector_reuse(self):
        for length in (257, 300, 512):
            self.assertEqual(f.sequence_eligibility(entry(length=length)), [])
        for length in (64, 256, 513):
            self.assertIn("length_outside_257_512", f.sequence_eligibility(entry(length=length)))
        self.assertEqual(inspect.getfile(f.P5.labels_and_pair), str(f.SELECTOR))
        self.assertEqual(f.P5.SEED, f.SALT)
        sample = entry()
        pair, sites, _, counts = f.P5.labels_and_pair(sample)
        self.assertEqual(counts["legal_pair_count"], 1)
        self.assertEqual(pair["h_site"]["q_0based"], 7)
        self.assertEqual(pair["s_site"]["q_0based"], 292)
        self.assertFalse(pair["both_plusminus15_available"])
        self.assertEqual(pair["pair_sha256"], f.digest(f"pilot5a-pair:{pair['pair_key']}:{f.SALT}"))

    def test_P5_unchanged_exact_evidence_conflict_and_coverage_rules(self):
        original = entry()
        variants = []
        uncertain = copy.deepcopy(original)
        uncertain["features"][0]["location"]["start"]["modifier"] = "UNKNOWN"
        variants.append(uncertain)
        isoform = copy.deepcopy(original)
        isoform["features"][0]["location"]["sequenceId"] = "SYNTH-2"
        variants.append(isoform)
        unsupported = copy.deepcopy(original)
        unsupported["features"][0]["evidences"][0]["source"] = "other"
        variants.append(unsupported)
        conflict = copy.deepcopy(original)
        extra = copy.deepcopy(conflict["features"][0])
        extra["type"] = "Beta strand"
        conflict["features"].append(extra)
        variants.append(conflict)
        unknown_method = copy.deepcopy(original)
        unknown_method["uniProtKBCrossReferences"][0]["properties"][0]["value"] = "prediction"
        variants.append(unknown_method)
        missing_chain = copy.deepcopy(original)
        missing_chain["uniProtKBCrossReferences"][0]["properties"][1]["value"] = "-=1-300"
        variants.append(missing_chain)
        invalid_eco = copy.deepcopy(original)
        invalid_eco["features"][0]["evidences"][0]["evidenceCode"] = "ECO:bad"
        variants.append(invalid_eco)
        uncovered = copy.deepcopy(original)
        uncovered["uniProtKBCrossReferences"][0]["properties"][1]["value"] = "A=9-300"
        variants.append(uncovered)
        non_A = copy.deepcopy(original)
        non_A["sequence"]["value"] = "A" * 7 + "C" + "A" * 292
        variants.append(non_A)
        insufficient_local = copy.deepcopy(original)
        insufficient_local["features"][0]["location"] = {"start": {"value": 7}, "end": {"value": 7}}
        variants.append(insufficient_local)
        for sample in variants:
            self.assertIsNone(f.P5.labels_and_pair(sample)[0])

    def test_sequence_schema_eligibility_rejections(self):
        sample = entry()
        sample["sequence"]["length"] = 299
        self.assertIn("source_sequence_length_mismatch", f.sequence_eligibility(sample))
        sample = entry()
        sample["sequence"]["value"] = "X" + sample["sequence"]["value"][1:]
        self.assertIn("nonstandard_amino_acid", f.sequence_eligibility(sample))
        sample["entryType"] = "UniProtKB unreviewed (TrEMBL)"
        self.assertIn("source_not_reviewed", f.sequence_eligibility(sample))

    def test_fixed_query_next_and_selection_budget_no_fallback(self):
        self.assertEqual(f.validate_url(f.initial_url(), initial=True), f.initial_url())
        official = f.initial_url() + "&cursor=synthetic"
        self.assertEqual(f.next_url(f'<{official}>; rel="next"'), official)
        with self.assertRaises(f.SourceFailure):
            f.validate_url(official.replace("512", "513"))
        with self.assertRaises(f.SourceFailure):
            f.validate_url(official.replace("rest.uniprot.org", "example.org"))
        with self.assertRaises(f.SourceFailure):
            f.validate_url(official + "&cursor=duplicate")
        pair = f.P5.labels_and_pair(entry())[0]
        reps = [{"id": f"SYNTH{i}", "sequence": "A" * 300, "pfam": [], "pair": pair,
                 "component_key": f"acc:SYNTH{i}", "component_sha256": f.digest(f"c{i}"),
                 "known_group_id": f"g{i}", "protein_sha256": f.digest(f"p{i}")} for i in range(f.TARGET + 1)]
        selected, unused = f.freeze_selected(reps)
        reversed_selected, _ = f.freeze_selected(list(reversed(reps)))
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(len(selected), 1304)
        self.assertEqual(len(unused), 1)
        self.assertEqual(f.freeze_selected(reps[:1303])[0], [])
        self.assertEqual((f.PAGES, f.ATTEMPTS, f.TOTAL), (28, 3, 13658))
        self.assertTrue(all(r["a_site"]["q_0based"] < r["b_site"]["q_0based"] for r in selected))

    def test_HTTP_three_total_attempts_same_URL_and_no_partial_fallback(self):
        url = f.initial_url()
        opener = Mock()
        opener.open.side_effect = [HTTPError(url, 503, "synthetic", {}, io.BytesIO(b"discard")) for _ in range(3)]
        receipt = {"attempts": [], "pages": []}
        with patch.object(f, "write_json"), patch.object(f.time, "sleep") as sleep:
            with self.assertRaises(f.SourceFailure) as failure:
                f.fetch_page(opener, url, 1, receipt)
        self.assertEqual(failure.exception.status, "NETWORK_OR_HTTP_STOP")
        self.assertEqual(opener.open.call_count, 3)
        self.assertEqual([r["attempt"] for r in receipt["attempts"]], [1, 2, 3])
        self.assertTrue(all(r["request_url"] == url for r in receipt["attempts"]))
        self.assertTrue(all(call.args[0].full_url == url for call in opener.open.call_args_list))
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual(receipt["pages"], [])
        opener = Mock()
        opener.open.side_effect = HTTPError(url, 403, "synthetic", {}, io.BytesIO(b"discard"))
        with patch.object(f, "write_json"), patch.object(f.time, "sleep") as sleep:
            with self.assertRaises(f.SourceFailure):
                f.fetch_page(opener, url, 1, {"attempts": [], "pages": []})
        self.assertEqual(opener.open.call_count, 1)
        sleep.assert_not_called()

    def test_run_holds_qualification_when_new_exactseq_merges_D_groups(self):
        old = {"components": [{"accessions": ["OLD0"], "pfam": []},
                              {"accessions": ["OLD1"], "pfam": ["PF00002"]}],
               "selected_pair_groups": [{"origin": "a", "id": "OLD0", "old_component_index": 0},
                                        {"origin": "b", "id": "OLD1", "old_component_index": 1}]}
        baseline = [{**r, "current_known_union_group": f"old-union-{i + 1:06d}"}
                    for i, r in enumerate(old["selected_pair_groups"])]
        new = {"primaryAccession": "BRIDGE", "sequence": {"value": "SEQ"},
               "uniProtKBCrossReferences": [{"database": "Pfam", "id": "PF00002"}]}
        values = {"OLD_POOL_GRAPH.json": old,
                  "OLD_EXACT_SEQUENCE_INDEX.json": {"sequence_sha256_to_old_component": {f.digest("SEQ"): 0}},
                  "OLD_SELECTED_UNION_GROUPS.json": {"selected_pair_group_mapping": baseline}}
        headers = {"x-uniprot-release": f.RELEASE[0], "x-uniprot-release-date": f.RELEASE[1],
                   "x-total-results": "1", "content-type": "application/json"}
        # Every source/network/file side effect is mocked. The real union and D gate run.
        with patch.object(f, "TOTAL", 1), patch.object(f, "PAGES", 1), \
                patch.object(f, "require_lock"), patch.object(f, "file_sha", return_value="synthetic"), \
                patch.object(f, "metadata_rows", return_value=[{"id": "BRIDGE"}]), \
                patch.object(f, "build_opener"), \
                patch.object(f, "fetch_page", return_value=([new], headers, None)), \
                patch.object(f, "read_json", side_effect=lambda path: values[path.name]), \
                patch.object(f, "write_json") as write, patch.object(f, "write_jsonl"), \
                patch.object(f, "qualify") as qualification, patch("builtins.print"):
            f.run(f.OUT / "synthetic-lock.json")
        qualification.assert_not_called()
        reports = [call.args[1] for call in write.call_args_list if call.args[0].name == "GRAPH_RESULT.json"]
        self.assertEqual(reports[0]["status"], "D_GROUPS_CHANGED_QUALIFICATION_HELD")
        self.assertEqual(reports[0]["D_group_count"], 1)


if __name__ == "__main__":
    unittest.main()
