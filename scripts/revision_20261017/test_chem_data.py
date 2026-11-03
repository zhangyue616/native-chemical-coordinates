"""Finite synthetic acceptance; never opens the actual source gzip or T inputs."""
from __future__ import annotations

import collections
import csv
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

import chem_data as cd
from semantic_mapper import SemanticMapper


def run():
    cd.ENGINEERING.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name, passed, evidence):
        checks.append({"case": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})
    concepts = cd.load_definitions()
    counts = collections.Counter(c["source"] for c in concepts)
    check("91_frozen_definitions_only", len(concepts) == 91 and counts == {"topology": 14, "fg_hierarchy": 10, "fragment_descriptors": 67}
          and all(set(c) == {"id", "name", "source", "kind", "definition", "partition"} for c in concepts), dict(counts))
    golden_digest = "0071e407813d6a548e68d3fe284cf8c4199ce860e15efd3cf39b040a3d95cc5e"
    check("fixed_new_salt_hash_and_threshold", cd.cid_digest("24586").hex() == golden_digest
          and cd.candidate_bucket(bytes.fromhex(golden_digest)) == 1533 and cd.THRESHOLD == 2000,
          {"cid": 24586, "expected_full_digest": golden_digest, "bucket": 1533})
    first = (bytes.fromhex(golden_digest), 24586, "CCCCC", 1)
    smaller_bucket = (bytes.fromhex("2bec418c7422d46ba20ce47a39ae5a04775d2c026c590863b985da67681808d2"), 94522, "CCCCN", 2)
    tie = (first[0], 1, "CCCCO", 3)
    check("full_digest_sort_not_bucket_numeric_tiebreak", [c[1] for c in sorted([smaller_bucket, first, tie], key=cd.candidate_sort_key)] == [1, 24586, 94522],
          "full digest wins over bucket34; numeric CID breaks an injected exact-digest tie")
    check("fixed_split_boundaries", [cd.split_for_rank(r) for r in (0, 4999, 5000, 14999)] == ["D", "D", "T", "T"], "0-4999 D;5000-14999 T")
    try:
        cd.split_for_rank(15000)
        check("split_outside_rank_rejected", False, "unexpected pass")
    except cd.DataError:
        check("split_outside_rank_rejected", True, "no third split or refill beyond target")
    std_cases = [
        ("largest_fragment", "CCCCC.[Na+]", "CCCCC", None),
        ("first_fragment_tie", "CCCCC.NCCCC", "CCCCC", None),
        ("first_fragment_tie_reversed", "NCCCC.CCCCC", "CCCCN", None),
        ("RemoveHs", "[H]CCCCC", "CCCCC", None),
        ("small", "CCO", None, "heavy_atoms"),
        ("isotope", "[13CH3]CCCC", None, "radical_or_isotope"),
        ("radical", "[CH2]CCCC", None, "radical_or_isotope"),
        ("element", "CCCC[SeH]", None, "element"),
        ("tripeptide", "NCC(=O)NCC(=O)NCC(=O)O", None, "bio_structure"),
        ("nucleoside", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", None, "bio_structure"),
    ]
    for name, source, expected, reason in std_cases:
        actual, why = cd.standardize(source)
        check("standardize:" + name, (actual, why) == (expected, reason), {"expected_reason": reason, "actual_reason": why, "canonical_matches_expected": actual == expected})
    mapper = SemanticMapper()
    candidates = [(bytes([i]) * 32, i, smi, i) for i, smi in enumerate(
        ["CCCCC", "CCCCN", "CCCCO", "OCCCC", "CC(C)CC", "CCCCCl", "CCCCCBr"], 1)]
    stats = {"processed_prefix": 0, "accepted": 0, "excluded": collections.Counter(), "mapper_reasons": collections.Counter()}
    selected = list(cd.select_eligible(candidates, {1}, {cd.smiles_identity("CCCCN")}, mapper, 3, stats))
    check("exclude_old_CID_old_identity_new_canonical_duplicate", [r["cid"] for r, _ in selected] == [3, 5, 6]
          and stats["processed_prefix"] == 6 and stats["excluded"] == {"old_cid": 1, "old_canonical_identity": 1, "duplicate_new_canonical": 1},
          {"selected_cids": [r["cid"] for r, _ in selected], **stats})
    check("mapper_actual_Cl_anchor_kept", selected[-1][1]["accepted"] and selected[-1][1]["identity_lossy"]
          and selected[-1][1]["atoms"][-1]["graph_properties"]["atomic_number"] == 17,
          "actual fixed tokenizer; source Cl identity and offset remain traceable")
    class RejectFirst:
        def audit_molecule(self, smiles, **kwargs):
            return {"accepted": False, "exclusion_reasons": ["synthetic_ambiguous_anchor"]} if smiles == "CCCCO" else mapper.audit_molecule(smiles, **kwargs)
    rejected_stats = {"processed_prefix": 0, "accepted": 0, "excluded": collections.Counter(), "mapper_reasons": collections.Counter()}
    after_mapper = list(cd.select_eligible(candidates[2:], set(), set(), RejectFirst(), 1, rejected_stats))
    check("mapper_reject_does_not_occupy_split_rank", after_mapper[0][0]["cid"] == 5 and after_mapper[0][0]["rank"] == 0
          and rejected_stats["excluded"] == {"mapper_gate": 1, "duplicate_new_canonical": 1}, rejected_stats)
    def stop_generator():
        yield candidates[4]
        raise AssertionError("candidate beyond frozen accepted prefix was consumed")
    limited_stats = {"processed_prefix": 0, "accepted": 0, "excluded": collections.Counter(), "mapper_reasons": collections.Counter()}
    check("stop_immediately_at_target", len(list(cd.select_eligible(stop_generator(), set(), set(), mapper, 1, limited_stats))) == 1,
          "does not consume next source candidate after target")
    labeler = cd.FrozenLabeler(concepts)
    labels, occurrences = labeler.molecule("O=C(O)CC(=O)O")
    columns = {c["name"]: i for i, c in enumerate(concepts)}
    acid = columns["fg:CarboxylicAcid"]
    carbonyl = columns["frag:fr_C=O"]
    check("SMARTS_all_match_atom_union_vs_occurrence", np.flatnonzero(labels[:, acid]).tolist() == [0, 1, 2, 4, 5, 6]
          and int(occurrences[acid]) == 2 and np.flatnonzero(labels[:, carbonyl]).tolist() == [0, 1, 4, 5]
          and int(occurrences[carbonyl]) == 2,
          {"acid_positive_atom_ids": np.flatnonzero(labels[:, acid]).tolist(), "acid_unique_matches": int(occurrences[acid]),
           "carbonyl_positive_atom_ids": np.flatnonzero(labels[:, carbonyl]).tolist(), "carbonyl_unique_matches": int(occurrences[carbonyl])})
    overlap_labels, overlap_occ = labeler.molecule("CCCCC")
    alkane = columns["frag:fr_unbrch_alkane"]
    check("overlapping_matches_deduplicate_atoms_only", np.flatnonzero(overlap_labels[:, alkane]).tolist() == [0, 1, 2, 3, 4]
          and int(overlap_occ[alkane]) == 2, "two overlapping four-atom occurrences; five unique positive atoms")
    spiro = cd.topology_labels(Chem.MolFromSmiles("C1CCC2(CC1)CCCC2"))
    check("topology_preserves_at_least_two_RDKit_rings_semantics", spiro["topo:spiro_atom"] == {3}
          and spiro["topo:fused_ring_atom"] == {3}, "historical fused name includes the spiro atom because membership in >=2 RDKit rings")
    check("function_occurrence_not_atom_count", all(occurrences[i] == -1 for i, c in enumerate(concepts) if c["kind"] == "function"), "topology function occurrence columns use -1 sentinel")
    synthetic_positive = np.zeros((500, 91), dtype=np.int32)
    eligible_ids = [0, 1, 2, 3, 14, 15, 16, 17, 24, 25, 26, 27, 28]
    synthetic_positive[:200, eligible_ids] = 1
    coverage, panel = cd.coverage_and_panel(concepts, synthetic_positive)
    check("coverage_C_all_qualified_ID_order", coverage["C_ids"] == eligible_ids and coverage["coverage_gate"] == "PASS", coverage["C_ids"])
    check("panel_round_robin_not_effect_selected", panel["panel_ids"] == [0, 14, 24, 1, 15, 25, 2, 16, 26, 3, 17, 27], panel["panel_ids"])
    check("mechanism_first3_events_no_replacement", panel["mechanism_candidate_ids"] == [24, 25, 26], panel["mechanism_candidate_ids"])
    failed, _ = cd.coverage_and_panel(concepts, np.zeros((500, 91), dtype=np.int32))
    check("coverage_failure_no_rescue", failed["coverage_gate"] == "FAIL_FEWER_THAN_12" and failed["C_ids"] == [], "no threshold change or added concepts")
    collisions = cd.collision_counts({"D": collections.Counter({"a": 2, "b": 1}), "T": collections.Counter({"a": 3, "c": 1})})
    check("token_collisions_count_only_no_narrowing", collisions["D"]["pairs"] == 1 and collisions["T"]["pairs"] == 3
          and collisions["cross_D_T"]["pairs"] == 6 and not collisions["collision_changes_eligibility"], collisions)
    with tempfile.TemporaryDirectory(prefix="synthetic_", dir=cd.ENGINEERING) as tmp:
        tmp = Path(tmp)
        out = tmp / "D"
        records = [{"split": "D", "cid": 9001, "smiles": "O=C(O)CC(=O)O"}, {"split": "D", "cid": 9002, "smiles": "CCCCC"}]
        info = cd.label_development(records, concepts, out)
        with np.load(out / "labels_D.npz", allow_pickle=False) as lab, np.load(out / "atoms_D.npz", allow_pickle=False) as atoms, \
                np.load(out / "occurrence_counts_D.npz", allow_pickle=False) as occ:
            check("saved_D_schema_and_atom_order", lab["labels"].shape == (12, 91) and lab["labels"].dtype == np.bool_
                  and atoms["mol_offsets"].tolist() == [0, 7, 12] and atoms["atom_idx"].tolist() == list(range(7)) + list(range(5))
                  and atoms["cid"].tolist() == [9001] * 7 + [9002] * 5 and occ["counts"].shape == (2, 91)
                  and int(occ["counts"][0, acid]) == 2 and int(occ["positive_atom_counts"][0, acid]) == 6,
                  {"atoms_D": info["atoms_D"], "label_shape": list(lab["labels"].shape), "occurrence_shape": list(occ["counts"].shape)})
        try:
            cd.label_development([{"split": "T", "cid": 9003, "smiles": "CCCCC"}], concepts, tmp / "T")
            check("T_label_guard_before_content_parse_or_write", False, "unexpected T label permission")
        except cd.DataError:
            check("T_label_guard_before_content_parse_or_write", not (tmp / "T").exists(), "rejects T row before output creation")
        try:
            cd.require_data_lock(tmp / "MISSING_LOCK.json", tmp / "MISSING_READY.json")
            check("missing_root_lock_blocks_source_access", False, "missing lock passed")
        except cd.DataError:
            check("missing_root_lock_blocks_source_access", True, "no gzip operation is reached")
        ready_path, lock_path = tmp / "READY.json", tmp / "LOCK.json"
        cd.dump_json(ready_path, {"synthetic": True})
        cd.dump_json(lock_path, {"status": "LOCKED", "data_builder_ready_sha256": "0" * 64})
        try:
            cd.require_data_lock(lock_path, ready_path)
            check("wrong_ready_SHA_lock_rejected", False, "wrong SHA passed")
        except cd.DataError:
            check("wrong_ready_SHA_lock_rejected", True, "no source read")
        cd.dump_json(lock_path, {"status": "LOCKED", "data_builder_ready_sha256": cd.file_sha(ready_path)})
        check("correct_ready_SHA_lock_accepted", cd.require_data_lock(lock_path, ready_path) == {"synthetic": True}, "pure lock fixture only")
    report = {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "scope": "finite synthetic builder/exclusion/split/standardization/mapper/91-definition labeling/lock checks",
        "counts": {"checks": len(checks), "passed": sum(c["status"] == "PASS" for c in checks), "failed": sum(c["status"] == "FAIL" for c in checks)},
        "runtime": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
        "execution_ledger": {"actual_source_gzip_reads": 0, "actual_new_D_or_T_reads": 0, "T_label_calls": 0,
                             "model_loads": 0, "model_forwards": 0, "fits": 0, "bootstrap": 0},
        "checks": checks}
    cd.dump_json(cd.ENGINEERING / "ENGINEERING_CHECKS.json", report)
    print(json.dumps({"status": report["status"], **report["counts"], "failures": [c for c in checks if c["status"] == "FAIL"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    with rdBase.BlockLogs():
        raise SystemExit(run())
