"""One locked long-protein source gate; no model/activation/statistical output.

The P5 evidence/pair selector is imported read-only in an isolated module.
Only the outer length population and the prelocked hash salt change.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/revision_20261017/protein_fresh"
META = ROOT / "data/revision_20261017/protein_metadata"
PROTOCOL = ROOT / "docs/revision_20261017/PROTEIN_FRESH_DATA_PROTOCOL.md"
PRECISION = ROOT / "results/revision_20261017/protein_D/PRECISION_PLAN.json"
SELECTOR = ROOT / "scripts/prepare_pilot5a_data.py"
SALT = "protein-long-transfer-20261017-v1"
QUERY = "(reviewed:true) AND (length:[257 TO 512]) AND (database:pdb)"
FIELDS = "accession,sequence,length,xref_pfam,xref_pdb,ft_helix,ft_strand"
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
RELEASE = ["2026_03", "02-September-2026"]
TOTAL, PAGES, ATTEMPTS, TARGET = 13658, 28, 3, 1304
AA = set("ACDEFGHIKLMNPQRSTVWY")
CONFIG = {"endpoint": ENDPOINT, "query": QUERY, "fields": FIELDS, "sort": "accession asc",
          "format": "json", "page_size": 500, "source_records_exact": TOTAL,
          "successful_pages_exact": PAGES, "maximum_attempts_per_URL": ATTEMPTS,
          "transient_retry_policy": "initial attempt plus at most two retries, same URL only",
          "release": RELEASE, "length_inclusive": [257, 512], "salt": SALT,
          "target_final_fresh_groups": TARGET, "old_raw_records": 16896,
          "new_model_loads": 0, "new_model_forwards": 0, "new_predictions": 0,
          "new_query_or_partial_cohort_fallback": False,
          "qualification_requires_D_groups_unchanged": True}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


class SourceFailure(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def load_selector():
    spec = importlib.util.spec_from_file_location("_protein_fresh_p5_selector", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEED = SALT
    return module


P5 = load_selector()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def project(entry):
    seqinfo = entry.get("sequence")
    seq = seqinfo.get("value") if isinstance(seqinfo, dict) else None
    refs = entry.get("uniProtKBCrossReferences", [])
    refs = refs if isinstance(refs, list) else []
    return {"id": entry["primaryAccession"], "sequence": seq if isinstance(seq, str) else None,
            "pfam": sorted({x["id"] for x in refs if isinstance(x, dict) and x.get("database") == "Pfam"
                            and isinstance(x.get("id"), str) and x["id"]})}


def sequence_eligibility(entry):
    seqinfo = entry.get("sequence")
    seq = seqinfo.get("value") if isinstance(seqinfo, dict) else None
    reasons = []
    if not isinstance(seq, str) or not seq:
        reasons.append("missing_sequence")
    else:
        if not 257 <= len(seq) <= 512:
            reasons.append("length_outside_257_512")
        if set(seq) - AA:
            reasons.append("nonstandard_amino_acid")
        if entry["sequence"].get("length") != len(seq):
            reasons.append("source_sequence_length_mismatch")
    if entry.get("entryType") != "UniProtKB reviewed (Swiss-Prot)":
        reasons.append("source_not_reviewed")
    return reasons


def metadata_rows():
    return [json.loads(s) for s in (META / "metadata_projection.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]


def old_sequence_index(old):
    """One necessary old sequence-key augmentation; no old graph rebuild/hash pass."""
    destination = OUT / "OLD_EXACT_SEQUENCE_INDEX.json"
    if destination.exists():
        raise RuntimeError("Old sequence projection already exists; reuse it rather than reread old raw")
    by_id = {accession: i for i, comp in enumerate(old["components"]) for accession in comp["accessions"]}
    first, count, sources = {}, 0, []
    for source in old["sources"]:
        name = source["path"]
        if name != "data/proteins.jsonl" and "/raw/" not in name:
            continue
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.stat().st_size != source["current_bytes"]:
            raise SourceFailure("OLD_INPUT_DRIFT", "Old source path/size differs from accepted metadata graph")
        if path.stat().st_mtime_ns != source["current_mtime_ns"]:
            raise SourceFailure("OLD_INPUT_DRIFT", "Old source mtime differs from accepted metadata graph")
        if name == "data/proteins.jsonl":
            rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
            nodes = [{"id": r["id"], "sequence": r["sequence"], "pfam": r["pfam"]} for r in rows]
        else:
            rows = read_json(path)["results"]
            nodes = [project(r) for r in rows]
        if len(nodes) != source["records"]:
            raise SourceFailure("OLD_INPUT_DRIFT", "Old source raw count differs from accepted receipt")
        for node in nodes:
            group = by_id[node["id"]]
            if not isinstance(node["sequence"], str) or not node["sequence"]:
                raise SourceFailure("OLD_INPUT_DRIFT", "Missing old sequence needed for exact-sequence projection")
            if not set(node["pfam"]).issubset(old["components"][group]["pfam"]):
                raise SourceFailure("OLD_INPUT_DRIFT", "Old Pfam projection differs from existing supernode")
            key = digest(node["sequence"])
            if key in first and first[key] != group:
                raise SourceFailure("OLD_INPUT_DRIFT", "Existing old supernodes disagree on exact sequence")
            first[key] = group
        count += len(nodes)
        sources.append({"path": name, "records": len(nodes), "old_full_file_SHA_recomputed": False})
    if count != 16896 or len(first) != old["old_unique_exact_sequences"] or len(set(first.values())) != len(old["components"]):
        raise SourceFailure("OLD_INPUT_DRIFT", "Old raw projection does not total 16896")
    result = {"created_utc": utc(), "raw_records_read_once": count,
              "sequence_sha256_to_old_component": first, "sources_reused": sources,
              "existing_supernodes_reused": True, "old_full_file_SHA_recomputations": 0,
              "only_new_purpose": "Exact-sequence attachment key omitted from metadata-only supernodes"}
    write_json(destination, result, exclusive=True)
    return result


def full_union(old, seq_index, nodes):
    old_n = len(old["components"])
    graph = UnionFind(old_n + len(nodes))
    first_id, first_pf = {}, {}
    first_seq = dict(seq_index)
    for i, comp in enumerate(old["components"]):
        for accession in comp["accessions"]:
            first_id[accession] = i
        for pfam in comp["pfam"]:
            first_pf[pfam] = i
    for offset, node in enumerate(nodes):
        index = old_n + offset
        keys = (([node["id"]], first_id), (node["pfam"], first_pf),
                ([digest(node["sequence"])] if node.get("sequence") else [], first_seq))
        for values, lookup in keys:
            for value in values:
                if value in lookup:
                    graph.union(index, lookup[value])
                else:
                    lookup[value] = index
    old_by_root, new_by_root = defaultdict(list), defaultdict(list)
    for i in range(old_n):
        old_by_root[graph.find(i)].append(i)
    for i in range(len(nodes)):
        new_by_root[graph.find(old_n + i)].append(i)
    components, join = [], {}
    for root, indices in new_by_root.items():
        old_indices = old_by_root.get(root, [])
        ids = sorted({nodes[i]["id"] for i in indices})
        all_ids = sorted(set(ids).union(*(old["components"][i]["accessions"] for i in old_indices)))
        key = "acc:" + ",".join(all_ids)
        component = {"component_key": key, "component_sha256": digest(f"pilot5a-component:{key}:{SALT}"),
                     "new_member_accessions": ids, "new_raw_indices": indices,
                     "old_component_indices": old_indices, "connected_to_old_pool": bool(old_indices),
                     "known_group_id": (f"old-union-{min(old_indices) + 1:06d}" if old_indices else "fresh-" + digest(key)),
                     "pfam": sorted({p for i in indices for p in nodes[i]["pfam"]}.union(
                         *(old["components"][i]["pfam"] for i in old_indices)))}
        components.append(component)
        for index in indices:
            join[index] = component
    mapping = []
    for pair in old["selected_pair_groups"]:
        root = graph.find(pair["old_component_index"])
        mapping.append({**pair, "current_known_union_group": f"old-union-{min(old_by_root[root]) + 1:06d}"})
    if len(join) != len(nodes):
        raise SourceFailure("GRAPH_JOIN_FAILED", "Not every full raw record has exactly one component")
    return sorted(components, key=lambda c: c["component_key"]), join, mapping


def groups_unchanged(baseline, current):
    old = {(r["origin"], r["id"]): r["current_known_union_group"] for r in baseline}
    new = {(r["origin"], r["id"]): r["current_known_union_group"] for r in current}
    if set(old) != set(new) or len(baseline) != len(old) or len(current) != len(new):
        raise SourceFailure("GRAPH_JOIN_FAILED", "Old selected pair group join is incomplete or duplicated")
    # The current known partition must not split; renamed canonical roots count as
    # changes too, because hash folds depend on the exact group identity.
    return old == new


def validate_url(url, initial=False):
    parsed = urlsplit(url)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    params = dict(pairs)
    required = {"query", "fields", "sort", "format", "size"} | (set() if initial else {"cursor"})
    if (parsed.scheme != "https" or parsed.netloc != "rest.uniprot.org" or parsed.path != "/uniprotkb/search"
            or parsed.fragment or parsed.username or parsed.password or set(params) != required
            or len(params) != len(pairs)):
        raise SourceFailure("SOURCE_URL_DRIFT", "Only the frozen public UniProt search URL is allowed")
    expected = {"query": QUERY, "fields": FIELDS, "sort": "accession asc", "format": "json", "size": "500"}
    if any(params.get(k) != v for k, v in expected.items()) or (not initial and not params.get("cursor")):
        raise SourceFailure("SOURCE_URL_DRIFT", "Source link changed query, fields, order, size or cursor")
    return url


def initial_url():
    return ENDPOINT + "?" + urlencode({"query": QUERY, "fields": FIELDS, "sort": "accession asc", "format": "json", "size": 500})


def next_url(link):
    found = re.findall(r'<([^>]+)>;\s*rel="next"', link or "")
    if len(found) > 1 or (not found and re.search(r"\bnext\b", link or "")):
        raise SourceFailure("SOURCE_URL_DRIFT", "Malformed or multiple next links")
    return validate_url(found[0]) if found else None


def check_page(rows, headers, expected_ids, offset):
    if [headers.get("x-uniprot-release"), headers.get("x-uniprot-release-date")] != RELEASE:
        raise SourceFailure("SOURCE_RELEASE_DRIFT", "Source release/date differs from the metadata snapshot")
    if headers.get("x-total-results") != str(TOTAL):
        raise SourceFailure("SOURCE_TOTAL_DRIFT", "Official total differs from frozen metadata query")
    if not headers.get("content-type", "").lower().startswith("application/json"):
        raise SourceFailure("SOURCE_SCHEMA_DRIFT", "Source is not JSON")
    if not isinstance(rows, list) or len(rows) != min(500, TOTAL - offset):
        raise SourceFailure("SOURCE_COUNT_DRIFT", "Full page record count differs from fixed traversal")
    ids = [row.get("primaryAccession") if isinstance(row, dict) else None for row in rows]
    if ids != expected_ids[offset:offset + len(rows)]:
        raise SourceFailure("SOURCE_IDS_DRIFT", "Source IDs/order differ from the complete metadata snapshot")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def is_transient(exc):
    reason = exc.reason if isinstance(exc, URLError) else exc
    return isinstance(reason, (TimeoutError, socket.timeout, ConnectionError, socket.gaierror)) or (
        isinstance(reason, OSError) and reason.errno in {errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
                                                        errno.EHOSTUNREACH, errno.ENETUNREACH})


def fetch_page(opener, url, number, receipt):
    for attempt in range(1, ATTEMPTS + 1):
        item = {"page": number, "attempt": attempt, "request_url": url, "started_utc": utc()}
        receipt["attempts"].append(item)
        receipt["inflight_page"] = number
        write_json(OUT / "http_provenance.json", receipt)
        try:
            with opener.open(Request(url, headers={"Accept": "application/json", "User-Agent": "BoundedProteinFreshSource/20261017"}), timeout=45) as response:
                body, status, actual = response.read(), response.status, response.geturl()
                headers = {key: response.headers.get(key) for key in ("content-type", "date", "x-uniprot-release",
                          "x-uniprot-release-date", "x-total-results")}
                link = response.headers.get("link")
            path = OUT / "raw" / f"page{number:02d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(body)
            page = {"page": number, "request_url": url, "http_status": status, "headers": headers,
                    "raw_path": str(path.relative_to(ROOT)), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            receipt["pages"].append(page)
            item.update(outcome="received", http_status=status)
            receipt.pop("inflight_page", None)
            write_json(OUT / "http_provenance.json", receipt)
            if status != 200 or actual != url:
                raise SourceFailure("SOURCE_URL_DRIFT", "Unexpected response status or URL")
            following = next_url(link)
            page["official_next_url"] = following
            write_json(OUT / "http_provenance.json", receipt)
            return json.loads(body)["results"], headers, following
        except HTTPError as exc:
            exc.read()  # Never retain arbitrary proxy/error bodies or secret-bearing text.
            item.update(outcome="HTTP_ERROR", http_status=exc.code)
            retry = exc.code in {429, 500, 502, 503, 504}
        except (URLError, OSError, TimeoutError) as exc:
            item.update(outcome="NETWORK_ERROR", exception_class=type(exc).__name__)
            retry = is_transient(exc)
        write_json(OUT / "http_provenance.json", receipt)
        if not retry or attempt == ATTEMPTS:
            raise SourceFailure("NETWORK_OR_HTTP_STOP", "Frozen URL failed; no automatic resume or source change")
        time.sleep(min(attempt, 2))
    raise SourceFailure("NETWORK_OR_HTTP_STOP", "Attempt budget exhausted")


def freeze_selected(representatives):
    ordered = sorted(representatives, key=lambda r: (r["component_sha256"], r["component_key"]))
    if len(ordered) < TARGET:
        return [], ordered
    selected = []
    for row in ordered[:TARGET]:
        pair = row["pair"]
        a, b = sorted((pair["h_site"], pair["s_site"]), key=lambda s: s["q_0based"])
        selected.append({"id": row["id"], "sequence": row["sequence"], "pfam": row["pfam"],
                         "component_key": row["component_key"], "component_sha256": row["component_sha256"],
                         "known_group_id": row["known_group_id"], "protein_sha256": row["protein_sha256"],
                         "no_Pfam": not row["pfam"], "split": "T_confirm_long", "pair_order": "canonical_q_ascending", "a_site": a, "b_site": b,
                         "y_a": int(a["label"] == "HELIX"), "y_b": int(b["label"] == "HELIX"),
                         **{key: value for key, value in pair.items() if key != "id"}})
    return selected, ordered[TARGET:]


def qualify(entries, nodes, components, join):
    stages = Counter({key: 0 for key in ("old_pool_connected_raw_excluded", "old_disconnected_raw_records",
        "standard_reviewed_long_sequence_records", "with_PDB_crossreference", "eligible_pair_protein_records",
        "eligible_final_fresh_components", "selected_components")})
    stages["raw_records"] = len(entries)
    reasons, evidence_reasons, site_reasons, feature_reasons = Counter(), Counter(), Counter(), Counter()
    by_component, audits, features, site_rows = defaultdict(list), [], [], []
    legal_pairs = 0
    for i, (entry, node) in enumerate(zip(entries, nodes)):
        component = join[i]
        why = []
        counts = None
        if component["connected_to_old_pool"]:
            stages["old_pool_connected_raw_excluded"] += 1
            why.append("component_connected_to_old_pool")
        else:
            stages["old_disconnected_raw_records"] += 1
            why.extend(sequence_eligibility(entry))
            if not why:
                stages["standard_reviewed_long_sequence_records"] += 1
                if not any(x.get("database") == "PDB" for x in entry.get("uniProtKBCrossReferences", [])):
                    why.append("no_PDB_crossreference")
                else:
                    stages["with_PDB_crossreference"] += 1
                    pair, sites, feature_audit, counts = P5.labels_and_pair(entry)
                    features.extend({"raw_index": i, **r} for r in feature_audit)
                    for feature in feature_audit:
                        feature_reasons.update(feature["exclusion_reasons"])
                    for label in sites:
                        site_rows.extend({"id": node["id"], "raw_index": i, **r} for r in sites[label])
                    legal_pairs += counts["legal_pair_count"]
                    evidence_reasons.update(counts["evidence_exclusion_reason_counts_nonexclusive"])
                    site_reasons.update(counts["site_exclusion_reason_counts_nonexclusive"])
                    if not sites["HELIX"]:
                        why.append("no_eligible_HELIX_A_site")
                    if not sites["STRAND"]:
                        why.append("no_eligible_STRAND_A_site")
                    if pair is None:
                        why.append("no_legal_pair")
                    if not why:
                        stages["eligible_pair_protein_records"] += 1
                        by_component[component["component_key"]].append({**node, "pair": pair,
                            "component_key": component["component_key"], "known_group_id": component["known_group_id"],
                            "component_sha256": component["component_sha256"],
                            "protein_sha256": digest(f"pilot5a-protein:{node['id']}:{SALT}")})
        reasons.update(why)
        audits.append({"raw_index": i, "id": node["id"], "length": len(node["sequence"]) if node["sequence"] else None,
                       "component_key": component["component_key"], "connected_to_old_pool": component["connected_to_old_pool"],
                       "exclusion_reasons": why, "label_counts": counts})
    representatives = [min(rows, key=lambda r: (r["protein_sha256"], r["id"])) for rows in by_component.values()]
    selected, unused = freeze_selected(representatives)
    stages.update(eligible_final_fresh_components=len(representatives), selected_components=len(selected))
    write_jsonl(OUT / "candidate_filter_audit.jsonl", audits)
    write_jsonl(OUT / "feature_evidence_audit.jsonl", features)
    write_jsonl(OUT / "eligible_site_audit.jsonl", site_rows)
    write_jsonl(OUT / "unused_component_inventory.jsonl", [{key: row[key] for key in
                  ("id", "component_key", "component_sha256", "known_group_id")} for row in unused])
    enough = len(representatives) >= TARGET
    report = {"status": "FRESH_DATA_PASS_NO_MODEL_OUTPUT" if enough else "FRESH_DATA_INSUFFICIENT",
              "created_utc": utc(), "target": TARGET, "source_complete": True, "raw_records": len(entries),
              "stage_survivors": dict(stages), "raw_no_Pfam": sum(not n["pfam"] for n in nodes),
              "candidate_legal_pair_combinations": legal_pairs, "eligible_final_fresh_components": len(representatives),
              "selected_components": len(selected), "unused_eligible_components": len(unused),
              "raw_sequence_lengths": dict(Counter(str(len(n["sequence"])) if n["sequence"] else "missing" for n in nodes)),
              "selected_sequence_lengths": dict(Counter(str(len(r["sequence"])) for r in selected)),
              "record_exclusion_counts_nonexclusive": dict(reasons), "evidence_exclusion_counts_nonexclusive": dict(evidence_reasons),
              "feature_exclusion_counts_nonexclusive": dict(feature_reasons),
              "site_exclusion_counts_nonexclusive": dict(site_reasons), "raw_group_join_count": len(join),
              "D_groups_unchanged": True, "D_precision_target_used": TARGET,
              "new_model_loads": 0, "new_model_forwards": 0, "new_predictions": 0,
              "confirmation_p_values": 0, "limitations": "Known ID/exact-sequence/Pfam isolation only; no remote-homology/pretraining independence guarantee"}
    if enough:
        if len({r["known_group_id"] for r in selected}) != TARGET or any(not 257 <= len(r["sequence"]) <= 512 for r in selected):
            raise SourceFailure("SELECTION_INTEGRITY_ERROR", "Selected final group count/length failed")
        write_jsonl(OUT / "proteins.jsonl", selected)
        cases, pairs = [], []
        for row in selected:
            start = len(cases)
            for site in (row["a_site"], row["b_site"]):
                q, label = site["q_0based"], site["label"]
                cases.append({"case_id": f"{row['id']}:{label}:{q}", "id": row["id"], "sequence": row["sequence"],
                              "component": row["component_key"], "known_group_id": row["known_group_id"],
                              "q": q, "token_q_0based": q + 1, "label": label, "y": float(label == "HELIX")})
            pairs.append({"id": row["id"], "group": row["known_group_id"], "a_index": start,
                          "b_index": start + 1, "y_a": row["y_a"], "pair_key": row["pair_key"]})
        write_json(OUT / "SEALED_FRESH_CASES.json", {"status": "INPUTS_SEALED_NO_MODEL_OUTPUT", "created_utc": utc(),
                    "cases": cases, "pairs": pairs, "selected_proteins_sha256": file_sha(OUT / "proteins.jsonl"),
                    "T_OUTPUT_authorized": False}, exclusive=True)
        report["sealed_cases_sha256"] = file_sha(OUT / "SEALED_FRESH_CASES.json")
    write_json(OUT / "FRESH_DATA_RESULT.json", report, exclusive=True)
    return report


def bindings():
    return {"script": Path(__file__), "protocol": PROTOCOL, "test": Path(__file__).with_name("test_protein_fresh_data.py"),
            "readonly_P5_selector": SELECTOR, "metadata_projection": META / "metadata_projection.jsonl",
            "metadata_result": META / "METADATA_GATE_RESULT.json", "old_graph": META / "OLD_POOL_GRAPH.json",
            "current_D_groups": META / "OLD_SELECTED_UNION_GROUPS.json", "precision_plan": PRECISION,
            "old_exactseq_index": OUT / "OLD_EXACT_SEQUENCE_INDEX.json", "selftest": OUT / "SYNTHETIC_CHECKS.json"}


def prepare():
    if ((OUT / "RUN_STARTED.json").exists() or (OUT / "http_provenance.json").exists()
            or ((OUT / "raw").exists() and any((OUT / "raw").iterdir()))):
        raise RuntimeError("Source run already started; no automatic preparation/resume")
    result = read_json(META / "METADATA_GATE_RESULT.json")
    rows = metadata_rows()
    ids = [r["id"] for r in rows]
    if not result.get("complete") or result["actual_raw_response_records"] != TOTAL or len(ids) != TOTAL or ids != sorted(set(ids)):
        raise SourceFailure("METADATA_INPUT_DRIFT", "Expected complete ordered 13658-ID metadata snapshot")
    precision = read_json(PRECISION)
    if (precision["status"] != "PRECISION_TARGET_FEASIBLE_PENDING_FRESH_SOURCE"
            or any(precision[key] != TARGET for key in ("N_req", "authorized_target_within_cap", "requested_target"))
            or precision["maximum_target"] != 1536 or precision["pair_count"] != 432 or precision["group_count"] != 421):
        raise SourceFailure("PRECISION_INPUT_DRIFT", "Only the frozen 1304-group precision target is authorized")
    old = read_json(META / "OLD_POOL_GRAPH.json")
    baseline = read_json(META / "OLD_SELECTED_UNION_GROUPS.json")["selected_pair_group_mapping"]
    if (old["old_raw_records"] != 16896 or len(old["components"]) != 3967
            or len(old["selected_pair_groups"]) != 432 or len(baseline) != 432
            or len({(r["origin"], r["id"]) for r in baseline}) != 432
            or len({r["current_known_union_group"] for r in baseline}) != 421
            or {(r["origin"], r["id"], r["old_component_index"]) for r in baseline}
            != {(r["origin"], r["id"], r["old_component_index"]) for r in old["selected_pair_groups"]}):
        raise SourceFailure("D_GROUP_INPUT_DRIFT", "Expected exact 432-pair/421-current-group D join and 3967 old supernodes")
    if not (OUT / "OLD_EXACT_SEQUENCE_INDEX.json").exists():
        old_sequence_index(old)
    check = read_json(OUT / "SYNTHETIC_CHECKS.json")
    if (check["status"] != "PASS" or check["script_sha256"] != file_sha(__file__)
            or check["test_sha256"] != file_sha(Path(__file__).with_name("test_protein_fresh_data.py"))
            or check["readonly_P5_selector_sha256"] != file_sha(SELECTOR)):
        raise SourceFailure("CHECK_REQUIRED", "Current finite synthetic checks must pass")
    ready = {"status": "FRESH_SOURCE_READY", "created_utc": utc(), "config": CONFIG,
             "bindings": {key: {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)} for key, path in bindings().items()},
             "metadata_accession_count": len(ids), "new_HTTP_requests": 0, "new_labels_downloaded": 0,
             "D_pair_count": len(baseline), "D_known_group_count": 421,
             "old_sequence_projection_read_once": 16896, "old_full_file_SHA_recomputations": 0,
             "awaiting": "root-owned SOURCE_DATA_GO lock bound to this READY before any request"}
    write_json(OUT / "FRESH_SOURCE_READY.json", ready)
    print(json.dumps({"status": ready["status"], "ready_sha256": file_sha(OUT / "FRESH_SOURCE_READY.json"), "target": TARGET}))


def require_lock(path):
    lock, ready = read_json(path), read_json(OUT / "FRESH_SOURCE_READY.json")
    if (lock.get("status") != "SOURCE_DATA_GO" or lock.get("source_fetch_authorized") is not True
            or lock.get("model_forward_authorized") is not False or lock.get("T_output_authorized") is not False
            or lock.get("ready_sha256") != file_sha(OUT / "FRESH_SOURCE_READY.json")):
        raise SourceFailure("LOCK_REQUIRED", "Explicit root-owned source-only data lock required")
    if ready["config"] != CONFIG:
        raise SourceFailure("LOCK_DRIFT", "Ready configuration changed")
    try:
        locked_at = datetime.fromisoformat(lock["created_utc"])
        ready_at = datetime.fromisoformat(ready["created_utc"])
        if locked_at.utcoffset() is None or ready_at.utcoffset() is None or locked_at < ready_at:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise SourceFailure("LOCK_TIMING_ERROR", "Source lock must have a timezone-aware timestamp after READY") from None
    for key, binding in ready["bindings"].items():
        if file_sha(ROOT / binding["path"]) != binding["sha256"]:
            raise SourceFailure("LOCK_DRIFT", f"Bound source input changed: {key}")
    return lock


def run(lock_path):
    require_lock(lock_path)
    write_json(OUT / "RUN_STARTED.json", {"created_utc": utc(), "lock_sha256": file_sha(lock_path),
               "T_output_authorized": False}, exclusive=True)
    receipt = {"configuration": CONFIG, "started_utc": utc(), "complete": False,
               "pages": [], "attempts": [], "raw_records": 0, "terminal_failure": None}
    write_json(OUT / "http_provenance.json", receipt)
    try:
        expected_ids = [r["id"] for r in metadata_rows()]
        entries, url, used_urls = [], validate_url(initial_url(), initial=True), set()
        opener = build_opener(NoRedirect())  # Inherited networking; never inspect/change proxy/VPN state.
        for page in range(1, PAGES + 1):
            if url in used_urls:
                raise SourceFailure("SOURCE_URL_DRIFT", "Repeated official pagination URL")
            used_urls.add(url)
            rows, headers, following = fetch_page(opener, url, page, receipt)
            check_page(rows, headers, expected_ids, len(entries))
            entries.extend(rows)
            receipt["raw_records"] = len(entries)
            write_json(OUT / "http_provenance.json", receipt)
            print(json.dumps({"source_page": page, "raw_records": len(entries), "expected_total": TOTAL}), flush=True)
            if page < PAGES and following is None:
                raise SourceFailure("SOURCE_COUNT_DRIFT", "Official next ended before complete source")
            if page == PAGES and following is not None:
                raise SourceFailure("SOURCE_COUNT_DRIFT", "Official next continues beyond frozen source total")
            url = following
        if len(entries) != TOTAL or [e["primaryAccession"] for e in entries] != expected_ids:
            raise SourceFailure("SOURCE_IDS_DRIFT", "Full source ID inventory differs")
        receipt.update(complete=True, completed_utc=utc())
        write_json(OUT / "http_provenance.json", receipt)
        old = read_json(META / "OLD_POOL_GRAPH.json")
        seq = read_json(OUT / "OLD_EXACT_SEQUENCE_INDEX.json")["sequence_sha256_to_old_component"]
        nodes = [project(entry) for entry in entries]
        components, join, mapping = full_union(old, seq, nodes)
        baseline = read_json(META / "OLD_SELECTED_UNION_GROUPS.json")["selected_pair_group_mapping"]
        unchanged = groups_unchanged(baseline, mapping)
        write_json(OUT / "OLD_SELECTED_FINAL_GROUPS.json", {"selected_pair_group_mapping": mapping,
                   "old_selected_pair_count": len(mapping), "final_group_count": len({r["current_known_union_group"] for r in mapping}),
                   "D_groups_unchanged": unchanged, "grouping_keys": ["ID", "exact-sequence", "Pfam"]}, exclusive=True)
        write_jsonl(OUT / "components.jsonl", components)
        write_jsonl(OUT / "raw_group_join.jsonl", [{"raw_index": i, "id": n["id"], "known_group_id": join[i]["known_group_id"],
                    "connected_to_old_pool": join[i]["connected_to_old_pool"], "component_key": join[i]["component_key"]} for i, n in enumerate(nodes)])
        graph_report = {"status": "GRAPH_PASS_D_GROUPS_UNCHANGED" if unchanged else "D_GROUPS_CHANGED_QUALIFICATION_HELD",
                        "raw_records": len(entries), "raw_group_join_count": len(join), "old_raw_count_reused": 16896,
                        "all_raw_used_as_bridges_before_eligibility": True, "D_group_count": len({r["current_known_union_group"] for r in mapping}),
                        "current_D_group_count": len({r["current_known_union_group"] for r in baseline}),
                        "new_components_disconnected_old": sum(not c["connected_to_old_pool"] for c in components),
                        "new_raw_connected_old": sum(join[i]["connected_to_old_pool"] for i in range(len(nodes))),
                        "D_groups_unchanged": unchanged, "new_model_forwards": 0, "new_predictions": 0}
        write_json(OUT / "GRAPH_RESULT.json", graph_report, exclusive=True)
        if not unchanged:
            print(json.dumps(graph_report))
            return
        report = qualify(entries, nodes, components, join)
        print(json.dumps({key: report[key] for key in ("status", "target", "raw_records", "candidate_legal_pair_combinations", "eligible_final_fresh_components", "selected_components")}))
    except Exception as exc:
        status = exc.status if isinstance(exc, SourceFailure) else "SOURCE_OR_BUILD_ERROR"
        receipt.update(terminal_failure={"status": status, "exception_class": type(exc).__name__}, stopped_utc=utc())
        write_json(OUT / "http_provenance.json", receipt)
        write_json(OUT / "STOP_RECEIPT.json", {"status": status, "created_utc": utc(),
                   "exception_class": type(exc).__name__, "source_complete": receipt["complete"],
                   "valid_insufficiency_conclusion": False, "new_model_forwards": 0,
                   "next_action": "Root review; no automatic restart, new query, source expansion or partial-data claim"}, exclusive=True)
        print(json.dumps({"status": status, "exception_class": type(exc).__name__, "partial_data_not_usable": True}))
        raise SystemExit(1) from None


def selftest():
    import unittest
    import test_protein_fresh_data
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_protein_fresh_data))
    write_json(OUT / "SYNTHETIC_CHECKS.json", {"status": "PASS" if result.wasSuccessful() else "FAIL", "created_utc": utc(),
               "tests_run": result.testsRun, "script_sha256": file_sha(__file__), "readonly_P5_selector_sha256": file_sha(SELECTOR),
               "test_sha256": file_sha(Path(__file__).with_name("test_protein_fresh_data.py")),
               "new_HTTP_requests": 0, "model_forwards": 0, "scope": "Finite graph/identity/length/P5 selector/selection budget checks only"})
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selftest", "prepare", "run"))
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "prepare":
        prepare()
    elif args.lock is None:
        parser.error("run requires --lock <root-owned source lock>")
    else:
        run(args.lock)


if __name__ == "__main__":
    main()
