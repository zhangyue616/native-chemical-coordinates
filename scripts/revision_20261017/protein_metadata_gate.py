"""One bounded, metadata-only UniProt feasibility gate; no model dependencies.

prepare is offline.  fetch requires the root-owned DATA_RULE_LOCK and is not
resumable.  Neither action chooses a confirmation cohort or reads new labels.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/revision_20261017/protein_metadata"
PROTOCOL = ROOT / "docs/revision_20261017/PROTEIN_METADATA_PROTOCOL.md"
SCRIPT = Path(__file__).resolve()
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
QUERY = "(reviewed:true) AND (length:[257 TO 512]) AND (database:pdb)"
FIELDS = "accession,length,xref_pfam"
SORT = "accession asc"
RELEASE = ["2026_03", "02-September-2026"]
MAX_RAW = 16384
MAX_PAGES = 33
MAX_ATTEMPTS = 3
PAGE_SIZE = 500
OLD_COUNT = 16896
ACC = re.compile(r"[A-Z0-9]{6}(?:[A-Z0-9]{4})?\Z")
PFAM = re.compile(r"PF[0-9]{5}\Z")
SAFE_HEADERS = {
    "date", "content-type", "cache-control", "x-total-results",
    "x-uniprot-release", "x-uniprot-release-date", "x-api-deployment-date",
}
CONFIG = {
    "endpoint": ENDPOINT, "query": QUERY, "fields": FIELDS, "sort": SORT,
    "format": "json", "expected_release": RELEASE,
    "maximum_raw_response_records": MAX_RAW,
    "maximum_successful_pages": MAX_PAGES,
    "initial_page_size": PAGE_SIZE, "maximum_attempts_per_url": MAX_ATTEMPTS,
    "http_timeout_seconds": 45,
    "old_pool_raw_records": OLD_COUNT,
    "new_graph_keys": ["accession", "Pfam"],
    "old_graph_keys": ["accession", "exact_sequence", "Pfam"],
    "thresholds_are_upper_bound_checks_only": [512, 1536],
    "new_sequence_values_requested": False,
    "new_structural_labels_requested": False,
    "new_pdb_crossreference_or_mapping_requested": False,
    "automatic_resume_or_requery": False,
}


class GateFailure(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code, self.detail = code, detail


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path):
    return path.resolve().relative_to(ROOT).as_posix()


def project_path(name):
    path = (ROOT / name).resolve()
    if ROOT != path and ROOT not in path.parents:
        raise GateFailure("OLD_POOL_ERROR", "Old source path escapes project")
    return path


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


def compress_old(nodes):
    """Full old ID/exact-sequence/Pfam graph, then lossless supernodes."""
    graph = UnionFind(len(nodes))
    first = {"id": {}, "sequence": {}, "pfam": {}}
    for i, node in enumerate(nodes):
        for kind, values in (("id", [node["id"]]), ("sequence", [node["sequence"]]),
                             ("pfam", node["pfam"])):
            for value in values:
                if value in first[kind]:
                    graph.union(i, first[kind][value])
                else:
                    first[kind][value] = i
    groups = defaultdict(list)
    for i in range(len(nodes)):
        groups[graph.find(i)].append(i)
    components = []
    for members in groups.values():
        components.append({
            "raw_count": len(members),
            "accessions": sorted({nodes[i]["id"] for i in members}),
            "pfam": sorted({p for i in members for p in nodes[i]["pfam"]}),
        })
    components.sort(key=lambda x: x["accessions"])
    return {
        "old_raw_records": len(nodes),
        "old_unique_accessions": len(first["id"]),
        "old_unique_exact_sequences": len(first["sequence"]),
        "old_unique_pfam_ids": len(first["pfam"]),
        "old_raw_without_pfam": sum(not n["pfam"] for n in nodes),
        "old_graph_components_pre_new": len(components),
        "components": components,
    }


def old_node(entry, legacy=False):
    try:
        accession = entry["id"] if legacy else entry["primaryAccession"]
        sequence = entry["sequence"] if legacy else entry["sequence"]["value"]
        if legacy:
            pfam = entry["pfam"]
        else:
            pfam = [x["id"] for x in entry.get("uniProtKBCrossReferences", [])
                    if x.get("database") == "Pfam"]
        if not isinstance(accession, str) or not accession:
            raise ValueError()
        if not isinstance(sequence, str) or not sequence:
            raise ValueError()
        if not isinstance(pfam, list) or any(not isinstance(x, str) or not x for x in pfam):
            raise ValueError()
        return {"id": accession, "sequence": sequence, "pfam": sorted(set(pfam))}
    except (KeyError, TypeError, ValueError, AttributeError):
        raise GateFailure("OLD_POOL_ERROR", "Old ID/sequence/Pfam projection has invalid schema") from None


def load_old_once():
    """Reuse previous declared identities, inspect counts/schema, no raw rehash."""
    identity_source = ROOT / "data/pilot5a/DATA_RULE_LOCK.json"
    identities = {x["path"]: x for x in read_json(identity_source)["old_pool_sources"]}
    for asset in read_json(ROOT / "data/pilot5a/asset_manifest.json")["assets"]:
        identities.setdefault(asset["path"], asset)
    sources = []

    def binding(path, records=None, declared=None, identity_path=None):
        path = path.resolve()
        existing = declared or identities.get(rel(path), {})
        if not existing.get("sha256"):
            raise GateFailure("OLD_POOL_ERROR", "Existing source identity is missing")
        st = path.stat()
        if "bytes" in existing and st.st_size != existing["bytes"]:
            raise GateFailure("OLD_POOL_ERROR", "Old source size differs from existing manifest")
        item = {
            "path": rel(path), "existing_declared_sha256": existing["sha256"],
            "identity_source": identity_path or rel(identity_source),
            "current_bytes": st.st_size, "current_mtime_ns": st.st_mtime_ns,
            "full_sha256_recomputed_this_gate": False,
        }
        if records is not None:
            item["records"] = records
        sources.append(item)

    path = ROOT / "data/proteins.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 512:
        raise GateFailure("OLD_POOL_ERROR", "Legacy old pool is not 512 raw rows")
    binding(path, 512)
    nodes = [old_node(x, legacy=True) for x in rows]
    del rows
    counts = {"pilot1": 512}
    for pilot in ("pilot2", "pilot3a", "pilot4a", "pilot5a"):
        receipt_path = ROOT / "data" / pilot / "http_provenance.json"
        receipt = read_json(receipt_path)
        binding(receipt_path, identity_path=("data/pilot5a/asset_manifest.json" if pilot == "pilot5a" else None))
        count = 0
        for page in receipt["pages"]:
            path = project_path(page["path"])
            if path.parent != (ROOT / "data" / pilot / "raw").resolve():
                raise GateFailure("OLD_POOL_ERROR", "Old raw source is outside its expected raw directory")
            raw_entries = read_json(path)["results"]
            if not isinstance(raw_entries, list) or len(raw_entries) != page["records"]:
                raise GateFailure("OLD_POOL_ERROR", "Old raw count differs from prior receipt")
            headers = {k.lower(): v for k, v in page.get("headers", {}).items()}
            if [headers.get("x-uniprot-release"), headers.get("x-uniprot-release-date")] != RELEASE:
                raise GateFailure("OLD_POOL_ERROR", "Old raw receipt is not the frozen release")
            nodes.extend(old_node(x) for x in raw_entries)
            count += len(raw_entries)
            binding(path, len(raw_entries), page, rel(receipt_path))
        if count != 4096:
            raise GateFailure("OLD_POOL_ERROR", "Each old pilot must contribute all 4096 raw rows")
        counts[pilot] = count
    if len(nodes) != OLD_COUNT:
        raise GateFailure("OLD_POOL_ERROR", "Old full raw pool does not total 16896")
    compressed = compress_old(nodes)
    old_group_by_id = {accession: i for i, component in enumerate(compressed["components"])
                       for accession in component["accessions"]}
    selected_pairs = []
    selected_counts = {}
    for pilot, expected in (("pilot3a", 176), ("pilot4a", 128), ("pilot5a", 128)):
        path = ROOT / "data" / pilot / "proteins.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected or len({row.get("id") for row in rows}) != expected:
            raise GateFailure("OLD_POOL_ERROR", "Old selected pair ID inventory differs from 176/128/128")
        for row in rows:
            accession, prior_key = row.get("id"), row.get("component_key")
            if accession not in old_group_by_id or not isinstance(prior_key, str) or not prior_key:
                raise GateFailure("OLD_POOL_ERROR", "Old selected pair has no full-raw union group or prior key")
            index = old_group_by_id[accession]
            selected_pairs.append({"origin": pilot, "id": accession,
                                   "prior_component_key": prior_key,
                                   "old_component_index": index,
                                   "old_union_group": f"old-union-{index + 1:06d}"})
        selected_counts[pilot] = len(rows)
        assets = read_json(ROOT / "data" / pilot / "asset_manifest.json")["assets"]
        declared = next((item for item in assets if item["path"] == rel(path)), None)
        binding(path, expected, declared, f"data/{pilot}/asset_manifest.json")
    compressed.update({
        "created_utc": utc(), "old_raw_by_origin": counts, "sources": sources,
        "method": "All 16896 old raw nodes joined by ID/exact-sequence/any known Pfam; connected components compressed losslessly for new ID/Pfam attachment.",
        "old_structural_label_values_used": 0,
        "old_sequence_values_used_for_old_graph_only": len(nodes),
        "old_raw_full_file_hash_recomputations": 0,
        "selected_pair_groups": selected_pairs,
        "old_selected_pair_counts_by_origin": selected_counts,
        "old_selected_pair_count": len(selected_pairs),
        "old_selected_prior_cohort_groups": len({(x["origin"], x["prior_component_key"]) for x in selected_pairs}),
        "old_selected_global_groups_pre_new": len({x["old_component_index"] for x in selected_pairs}),
    })
    return compressed


def union_summary(old, new):
    old_n = len(old["components"])
    graph = UnionFind(old_n + len(new))
    first_id, first_pf = {}, {}
    for i, comp in enumerate(old["components"]):
        for accession in comp["accessions"]:
            first_id[accession] = i
        for pf in comp["pfam"]:
            first_pf[pf] = i
    old_ids, old_pfams = set(first_id), set(first_pf)
    for offset, row in enumerate(new):
        i = old_n + offset
        for values, first in (([row["id"]] if row.get("id") else [], first_id),
                              (row["pfam"], first_pf)):
            for value in values:
                if value in first:
                    graph.union(i, first[value])
                else:
                    first[value] = i
    old_roots = {graph.find(i) for i in range(old_n)}
    new_groups = defaultdict(list)
    for offset, row in enumerate(new):
        new_groups[graph.find(old_n + offset)].append(row)
    touched = [row for root, rows in new_groups.items() if root in old_roots for row in rows]
    fresh_groups = [rows for root, rows in new_groups.items() if root not in old_roots]
    fresh = [row for rows in fresh_groups for row in rows]
    ids = [x["id"] for x in new if x.get("id")]
    pfams = {p for x in new for p in x["pfam"]}
    direct = sum(bool(x.get("id") in old_ids or old_pfams.intersection(x["pfam"])) for x in new)
    old_indices_by_root = defaultdict(list)
    for index in range(old_n):
        old_indices_by_root[graph.find(index)].append(index)
    selected_mapping = []
    for pair in old.get("selected_pair_groups", []):
        root = graph.find(pair["old_component_index"])
        selected_mapping.append({**pair, "current_known_union_group": f"old-union-{min(old_indices_by_root[root]) + 1:06d}"})
    selected_group_count = len({x["current_known_union_group"] for x in selected_mapping})
    return {
        **{k: v for k, v in old.items() if k.startswith("old_")},
        "new_raw_records_in_graph": len(new), "new_unique_accessions": len(set(ids)),
        "new_duplicate_accession_raw_rows": len(ids) - len(set(ids)),
        "new_raw_without_accession": len(new) - len(ids),
        "new_unique_pfam_ids": len(pfams),
        "new_raw_with_pfam": sum(bool(x["pfam"]) for x in new),
        "new_raw_without_pfam": sum(not x["pfam"] for x in new),
        "new_invalid_metadata_raw_rows": sum(not x["metadata_valid"] for x in new),
        "new_invalid_reasons": dict(Counter(r for x in new for r in x["invalid_reasons"])),
        "new_raw_directly_overlapping_old_id_or_pfam": direct,
        "new_raw_connected_to_old_after_full_union": len(touched),
        "new_raw_only_indirectly_connected_to_old": len(touched) - direct,
        "new_unique_ids_connected_to_old": len({x["id"] for x in touched if x.get("id")}),
        "old_components_after_new_bridges": len(old_roots),
        "new_raw_not_connected_to_old": len(fresh),
        "new_unique_ids_not_connected_to_old": len({x["id"] for x in fresh if x.get("id")}),
        "new_raw_without_pfam_not_connected_to_old": sum(not x["pfam"] for x in fresh),
        "new_components_disconnected_from_old_upper_bound": len(fresh_groups),
        "new_components_disconnected_from_old_with_valid_metadata": sum(any(x["metadata_valid"] for x in g) for g in fresh_groups),
        "upper_bound_includes_invalid_only_components": True,
        "old_selected_global_groups_after_new_metadata": selected_group_count,
        "old_selected_groups_merged_by_new_metadata": old.get("old_selected_global_groups_pre_new", 0) - selected_group_count,
        "selected_pair_group_mapping": selected_mapping,
    }


def project_new(entry, page, offset, unexpected):
    bad = []
    if not isinstance(entry, dict):
        return {"raw_index": offset, "page": page, "id": None, "length": None,
                "pfam": [], "metadata_valid": False, "invalid_reasons": ["not_object"]}
    for key in entry:
        if key not in {"entryType", "primaryAccession", "sequence", "uniProtKBCrossReferences"}:
            unexpected["entry." + key] += 1
    accession = entry.get("primaryAccession")
    if not isinstance(accession, str) or not ACC.fullmatch(accession):
        bad.append("invalid_or_missing_accession")
        accession = accession if isinstance(accession, str) and accession else None
    seq = entry.get("sequence", {})
    if isinstance(seq, dict):
        for key in seq:
            if key != "length":
                unexpected["sequence." + key] += 1
        length = seq.get("length")
    else:
        length = None
        unexpected["sequence.non_object"] += 1
    if type(length) is not int or not 257 <= length <= 512:
        bad.append("invalid_or_out_of_range_length")
        length = length if type(length) is int else None
    refs = entry.get("uniProtKBCrossReferences", [])
    pfams = set()
    if not isinstance(refs, list):
        bad.append("invalid_crossreference_container")
        refs = []
    for ref in refs:
        if not isinstance(ref, dict):
            bad.append("invalid_crossreference_object")
            continue
        if ref.get("database") != "Pfam":
            unexpected["crossreference.non_Pfam_database"] += 1
            continue
        pf = ref.get("id")
        if not isinstance(pf, str) or not PFAM.fullmatch(pf):
            bad.append("invalid_Pfam_id")
            continue
        pfams.add(pf)
        for key in ref:
            if key not in {"database", "id"}:
                unexpected["Pfam_crossreference." + key] += 1
    # No access to sequence.value, features, PDB mapping, or activations here.
    return {"raw_index": offset, "page": page, "id": accession, "length": length,
            "pfam": sorted(pfams), "metadata_valid": not bad,
            "invalid_reasons": sorted(set(bad))}


def next_size(raw_count, successful_pages):
    if raw_count >= MAX_RAW or successful_pages >= MAX_PAGES:
        return 0
    return min(PAGE_SIZE, MAX_RAW - raw_count)


def validate_url(url, require_cursor=False):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "rest.uniprot.org" or parsed.path != "/uniprotkb/search" or parsed.fragment:
        raise GateFailure("SOURCE_URL_ERROR", "URL is outside the frozen official endpoint")
    try:
        pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise GateFailure("SOURCE_URL_ERROR", "Malformed official URL parameters") from None
    q = dict(pairs)
    if len(q) != len(pairs) or set(q) - {"query", "format", "fields", "sort", "size", "cursor"}:
        raise GateFailure("SOURCE_URL_ERROR", "Unexpected or repeated URL parameters")
    fixed = {"query": QUERY, "format": "json", "fields": FIELDS, "sort": SORT}
    if any(q.get(k) != v for k, v in fixed.items()):
        raise GateFailure("SOURCE_URL_ERROR", "Official URL changed frozen query/fields/sort/format")
    if not q.get("size", "").isdigit() or not 1 <= int(q["size"]) <= PAGE_SIZE:
        raise GateFailure("SOURCE_URL_ERROR", "Invalid requested page size")
    if require_cursor and not q.get("cursor"):
        raise GateFailure("SOURCE_URL_ERROR", "Official next URL has no cursor")
    return pairs


def resized_official_next(url, size):
    pairs = validate_url(url, require_cursor=True)
    return ENDPOINT + "?" + urllib.parse.urlencode([(k, str(size) if k == "size" else v) for k, v in pairs])


def get_next(link):
    if not link:
        return None
    candidates = re.findall(r'<([^>]+)>\s*;\s*rel="?next"?', link)
    if len(candidates) != 1:
        raise GateFailure("SOURCE_URL_ERROR", "Unrecognized or repeated official next link")
    validate_url(candidates[0], require_cursor=True)
    return candidates[0]


def start_url():
    return ENDPOINT + "?" + urllib.parse.urlencode({
        "query": QUERY, "format": "json", "fields": FIELDS, "sort": SORT, "size": PAGE_SIZE,
    })


def synthetic_checks():
    def o(i, seq, pf):
        return {"id": i, "sequence": seq, "pfam": pf}

    def n(i, pf, valid=True):
        return {"id": i, "pfam": pf, "metadata_valid": valid, "invalid_reasons": []}

    old = compress_old([o("OLD001", "SAME", ["PF00001"]), o("OLD002", "SAME", ["PF00002"])])
    bridged = union_summary(old, [n("NEW001", ["PF00003"]), n("NEW001", ["PF00002"]), n("NEW002", ["PF00003"])])
    separate = union_summary(old, [n("NEW003", []), n("NEW004", []), n("NEW005", ["PF00004"]), n("NEW006", ["PF00004"]), n(None, [], False)])
    old_separate = compress_old([o("OLD001", "ONE", ["PF00001"]), o("OLD002", "TWO", ["PF00002"])])
    old_separate["old_selected_global_groups_pre_new"] = 2
    old_separate["selected_pair_groups"] = [
        {"id": "OLD001", "old_component_index": 0}, {"id": "OLD002", "old_component_index": 1}]
    selected_bridge = union_summary(old_separate, [n("NEW008", ["PF00001", "PF00002"])])
    unexpected = Counter()
    projected = project_new({"primaryAccession": "NEW007", "sequence": {"length": 300, "value": "DO_NOT_USE"},
                             "features": [{"type": "HELIX"}], "uniProtKBCrossReferences": [
                                 {"database": "Pfam", "id": "PF00005"}, {"database": "PDB", "id": "IGNORE"}]}, 1, 1, unexpected)
    sizes, total, pages = [], 0, 0
    while next_size(total, pages):
        sizes.append(next_size(total, pages))
        total += sizes[-1]
        pages += 1
    rejected = False
    try:
        validate_url(start_url().replace("accession%2Clength%2Cxref_pfam", "accession%2Csequence"))
    except GateFailure:
        rejected = True
    checks = {
        "old_exact_sequence_bridge_preserved": old["old_graph_components_pre_new"] == 1,
        "new_metadata_mergers_of_old_selected_groups_reported": selected_bridge["old_selected_groups_merged_by_new_metadata"] == 1 and selected_bridge["old_selected_global_groups_after_new_metadata"] == 1,
        "duplicate_accession_and_Pfam_transitive_old_bridge_excluded": bridged["new_raw_connected_to_old_after_full_union"] == 3 and bridged["new_components_disconnected_from_old_upper_bound"] == 0,
        "missing_Pfam_and_invalid_raw_preserved_in_upper_bound": separate["new_raw_without_pfam"] == 3 and separate["new_components_disconnected_from_old_upper_bound"] == 4,
        "new_unrequested_sequence_labels_mapping_not_used": "sequence" not in projected and "features" not in projected and projected["pfam"] == ["PF00005"] and unexpected["sequence.value"] == 1 and unexpected["entry.features"] == 1 and unexpected["crossreference.non_Pfam_database"] == 1,
        "raw_budget_exact_16384_in_33_pages_last384": total == MAX_RAW and pages == MAX_PAGES and sizes[-1] == 384,
        "all_raw_counted_without_filtering_in_budget": next_size(16383, 32) == 1,
        "query_field_changes_rejected": rejected,
        "official_next_only_size_adjusted": dict(validate_url(resized_official_next(start_url() + "&cursor=synthetic", 384), True))["size"] == "384",
        "http_requests": 0,
    }
    if not all(value for key, value in checks.items() if key != "http_requests"):
        raise GateFailure("SYNTHETIC_CHECK_FAILED", "A bounded metadata/graph synthetic check failed")
    return {"created_utc": utc(), "passed": True, "checks": checks}


def prepare():
    if (OUT / "DATA_RULE_LOCK.json").exists() or (OUT / "http_provenance.json").exists():
        raise GateFailure("ALREADY_LOCKED_OR_ATTEMPTED", "Do not modify a locked or attempted gate")
    OUT.mkdir(parents=True, exist_ok=True)
    checks = synthetic_checks()
    old = load_old_once()
    old_path = OUT / "OLD_POOL_GRAPH.json"
    write_json(old_path, old)
    pre_groups = union_summary(old, [])
    write_json(OUT / "OLD_SELECTED_UNION_GROUPS_PRE_HTTP.json", {
        "old_selected_pair_count": old["old_selected_pair_count"],
        "prior_cohort_groups": old["old_selected_prior_cohort_groups"],
        "global_known_groups": old["old_selected_global_groups_pre_new"],
        "prior_groups_merged_by_full_old_raw_union": old["old_selected_prior_cohort_groups"] - old["old_selected_global_groups_pre_new"],
        "selected_pair_group_mapping": pre_groups["selected_pair_group_mapping"],
        "structural_labels_or_model_outputs_used": 0,
    })
    write_json(OUT / "SYNTHETIC_CHECKS.json", checks)
    inputs = {
        "created_utc": utc(), "status": "FINAL_READY_HTTP_ZERO",
        "configuration": CONFIG, "script_path": rel(SCRIPT), "script_sha256": sha(SCRIPT.read_bytes()),
        "protocol_path": rel(PROTOCOL), "protocol_sha256": sha(PROTOCOL.read_bytes()),
        "old_pool_graph_path": rel(old_path), "old_pool_graph_sha256": sha(old_path.read_bytes()),
        "old_pool_source_bindings": old["sources"],
        "old_pool_summary": {k: v for k, v in old.items() if k.startswith("old_")},
        "synthetic_checks_path": rel(OUT / "SYNTHETIC_CHECKS.json"),
        "http_requests_before_lock": 0,
        "root_lock_required_fields": ["status=DATA_RULE_LOCK", "frozen_utc", "inputs_sha256", "http_requests_before_lock=0"],
        "new_T_or_scientific_route_authorized": False,
    }
    write_json(OUT / "DATA_RULE_INPUTS.json", inputs)
    print(json.dumps({"status": inputs["status"], "inputs_path": rel(OUT / "DATA_RULE_INPUTS.json"),
                      "inputs_sha256": sha((OUT / "DATA_RULE_INPUTS.json").read_bytes()),
                      "old_raw_records": old["old_raw_records"], "old_graph_components": old["old_graph_components_pre_new"],
                      "old_selected_prior_groups": old["old_selected_prior_cohort_groups"],
                      "old_selected_global_union_groups": old["old_selected_global_groups_pre_new"],
                      "synthetic_checks_passed": True, "HTTP": 0}, ensure_ascii=False))


def verify_lock():
    lock_path = OUT / "DATA_RULE_LOCK.json"
    inputs_path = OUT / "DATA_RULE_INPUTS.json"
    lock, inputs = read_json(lock_path), read_json(inputs_path)
    if lock.get("status") != "DATA_RULE_LOCK" or lock.get("http_requests_before_lock") != 0 or not lock.get("frozen_utc"):
        raise GateFailure("LOCK_ERROR", "A root-owned pre-HTTP data-rule lock is required")
    try:
        frozen = datetime.fromisoformat(lock["frozen_utc"])
        if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
            raise ValueError()
    except (TypeError, ValueError):
        raise GateFailure("LOCK_ERROR", "Lock time is invalid or in the future") from None
    if lock.get("inputs_sha256") != sha(inputs_path.read_bytes()) or inputs.get("configuration") != CONFIG:
        raise GateFailure("LOCK_ERROR", "Frozen inputs/configuration differ")
    for path, key in ((SCRIPT, "script_sha256"), (PROTOCOL, "protocol_sha256"), (OUT / "OLD_POOL_GRAPH.json", "old_pool_graph_sha256")):
        if sha(path.read_bytes()) != inputs[key]:
            raise GateFailure("LOCK_ERROR", "Prepared script/protocol/old-graph changed after lock")
    for binding in inputs["old_pool_source_bindings"]:
        st = project_path(binding["path"]).stat()
        if st.st_size != binding["current_bytes"] or st.st_mtime_ns != binding["current_mtime_ns"]:
            raise GateFailure("OLD_POOL_DRIFT", "Old source size or modification time changed after preparation")
    return lock, sha(lock_path.read_bytes()), read_json(OUT / "OLD_POOL_GRAPH.json")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def safe_headers(headers):
    return {k.lower(): v for k, v in headers.items() if k.lower() in SAFE_HEADERS}


def transient_network_error(exc):
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, (TimeoutError, socket.timeout, ConnectionError, socket.gaierror)) or (
        isinstance(reason, OSError) and reason.errno in {
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED, errno.EHOSTUNREACH, errno.ENETUNREACH,
        })


def fetch_page(opener, url, page_number, receipt):
    attempts = []
    for number in range(1, MAX_ATTEMPTS + 1):
        attempt = {"attempt": number, "started_utc": utc()}
        attempts.append(attempt)
        receipt["http_attempts"] += 1
        receipt["inflight"] = {"page": page_number, "request_url": url, "attempts": attempts}
        write_json(OUT / "http_provenance.json", receipt)
        started = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "BoundedProteinMetadataGate/20261017"}, method="GET")
            with opener.open(req, timeout=CONFIG["http_timeout_seconds"]) as response:
                body = response.read()
                status = response.status
                response_url = response.geturl()
                headers = safe_headers(response.headers)
                link = response.headers.get("Link")
            attempt.update({"outcome": "received", "http_status": status, "elapsed_seconds": round(time.monotonic() - started, 6)})
            raw_path = OUT / "raw" / f"page{page_number:02d}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(body)
            page = {"page": page_number, "request_url": url, "request_started_utc": attempt["started_utc"],
                    "http_status": status, "retrieved_utc": utc(), "headers": headers,
                    "raw_path": rel(raw_path), "raw_sha256": sha(body), "bytes": len(body),
                    "records": None, "attempts": attempts}
            receipt["pages"].append(page)
            receipt["successful_http_pages"] += 1
            receipt.pop("inflight", None)
            write_json(OUT / "http_provenance.json", receipt)
            validate_url(response_url)
            if response_url != url:
                raise GateFailure("SOURCE_URL_ERROR", "Unexpected response URL change")
            page["response_url"] = response_url
            return body, page, link
        except urllib.error.HTTPError as exc:
            # HTTP bodies are retained without copying arbitrary error text or proxy credentials.
            error_body = exc.read()
            error_path = OUT / "raw" / f"page{page_number:02d}_attempt{number:02d}_http_error.bin"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_bytes(error_body)
            attempt.update({"outcome": "http_error", "http_status": exc.code,
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                            "error_body_path": rel(error_path), "error_body_sha256": sha(error_body),
                            "headers": safe_headers(exc.headers)})
            retry = exc.code in {429, 500, 502, 503, 504}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            attempt.update({"outcome": "network_error", "exception_class": type(exc).__name__,
                            "elapsed_seconds": round(time.monotonic() - started, 6)})
            retry = transient_network_error(exc)
        write_json(OUT / "http_provenance.json", receipt)
        if not retry or number == MAX_ATTEMPTS:
            raise GateFailure("NETWORK_OR_HTTP_STOP", "Frozen URL failed; no new query or proxy change is permitted")
        time.sleep(min(number, 2))
    raise GateFailure("NETWORK_OR_HTTP_STOP", "Maximum attempts exhausted")


def scope_counts(receipt, old, new, unexpected):
    return {
        "http_attempts": receipt["http_attempts"], "successful_http_pages": receipt["successful_http_pages"],
        "old_raw_records_used_for_exclusion": old["old_raw_records"],
        "old_raw_labels_used": 0, "old_full_raw_rehashes": 0,
        "new_metadata_projected_rows": len(new),
        "new_sequence_values_requested": 0, "new_sequence_values_used": 0,
        "new_structure_label_fields_requested": 0, "new_structure_label_values_used": 0,
        "new_PDB_mapping_fields_requested": 0, "new_PDB_mapping_values_used": 0,
        "new_unrequested_sequence_value_records_returned": unexpected.get("sequence.value", 0),
        "new_unrequested_feature_records_returned": unexpected.get("entry.features", 0),
        "activation_arrays_read": 0, "model_loads": 0, "model_downloads": 0,
        "model_forwards": 0, "fit_count": 0, "new_T_selections_written": 0,
        "old_scientific_tests_rerun": 0, "shared_memory_writes": 0,
        "unexpected_fields_counts_only_no_values_used": dict(sorted(unexpected.items())),
    }


def fetch():
    if (OUT / "http_provenance.json").exists() or ((OUT / "raw").exists() and any((OUT / "raw").iterdir())):
        raise GateFailure("ALREADY_ATTEMPTED", "This one-shot gate already has an HTTP receipt/raw response; do not retry")
    lock, lock_sha, old = verify_lock()
    receipt = {"started_utc": utc(), "complete": False, "status": "RUNNING_METADATA_ONLY",
               "configuration": CONFIG, "lock_path": rel(OUT / "DATA_RULE_LOCK.json"),
               "lock_sha256": lock_sha, "frozen_utc": lock["frozen_utc"],
               "http_attempts": 0, "successful_http_pages": 0, "raw_response_records": 0,
               "official_source_total_results": None, "pages": []}
    write_json(OUT / "http_provenance.json", receipt)
    new, unexpected, seen_urls = [], Counter(), set()
    url = start_url()
    previous_accession = None
    opener = urllib.request.build_opener(NoRedirect())
    failure = None
    try:
        while True:
            validate_url(url)
            if url in seen_urls:
                raise GateFailure("SOURCE_URL_ERROR", "Official pagination repeated a previously requested URL")
            seen_urls.add(url)
            body, page, link = fetch_page(opener, url, len(receipt["pages"]) + 1, receipt)
            try:
                envelope = json.loads(body)
                entries = envelope["results"]
                if not isinstance(entries, list):
                    raise TypeError()
            except (ValueError, KeyError, TypeError):
                raise GateFailure("SUCCESS_RESPONSE_PARSE_STOP", "Successful response could not be parsed; retained and stopped without retry") from None
            page["records"] = len(entries)
            receipt["raw_response_records"] += len(entries)
            for key in envelope:
                if key != "results":
                    unexpected["envelope." + key] += 1
            reported = page["headers"].get("x-total-results", "")
            if not reported.isdigit():
                raise GateFailure("SOURCE_TOTAL_ERROR", "Official source total is missing or invalid")
            page["official_source_total_results"] = int(reported)
            if receipt["official_source_total_results"] is None:
                receipt["official_source_total_results"] = int(reported)
            elif receipt["official_source_total_results"] != int(reported):
                raise GateFailure("SOURCE_DRIFT", "Official source total changed during bounded traversal")
            current_release = [page["headers"].get("x-uniprot-release"), page["headers"].get("x-uniprot-release-date")]
            if current_release != RELEASE:
                raise GateFailure("SOURCE_DRIFT", "Official release differs from 2026_03 / 02-September-2026")
            requested_size = int(dict(validate_url(url))["size"])
            page["requested_size"] = requested_size
            if len(entries) > requested_size or receipt["raw_response_records"] > MAX_RAW or len(receipt["pages"]) > MAX_PAGES:
                raise GateFailure("RAW_BUDGET_OR_PAGE_ERROR", "Server response exceeded frozen requested/raw/page budget")
            next_url = get_next(link)
            if next_url:
                page["official_next_url"] = next_url
            total = receipt["official_source_total_results"]
            if receipt["raw_response_records"] > total or (not next_url and receipt["raw_response_records"] != total) or (next_url and receipt["raw_response_records"] >= total):
                raise GateFailure("SOURCE_PAGINATION_ERROR", "Official total and next/exhaustion disagree")
            if not entries and next_url:
                raise GateFailure("SOURCE_URL_ERROR", "Empty response unexpectedly has a next page")
            for entry in entries:
                row = project_new(entry, page["page"], len(new) + 1, unexpected)
                accession = row["id"]
                if accession and ACC.fullmatch(accession):
                    if previous_accession is not None and accession < previous_accession:
                        raise GateFailure("ACCESSION_ORDER_ERROR", "Valid accessions are not nondecreasing")
                    previous_accession = accession
                new.append(row)
            # Persist the minimal projection only.  It is not a candidate/T selection.
            with (OUT / "metadata_projection.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                for row in new[-len(entries):] if entries else []:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            write_json(OUT / "http_provenance.json", receipt)
            print(json.dumps({"page": page["page"], "raw_received": receipt["raw_response_records"],
                              "official_source_total": receipt["official_source_total_results"], "metadata_only": True}), flush=True)
            if not next_url:
                receipt["stop_reason"] = "OFFICIAL_SOURCE_EXHAUSTED"
                break
            size = next_size(receipt["raw_response_records"], receipt["successful_http_pages"])
            if not size:
                receipt["stop_reason"] = "FROZEN_RAW_OR_PAGE_CAP_REACHED"
                break
            url = resized_official_next(next_url, size)
        receipt["complete"] = True
        receipt["status"] = "METADATA_TRAVERSAL_COMPLETE"
    except GateFailure as exc:
        failure = exc
        receipt.update({"status": exc.code, "stop_reason": exc.detail})
    except Exception as exc:
        # Exception text may contain local network configuration; preserve category only.
        failure = GateFailure("LOCAL_EXECUTION_ERROR", "Local execution failed; no new query or automatic resume")
        receipt.update({"status": failure.code, "stop_reason": failure.detail, "exception_class": type(exc).__name__})
    receipt["finished_utc"] = utc()
    receipt["read_only_scope_counts"] = scope_counts(receipt, old, new, unexpected)
    write_json(OUT / "http_provenance.json", receipt)
    graph = union_summary(old, new)
    selected_mapping = graph.pop("selected_pair_group_mapping")
    write_json(OUT / "OLD_SELECTED_UNION_GROUPS.json", {
        "complete_metadata_traversal": receipt["complete"],
        "old_selected_pair_count": old["old_selected_pair_count"],
        "prior_cohort_groups": old["old_selected_prior_cohort_groups"],
        "global_groups_before_new_metadata": old["old_selected_global_groups_pre_new"],
        "current_known_groups_after_seen_new_metadata": graph["old_selected_global_groups_after_new_metadata"],
        "additional_group_mergers_from_seen_new_metadata": graph["old_selected_groups_merged_by_new_metadata"],
        "selected_pair_group_mapping": selected_mapping,
        "structural_labels_or_model_outputs_used": 0,
        "limitation": "Known metadata grouping only; no CV, fitting or claim of remote-homology independence.",
    })
    upper = graph["new_components_disconnected_from_old_upper_bound"]
    summary = {
        "created_utc": utc(), "status": receipt["status"], "complete": receipt["complete"],
        "official_source_total_results": receipt["official_source_total_results"],
        "actual_raw_response_records": receipt["raw_response_records"],
        "raw_records_projected_into_graph": len(new), "stop_reason": receipt.get("stop_reason"),
        "graph_summary": graph,
        "old_selected_pair_current_union_groups_path": rel(OUT / "OLD_SELECTED_UNION_GROUPS.json"),
        "threshold_checks": {str(n): {
            "upper_bound_itself_insufficient": upper < n if receipt["complete"] else None,
            "not_excluded_by_metadata_upper_bound": upper >= n if receipt["complete"] else None,
        } for n in (512, 1536)},
        "scope_of_upper_bound": "Only the frozen first-at-most-16384 raw traversal, after full-old-union ID/Pfam exclusion; partial failure counts cannot rule out the unfinished route.",
        "conclusion": ("FROZEN_METADATA_ROUTE_INSUFFICIENT_EVEN_FOR_512" if upper < 512 else "METADATA_UPPER_BOUND_DOES_NOT_ESTABLISH_ELIGIBLE_T") if receipt["complete"] else "NO_COMPLETE_FEASIBILITY_CONCLUSION",
        "read_only_scope_counts": receipt["read_only_scope_counts"],
        "limitations": [
            "Official query total is not usable N.  Remaining known-graph components are an upper bound, never an eligible T.",
            "No new exact sequences were requested or used. Additional exact-sequence overlaps can merge components or connect them to old data, only reducing this bound.",
            "No new structural labels, PDB mappings, centered-A pairs, activations or model fits were evaluated. Future label/mapping/pair eligibility can only reduce this bound.",
            "Missing Pfam is explicit; known Pfam/ID isolation does not establish remote-homology independence or absence of pretraining overlap.",
            "All invalid/duplicate/old raw records consume the frozen budget; invalid-only residual components remain in the conservative upper bound.",
            "Old identities are reused from existing manifests/receipts with counts/schema/stat checks; old full-raw hashes were not recomputed.",
        ],
    }
    write_json(OUT / "METADATA_GATE_RESULT.json", summary)
    print(json.dumps({"status": summary["status"], "complete": summary["complete"],
                      "official_total": summary["official_source_total_results"], "actual_raw": summary["actual_raw_response_records"],
                      "component_upper_bound": upper, "conclusion": summary["conclusion"],
                      "result_path": rel(OUT / "METADATA_GATE_RESULT.json")}, ensure_ascii=False), flush=True)
    return 2 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fetch"))
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare()
            return 0
        return fetch()
    except GateFailure as exc:
        print(json.dumps({"status": exc.code, "detail": exc.detail, "new_http_started_by_this_invocation": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
