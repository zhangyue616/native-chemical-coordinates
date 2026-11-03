"""Post hoc H4 legacy-position versus independent span/graph audit; local only.

No model APIs or weight files are imported, opened, or called. Existing mapper
engineering receipts are reused; only the local-tokenizer adapter is checked.
"""
from __future__ import annotations

import time

PROCESS_START = time.process_time()
WALL_START = time.monotonic()

import argparse
import ast
import collections
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("NATIVE_UNIT_LEGACY_ROOT", ROOT))
MODEL_ROOT = Path(os.environ.get("NEURONAUDIT_MODEL_ROOT", ROOT / "external"))
OUT = ROOT / "results/strengthening_20261025/mapping"
PROTOCOL = ROOT / "docs/strengthening_20261025/A_MAPPING_PROTOCOL.md"
SCRIPT = Path(__file__).resolve()
H4 = SOURCE_ROOT / "data/derived/lineB/corpus/H4_used.tsv"
LEGACY = SOURCE_ROOT / "scripts/lineB/models.py"
REFERENCE = ROOT / "scripts/revision_20261017/semantic_mapper.py"
REUSED_CHECKS = ROOT / "results/revision_20261017/mapper/ENGINEERING_CHECKS.json"
DOWNLOAD_LOG = SOURCE_ROOT / "models/download_log.csv"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json", "added_tokens.json")
FAMILIES = {
    "chemberta_mlm": ("ChemBERTa-77M-MLM", MODEL_ROOT / "models/chemberta_77m_mlm"),
    "chemberta_mtr": ("ChemBERTa-77M-MTR", MODEL_ROOT / "models/chemberta_77m_mtr"),
}
CATEGORIES = ("verified_unique_anchor_match", "ambiguous/unverifiable", "hardfail",
              "actual_position_disagreement")
MAX_CPU = 1800.0
STOP_CPU = 1760.0
NETWORK_ATTEMPTS = []

# Fail closed even if an imported package were to ignore the HF offline flags.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def network_forbidden(*args, **kwargs):
    NETWORK_ATTEMPTS.append("blocked_network_call")
    raise RuntimeError("network is forbidden in this local audit")


socket.socket.connect = network_forbidden
socket.socket.connect_ex = network_forbidden
socket.create_connection = network_forbidden
socket.getaddrinfo = network_forbidden


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest_bytes(Path(path).read_bytes())


def dump(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stat_record(path):
    path = Path(path)
    info = path.stat()
    return {"path": path.as_posix(), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "sha256": sha(path)}


def load_ledger():
    path = OUT / "CPU_LEDGER.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "h4-mapping-cpu-ledger/1", "hard_limit_process_cpu_seconds": MAX_CPU,
        "soft_stop_process_cpu_seconds": STOP_CPU, "cumulative_process_cpu_seconds": 0.0,
        "entries": [], "scope": "all invocations of the new local mapping runner, including imports, preparation, adapter checks, audit and output serialization",
        "model_loads": 0, "model_forwards": 0, "model_weight_reads": 0,
        "training_calls": 0, "fits": 0, "bootstrap_calls": 0, "paid_calls": 0,
        "network_requests": 0,
    }


def cpu_used(prior):
    return prior + time.process_time() - PROCESS_START


def budget_check(prior):
    if cpu_used(prior) >= STOP_CPU:
        raise BudgetStop("preemptive CPU stop with reserved output-finalization time")


class BudgetStop(RuntimeError):
    pass


def finish_ledger(ledger, phase, outcome):
    entry = {"phase": phase, "outcome": outcome, "finished_utc": utc(),
             "process_cpu_seconds": time.process_time() - PROCESS_START,
             "wall_seconds": time.monotonic() - WALL_START,
             "network_attempts_blocked": len(NETWORK_ATTEMPTS)}
    ledger["entries"].append(entry)
    ledger["cumulative_process_cpu_seconds"] += entry["process_cpu_seconds"]
    ledger["hard_limit_respected"] = ledger["cumulative_process_cpu_seconds"] <= MAX_CPU
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "CPU_LEDGER.events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    dump(OUT / "CPU_LEDGER.json", ledger)
    print(json.dumps({"phase": phase, "outcome": outcome, "process_cpu_seconds": entry["process_cpu_seconds"],
                      "cumulative_process_cpu_seconds": ledger["cumulative_process_cpu_seconds"]}), flush=True)


def source_functions():
    source = LEGACY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    required = ("chemberta_tokenizer", "chemberta_atom_positions", "_extract_chemberta", "extract_layers")
    # Actual source naming differs across historical versions: bind all relevant
    # source text, while requiring the two exact rule functions used here.
    if not all(k in funcs for k in ("chemberta_tokenizer", "chemberta_atom_positions", "_extract_chemberta")):
        raise RuntimeError("required historical tokenizer/position functions absent")
    parsed_tok = ast.parse(funcs["chemberta_tokenizer"])
    subscript_keys = [n.slice.value for n in ast.walk(parsed_tok)
                      if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)]
    if "chemberta_mlm" not in subscript_keys:
        raise RuntimeError("historical shared MLM tokenizer path no longer established")
    return {k: funcs[k] for k in required if k in funcs}


def prepare(prior):
    if (OUT / "AUDIT_LOCK.json").exists():
        raise RuntimeError("lock already exists; do not overwrite an existing audit")
    budget_check(prior)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(H4.open(encoding="utf-8", newline=""), delimiter="\t"))
    if len(rows) != 10000:
        raise RuntimeError(f"H4 denominator changed: {len(rows)}")
    download_rows = list(csv.DictReader(DOWNLOAD_LOG.open(encoding="utf-8-sig", newline="")))
    receipts = {}
    for family, (model_name, asset) in FAMILIES.items():
        files = {}
        for name in (*TOKENIZER_FILES, "config.json"):
            matches = [r for r in download_rows if r["model"] == model_name and r["file"] == name]
            if len(matches) != 1 or matches[0]["ok"] != "True":
                raise RuntimeError(f"missing or nonunique historical receipt: {family}/{name}")
            old = matches[0]
            rec = stat_record(asset / name)
            if rec["sha256"] != old["sha256"].lower() or rec["bytes"] != int(old["bytes"]):
                raise RuntimeError(f"historical bytes mismatch: {family}/{name}")
            files[name] = {**rec, "historical_receipt": old}
        receipts[family] = {"repository": "DeepChem/" + model_name, "asset_dir": asset.as_posix(),
                            "revision": "not pinned in original resolve/main download receipt",
                            "files": files}
    equal = {name: receipts["chemberta_mlm"]["files"][name]["sha256"] ==
             receipts["chemberta_mtr"]["files"][name]["sha256"] for name in TOKENIZER_FILES}
    if not all(equal.values()):
        raise RuntimeError("original MLM/MTR tokenizer bytes differ; affected historical equivalence is unresolved")
    old_checks = json.loads(REUSED_CHECKS.read_text(encoding="utf-8"))
    if old_checks["status"] != "PASS" or old_checks["counts"] != {
            "checks": 47, "passed": 47, "failed": 0, "golden_fixture_families": 16}:
        raise RuntimeError("47/47 reusable mapper receipt is not the expected passing receipt")
    funcs = source_functions()
    bindings = {
        "schema": "h4-mapping-local-input-binding/1", "prepared_utc": utc(),
        "scientific_audit_rows_processed_at_lock": 0,
        "h4": {**stat_record(H4), "rows": len(rows), "columns": list(rows[0])},
        "legacy_source": stat_record(LEGACY), "legacy_functions": funcs,
        "historical_tokenizer_routing": "both MLM and MTR activation paths call chemberta_tokenizer(), which loads original chemberta_77m_mlm",
        "legacy_first_letter_only_receipt": stat_record(SOURCE_ROOT / "lineB/results_posthoc/alignment_check.json"),
        "legacy_alignment_check_source": stat_record(SOURCE_ROOT / "scripts/lineB/check_chemberta_alignment.py"),
        "strict_mapper_source": stat_record(REFERENCE), "reused_mapper_47_receipt": stat_record(REUSED_CHECKS),
        "download_log": stat_record(DOWNLOAD_LOG), "tokenizers": receipts,
        "mlm_mtr_tokenizer_file_byte_equality": equal,
        "protocol": stat_record(PROTOCOL), "runner": stat_record(SCRIPT),
        "forbidden": ["network", "model loads or forwards", "model weight reads", "H4 exclusions",
                      "old p recomputation", "layer selection", "encoding model activations", "old writes", "paid APIs", "installation", "git"],
    }
    dump(OUT / "INPUT_BINDING.json", bindings)
    dump(OUT / "AUDIT_LOCK.json", {
        "schema": "h4-mapping-audit-lock/1", "locked_utc": utc(),
        "state": "PROTOCOL_AND_LOCAL_INPUTS_LOCKED_BEFORE_AUDIT", "planned_model_rows": 20000,
        "input_binding_sha256": sha(OUT / "INPUT_BINDING.json"),
        "protocol_sha256": bindings["protocol"]["sha256"], "runner_sha256": bindings["runner"]["sha256"],
    })
    print("Prepared protocol/local-input lock; scientific audit rows=0", flush=True)


def checked_binding():
    lock = json.loads((OUT / "AUDIT_LOCK.json").read_text(encoding="utf-8"))
    if sha(OUT / "INPUT_BINDING.json") != lock["input_binding_sha256"]:
        raise RuntimeError("input binding differs from lock")
    binding = json.loads((OUT / "INPUT_BINDING.json").read_text(encoding="utf-8"))
    if sha(PROTOCOL) != lock["protocol_sha256"] or sha(SCRIPT) != lock["runner_sha256"]:
        raise RuntimeError("protocol/runner differs from pre-audit lock")
    recs = [binding[k] for k in ("h4", "legacy_source", "strict_mapper_source", "reused_mapper_47_receipt", "download_log")]
    recs += [r for token in binding["tokenizers"].values() for r in token["files"].values()]
    for rec in recs:
        info = Path(rec["path"]).stat()
        if (info.st_size, info.st_mtime_ns) != (rec["bytes"], rec["mtime_ns"]):
            raise RuntimeError("bound input stat drift: " + rec["path"])
    return binding, lock


def imports():
    from rdkit import Chem, rdBase
    import tokenizers
    from tokenizers import Tokenizer
    import transformers
    from transformers import AutoTokenizer
    spec = importlib.util.spec_from_file_location("h4_reference_semantic_mapper", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Chem, rdBase, tokenizers, Tokenizer, transformers, AutoTokenizer, module


def make_adapter(family, binding, deps):
    Chem, _, _, Tokenizer, _, AutoTokenizer, module = deps
    identity = binding["tokenizers"][family]
    asset = Path(identity["asset_dir"])
    path = asset / "tokenizer.json"
    specification = json.loads(path.read_text(encoding="utf-8"))
    if specification["normalizer"] is not None or specification["pre_tokenizer"]["type"] != "ByteLevel":
        raise RuntimeError("offset fidelity contract not established for local tokenizer components")

    class LocalH4Mapper(module.SemanticMapper):
        def __init__(self):
            # Parent __init__ binds the later MTR download provenance. This
            # adapter binds the original H4 files instead; attribution is unchanged.
            self.tokenizer = Tokenizer.from_file(str(path))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
            self.special_ids = {x["id"] for x in specification["added_tokens"] if x["special"]}
            self.pad_id = self.tokenizer.token_to_id("[PAD]")
            self.unk_id = self.tokenizer.token_to_id("[UNK]")
            self.identity = {"repository": identity["repository"], "revision": identity["revision"],
                             "local_asset_dir": asset.as_posix(),
                             "tokenizer_sha256": identity["files"]["tokenizer.json"]["sha256"],
                             "provenance": "original H4 download_log.csv, matching local bytes"}

    mapper = LocalH4Mapper()
    tokenizer = AutoTokenizer.from_pretrained(str(asset), local_files_only=True)
    rule = binding["legacy_functions"]["chemberta_atom_positions"]
    namespace = {"Chem": Chem, "chemberta_tokenizer": lambda: tokenizer}
    exec(compile(rule, str(LEGACY) + ":chemberta_atom_positions", "exec"), namespace)
    return mapper, tokenizer, namespace["chemberta_atom_positions"]


def hf_raw(tokenizer, smiles):
    enc = tokenizer(smiles, return_offsets_mapping=True, return_special_tokens_mask=True)
    return {"ids": list(enc["input_ids"]),
            "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"]),
            "offsets": [list(x) for x in enc["offset_mapping"]],
            "attention_mask": list(enc["attention_mask"]),
            "special_tokens_mask": list(enc["special_tokens_mask"])}


def backend_raw(mapper, smiles):
    enc = mapper.tokenizer.encode(smiles, add_special_tokens=True)
    return {"ids": list(enc.ids), "tokens": list(enc.tokens),
            "offsets": [list(x) for x in enc.offsets],
            "attention_mask": list(enc.attention_mask),
            "special_tokens_mask": list(enc.special_tokens_mask)}


def encoding_fingerprint(raw):
    return digest_bytes(json.dumps(raw, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def adapter_check(prior):
    if (OUT / "ADAPTER_CHECKS.json").exists():
        raise RuntimeError("adapter receipt already exists; do not rerun routine checks")
    binding, lock = checked_binding()
    deps = imports()
    fixtures = ("CCO", "CC(Cl)Br", "[13CH3][NH2+]C(=O)[O-]", "C[C@H](O)C(=O)O", "c1cc[nH]c1")
    checks = []
    raws = {}
    for family in FAMILIES:
        budget_check(prior)
        mapper, tokenizer, old_fn = make_adapter(family, binding, deps)
        raws[family] = []
        for smiles in fixtures:
            raw = hf_raw(tokenizer, smiles)
            independently_encoded = backend_raw(mapper, smiles)
            report = mapper.map(smiles)
            expected_alpha = [i for i, t in enumerate(raw["tokens"])
                              if t not in set(tokenizer.all_special_tokens) and t.isalpha()]
            default_mol = deps[0].MolFromSmiles(smiles)
            expected_old = expected_alpha if default_mol and len(expected_alpha) == default_mol.GetNumAtoms() and len(raw["ids"]) <= 512 else None
            checks.extend([
                {"family": family, "fixture": smiles, "check": "local_hf_vs_raw_backend_all_encoding_fields", "pass": raw == independently_encoded},
                {"family": family, "fixture": smiles, "check": "extracted_original_rule_executes_unchanged", "pass": old_fn(smiles) == expected_old},
                {"family": family, "fixture": smiles, "check": "injected_tokenizer_preserves_reference_attribution", "pass": report["input_ids"] == raw["ids"] and len(report["atoms"]) == default_mol.GetNumAtoms()},
            ])
            raws[family].append(raw)
    checks.append({"check": "two_original_tokenizer_paths_identical_fixture_encoding",
                   "pass": raws["chemberta_mlm"] == raws["chemberta_mtr"]})
    rec = {"schema": "h4-original-tokenizer-adapter-check/1", "finished_utc": utc(),
           "status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
           "counts": {"checks": len(checks), "passed": sum(x["pass"] for x in checks)},
           "checks": checks, "fixtures_are_synthetic": True, "scientific_h4_rows_processed": 0,
           "reused_reference_mapper_checks": {"counts": {"checks": 47, "passed": 47},
                                               "path": REUSED_CHECKS.as_posix(), "rerun": False},
           "runtime": {"python": platform.python_version(), "rdkit": deps[1].rdkitVersion,
                       "tokenizers": deps[2].__version__, "transformers": deps[4].__version__},
           "input_binding_sha256": lock["input_binding_sha256"]}
    dump(OUT / "ADAPTER_CHECKS.json", rec)
    if rec["status"] != "PASS":
        raise RuntimeError("new adapter check failed; affected scientific audit blocked")
    print("Adapter checks passed: " + str(rec["counts"]), flush=True)


def prefix_counts(items):
    return collections.Counter(x.split(":", 1)[0] for x in items)


def original_graph_equivalent(default_mol, report):
    if default_mol is None or default_mol.GetNumAtoms() != len(report["graph"]["nodes"]):
        return False
    nodes = report["graph"]["nodes"]
    for atom in default_mol.GetAtoms():
        ref = nodes[atom.GetIdx()]
        props = (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetIsAromatic(), str(atom.GetChiralTag()))
        if props != tuple(ref[k] for k in ("atomic_number", "isotope", "formal_charge", "aromatic", "chiral_tag")):
            return False
    edges = {(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))), b.GetBondTypeAsDouble(), str(b.GetStereo()), str(b.GetBondDir()))
             for b in default_mol.GetBonds()}
    ref_edges = {(tuple(e["graph_atom_ids"]), e["bond_order"], e["stereo"], e["direction"]) for e in report["graph"]["edges"]}
    return edges == ref_edges


def one_row(index, row, family, mapper, tokenizer, old_fn, Chem, module):
    smiles = row.get("smiles", "")
    detail = {"row_index": index, "family": family, "input_row": row,
              "category": None, "reason_subtypes": [], "legacy_positions": None,
              "strict_positions_by_graph_index": None, "disagreement_graph_indices": [],
              "candidate_disagreement_graph_indices": [], "strict": None}
    try:
        if not smiles:
            detail["category"] = "hardfail"
            detail["reason_subtypes"].append("missing_smiles")
            return detail
        raw = hf_raw(tokenizer, smiles)
        backend = backend_raw(mapper, smiles)
        detail["encoding_fingerprint"] = encoding_fingerprint(raw)
        detail["encoding_fields_equal"] = raw == backend
        detail["historical_hf_encoding"] = raw
        detail["legacy_positions"] = old_fn(smiles)
        default_mol = Chem.MolFromSmiles(smiles)
        detail["legacy_graph_atom_count"] = default_mol.GetNumAtoms() if default_mol else None
        if not detail["encoding_fields_equal"]:
            detail["category"] = "hardfail"
            detail["reason_subtypes"].append("local_hf_vs_strict_backend_encoding_mismatch")
            detail["strict_backend_encoding"] = backend
            return detail
        if default_mol is None:
            detail["category"] = "hardfail"
            detail["reason_subtypes"].append("rdkit_default_parse_failed")
            return detail
        try:
            report = mapper.map(smiles, max_tokens=512)
            detail["strict"] = report
        except (module.MappingError, RuntimeError) as exc:
            detail["strict_exception"] = {"type": type(exc).__name__, "message": str(exc)}
            detail["category"] = "hardfail" if detail["legacy_positions"] is None else "ambiguous/unverifiable"
            detail["reason_subtypes"].append("strict_source_graph_contract_unsupported:" + str(exc))
            if detail["legacy_positions"] is None:
                detail["reason_subtypes"].append("legacy_rule_returned_none")
            return detail
        atoms = sorted(report["atoms"], key=lambda a: a["graph_atom_id"])
        if [a["graph_atom_id"] for a in atoms] != list(range(len(atoms))):
            detail["category"] = "hardfail"
            detail["reason_subtypes"].append("strict_graph_index_not_complete_unique_range")
            return detail
        candidates = [list(a["anchor_token_indices"]) for a in atoms]
        detail["strict_anchor_candidates_by_graph_index"] = candidates
        strict_pos = [p[0] if len(p) == 1 else None for p in candidates]
        detail["strict_positions_by_graph_index"] = strict_pos
        pos = detail["legacy_positions"]
        detail["legacy_strict_graph_index_contract_equal"] = original_graph_equivalent(default_mol, report)
        if pos is not None and len(pos) == len(strict_pos):
            detail["candidate_disagreement_graph_indices"] = [i for i, (a, b) in enumerate(zip(pos, strict_pos)) if b is not None and a != b]
        if pos is None:
            detail["category"] = "hardfail"
            detail["reason_subtypes"].append("legacy_rule_returned_none")
            alpha_count = sum(t not in set(tokenizer.all_special_tokens) and t.isalpha() for t in raw["tokens"])
            if alpha_count != default_mol.GetNumAtoms():
                detail["reason_subtypes"].append("legacy_alpha_count_not_rdkit_atom_count")
            if len(raw["ids"]) > 512:
                detail["reason_subtypes"].append("legacy_over_512_token_window")
        elif not detail["legacy_strict_graph_index_contract_equal"]:
            detail["category"] = "hardfail"
            detail["reason_subtypes"].append("legacy_strict_graph_index_contract_not_identical")
        elif report["element_anchor_position_status"] != "position-aligned" or not all(len(p) == 1 for p in candidates):
            detail["category"] = "ambiguous/unverifiable"
            detail["reason_subtypes"].append("strict_unique_element_anchor_not_established")
            detail["reason_subtypes"].extend("strict_anchor_" + x for x in sorted(prefix_counts(report["anchor_errors"])))
            if any(not p for p in candidates):
                detail["reason_subtypes"].append("strict_atom_without_anchor")
            if any(len(p) > 1 for p in candidates):
                detail["reason_subtypes"].append("strict_atom_with_multiple_anchors")
        elif pos != strict_pos:
            detail["category"] = "actual_position_disagreement"
            detail["disagreement_graph_indices"] = detail["candidate_disagreement_graph_indices"]
            detail["reason_subtypes"].append("verified_graph_atom_anchor_position_differs")
        else:
            detail["category"] = "verified_unique_anchor_match"
            detail["reason_subtypes"].append("all_graph_indices_have_equal_unique_verified_anchor")
            if report["errors"]:
                detail["reason_subtypes"].append("full_offset_defects_retained_anchor_unaffected")
            if report["identity_fidelity"] == "identity-lossy":
                detail["reason_subtypes"].append("atom_identity_loss_retained")
        return detail
    except Exception as exc:
        detail["category"] = "hardfail"
        detail["reason_subtypes"].append("unexpected_execution_error:" + type(exc).__name__)
        detail["exception"] = {"message": str(exc), "traceback": traceback.format_exc()}
        return detail


def initial_summary(family, rows):
    return {"family": family, "original_molecule_denominator": len(rows), "processed_molecules": 0,
            "unprocessed_molecules": len(rows), "category_counts": {c: 0 for c in CATEGORIES},
            "reason_subtype_molecule_counts": collections.Counter(),
            "legacy_parseable_atom_count": 0, "legacy_position_atom_count": 0,
            "strict_graph_atom_count": 0, "strict_unique_anchor_atom_count": 0,
            "definitively_comparable_atom_count": 0, "actual_position_disagreement_atom_count": 0,
            "full_offset_blocked_molecule_count": 0, "element_anchor_blocked_molecule_count": 0,
            "identity_lossy_molecule_count": 0, "identity_lossy_atom_count": 0,
            "lost_attribute_atom_counts": collections.Counter(),
            "full_offset_error_occurrence_counts": collections.Counter(),
            "full_offset_error_molecule_counts": collections.Counter(),
            "anchor_error_occurrence_counts": collections.Counter(),
            "anchor_error_molecule_counts": collections.Counter(),
            "lost_syntax_molecule_count": 0, "encoding_mismatch_molecule_count": 0,
            "cross_original_tokenizer_encoding_mismatch_count": 0,
            "exception_molecule_count": 0, "first_actual_disagreement": None,
            "input_duplicates": {
                "cid_extra_occurrences": len(rows) - len({r.get("cid", "") for r in rows}),
                "smiles_extra_occurrences": len(rows) - len({r.get("smiles", "") for r in rows}),
                "whole_row_extra_occurrences": len(rows) - len({tuple(sorted(r.items())) for r in rows}),
                "policy": "all original rows retained"}}


def accumulate(summary, detail):
    summary["processed_molecules"] += 1
    summary["unprocessed_molecules"] -= 1
    summary["category_counts"][detail["category"]] += 1
    summary["reason_subtype_molecule_counts"].update(set(detail["reason_subtypes"]))
    summary["legacy_parseable_atom_count"] += detail.get("legacy_graph_atom_count") or 0
    summary["legacy_position_atom_count"] += len(detail.get("legacy_positions") or [])
    if detail["category"] in ("verified_unique_anchor_match", "actual_position_disagreement"):
        summary["definitively_comparable_atom_count"] += len(detail["strict_positions_by_graph_index"])
    summary["actual_position_disagreement_atom_count"] += len(detail["disagreement_graph_indices"])
    if detail.get("encoding_fields_equal") is False:
        summary["encoding_mismatch_molecule_count"] += 1
    if detail.get("cross_original_tokenizer_encoding_equal") is False:
        summary["cross_original_tokenizer_encoding_mismatch_count"] += 1
    if "exception" in detail or "strict_exception" in detail:
        summary["exception_molecule_count"] += 1
    report = detail.get("strict")
    if report:
        summary["strict_graph_atom_count"] += len(report["atoms"])
        summary["strict_unique_anchor_atom_count"] += sum(len(a["anchor_token_indices"]) == 1 for a in report["atoms"])
        summary["full_offset_blocked_molecule_count"] += report["position_status"] == "BLOCKED"
        summary["element_anchor_blocked_molecule_count"] += report["element_anchor_position_status"] == "BLOCKED"
        identity_lossy = [a for a in report["atoms"] if a["fidelity"] == "identity-lossy"]
        summary["identity_lossy_molecule_count"] += bool(identity_lossy)
        summary["identity_lossy_atom_count"] += len(identity_lossy)
        for atom in report["atoms"]:
            summary["lost_attribute_atom_counts"].update(atom["lost_attributes"])
        errors, anchor_errors = prefix_counts(report["errors"]), prefix_counts(report["anchor_errors"])
        summary["full_offset_error_occurrence_counts"].update(errors)
        summary["full_offset_error_molecule_counts"].update(errors.keys())
        summary["anchor_error_occurrence_counts"].update(anchor_errors)
        summary["anchor_error_molecule_counts"].update(anchor_errors.keys())
        summary["lost_syntax_molecule_count"] += bool(report["lost_syntax"])
    if detail["category"] == "actual_position_disagreement" and summary["first_actual_disagreement"] is None:
        summary["first_actual_disagreement"] = {k: detail[k] for k in ("row_index", "input_row", "legacy_positions", "strict_positions_by_graph_index", "disagreement_graph_indices")}
        print("SUBSTANTIVE_POSITION_DISAGREEMENT " + json.dumps(summary["first_actual_disagreement"], ensure_ascii=False), flush=True)


def run(prior):
    if any((OUT / (family + ".tsv")).exists() for family in FAMILIES):
        raise RuntimeError("scientific output already exists; do not rerun/overwrite")
    binding, lock = checked_binding()
    adapter = json.loads((OUT / "ADAPTER_CHECKS.json").read_text(encoding="utf-8"))
    if adapter["status"] != "PASS" or adapter["input_binding_sha256"] != lock["input_binding_sha256"]:
        raise RuntimeError("passing adapter receipt for this lock is required")
    deps = imports()
    Chem, _, _, _, _, _, module = deps
    rows = list(csv.DictReader(H4.open(encoding="utf-8", newline=""), delimiter="\t"))
    summaries = {family: initial_summary(family, rows) for family in FAMILIES}
    summary = {"schema": "h4-strict-mapping-posthoc-audit/1", "started_utc": utc(), "state": "RUNNING",
               "input_binding_sha256": lock["input_binding_sha256"], "protocol_sha256": lock["protocol_sha256"],
               "planned_model_molecule_denominator": 2 * len(rows), "models": summaries,
               "historical_tokenizer_routing": binding["historical_tokenizer_routing"],
               "interpretation_limit": "element-anchor positions only; identity loss and full-offset defects are retained separately; no old result or scientific population changed",
               "reused_reference_mapper_engineering_checks": "47/47, not rerun",
               "adapter_checks": adapter["counts"],
               "runtime": adapter["runtime"], "process_cpu_prior_to_run": prior}
    mlm_fingerprints = []
    csv_fields = ["family", "row_index", "rank", "cid", "hash", "smiles", "category", "reason_subtypes_json",
                  "legacy_graph_atom_count", "strict_graph_atom_count", "legacy_positions_json",
                  "strict_positions_by_graph_index_json", "disagreement_graph_indices_json",
                  "candidate_disagreement_graph_indices_json", "strict_anchor_status", "full_offset_status",
                  "identity_lossy_atom_count", "full_offset_errors_json", "anchor_errors_json", "encoding_fingerprint",
                  "encoding_fields_equal", "cross_original_tokenizer_encoding_equal"]
    try:
        for family in FAMILIES:
            budget_check(prior)
            mapper, tokenizer, old_fn = make_adapter(family, binding, deps)
            with (OUT / (family + ".tsv")).open("w", encoding="utf-8", newline="") as table, \
                    gzip.open(OUT / (family + ".details.jsonl.gz"), "wt", encoding="utf-8", compresslevel=1) as details:
                writer = csv.DictWriter(table, fieldnames=csv_fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for index, row in enumerate(rows):
                    budget_check(prior)
                    detail = one_row(index, row, family, mapper, tokenizer, old_fn, Chem, module)
                    if family == "chemberta_mlm":
                        mlm_fingerprints.append(detail.get("encoding_fingerprint"))
                    else:
                        same = bool(detail.get("encoding_fingerprint")) and detail["encoding_fingerprint"] == mlm_fingerprints[index]
                        detail["cross_original_tokenizer_encoding_equal"] = same
                        if not same:
                            detail["category"] = "hardfail"
                            detail["reason_subtypes"].append("mtr_original_asset_encoding_differs_from_historical_shared_mlm")
                            detail["disagreement_graph_indices"] = []
                    report = detail.get("strict") or {}
                    json_field = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    writer.writerow({"family": family, "row_index": index, **{k: row.get(k, "") for k in ("rank", "cid", "hash", "smiles")},
                                     "category": detail["category"], "reason_subtypes_json": json_field(detail["reason_subtypes"]),
                                     "legacy_graph_atom_count": detail.get("legacy_graph_atom_count"),
                                     "strict_graph_atom_count": len(report["atoms"]) if report else None,
                                     "legacy_positions_json": json_field(detail["legacy_positions"]),
                                     "strict_positions_by_graph_index_json": json_field(detail["strict_positions_by_graph_index"]),
                                     "disagreement_graph_indices_json": json_field(detail["disagreement_graph_indices"]),
                                     "candidate_disagreement_graph_indices_json": json_field(detail["candidate_disagreement_graph_indices"]),
                                     "strict_anchor_status": report.get("element_anchor_position_status", "NOT_EVALUATED"),
                                     "full_offset_status": report.get("position_status", "NOT_EVALUATED"),
                                     "identity_lossy_atom_count": sum(a["fidelity"] == "identity-lossy" for a in report.get("atoms", [])),
                                     "full_offset_errors_json": json_field(report.get("errors", [])),
                                     "anchor_errors_json": json_field(report.get("anchor_errors", [])),
                                     "encoding_fingerprint": detail.get("encoding_fingerprint", ""),
                                     "encoding_fields_equal": detail.get("encoding_fields_equal"),
                                     "cross_original_tokenizer_encoding_equal": detail.get("cross_original_tokenizer_encoding_equal", "historical_shared_path")})
                    details.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
                    accumulate(summaries[family], detail)
                    if (index + 1) % 1000 == 0:
                        print(json.dumps({"family": family, "processed": index + 1, "cpu_cumulative": cpu_used(prior), "categories": summaries[family]["category_counts"]}), flush=True)
                        dump(OUT / "SUMMARY.json", summary)
            print("Completed " + family, flush=True)
        summary["state"] = "COMPLETE_AUDIT"
    except BudgetStop as exc:
        summary["state"] = "PARTIAL_BUDGET_STOP"
        summary["stop_reason"] = str(exc)
    except Exception as exc:
        summary["state"] = "PARTIAL_EXECUTION_FAILURE"
        summary["stop_reason"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        summary["finished_utc"] = utc()
        summary["process_cpu_through_final_summary"] = cpu_used(prior)
        summary["processed_model_molecules"] = sum(s["processed_molecules"] for s in summaries.values())
        summary["unprocessed_model_molecules"] = sum(s["unprocessed_molecules"] for s in summaries.values())
        summary["classification_sum_checks"] = {
            family: sum(s["category_counts"].values()) == s["processed_molecules"] and
            s["processed_molecules"] + s["unprocessed_molecules"] == s["original_molecule_denominator"]
            for family, s in summaries.items()}
        dump(OUT / "SUMMARY.json", summary)
        print(json.dumps({"state": summary["state"], "processed": summary["processed_model_molecules"],
                          "unprocessed": summary["unprocessed_model_molecules"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "adapter-check", "run"))
    args = parser.parse_args()
    ledger = load_ledger()
    prior = ledger["cumulative_process_cpu_seconds"]
    outcome = "FAILED"
    try:
        budget_check(prior)
        {"prepare": prepare, "adapter-check": adapter_check, "run": run}[args.phase](prior)
        outcome = "FINISHED"
    finally:
        finish_ledger(ledger, args.phase, outcome)


if __name__ == "__main__":
    main()
