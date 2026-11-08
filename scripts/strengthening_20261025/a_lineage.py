"""Post hoc identity lineage and structure-only D-unseen-scaffold T freeze.

No model, label, score, bootstrap, raw PubChem, or network input is used.
"""
from __future__ import annotations

import time

PROCESS_START_CPU = time.process_time()
PROCESS_START_WALL = time.perf_counter()

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import csv
import hashlib
import io
import json
import platform
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/strengthening_20261025/lineage"
PROTOCOL = ROOT / "docs/strengthening_20261025/A_LINEAGE_PROTOCOL.md"
INPUTS = {
    "H4": ((Path(os.environ.get("NATIVE_UNIT_LEGACY_ROOT", ROOT)) / "data/derived/lineB/corpus/H4_used.tsv"), 10000, 0),
    "D": (ROOT / "data/revision_20261017/chemistry/development/D.tsv", 5000, 0),
    "T": (ROOT / "data/revision_20261017/chemistry/sealed/T.tsv", 10000, 5000),
}
KINDS = ("cid_identity", "canonical_isomeric_smiles", "canonical_nonisomeric_smiles",
         "murcko_scaffold_smiles")
PAIRS = (("H4", "D"), ("H4", "T"), ("D", "T"))
RULE = "T row included iff successful nonempty achiral Bemis-Murcko scaffold is absent from D nonempty successful scaffold set; retain all such rows; H4 overlap is not an exclusion"
START_UTC = datetime.now(timezone.utc).isoformat()
CAP_CPU_SECONDS = 900
PERSISTENCE_RESERVE_SECONDS = 5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cpu_seconds():
    return time.process_time() - PROCESS_START_CPU


def check_budget():
    if cpu_seconds() >= CAP_CPU_SECONDS - PERSISTENCE_RESERVE_SECONDS:
        raise TimeoutError("895 CPU second guard reached; reserve 5 seconds for partial and ledger")


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def identity_row(dataset, index, source):
    row = dict(source)
    row.update(dataset=dataset, row_index=index, status="ok", parse_error="",
               identity_error="", scaffold_error="", murcko_is_empty=0)
    for kind in KINDS:
        row[kind] = ""
    try:
        cid = int(source["cid"])
        if cid <= 0:
            raise ValueError("CID must be a positive decimal integer")
        row["cid_identity"] = str(cid)
    except Exception as exc:
        row["identity_error"] = f"cid:{type(exc).__name__}:{exc}"
        row["status"] = "identity_failure"
    try:
        with rdBase.CaptureErrorLog() as error_log:
            mol = Chem.MolFromSmiles(source["smiles"])
        if mol is None:
            raise ValueError("MolFromSmiles returned None; " + " | ".join(error_log))
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() == 0:
            raise ValueError("parsed graph has zero atoms")
    except Exception as exc:
        row["parse_error"] = f"parse_or_RemoveHs:{type(exc).__name__}:{exc}"
        row["status"] = "parse_failure"
        return row
    try:
        row["canonical_isomeric_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        nonstereo = Chem.Mol(mol)
        Chem.RemoveStereochemistry(nonstereo)
        row["canonical_nonisomeric_smiles"] = Chem.MolToSmiles(nonstereo, canonical=True, isomericSmiles=False)
    except Exception as exc:
        row["identity_error"] += f";smiles_identity:{type(exc).__name__}:{exc}"
        row["status"] = "identity_failure"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        Chem.RemoveStereochemistry(scaffold)
        row["murcko_scaffold_smiles"] = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
        row["murcko_is_empty"] = int(scaffold.GetNumAtoms() == 0)
        if bool(row["murcko_scaffold_smiles"]) == bool(row["murcko_is_empty"]):
            raise ValueError("empty-scaffold graph/string disagreement")
    except Exception as exc:
        row["murcko_scaffold_smiles"] = ""
        row["murcko_is_empty"] = 0
        row["scaffold_error"] = f"murcko:{type(exc).__name__}:{exc}"
        row["status"] = "scaffold_failure"
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "CPU_LEDGER.json"
    if ledger_path.exists() or (OUT / "FREEZE.json").exists():
        raise RuntimeError("existing A1 ledger/freeze: refuse automatic rerun or overwrite")
    ledger = {
        "status": "RUNNING", "start_utc": START_UTC,
        "cpu_seconds_hard_limit": CAP_CPU_SECONDS,
        "persistence_reserve_seconds": PERSISTENCE_RESERVE_SECONDS,
        "cpu_clock": "time.process_time(); single process cumulative from script start including imports",
        "command": f'"{sys.executable}" -B "{Path(__file__).resolve()}"',
        "argv": sys.argv, "cwd": str(Path.cwd()),
        "python": sys.version, "platform": platform.platform(),
        "rdkit": rdkit.__version__, "numpy": np.__version__,
        "thread_environment": {key: os.environ[key] for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "model_forward_calls": 0, "fit_calls": 0, "statistical_resamples": 0,
        "score_label_activation_files_read": [], "inputs": {}, "completed_rows": {},
        "output_directory": str(OUT),
    }
    save_json(ledger_path, ledger)
    source_rows = {}
    row_tables = {}
    fields = {}
    try:
        if rdkit.__version__ != "2026.03.6":
            raise RuntimeError(f"RDKit version drift: {rdkit.__version__}")
        if not PROTOCOL.is_file():
            raise RuntimeError("recorded A1 protocol missing")
        ledger["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        ledger["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for dataset, (path, expected_n, start_rank) in INPUTS.items():
            check_budget()
            raw = path.read_bytes()
            ledger["inputs"][dataset] = {
                "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "expected_n": expected_n,
            }
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
            fields[dataset] = reader.fieldnames
            if not {"rank", "cid", "smiles"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{dataset}: missing identity columns")
            source_rows[dataset] = list(reader)
            if len(source_rows[dataset]) != expected_n:
                raise ValueError(f"{dataset}: observed {len(source_rows[dataset])} rows != {expected_n}")
            if [int(row["rank"]) for row in source_rows[dataset]] != list(range(start_rank, start_rank + expected_n)):
                raise ValueError(f"{dataset}: rank/order drift")
            row_tables[dataset] = []
            for index, source in enumerate(source_rows[dataset]):
                check_budget()
                row_tables[dataset].append(identity_row(dataset, index, source))
                if (index + 1) % 500 == 0:
                    ledger["completed_rows"][dataset] = index + 1
                    ledger["elapsed_cpu_seconds"] = cpu_seconds()
                    ledger["elapsed_wall_seconds"] = time.perf_counter() - PROCESS_START_WALL
                    save_json(ledger_path, ledger)
            ledger["completed_rows"][dataset] = len(row_tables[dataset])
            print(f"{dataset}: {len(row_tables[dataset])} rows; cumulative CPU={cpu_seconds():.3f}s", flush=True)

        check_budget()
        counters = {kind: {dataset: Counter(row[kind] for row in rows if row[kind])
                           for dataset, rows in row_tables.items()} for kind in KINDS}
        sets = {kind: {dataset: set(counts) for dataset, counts in by_set.items()}
                for kind, by_set in counters.items()}
        summary = {
            "analysis": "A1_POST_HOC_IDENTITY_LINEAGE_AND_STRUCTURE_ONLY_SLICE",
            "rdkit": rdkit.__version__, "inputs": ledger["inputs"],
            "post_hoc": True, "not_new_independent_scaffold_split": True,
            "identity_rule": "saved SMILES -> MolFromSmiles(default sanitize) -> RemoveHs -> canonical isomeric; copies remove stereochemistry; no salt/neutralization/tautomer/largest-fragment operation",
            "empty_scaffold_policy": "count separately; excluded from all scaffold identity sets/intersections and from slice",
            "sets": {}, "intersections": {}, "model_forward_calls": 0,
            "fit_calls": 0, "statistical_resamples": 0,
        }
        for dataset, rows in row_tables.items():
            summary["sets"][dataset] = {
                "n_rows": len(rows),
                "parse_failure_rows": sum(bool(row["parse_error"]) for row in rows),
                "identity_failure_rows": sum(bool(row["identity_error"]) for row in rows),
                "scaffold_failure_rows": sum(bool(row["scaffold_error"]) for row in rows),
                "empty_scaffold_rows": sum(row["murcko_is_empty"] for row in rows),
                "saved_smiles_equal_common_isomeric_rows": sum(row["smiles"] == row["canonical_isomeric_smiles"] for row in rows),
                "identity_counts": {},
            }
            for kind in KINDS:
                counts = counters[kind][dataset]
                valid_rows = sum(counts.values())
                summary["sets"][dataset]["identity_counts"][kind] = {
                    "rows_with_identity": valid_rows, "unique_identities": len(counts),
                    "duplicate_excess_rows": valid_rows - len(counts),
                    "duplicate_groups": sum(count > 1 for count in counts.values()),
                    "rows_in_duplicate_groups": sum(count for count in counts.values() if count > 1),
                    "rows_without_identity": len(rows) - valid_rows,
                    "failure_rows": (sum(bool(row["parse_error"] or row["scaffold_error"]) for row in rows)
                                     if kind == "murcko_scaffold_smiles" else len(rows) - valid_rows),
                }
        intersections = []
        for kind in KINDS:
            summary["intersections"][kind] = {}
            for left, right in PAIRS:
                shared = sets[kind][left] & sets[kind][right]
                summary["intersections"][kind][f"{left}-{right}"] = {
                    "unique_identities": len(shared),
                    "left_rows": sum(counters[kind][left][value] for value in shared),
                    "right_rows": sum(counters[kind][right][value] for value in shared),
                }
            triple = sets[kind]["H4"] & sets[kind]["D"] & sets[kind]["T"]
            summary["intersections"][kind]["H4-D-T"] = {"unique_identities": len(triple)}
            shared_union = set().union(*(sets[kind][left] & sets[kind][right] for left, right in PAIRS))
            for value in sorted(shared_union):
                intersection = {"identity_kind": kind, "identity": value}
                for dataset in INPUTS:
                    intersection[f"{dataset}_row_count"] = counters[kind][dataset][value]
                intersections.append(intersection)
            for dataset, rows in row_tables.items():
                for row in rows:
                    present = []
                    for target in INPUTS:
                        member = int(bool(row[kind]) and row[kind] in sets[kind][target])
                        row[f"{kind}__in_{target}"] = member
                        present.append(member)
                    row[f"{kind}__shared_with_another_set"] = int(sum(present) >= 2)
                    row[f"{kind}__in_all_three"] = int(sum(present) == 3)

        t_rows = row_tables["T"]
        d_scaffolds = sets["murcko_scaffold_smiles"]["D"]
        mask = np.array([bool(row["murcko_scaffold_smiles"]) and
                         row["murcko_scaffold_smiles"] not in d_scaffolds for row in t_rows], dtype=np.bool_)
        for row, included in zip(t_rows, mask):
            row["D_unseen_scaffold_included"] = int(included)
            row["D_unseen_scaffold_reason"] = ("included_nonempty_scaffold_absent_D" if included else
                "failed_structure_or_scaffold" if row["parse_error"] or row["scaffold_error"] else
                "empty_scaffold" if row["murcko_is_empty"] else "nonempty_scaffold_present_D")
        mask_fields = ["row_index", "rank", "cid", "smiles", "canonical_isomeric_smiles",
                       "canonical_nonisomeric_smiles", "murcko_scaffold_smiles",
                       "D_unseen_scaffold_included", "D_unseen_scaffold_reason"]
        for dataset, rows in row_tables.items():
            extra = [key for key in rows[0] if key not in fields[dataset]]
            save_tsv(OUT / f"{dataset}_IDENTITY_ROWS.tsv", rows, fields[dataset] + extra)
        save_tsv(OUT / "INTERSECTIONS.tsv", intersections,
                 ["identity_kind", "identity", "H4_row_count", "D_row_count", "T_row_count"])
        save_tsv(OUT / "T_D_UNSEEN_SCAFFOLD_MASK.tsv", t_rows, mask_fields)
        save_tsv(OUT / "T_D_UNSEEN_SCAFFOLD.tsv", [row for row, included in zip(t_rows, mask) if included], mask_fields)
        mask_path = OUT / "T_D_UNSEEN_SCAFFOLD_MASK.npy"
        np.save(mask_path, mask, allow_pickle=False)
        mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        slice_summary = {
            "rule": RULE, "n_T": len(t_rows), "n_included": int(mask.sum()),
            "n_excluded": int((~mask).sum()), "exclusion_reasons": dict(Counter(
                row["D_unseen_scaffold_reason"] for row, included in zip(t_rows, mask) if not included)),
            "included_unique_scaffolds": len({row["murcko_scaffold_smiles"] for row, included in zip(t_rows, mask) if included}),
            "included_rows_with_H4_nonempty_scaffold": sum(
                bool(included) and row["murcko_scaffold_smiles"] in sets["murcko_scaffold_smiles"]["H4"]
                for row, included in zip(t_rows, mask)),
        }
        summary["slice"] = slice_summary

        # One consolidated engineering check; no second molecular parsing or hash pass.
        check_budget()
        checks = {"input_expected_counts_and_ranks": True}
        checks["row_order_and_original_identities_preserved"] = all(
            len(rows) == len(source_rows[dataset]) and all(
                row["row_index"] == index and all(row[field] == source[field] for field in fields[dataset])
                for index, (row, source) in enumerate(zip(rows, source_rows[dataset])))
            for dataset, rows in row_tables.items())
        checks["identity_counts_and_intersection_bounds"] = all(
            summary["sets"][dataset]["identity_counts"][kind]["unique_identities"] <=
            summary["sets"][dataset]["identity_counts"][kind]["rows_with_identity"] <= len(rows)
            for kind in KINDS for dataset, rows in row_tables.items()) and all(
            len(sets[kind][left] & sets[kind][right]) <= min(len(sets[kind][left]), len(sets[kind][right]))
            for kind in KINDS for left, right in PAIRS)
        checks["scaffold_success_empty_failure_partition"] = all(
            sum(bool(row["murcko_scaffold_smiles"]) for row in rows) +
            sum(row["murcko_is_empty"] for row in rows) +
            sum(bool(row["parse_error"] or row["scaffold_error"]) for row in rows) == len(rows)
            for rows in row_tables.values())
        checks["mask_dtype_length_and_frozen_rule"] = (
            mask.dtype == np.bool_ and mask.shape == (10000,) and all(
                bool(included) == (bool(row["murcko_scaffold_smiles"]) and row["murcko_scaffold_smiles"] not in d_scaffolds)
                for row, included in zip(t_rows, mask)))
        checks["saved_mask_readback_equal"] = np.array_equal(np.load(mask_path, allow_pickle=False), mask)
        checks["no_empty_scaffold_in_sets_or_selected_rows"] = all(
            "" not in sets["murcko_scaffold_smiles"][dataset] for dataset in INPUTS) and all(
                row["murcko_scaffold_smiles"] for row, included in zip(t_rows, mask) if included)
        summary["engineering_checks"] = checks
        summary["engineering_status"] = "PASS" if all(checks.values()) else "FAIL"
        save_json(OUT / "SUMMARY.json", summary)
        if not all(checks.values()):
            raise AssertionError("consolidated engineering checks failed")
        freeze = {
            "freeze_status": "STRUCTURE_ONLY_POST_HOC_SLICE_FROZEN", "utc": utc_now(),
            "rule": RULE, "rdkit": rdkit.__version__, "inputs": ledger["inputs"],
            "mask_file": str(mask_path), "mask_sha256": mask_sha, "mask_dtype": "bool",
            "mask_shape": list(mask.shape), "n_T": len(t_rows), "n_selected": int(mask.sum()),
            "empty_T_scaffold_rows": summary["sets"]["T"]["empty_scaffold_rows"],
            "failed_T_structure_or_scaffold_rows": sum(bool(row["parse_error"] or row["scaffold_error"]) for row in t_rows),
            "row_order": "exact T.tsv order; row_index 0..9999; retain original rank 5000..14999, CID, saved SMILES",
            "H4_overlap_excluded": False, "new_independent_scaffold_split": False,
            "no_scores_or_labels_read_declaration": "This executable only read the three enumerated identity TSVs and its protocol/source. No labels, scores, AP, model activations, outputs, or downstream statistical results were read to select the mask.",
            "engineering_checks": checks,
        }
        save_json(OUT / "FREEZE.json", freeze)
        ledger["status"] = "COMPLETED"
        ledger["engineering_status"] = "PASS"
        ledger["n_slice"] = int(mask.sum())
        ledger["mask_sha256"] = mask_sha
        print(json.dumps({"status": ledger["status"], "slice": slice_summary,
                          "intersections": summary["intersections"]}, indent=2), flush=True)
    except Exception as exc:
        ledger["status"] = "PARTIAL_BUDGET_STOP" if isinstance(exc, TimeoutError) else "PARTIAL_FAILED"
        ledger["error"] = f"{type(exc).__name__}: {exc}"
        ledger["traceback"] = traceback.format_exc()
        for dataset, rows in row_tables.items():
            if rows:
                extra = [key for key in rows[0] if key not in fields[dataset]]
                save_tsv(OUT / f"{dataset}_PARTIAL_ROWS.tsv", rows, fields[dataset] + extra)
        print(ledger["traceback"], file=sys.stderr, flush=True)
        raise
    finally:
        ledger["end_utc"] = utc_now()
        ledger["elapsed_cpu_seconds"] = cpu_seconds()
        ledger["elapsed_wall_seconds"] = time.perf_counter() - PROCESS_START_WALL
        ledger["completed_rows"] = {dataset: len(rows) for dataset, rows in row_tables.items()}
        ledger["outputs"] = sorted(path.name for path in OUT.iterdir() if path.is_file())
        save_json(ledger_path, ledger)


if __name__ == "__main__":
    main()
