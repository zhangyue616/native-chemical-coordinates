"""Lock-gated T labels and frozen-C event counts; no model or network calls.

``prepare`` uses D metadata and the accepted input manifest only. It never opens
sealed T files. ``build`` requires the independent root D_OUTPUT_LOCK before any
T input read. A failed event gate retains every frozen concept and all outputs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import rdBase

from chem_data import DataError, FrozenLabeler, TSV_FIELDS, dump_json, file_sha, smiles_identity, utc_now

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/revision_20261017/chemistry"
ENGINEERING = ROOT / "results/revision_20261017/chemistry_T_data_engineering"
READY = ENGINEERING / "T_DATA_READY.json"
CHECKS = ENGINEERING / "SYNTHETIC_CHECKS.json"
DEFAULT_LOCK = ROOT / "docs/revision_20261017/D_OUTPUT_LOCK.json"
BUILD_RECEIPT = BASE / "corpus/BUILD_RECEIPT.json"
WINNERS = ROOT / "results/revision_20261017/chemistry_D/D_FINAL_WINNERS.npz"
CONCEPTS = BASE / "corpus/concepts_91.json"
PANEL = BASE / "development/panel.json"
OUT = BASE / "test"
CONFIG = {"schema": "chemistry-T-data/1", "n_T": 10000, "first_rank": 5000,
          "n_definitions": 91, "positive_molecules_min": 20,
          "all_negative_molecules_min": 100, "positive_atoms_min": 50,
          "failed_concepts_retained": True, "extra_T_sampling_allowed": False,
          "labels": "FrozenLabeler; exact saved canonical graph; all-match atom union",
          "SMARTS_occurrences": "unique complete matches; function columns=-1",
          "T_input_copy": False, "model_forwards": 0}
OUTPUT_NAMES = ("INPUT_POINTER.json", "labels_T.npz", "atoms_T.npz",
                "occurrence_counts_T.npz", "T_EVENT_GATE.json", "T_DATA_RECEIPT.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime():
    return {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__}


def frozen_metadata():
    """No access to sealed file contents or source gzip, including for hashes."""
    build, concepts, panel = read_json(BUILD_RECEIPT), read_json(CONCEPTS), read_json(PANEL)
    if build.get("status") != "PASS_DATA_BUILD" or build.get("n_T") != CONFIG["n_T"]:
        raise DataError("accepted fixed D/T build is required")
    if len(concepts) != 91 or len({c["id"] for c in concepts}) != 91:
        raise DataError("frozen 91-definition identity/schema failed")
    if collections.Counter(c["source"] for c in concepts) != {"topology": 14, "fg_hierarchy": 10, "fragment_descriptors": 67}:
        raise DataError("frozen definition source counts drifted")
    ids = [c["id"] for c in concepts]
    c_ids = panel["C_ids"]
    if ids != sorted(ids) or not c_ids or c_ids != sorted(set(c_ids)) or not set(c_ids) <= set(ids):
        raise DataError("frozen concept IDs/C join failed")
    for path in (CONCEPTS, PANEL):
        if build["outputs"][path.relative_to(ROOT).as_posix()]["sha256"] != file_sha(path):
            raise DataError("frozen D concept metadata identity drift")
    manifest = {}
    for name in ("T.tsv", "anchors_T.jsonl.gz"):
        relative = (BASE / "sealed" / name).relative_to(ROOT).as_posix()
        evidence = build["outputs"][relative]
        manifest[relative] = {"bytes": evidence["bytes"], "sha256": evidence["sha256"]}
    return concepts, panel, manifest


def input_fingerprints():
    paths = (Path(__file__), Path(__file__).with_name("test_chem_T_data.py"),
             Path(__file__).with_name("chem_data.py"), CHECKS, BUILD_RECEIPT, CONCEPTS, PANEL,
             ROOT / "docs/revision_20261017/CHEMISTRY_DATA_PROTOCOL.md",
             ROOT / "docs/revision_20261017/CHEMISTRY_CONFIRMATION_PROTOCOL.md")
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in paths}


def prepare_ready():
    if DEFAULT_LOCK.exists():
        raise DataError("D_OUTPUT_LOCK already exists; do not replace its READY binding")
    if read_json(CHECKS).get("status") != "PASS_SYNTHETIC_T_DATA":
        raise DataError("T-data synthetic checks are not PASS")
    concepts, panel, manifest = frozen_metadata()
    if rdBase.rdkitVersion != "2026.03.6":
        raise DataError("locked RDKit version drift")
    ready = {"status": "READY_AWAIT_ROOT_D_OUTPUT_LOCK_AND_T_GO", "prepared_at_utc": utc_now(),
             "config": CONFIG, "runtime": runtime(), "files": input_fingerprints(),
             "frozen_C_ids": panel["C_ids"], "frozen_panel_ids": panel["panel_ids"],
             "all_definition_ids": [c["id"] for c in concepts], "T_inputs_from_build_manifest": manifest,
             "final_winners_path": WINNERS.relative_to(ROOT).as_posix(),
             "lock_contract": {"path": DEFAULT_LOCK.relative_to(ROOT).as_posix(),
                 "status": "D_OUTPUT_LOCKED", "T_scientific_processing_authorized": True,
                 "required_SHA_fields": ["data_build_receipt_sha256", "D_final_winners_sha256", "T_data_ready_sha256"]},
             "output_dir": OUT.relative_to(ROOT).as_posix(), "output_names": list(OUTPUT_NAMES),
             "NPZ_schema": {"labels_T.npz": ["labels", "concept_ids"],
                 "atoms_T.npz": ["mol_idx", "atom_idx", "mol_heavy", "cid", "mol_offsets"],
                 "occurrence_counts_T.npz": ["counts", "positive_atom_counts", "concept_ids", "is_smarts"]},
             "execution_ledger": {"actual_T_input_reads": 0, "T_label_calls": 0,
                 "T_concept_count_calls": 0, "model_loads": 0, "model_forwards": 0, "fits": 0}}
    dump_json(READY, ready)
    return {"status": ready["status"], "ready_path": str(READY), "T_data_ready_sha256": file_sha(READY)}


def validate_lock_fields(lock):
    if lock.get("status") != "D_OUTPUT_LOCKED" or lock.get("T_scientific_processing_authorized") is not True:
        raise DataError("STOP: root D_OUTPUT_LOCK and explicit T science authorization are required")
    for field in ("data_build_receipt_sha256", "D_final_winners_sha256", "T_data_ready_sha256"):
        value = lock.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise DataError("STOP: missing/malformed lock SHA field " + field)


def require_output_lock(path=DEFAULT_LOCK):
    lock = read_json(path)
    validate_lock_fields(lock)
    if lock["T_data_ready_sha256"] != file_sha(READY):
        raise DataError("STOP: T data READY binding mismatch")
    ready = read_json(READY)
    if ready["config"] != CONFIG or ready["runtime"] != runtime() or ready["files"] != input_fingerprints():
        raise DataError("STOP: T implementation/runtime/input drift since READY")
    if lock["data_build_receipt_sha256"] != file_sha(BUILD_RECEIPT):
        raise DataError("STOP: data BUILD_RECEIPT binding mismatch")
    if lock["D_final_winners_sha256"] != file_sha(WINNERS):
        raise DataError("STOP: final D winner binding mismatch")
    with np.load(WINNERS, allow_pickle=False) as saved:
        winner_ids = saved["concept_ids"].tolist()
    concepts, panel, manifest = frozen_metadata()
    if winner_ids != ready["frozen_C_ids"] or panel["C_ids"] != winner_ids or manifest != ready["T_inputs_from_build_manifest"]:
        raise DataError("STOP: final winner/frozen-C/input manifest join mismatch")
    return ready, concepts, lock


def event_gate(concept_ids, positive_atom_counts, frozen_C_ids):
    """Join by original concept ID; retain every C entry, including failures."""
    ids, c_ids = list(map(int, concept_ids)), list(map(int, frozen_C_ids))
    counts = np.asarray(positive_atom_counts)
    if len(set(ids)) != len(ids) or len(set(c_ids)) != len(c_ids) or not c_ids or not set(c_ids) <= set(ids):
        raise DataError("invalid/unknown/duplicate frozen concept ID")
    if counts.ndim != 2 or counts.shape[1] != len(ids) or not len(counts) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise DataError("invalid molecule-by-concept positive atom counts")
    columns, rows = {cid: column for column, cid in enumerate(ids)}, []
    for cid in c_ids:
        values = counts[:, columns[cid]]
        pos, total, maximum = int(np.count_nonzero(values)), int(values.sum(dtype=np.int64)), int(values.max())
        neg = len(counts) - pos
        failures = []
        if pos < CONFIG["positive_molecules_min"]:
            failures.append("positive_molecules_below_20")
        if neg < CONFIG["all_negative_molecules_min"]:
            failures.append("all_negative_molecules_below_100")
        if total < CONFIG["positive_atoms_min"]:
            failures.append("positive_atoms_below_50")
        rows.append({"id": cid, "positive_molecules_T": pos, "all_negative_molecules_T": neg,
            "positive_atoms_T": total, "max_single_molecule_positive_atom_count": maximum,
            "max_single_molecule_positive_atom_fraction": maximum / total if total else 0.0,
            "event_gate_pass": not failures, "failures": failures})
    passed = all(row["event_gate_pass"] for row in rows)
    return {"status": "PASS_T_EVENT_GATE" if passed else "FAIL_T_EVENT_GATE", "split": "T",
        "n_molecules": len(counts), "C_ids": c_ids, "concepts": rows,
        "thresholds": {"positive_molecules": 20, "all_negative_molecules": 100, "positive_atoms": 50},
        "joint_confirmation_event_gate_met": passed, "failed_concept_ids": [r["id"] for r in rows if not r["event_gate_pass"]],
        "C_or_panel_adjusted": False, "failed_concepts_removed": False, "extra_T_sampling_allowed": False,
        "downstream_mode": "FROZEN_C_CONFIRMATION_EVENTS_AVAILABLE" if passed else "ALL_FROZEN_C_DESCRIPTIVE_EVENT_GATE_FAILED"}


def read_test_inputs(ready):
    for relative, evidence in ready["T_inputs_from_build_manifest"].items():
        path = ROOT / relative
        if path.stat().st_size != evidence["bytes"] or file_sha(path) != evidence["sha256"]:
            raise DataError("locked T input/anchor identity drift")
    records = []
    with (BASE / "sealed/T.tsv").open(encoding="utf-8", newline="") as stream, \
            gzip.open(BASE / "sealed/anchors_T.jsonl.gz", "rt", encoding="utf-8") as anchors:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != TSV_FIELDS:
            raise DataError("T table schema drift")
        for mi, (row, line) in enumerate(itertools.zip_longest(reader, anchors)):
            if row is None or line is None:
                raise DataError("T input/anchor row counts disagree")
            anchor = json.loads(line)
            cid, rank = int(row["cid"]), int(row["rank"])
            if rank != CONFIG["first_rank"] + mi or rank != anchor["rank"] or cid != anchor["cid"]:
                raise DataError("T input/anchor molecule-order drift")
            identity = smiles_identity(row["smiles"])
            if identity != row["canonical_smiles_sha256"] or identity != anchor["canonical_smiles_sha256"]:
                raise DataError("T canonical identity drift")
            if anchor["gate"] != "PASS_POSITIONAL_ELEMENT_ANCHOR":
                raise DataError("T input lacks the accepted positional gate")
            records.append({**row, "cid": cid, "split": "T", "n_atoms": len(anchor["atoms"])})
    if len(records) != CONFIG["n_T"]:
        raise DataError("T fixed allocation count mismatch; no refill permitted")
    return records


def write_labels(records, concepts, output_dir, ledger=None):
    """Same NPZ keys/dtypes/order as D; uses the original frozen labeler."""
    if not records or any(r.get("split") != "T" for r in records):
        raise DataError("T label writer requires nonempty T records")
    labeler, labels, occurrences, positive_counts = FrozenLabeler(concepts), [], [], []
    molecule_indices, atom_indices, molecule_heavy, cids, offsets = [], [], [], [], [0]
    for mi, record in enumerate(records):
        label, occurrence = labeler.molecule(record["smiles"])
        if ledger is not None:
            ledger["T_label_calls"] += 1
        n = len(label)
        if "n_atoms" in record and n != record["n_atoms"]:
            raise DataError("T canonical label atom count disagrees with accepted anchors")
        labels.append(label); occurrences.append(occurrence)
        positive_counts.append(label.sum(axis=0, dtype=np.int32))
        molecule_indices.extend([mi] * n); atom_indices.extend(range(n))
        molecule_heavy.extend([n] * n); cids.extend([record["cid"]] * n)
        offsets.append(offsets[-1] + n)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concept_ids = np.array([c["id"] for c in concepts], dtype=np.int32)
    all_labels = np.concatenate(labels, axis=0)
    positive_counts = np.asarray(positive_counts, dtype=np.int32)
    np.savez_compressed(output_dir / "labels_T.npz", labels=all_labels, concept_ids=concept_ids)
    np.savez_compressed(output_dir / "atoms_T.npz", mol_idx=np.asarray(molecule_indices, np.int32),
        atom_idx=np.asarray(atom_indices, np.int16), mol_heavy=np.asarray(molecule_heavy, np.int16),
        cid=np.asarray(cids, np.int64), mol_offsets=np.asarray(offsets, np.int64))
    np.savez_compressed(output_dir / "occurrence_counts_T.npz", counts=np.asarray(occurrences, np.int32),
        positive_atom_counts=positive_counts, concept_ids=concept_ids,
        is_smarts=np.array([c["kind"] == "smarts" for c in concepts], dtype=np.bool_))
    return positive_counts, {"n_molecules_T": len(records), "n_atoms_T": len(all_labels),
        "labels_shape": list(all_labels.shape), "occurrence_shape": [len(records), len(concepts)],
        "order": "T.tsv order then canonical RDKit atom index; mol_idx resets to0 in T"}


def run_build(lock_path=DEFAULT_LOCK):
    ready, concepts, lock = require_output_lock(lock_path)  # Before any sealed-file read.
    if any((OUT / name).exists() for name in OUTPUT_NAMES):
        raise DataError("STOP: actual T outputs already exist; preserve partial/completed evidence")
    ledger = {"T_label_calls": 0, "T_concept_count_calls": 0,
              "model_loads": 0, "model_forwards": 0, "fits": 0, "bootstrap_or_inference": 0}
    receipt = {"status": "T_DATA_BUILD_STARTED", "started_at_utc": utc_now(),
        "D_output_lock_sha256": file_sha(lock_path), "T_data_ready_sha256": file_sha(READY),
        "data_build_receipt_sha256": lock["data_build_receipt_sha256"],
        "D_final_winners_sha256": lock["D_final_winners_sha256"],
        "T_inputs": ready["T_inputs_from_build_manifest"], "execution_ledger": ledger,
        "source_gzip_reads_or_rehashes": 0}
    dump_json(OUT / "T_DATA_RECEIPT.json", receipt)
    try:
        records = read_test_inputs(ready)
        positive_counts, dimensions = write_labels(records, concepts, OUT, ledger)
        gate = event_gate([c["id"] for c in concepts], positive_counts, ready["frozen_C_ids"])
        ledger["T_concept_count_calls"] = 1
        dump_json(OUT / "T_EVENT_GATE.json", gate)
        dump_json(OUT / "INPUT_POINTER.json", {"split": "T", "copied": False,
            "files": ready["T_inputs_from_build_manifest"], "identity_bound_by": "locked accepted BUILD_RECEIPT"})
        output_paths = [OUT / name for name in OUTPUT_NAMES if name != "T_DATA_RECEIPT.json"]
        receipt.update(status="PASS_T_DATA_BUILD", completed_at_utc=utc_now(), dimensions=dimensions,
            C_ids=ready["frozen_C_ids"], panel_ids=ready["frozen_panel_ids"],
            event_gate=gate["status"], joint_confirmation_event_gate_met=gate["joint_confirmation_event_gate_met"],
            failed_concept_ids=gate["failed_concept_ids"], C_or_panel_adjusted=False,
            outputs={p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in output_paths})
        dump_json(OUT / "T_DATA_RECEIPT.json", receipt)
        return {"status": receipt["status"], **dimensions, "event_gate": gate["status"],
                "C_size_unchanged": len(ready["frozen_C_ids"]), "model_forwards": 0}
    except Exception as exc:
        receipt.update(status="FAIL_T_DATA_BUILD_PARTIAL_PRESERVED", failed_at_utc=utc_now(), reason=str(exc))
        dump_json(OUT / "T_DATA_RECEIPT.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "build"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    try:
        result = prepare_ready() if args.action == "prepare" else run_build(args.lock)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError) as exc:
        print(json.dumps({"status": "STOP", "reason": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
