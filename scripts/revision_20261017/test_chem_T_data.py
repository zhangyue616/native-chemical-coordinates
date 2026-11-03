"""Finite synthetic T-event/ID/schema checks. Never reads actual sealed T."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from rdkit import rdBase

import chem_T_data as td


def run():
    checks = []
    def check(name, passed):
        checks.append({"case": name, "status": "PASS" if passed else "FAIL"})
    def rejects(name, call):
        try:
            call()
            check(name, False)
        except td.DataError:
            check(name, True)

    # The exact boundary is120 molecules:20 positive plus100 all-negative.
    counts = np.zeros((120, 3), dtype=np.int32)
    counts[:20, 1] = 1; counts[0, 1] = 31
    counts[:20, 0] = 3
    joined = td.event_gate([42, 7, 91], counts, [7, 42])
    check("three_thresholds_inclusive_at_boundary", joined["status"] == "PASS_T_EVENT_GATE"
          and joined["concepts"][0]["positive_atoms_T"] == 50)
    check("concept_ID_join_and_frozen_C_order", joined["C_ids"] == [7, 42]
          and [r["positive_atoms_T"] for r in joined["concepts"]] == [50, 60])
    check("max_single_molecule_positive_contribution", joined["concepts"][0]["max_single_molecule_positive_atom_count"] == 31
          and joined["concepts"][0]["max_single_molecule_positive_atom_fraction"] == 31 / 50)
    insufficient_pos = counts.copy(); insufficient_pos[19, 1] = 0
    check("19_positive_molecules_fail_without_other_rescue", td.event_gate([42, 7, 91], insufficient_pos, [7])["concepts"][0]["failures"] == ["positive_molecules_below_20", "positive_atoms_below_50"])
    check("99_negative_molecules_fail", td.event_gate([42, 7, 91], counts[:-1], [7])["concepts"][0]["failures"] == ["all_negative_molecules_below_100"])
    insufficient_atoms = counts.copy(); insufficient_atoms[:20, 1] = 2
    check("49_or_fewer_positive_atoms_fail", td.event_gate([42, 7, 91], insufficient_atoms, [7])["concepts"][0]["failures"] == ["positive_atoms_below_50"])
    failed = td.event_gate([42, 7, 91], counts, [7, 91, 42])
    check("failed_event_preserves_entire_frozen_C", failed["C_ids"] == [7, 91, 42]
          and len(failed["concepts"]) == 3 and failed["failed_concept_ids"] == [91]
          and not failed["C_or_panel_adjusted"] and not failed["failed_concepts_removed"]
          and not failed["extra_T_sampling_allowed"])
    check("zero_positive_contribution_finite", failed["concepts"][1]["max_single_molecule_positive_atom_count"] == 0
          and failed["concepts"][1]["max_single_molecule_positive_atom_fraction"] == 0)
    rejects("unknown_C_ID_rejected", lambda: td.event_gate([42, 7, 91], counts, [100]))
    rejects("duplicate_definition_ID_rejected", lambda: td.event_gate([42, 7, 7], counts, [7]))
    rejects("duplicate_frozen_C_ID_rejected", lambda: td.event_gate([42, 7, 91], counts, [7, 7]))
    rejects("negative_atom_count_rejected", lambda: td.event_gate([7], np.array([[-1]], dtype=np.int32), [7]))
    valid_lock = {"status": "D_OUTPUT_LOCKED", "T_scientific_processing_authorized": True,
        "data_build_receipt_sha256": "1" * 64, "D_final_winners_sha256": "2" * 64,
        "T_data_ready_sha256": "3" * 64}
    td.validate_lock_fields(valid_lock)
    rejects("missing_T_GO_rejected", lambda: td.validate_lock_fields({**valid_lock, "T_scientific_processing_authorized": False}))
    rejects("wrong_status_rejected", lambda: td.validate_lock_fields({**valid_lock, "status": "D_RUN_LOCKED"}))
    rejects("missing_winner_binding_rejected", lambda: td.validate_lock_fields({**valid_lock, "D_final_winners_sha256": None}))

    concepts = td.read_json(td.CONCEPTS)  # Definitions only; no D or T molecule read.
    td.ENGINEERING.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic_T_schema_", dir=td.ENGINEERING) as tmp:
        records = [{"split": "T", "cid": 9001, "smiles": "O=C(O)CC(=O)O", "n_atoms": 7},
                   {"split": "T", "cid": 9002, "smiles": "CCCCC", "n_atoms": 5}]
        positive_counts, dimensions = td.write_labels(records, concepts, Path(tmp))
        with np.load(Path(tmp) / "labels_T.npz", allow_pickle=False) as labels, \
                np.load(Path(tmp) / "atoms_T.npz", allow_pickle=False) as atoms, \
                np.load(Path(tmp) / "occurrence_counts_T.npz", allow_pickle=False) as occ:
            check("NPZ_keys_exactly_match_D", set(labels.files) == {"labels", "concept_ids"}
                and set(atoms.files) == {"mol_idx", "atom_idx", "mol_heavy", "cid", "mol_offsets"}
                and set(occ.files) == {"counts", "positive_atom_counts", "concept_ids", "is_smarts"})
            check("T_arrays_local_molecule_order_dtypes", labels["labels"].shape == (12, 91)
                and labels["labels"].dtype == np.bool_ and atoms["mol_offsets"].tolist() == [0, 7, 12]
                and atoms["cid"].tolist() == [9001] * 7 + [9002] * 5
                and atoms["mol_idx"].dtype == np.int32 and atoms["atom_idx"].dtype == np.int16
                and occ["counts"].shape == (2, 91) and occ["counts"].dtype == np.int32)
            acid = next(i for i, c in enumerate(concepts) if c["name"] == "fg:CarboxylicAcid")
            check("original_FrozenLabeler_occurrence_vs_atom_union", int(occ["counts"][0, acid]) == 2
                and int(positive_counts[0, acid]) == 6 and dimensions["n_atoms_T"] == 12)
            check("function_occurrence_sentinel_preserved", np.all(occ["counts"][:, ~occ["is_smarts"]] == -1))

    report = {"status": "PASS_SYNTHETIC_T_DATA" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "completed_at_utc": td.utc_now(), "runtime": td.runtime(),
        "counts": {"checks": len(checks), "passed": sum(c["status"] == "PASS" for c in checks),
                   "failed": sum(c["status"] == "FAIL" for c in checks)},
        "execution_ledger": {"actual_T_input_reads": 0, "actual_T_label_calls": 0,
            "synthetic_molecule_label_calls": 2, "model_loads": 0, "model_forwards": 0, "fits": 0},
        "checks": checks}
    td.dump_json(td.CHECKS, report)
    print(json.dumps({"status": report["status"], **report["counts"],
        "failures": [c for c in checks if c["status"] == "FAIL"]}, indent=2))
    return 0 if report["status"] == "PASS_SYNTHETIC_T_DATA" else 1


if __name__ == "__main__":
    with rdBase.BlockLogs():
        raise SystemExit(run())
