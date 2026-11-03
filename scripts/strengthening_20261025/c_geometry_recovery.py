"""Bounded numerical diagnosis of the single failed dose case; no scientific edits."""
from pathlib import Path
import argparse
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import c_paired_dose as c

DEST = c.OUT / "geometry_recovery"
ROW, T_INDEX, COLUMN, SCALE_INDEX = 200, 8381, 2, 2
GEOMETRY = [8380, 8381, 8382, 8383]


def diagnose(budget):
    if (DEST / "DIAGNOSTIC.json").exists():
        raise RuntimeError("One bounded diagnostic already exists; do not repeat")
    ready = c.read(c.OUT / "READY.json")
    item = ready["objects"][2]
    source = c.OUT / item["parameter_file"]
    if item["concept_id"] != 26 or c.sha(source) != item["parameter_sha256"]:
        raise ValueError("Frozen concept26 parameters differ")
    with np.load(source, allow_pickle=False) as z:
        p = {k: z[k].copy() for k in z.files}
    if int(p["T_molecule_indices"][ROW]) != T_INDEX or int(p["units"][COLUMN]) != 10:
        raise ValueError("Single failed-case identity differs")
    DEST.mkdir(exist_ok=True)
    audits = c.selected_audits(set(GEOMETRY))
    batch = c.batch_from_audits([audits[i] for i in GEOMETRY])
    model = c.ChembertaMTR.from_official_checkpoint()
    budget.reserve(4)
    baseline = model(batch, layer_index=c.PRIMARY_LAYER)
    alpha = np.zeros(4, np.float64)
    alpha[1] = p["alpha"][SCALE_INDEX, ROW, COLUMN]
    result, hook = c.actual_forward(model, batch, p, COLUMN, alpha, budget)
    q = item["q"]
    delta = float(result.normalized_predictions[1, q]) - float(baseline.normalized_predictions[1, q])
    dose = float(hook.dose[1])
    cached_a, _ = c.accepted_array(c.BASE / "activations_T/trained", "atom_a.npy")
    cached_q, _ = c.accepted_array(c.BASE / "activations_T/trained", "normalized_head.npy")
    with np.load(c.BASE / "test/atoms_T.npz", allow_pickle=False) as z:
        offsets = z["mol_offsets"]
    fresh_a = baseline.activations[1, batch.atom_mask[1]].numpy()
    old_a = np.asarray(cached_a[offsets[T_INDEX]:offsets[T_INDEX+1]])
    before = np.load(c.OUT / "concept_26/s1_unit10.npy", allow_pickle=False)
    target = p["target"][SCALE_INDEX]
    restored_error = abs(dose - float(target[ROW]))
    tolerance = 1e-6 * (1 + float(target.max()))
    cached_baseline = np.asarray(cached_q[T_INDEX])
    new_q = baseline.normalized_predictions[1].numpy()
    q_error = float(abs(float(new_q[q]) - float(cached_baseline[q])))
    fresh_full = np.sqrt(np.mean((p["means_model_float32"].astype(np.float64) - fresh_a[:, p["units"]].astype(np.float64))**2, axis=0)) * p["outgoing_norms"]
    np.savez(DEST / "DIAGNOSTIC_ARRAYS.npz", geometry=np.asarray(GEOMETRY),
             baseline=baseline.normalized_predictions.numpy(), intervention=result.normalized_predictions.numpy(),
             activation=fresh_a, dose=hook.dose.numpy(), alpha=alpha,
             failed_row_before=before[ROW], recovered_row=np.asarray([delta, dose]))
    report = {"status": "PASS" if restored_error <= tolerance and q_error <= 1e-5 else "NUMERIC_GATE_UNCLOSED",
        "created_utc": c.utc(), "concept_id": 26, "row": ROW, "T_index": T_INDEX,
        "geometry": GEOMETRY, "original_cache_batch_source": "chem_T_run.py extract_T range(...,10000,4)",
        "auxiliary_rows_enter_statistics": False, "new_evaluations": 8,
        "parameter_sha256": c.sha(source), "target": float(target[ROW]), "alpha": float(alpha[1]),
        "dose_tolerance_original_per_condition": tolerance,
        "before_response_and_dose": before[ROW].tolist(), "restored_response_and_dose": [delta, dose],
        "before_dose_error": float(abs(before[ROW, 1] - target[ROW])), "restored_dose_error": restored_error,
        "restored_vs_cached_all464_activation_max_abs": float(np.max(np.abs(fresh_a.astype(np.float64)-old_a))),
        "restored_vs_cached_all199_baseline_max_abs": float(np.max(np.abs(new_q.astype(np.float64)-cached_baseline))),
        "restored_vs_cached_frozen_q_baseline_abs": q_error,
        "cached_full_dose": p["full_dose"][ROW].tolist(), "restored_full_dose": fresh_full.tolist(),
        "saved_scientific_responses_modified": False,
        "acceptance_rule": "Only numerical gates; neither response sign nor H used"}
    c.write(DEST / "DIAGNOSTIC.json", report)
    return report


def restore(budget):
    diagnostic = c.read(DEST / "DIAGNOSTIC.json")
    if diagnostic["status"] != "PASS" or (DEST / "RESTORATION.json").exists():
        raise RuntimeError("Restoration requires passing one-time diagnostic and no prior restoration")
    ready = c.read(c.OUT / "READY.json")
    item = ready["objects"][2]
    source = c.OUT / item["parameter_file"]
    if c.sha(source) != item["parameter_sha256"]:
        raise ValueError("Frozen parameters changed")
    with np.load(source, allow_pickle=False) as z:
        p = {k: z[k].copy() for k in z.files}
    with np.load(DEST / "DIAGNOSTIC_ARRAYS.npz", allow_pickle=False) as z:
        baseline = z["baseline"][1].copy()
        first_row = z["recovered_row"].copy()
    audits = c.selected_audits(set(GEOMETRY))
    batch = c.batch_from_audits([audits[i] for i in GEOMETRY])
    model = c.ChembertaMTR.from_official_checkpoint()
    responses = np.empty((3, 6, 2), np.float64)
    responses[SCALE_INDEX, COLUMN] = first_row
    conditions = []
    for si, scale in enumerate(c.SCALES):
        for col in range(6):
            if (si, col) != (SCALE_INDEX, COLUMN):
                alpha = np.zeros(4, np.float64)
                alpha[1] = p["alpha"][si, ROW, col]
                result, hook = c.actual_forward(model, batch, p, col, alpha, budget)
                responses[si, col] = [float(result.normalized_predictions[1, item["q"]])-float(baseline[item["q"]]), float(hook.dose[1])]
            error = abs(float(responses[si, col, 1]) - float(p["target"][si, ROW]))
            tolerance = 1e-6 * (1+float(p["target"][si].max()))
            if error > tolerance:
                raise ArithmeticError("Restored condition still exceeds original dose gate")
            path = c.OUT / f"concept_26/s{scale:g}_unit{int(p['units'][col])}.npy"
            before = np.load(path, allow_pickle=False)[ROW].tolist() if path.exists() else None
            conditions.append({"scale": float(scale), "unit": int(p["units"][col]),
                "before_response_and_dose": before, "after_response_and_dose": responses[si,col].tolist(),
                "before_dose_error": abs(before[1]-float(p["target"][si,ROW])) if before else None,
                "after_dose_error": error, "original_tolerance": tolerance})
    np.savez(DEST / "RESTORED_ARRAYS.npz", baseline=baseline, responses=responses,
             geometry=np.asarray(GEOMETRY), row=np.asarray(ROW), T_index=np.asarray(T_INDEX))
    report = {"status": "ALL18_RESTORED_NUMERICALLY", "created_utc": c.utc(),
        "concept_id": 26, "row": ROW, "T_index": T_INDEX, "geometry": GEOMETRY,
        "parameter_sha256": c.sha(source), "baseline_reused_from_diagnostic": True,
        "new_evaluations_this_stage": 68, "total_geometry_evaluations": 76,
        "original_baseline_and_condition_arrays_preserved": True,
        "restored_arrays_sha256": c.sha(DEST / "RESTORED_ARRAYS.npz"), "conditions": conditions,
        "controller_authorization": "2026-10-25: restore this one molecule in baseline and all18 conditions, numerical-only amendment, no gate widening or scientific change"}
    c.write(DEST / "RESTORATION.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("diagnose", "restore"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    budget = c.Budget("single_case_geometry_" + args.stage)
    try:
        with threadpool_limits(limits=4):
            report = diagnose(budget) if args.stage == "diagnose" else restore(budget)
        budget.finish(report["status"])
        print(c.json.dumps(report), flush=True)
    except BaseException:
        budget.finish("FAILED_RETAINED")
        raise


if __name__ == "__main__":
    main()
