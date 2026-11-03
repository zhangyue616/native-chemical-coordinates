"""D-only anonymous-output and dose-matched native-unit selection.

This module accepts arrays from the frozen D run. It has no data loading,
network access, model call, or T entry point.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def ranks(values):
    """Average ranks, 1 based, with exact equality ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ranks require a finite vector")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    start = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    end = np.r_[start[1:], len(values)]
    result = np.empty_like(values)
    for left, right in zip(start, end):
        result[order[left:right]] = (left + 1 + right) / 2
    return result


def spearman_with_outputs(counts, outputs):
    counts, outputs = np.asarray(counts), np.asarray(outputs)
    if counts.ndim != 1 or outputs.shape != (len(counts), 199):
        raise ValueError("D counts/head shape mismatch")
    if not np.isfinite(counts).all() or not np.isfinite(outputs).all():
        raise ValueError("Nonfinite D counts/head")
    left = ranks(counts)
    left -= left.mean()
    norm_left = np.linalg.norm(left)
    result = np.full(199, np.nan)
    if norm_left <= EPS:
        return result
    for q in range(199):
        right = ranks(outputs[:, q])
        right -= right.mean()
        norm_right = np.linalg.norm(right)
        if norm_right > EPS:
            result[q] = float(np.dot(left, right) / (norm_left * norm_right))
    return result


def stable_ratio(numerator, denominator):
    if abs(denominator) <= EPS:
        return 1.0 if abs(numerator) <= EPS else float("inf")
    return float(numerator / denominator)


def choose_plan(concept_ids, native_candidates, occurrence_counts, outputs,
                atom_mean, atom_mean_abs, atom_sd, cls_mean, outgoing_norm):
    """Return all predetermined concepts, including failed object gates."""
    ids = list(map(int, concept_ids))
    candidates = np.asarray(native_candidates, dtype=np.int64)
    counts, outputs = np.asarray(occurrence_counts), np.asarray(outputs)
    if len(ids) > 3 or candidates.shape != (len(ids),) or counts.shape != (len(outputs), len(ids)):
        raise ValueError("Mechanism candidate list/count dimensions mismatch")
    arrays = [np.asarray(v, dtype=np.float64) for v in
              (atom_mean, atom_mean_abs, atom_sd, cls_mean, outgoing_norm)]
    if any(v.shape != (464,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Invalid 464-unit D moments")
    means, meanabs, sds, clsmeans, norms = arrays
    if np.any(sds < 0) or np.any(meanabs < 0) or np.any(norms < 0):
        raise ValueError("Negative D scale")
    units = candidates % 464
    if np.any(candidates < 0) or np.any(candidates >= 928):
        raise ValueError("Invalid D winner candidate")
    excluded_units = set(map(int, units))
    dose = sds * norms
    plans = []
    for index, (concept_id, candidate, unit) in enumerate(zip(ids, candidates, units)):
        correlations = spearman_with_outputs(counts[:, index], outputs)
        finite = np.flatnonzero(np.isfinite(correlations))
        if len(finite):
            best = float(np.max(np.abs(correlations[finite])))
            q = int(next(v for v in finite if abs(correlations[v]) >= best - 8 * np.finfo(float).eps))
            rho = float(correlations[q])
        else:
            q, rho = None, None
        eligible = []
        differences = {}
        for control in range(464):
            if control in excluded_units:
                continue
            ratios = {"mean_abs_a": stable_ratio(meanabs[control], meanabs[unit]),
                      "sd_a": stable_ratio(sds[control], sds[unit]),
                      "outgoing_norm": stable_ratio(norms[control], norms[unit]),
                      "residual_dose": stable_ratio(dose[control], dose[unit])}
            if all(0.5 <= ratios[k] <= 2.0 for k in ("mean_abs_a", "sd_a", "outgoing_norm")) and 0.8 <= ratios["residual_dose"] <= 1.25:
                eligible.append(control)
                differences[str(control)] = ratios
        controls = []
        if len(eligible) >= 5:
            controls = np.random.default_rng(44000 + concept_id).choice(eligible, 5, replace=False).tolist()
        failures = []
        if rho is None or abs(rho) < 0.5:
            failures.append("D_output_association_below_0.5_or_undefined")
        if len(eligible) < 5:
            failures.append("fewer_than_five_matched_controls")
        plans.append({
            "concept_id": concept_id, "candidate": int(candidate), "unit": int(unit),
            "pole": "positive" if candidate < 464 else "negative",
            "anonymous_output_q": q, "D_spearman": rho,
            "D_association_sign": None if rho is None else (1 if rho >= 0 else -1),
            "D_all_199_correlations": [float(v) if np.isfinite(v) else None for v in correlations],
            "official_descriptor_name_known": False, "candidate_replacement_allowed": False,
            "atom_mean": float(means[unit]), "cls_mean": float(clsmeans[unit]),
            "D_atom_mean_abs": float(meanabs[unit]), "D_atom_sd": float(sds[unit]),
            "outgoing_norm": float(norms[unit]), "D_residual_dose": float(dose[unit]),
            "matched_pool_size": len(eligible), "control_seed": 44000 + concept_id,
            "control_units": list(map(int, controls)),
            "control_atom_means": [float(means[j]) for j in controls],
            "control_D_residual_doses": [float(dose[j]) for j in controls],
            "control_D_ratios": [differences[str(j)] for j in controls],
            "status": "D_MECHANISM_OBJECT_LOCKABLE" if not failures else "D_MECHANISM_GATE_FAILED",
            "failure_reasons": failures,
        })
    return {"schema": "anonymous-native-mechanism-D/1", "objects": plans,
            "predetermined_concepts": ids, "maximum_candidates": 3,
            "selected_unit_duplicates": len(ids) - len(set(units.tolist())),
            "selected_output_duplicates": len([p for p in plans if p["anonymous_output_q"] is not None]) - len({p["anonymous_output_q"] for p in plans if p["anonymous_output_q"] is not None}),
            "T_reads": 0, "model_forwards": 0, "new_unit_search": False}


def selftest():
    assert np.array_equal(ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
    c = np.asarray([0, 0, 1, 1, 2, 3])
    outputs = np.tile(np.arange(6, dtype=float)[:, None], (1, 199))
    outputs[:, 0] = c
    outputs[:, 1] = -c
    output = choose_plan([24], [464 + 2], c[:, None], outputs,
                         np.zeros(464), np.ones(464), np.ones(464), np.zeros(464), np.ones(464))
    obj = output["objects"][0]
    assert obj["anonymous_output_q"] == 0 and obj["pole"] == "negative"
    assert obj["status"] == "D_MECHANISM_OBJECT_LOCKABLE" and 2 not in obj["control_units"]
    norms = np.full(464, 5.0); norms[2] = 1
    failed = choose_plan([24], [2], c[:, None], outputs,
                         np.zeros(464), np.ones(464), np.ones(464), np.zeros(464), norms)
    assert "fewer_than_five_matched_controls" in failed["objects"][0]["failure_reasons"]
    assert np.isnan(spearman_with_outputs(np.ones(6), outputs)).all()
    return {"status": "PASS_SYNTHETIC_ONLY", "checks": ["average_tie_ranks", "anonymous_lowest_output_tie",
            "negative_pole_same_physical_unit", "selected_unit_exclusion_and_five_controls",
            "insufficient_dose_pool_fails", "constant_count_undefined"], "real_D_or_T_reads": 0}


if __name__ == "__main__":
    import json
    from pathlib import Path
    result = selftest()
    path = Path(__file__).resolve().parents[2] / "results/revision_20261017/engineering/MECHANISM_PLAN_CHECKS.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
