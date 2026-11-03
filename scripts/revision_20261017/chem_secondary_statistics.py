"""Paired, molecule-level statistics for pre-specified chemistry secondary quantities."""
from __future__ import annotations
import numpy as np
from chem_mechanism_plan import ranks

MECHANISM_QUANTILES = (0.05 / (2 * 3), 1 - 0.05 / (2 * 3))


def interval(values, quantiles):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    ci = np.quantile(values[valid], quantiles, method="linear").tolist() if valid.any() else [None, None]
    return {"ci": ci, "replicates": len(values), "defined_replicates": int(valid.sum()),
            "undefined_replicates": int((~valid).sum()), "quantiles": list(quantiles),
            "quantile_method": "linear"}


def panel_brier(labels, predictions, molecules, weights):
    """Pooled-atom paired increment, then equal-weight concept mean per replicate.

    predictions columns are token/r0/L/Lj. Each bootstrap weight is a molecule
    multiplicity and therefore multiplies both positive and negative atoms.
    """
    labels = np.asarray(labels, np.float64)
    predictions = np.asarray(predictions, np.float64)
    molecules = np.asarray(molecules, np.int32)
    weights = np.asarray(weights)
    if labels.ndim != 2 or predictions.shape != (*labels.shape, 4) or molecules.shape != (len(labels),):
        raise ValueError("Panel Brier input geometry differs")
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Non-probability predictions")
    if weights.ndim != 2 or molecules.min() < 0 or molecules.max() >= weights.shape[1]:
        raise ValueError("Molecule bootstrap geometry differs")
    differences = np.square(labels - predictions[..., 2]) - np.square(labels - predictions[..., 3])
    sums = np.zeros((weights.shape[1], labels.shape[1]), np.float64)
    np.add.at(sums, molecules, differences)
    counts = np.bincount(molecules, minlength=weights.shape[1]).astype(np.float64)
    denominators = weights @ counts
    if np.any(denominators <= 0):
        raise ValueError("Empty bootstrap molecule sample")
    replicated_per_concept = (weights @ sums) / denominators[:, None]
    replicated = replicated_per_concept.mean(axis=1)
    point_per_concept = differences.mean(axis=0)
    return {"Delta_L_panel": float(point_per_concept.mean()), "per_concept": point_per_concept.tolist(),
            "panel_ci95": interval(replicated, (.025, .975)), "replicated_panel": replicated,
            "Brier_per_concept_reference": np.square(labels[..., None] - predictions).mean(axis=0).tolist(),
            "weighting": "pooled atoms within each concept; equal weight over frozen panel concepts",
            "inference": "secondary; cannot rescue primary IUT; per-concept values descriptive"}


def H_value(deltas, positive):
    deltas = np.asarray(deltas, np.float64)
    positive = np.asarray(positive, bool)
    if deltas.ndim != 2 or deltas.shape[1] < 6 or positive.shape != (len(deltas),):
        raise ValueError("Expected selected and five control columns")
    if not positive.any() or positive.all():
        raise ValueError("Both molecule strata required")
    contrast = np.abs(deltas[positive, :6]).mean(0) - np.abs(deltas[~positive, :6]).mean(0)
    return float(contrast[0] - np.median(contrast[1:6]))


def mechanism_effect(deltas, positive, *, replicates=2000, seed=45017, budget_check=None):
    deltas = np.asarray(deltas, np.float64)
    positive = np.asarray(positive, bool)
    if not np.isfinite(deltas).all():
        raise ValueError("Nonfinite mechanism output changes")
    point = H_value(deltas, positive)
    yes, no = np.flatnonzero(positive), np.flatnonzero(~positive)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, np.float64)
    mask = np.r_[np.ones(len(yes), bool), np.zeros(len(no), bool)]
    for index in range(replicates):
        if budget_check is not None and index % 32 == 0:
            budget_check()
        sampled = np.r_[rng.choice(yes, len(yes), replace=True), rng.choice(no, len(no), replace=True)]
        values[index] = H_value(deltas[sampled], mask)
    ci = interval(values, MECHANISM_QUANTILES)
    return {"H": point, "H_interval": ci, "family_size_fixed": 3,
            "positive_lower_bound": bool(ci["ci"][0] is not None and ci["ci"][0] > 0),
            "seed": seed, "resampling": "molecules with replacement separately within positive and negative strata",
            "n_positive": len(yes), "n_negative": len(no),
            "mean_abs_delta_positive": np.abs(deltas[positive]).mean(0).tolist(),
            "mean_abs_delta_negative": np.abs(deltas[~positive]).mean(0).tolist(),
            "signed_mean_delta_positive": deltas[positive].mean(0).tolist(),
            "signed_mean_delta_negative": deltas[~positive].mean(0).tolist(),
            "replicated_H": values}


def spearman(left, right):
    x, y = ranks(left), ranks(right)
    x -= x.mean(); y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-12 else float("nan")


def output_association(counts, output, D_sign, *, replicates=2000, seed=45018, budget_check=None):
    counts, output = np.asarray(counts), np.asarray(output)
    if counts.ndim != 1 or output.shape != counts.shape or D_sign not in (-1, 1):
        raise ValueError("Frozen output/count/sign identity differs")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, np.float64)
    for index in range(replicates):
        if budget_check is not None and index % 16 == 0:
            budget_check()
        sample = rng.integers(0, len(counts), size=len(counts))
        values[index] = spearman(counts[sample], output[sample])
    oriented = D_sign * values
    ci = interval(oriented, MECHANISM_QUANTILES)
    point = spearman(counts, output)
    return {"raw_spearman": point, "D_association_sign": D_sign, "oriented_spearman": D_sign * point,
            "raw_interval": interval(values, MECHANISM_QUANTILES), "oriented_interval": ci,
            "same_direction_positive_lower_bound": bool(ci["ci"][0] is not None and ci["ci"][0] > 0
                                                         and ci["undefined_replicates"] == 0),
            "family_size_fixed": 3, "molecules": len(counts), "seed": seed,
            "resampling": "whole T molecules; count and frozen output remain paired; rerank ties each replicate",
            "replicated_raw_rho": values}


def selftest():
    y = np.asarray([[1.], [0.], [1.]])
    p = np.tile(np.array([.5, .5, .5, .75]), (3, 1, 1))
    p[1, 0, 3] = .25
    result = panel_brier(y, p, [0, 0, 1], np.array([[1, 1], [2, 0], [0, 2]]))
    assert result["Delta_L_panel"] == .1875 and np.all(result["replicated_panel"] == .1875)
    unequal = p.copy(); unequal[2, 0, 3] = .5
    result = panel_brier(y, unequal, [0, 0, 1], np.array([[1, 1], [2, 0], [0, 2]]))
    assert np.allclose(result["replicated_panel"], [.125, .1875, 0.])
    d = np.zeros((4, 8)); d[:2, 0] = [3, -3]; d[2:, 0] = [1, -1]
    d[:2, 1:6] = 1
    assert H_value(d, [1, 1, 0, 0]) == 1.0
    result = mechanism_effect(d, [1, 1, 0, 0], replicates=8)
    assert result["H"] == 1 and result["signed_mean_delta_positive"][0] == 0
    assert result["mean_abs_delta_positive"][0] == 3
    assert abs(spearman([0, 0, 1, 2], [0, 0, 2, 3]) - 1) < 1e-14
    assert np.isnan(spearman([1, 1, 1], [0, 1, 2]))
    return {"status": "PASS_SYNTHETIC_ONLY", "checks": ["paired_Brier_orientation", "molecule_atom_denominator",
        "H_median_control_and_mean_absolute", "signed_mean_distinct", "exact_tie_Spearman", "constant_undefined"],
        "real_D_or_T_reads": 0, "model_forwards": 0}


if __name__ == "__main__":
    import json
    from pathlib import Path
    report = selftest()
    path = Path(__file__).resolve().parents[2] / "results/revision_20261017/engineering/SECONDARY_STATISTICS_CHECKS.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
