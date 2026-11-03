"""Frozen finite input tables; no target SMARTS or model fitting is used as input.

The radius-one/j refinement distributes the original category's prior mass
over observed bins. Consequently a constant j reproduces the base table.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

ALPHA = 20.0
QUANTILES = (0.2, 0.4, 0.6, 0.8)


def key(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def atom_signature(atom):
    return (atom.GetAtomicNum(), atom.GetFormalCharge(), bool(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), str(atom.GetHybridization()))


def input_keys(smiles, anchor_token_ids):
    """Construct exactly one row per RDKit atom, in molecule/atom order."""
    if len(smiles) != len(anchor_token_ids):
        raise ValueError("Molecule and anchor lists differ")
    rows = []
    for text, tokens in zip(smiles, anchor_token_ids):
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() != len(tokens):
            raise ValueError("Exact graph and previously validated anchor count differ")
        for atom, token in zip(mol.GetAtoms(), tokens):
            r0 = (atom.GetAtomicNum(), atom.GetFormalCharge(), bool(atom.GetIsAromatic()))
            neighbors = sorted((str(bond.GetBondType()), atom_signature(bond.GetOtherAtom(atom)))
                               for bond in atom.GetBonds())
            rows.append({"token": str(int(token)), "r0": key(r0),
                         "r1": key((atom_signature(atom), neighbors))})
    return rows


def _counts(keys, values):
    result = defaultdict(lambda: [0, 0.0])
    for category, value in zip(keys, values):
        result[category][0] += 1
        result[category][1] += float(value)
    return dict(result)


def fit_one(rows, labels, scalar, concept_id):
    labels = np.asarray(labels, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    if labels.ndim != 1 or scalar.shape != labels.shape or len(rows) != len(labels):
        raise ValueError("Readout input dimensions differ")
    if not len(rows) or not np.isfinite(scalar).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Invalid labels/scalar")
    pi = float(labels.mean())
    token_counts = _counts([r["token"] for r in rows], labels)
    r0_counts = _counts([r["r0"] for r in rows], labels)
    r1_counts = _counts([r["r1"] for r in rows], labels)
    p_token = {g: float((s + ALPHA * pi) / (n + ALPHA)) for g, (n, s) in token_counts.items()}
    p0 = {g: float((s + ALPHA * pi) / (n + ALPHA)) for g, (n, s) in r0_counts.items()}
    parents = {}
    for row in rows:
        if row["r1"] in parents and parents[row["r1"]] != row["r0"]:
            raise ValueError("Radius-one category does not determine its radius-zero parent")
        parents[row["r1"]] = row["r0"]
    p_l = {g: float((s + ALPHA * p0[parents[g]]) / (n + ALPHA))
           for g, (n, s) in r1_counts.items()}
    cuts = np.unique(np.quantile(scalar, QUANTILES, method="linear"))
    bins = np.searchsorted(cuts, scalar, side="right")
    joint_keys = [key((row["r1"], int(b))) for row, b in zip(rows, bins)]
    joint_counts = _counts(joint_keys, labels)
    joint_parents = {k: row["r1"] for k, row in zip(joint_keys, rows)}
    p_lj = {}
    for gbin, (n_bin, successes) in joint_counts.items():
        category = joint_parents[gbin]
        n_category = r1_counts[category][0]
        alpha_bin = ALPHA * n_bin / n_category
        p_lj[gbin] = float((successes + alpha_bin * p0[parents[category]]) / (n_bin + alpha_bin))
    return {
        "schema": "finite-input-reference/1", "concept_id": int(concept_id),
        "alpha": ALPHA, "global_prevalence": pi, "training_atoms": len(rows),
        "quantiles": list(QUANTILES), "quantile_method": "linear", "bin_side": "right",
        "bin_cuts": cuts.tolist(), "observed_bin_count": int(len(np.unique(bins))),
        "p_token": p_token, "p_r0": p0, "p_L": p_l, "p_Lj": p_lj,
        "counts_token": token_counts, "counts_r0": r0_counts,
        "counts_L": r1_counts, "counts_Lj": joint_counts,
        "category_counts": {"token": len(p_token), "r0": len(p0), "L": len(p_l), "Lj": len(p_lj)},
    }


def predict_one(model, rows, scalar):
    scalar = np.asarray(scalar, dtype=np.float64)
    if scalar.shape != (len(rows),) or not np.isfinite(scalar).all():
        raise ValueError("Invalid test scalar shape or values")
    bins = np.searchsorted(np.asarray(model["bin_cuts"]), scalar, side=model["bin_side"])
    prediction = np.empty((len(rows), 4), dtype=np.float64)
    missing = {name: 0 for name in ("token", "r0", "L", "Lj")}
    pi = model["global_prevalence"]
    for i, (row, b) in enumerate(zip(rows, bins)):
        p0 = model["p_r0"].get(row["r0"], pi)
        p_l = model["p_L"].get(row["r1"], p0)
        gbin = key((row["r1"], int(b)))
        prediction[i] = (model["p_token"].get(row["token"], pi), p0, p_l,
                         model["p_Lj"].get(gbin, p_l))
        missing["token"] += row["token"] not in model["p_token"]
        missing["r0"] += row["r0"] not in model["p_r0"]
        missing["L"] += row["r1"] not in model["p_L"]
        missing["Lj"] += gbin not in model["p_Lj"]
    report = {"atoms": len(rows), "unseen_counts": missing,
              "fallback_fractions": {g: (n / len(rows) if rows else None) for g, n in missing.items()},
              "columns": ["token", "r0", "L", "Lj"]}
    return prediction, report


def paired_brier_contributions(labels, p_l, p_lj):
    y, a, b = map(lambda v: np.asarray(v, dtype=np.float64), (labels, p_l, p_lj))
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("Paired Brier arrays differ")
    return (y - a) ** 2 - (y - b) ** 2


def selftest():
    # Hand-defined categories isolate the smoothing identity from RDKit logic.
    rows = [{"token": "1", "r0": "C", "r1": "same"} for _ in range(8)]
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    constant = fit_one(rows, y, np.ones(8), 7)
    pc, _ = predict_one(constant, rows, np.ones(8))
    assert np.array_equal(pc[:, 2], pc[:, 3])
    refined = fit_one(rows, y, np.arange(8), 7)
    pr, _ = predict_one(refined, rows, np.arange(8))
    assert abs(float(pr[:, 3].mean()) - float(pr[0, 2])) < 1e-14
    expected_first = (0 + (20 * 2 / 8) * 0.5) / (2 + 20 * 2 / 8)
    assert abs(pr[0, 3] - expected_first) < 1e-14
    unknown = [{"token": "2", "r0": "new0", "r1": "new1"}]
    pu, missing = predict_one(refined, unknown, np.asarray([2.0]))
    assert np.array_equal(pu, np.full((1, 4), 0.5)) and all(v == 1 for v in missing["unseen_counts"].values())
    graph_rows = input_keys(["CCO"], [[16, 16, 19]])
    assert len(graph_rows) == 3 and graph_rows[0]["r0"] == graph_rows[1]["r0"]
    assert graph_rows[0]["r1"] != graph_rows[1]["r1"]
    assert np.allclose(paired_brier_contributions([1, 0], [.5, .5], [.75, .25]), [.1875, .1875])
    return {"status": "PASS_SYNTHETIC_ONLY", "checks": [
        "constant_j_exactly_base", "bin_weighted_mean_recovers_base", "hand_smoothing_count",
        "unseen_fallback", "radius_one_distinguishes_same_element_environment", "paired_brier_orientation"],
        "real_D_or_T_reads": 0, "model_forwards": 0}


if __name__ == "__main__":
    report = selftest()
    out = Path(__file__).resolve().parents[2] / "results/revision_20261017/engineering/INPUT_REFERENCE_CHECKS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
