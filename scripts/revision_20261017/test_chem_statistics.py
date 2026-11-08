"""Tiny independent golden references, synthetic timing, and 512-token spot check.

No D/T path is opened. This is the one statistics engineering acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from chem_model import ChembertaMTR, NativeMeanIntervention, batch_from_audits
from chem_statistics import (CachedScalar, average_precision, bootstrap_summary, cluster_weights,
                             estimands, fixed_winner_ap, molecule_label_counts,
                             select_winners, weighted_prevalence)
from semantic_mapper import SemanticMapper

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_20261017/chemistry_D"


def direct_threshold_ap(scores, labels):
    """Independent tiny threshold definition, without sorting/cumulative kernels."""
    scores = np.asarray(scores, np.float32)
    labels = np.asarray(labels, bool)
    if not labels.sum():
        return float("nan")
    previous_recall, ap = 0.0, 0.0
    for threshold in sorted(set(scores.tolist()), reverse=True):
        predicted = scores >= threshold
        true = int(labels[predicted].sum())
        recall = true / int(labels.sum())
        ap += (recall - previous_recall) * true / int(predicted.sum())
        previous_recall = recall
    return ap


def assert_close(a, b, tolerance=2e-14):
    if not np.allclose(a, b, atol=tolerance, rtol=tolerance, equal_nan=True):
        raise AssertionError(f"different: {a!r} versus {b!r}")


def goldens():
    checks = []
    scores = np.array([.4, .4, -.2, .8, .8, -.2], np.float32)
    labels = np.array([1, 0, 1, 0, 1, 0], bool)
    molecule = np.array([0, 0, 1, 2, 2, 2], np.int32)
    weights = np.array([[2, 0, 1], [0, 1, 3], [1, 1, 1]], np.int32)
    cached = CachedScalar.prepare(scores, labels, molecule)
    observed = cached.weighted_ap(weights)[:, 0]
    negative = cached.weighted_ap(weights, positive_pole=False)[:, 0]
    expected, expected_negative = [], []
    for w in weights:
        indices = np.concatenate([np.tile(np.flatnonzero(molecule == m), w[m]) for m in range(3)])
        expected.append(direct_threshold_ap(scores[indices], labels[indices]))
        expected_negative.append(direct_threshold_ap(-scores[indices], labels[indices]))
    assert_close(observed, expected)
    assert_close(negative, expected_negative)
    checks.append({"check": "weighted_molecule_AP_matches_explicit_whole_molecule_replication_with_ties", "status": "PASS",
                   "observed": observed.tolist(), "reference": expected, "negative_pole": negative.tolist()})
    constant = np.zeros(6, np.float32)
    pos_counts, atom_counts = molecule_label_counts(labels, molecule, 3)
    prevalence = weighted_prevalence(pos_counts, atom_counts, weights)
    constant_ap = CachedScalar.prepare(constant, labels, molecule).weighted_ap(weights)
    assert_close(constant_ap, prevalence)
    assert_close(average_precision(np.array([0, 1], np.float32), np.array([1, 0])), [.5])
    assert_close(average_precision(np.array([0, 1], np.float32), np.array([1, 0]), positive_pole=False), [1.0])
    checks.append({"check": "constant_score_AP_equals_weighted_atom_prevalence_and_negative_pole", "status": "PASS"})
    assert_close(average_precision(scores, np.zeros(6, bool)), [np.nan])
    near = np.array([1, np.nextafter(np.float32(1), np.float32(2))], np.float32)
    assert_close(average_precision(near, [0, 1]), [1.0])
    checks.append({"check": "no_positive_is_NA_and_adjacent_float32_scores_are_not_merged", "status": "PASS"})
    constant_matrix = np.zeros((6, 464), np.float32)
    winner, ap = select_winners(constant_matrix, labels[:, None])
    if winner.tolist() != [0]:
        raise AssertionError("constant tie winner must be unit0 positive")
    tied = np.zeros((6, 464), np.float32)
    tied[:, 0] = -labels.astype(np.float32)
    tied[:, 1] = labels
    winner2, _ = select_winners(tied, labels[:, None])
    if winner2.tolist() != [464]:
        raise AssertionError("unit0 negative must precede unit1 positive")
    checks.append({"check": "winner_tie_unit_low_then_positive_pole", "status": "PASS",
                   "constant_winner": winner.tolist(), "opposite_pole_tie_winner": winner2.tolist()})
    ell = np.zeros((4, 10, 3), np.float64)
    ell[0, 0] = [-2, 3, -4]
    ell[1:, 0] = [[-10, 0, 0], [0, -10, 0], [0, 0, -10]]
    result = estimands(ell)
    assert_close(result["Theta"], -1)
    assert_close(result["G_init"], -1)
    # median of model-level means would incorrectly add 10/3.
    wrong_order = np.mean(ell[0, 0]) - np.median(np.mean(ell[1:, 0], axis=1))
    if abs(float(result["G_init"]) - wrong_order) < 1:
        raise AssertionError("fixture does not distinguish median order")
    checks.append({"check": "Theta_Ginit_per_concept_median_before_mean_and_negative_values_retained", "status": "PASS",
                   "Theta": float(result["Theta"]), "G_init": float(result["G_init"]), "wrong_order_value": float(wrong_order)})
    weights1 = cluster_weights(11, 20, 42016)
    weights2 = cluster_weights(11, 20, 42016)
    if not np.array_equal(weights1, weights2) or not np.all(weights1.sum(1) == 11):
        raise AssertionError("cluster resampling identity failed")
    replicated = np.broadcast_to(ell, (100, 4, 10, 3)).copy()
    replicated[0] = np.nan
    summary = bootstrap_summary(ell, replicated, precision=True)
    if summary["undefined_replicates"] != 1 or not summary["reliable_interval_gate"]:
        raise AssertionError("1 percent invalid boundary differs")
    replicated[1] = np.nan
    if bootstrap_summary(ell, replicated, precision=True)["precision_gate"]:
        raise AssertionError("more than1 percent invalid must fail")
    checks.append({"check": "shared_seeded_molecule_resampling_and_undefined_replicate_gate", "status": "PASS"})
    return checks


def timed_statistics():
    rng = np.random.default_rng(20261018)  # Synthetic timing only.
    molecules, atoms_per, concepts = 160, 12, 12
    n = molecules * atoms_per
    a = rng.normal(size=(n, 464)).astype(np.float32)
    y = (rng.random((n, concepts)) < np.linspace(.02, .4, concepts)).astype(bool)
    mol = np.repeat(np.arange(molecules, dtype=np.int32), atoms_per)
    qraw = np.random.default_rng(30000).standard_normal((464, 464))
    tick = time.perf_counter()
    q, r = np.linalg.qr(qraw)
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)
    q_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    rotated = a @ q.astype(np.float32)
    rotation_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    winners, ap = select_winners(rotated, y)
    selection_seconds = time.perf_counter() - tick
    weights = cluster_weights(molecules, 500, 42016)
    tick = time.perf_counter()
    aps = fixed_winner_ap(rotated, y, mol, winners, weights)
    bootstrap_seconds = time.perf_counter() - tick
    if not np.isfinite(aps).all():
        raise AssertionError("timing fixture unexpectedly undefined")
    # Conservative dimensional scaling is planning evidence, not an empirical guarantee.
    # Actual D counts are intentionally not read here: maximum 5000*50=250000 atoms,
    # precision1000*50=50000, C<=91, 40 network/bases and an additional fullD pass.
    dfit_scale = (200000 / n) * (91 / concepts) * (math.log2(200000) / math.log2(n)) * 40
    full_scale = (250000 / n) * (91 / concepts) * (math.log2(250000) / math.log2(n)) * 40
    bootstrap_scale = (50000 / n) * (91 / concepts) * 40
    return {"synthetic_only": True, "atoms": n, "molecules": molecules, "concepts": concepts,
            "width": 464, "candidates": 928, "bootstrap_replicates": 500,
            "QR_float64_seconds": q_seconds, "rotation_float32_seconds": rotation_seconds,
            "all_candidate_all_concept_selection_seconds": selection_seconds,
            "fixed_winner_500_cluster_bootstrap_seconds": bootstrap_seconds,
            "distinct_selected_candidates": len(set(winners.tolist())),
            "planning_worst_geometry_atoms_D": 250000, "planning_max_C": 91,
            "linear_work_plus_sort_scaling_selection_hours_Dfit_and_fullD": selection_seconds * (dfit_scale + full_scale) / 3600,
            "linear_work_scaling_bootstrap_hours_Dprecision": bootstrap_seconds * bootstrap_scale / 3600,
            "scaling_limitation": "rough operation scaling; actual data/cache/ties/CPU contention not measured; hard stops do not imply completion"}


def token512_spotcheck(seconds):
    model, mapper = ChembertaMTR.from_official_checkpoint(), SemanticMapper()
    strings = ["C" * 50, "CC(C)" * 16 + "CC", "c1ccccc1" * 8, "CCO"]
    audits = [mapper.audit_molecule(s, require_canonical=False) for s in strings]
    batch = batch_from_audits(audits, pad_to=512)
    unpadded = batch_from_audits(audits)
    a, b = model(batch), model(unpadded)
    diff = float((a.normalized_predictions - b.normalized_predictions).abs().max())
    if diff > 3e-5:
        raise AssertionError("512-right-pad output invariance failed")
    patch = NativeMeanIntervention((0,), torch.tensor([0.0]), batch.scope("atoms"))
    records = []
    for name, intervention in (("base", None), ("single_native_unit", patch)):
        for _ in range(2):
            model(batch, intervention=intervention)
        started = datetime.now(timezone.utc).isoformat()
        tick, iterations = time.perf_counter(), 0
        while time.perf_counter() - tick < seconds:
            model(batch, intervention=intervention)
            iterations += 1
        elapsed = time.perf_counter() - tick
        records.append({"mode": name, "batch_size": 4, "tokens": 512, "iterations": iterations,
                        "elapsed_wall_seconds": elapsed, "started_utc": started,
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "molecules_per_second": iterations * 4 / elapsed,
                        "seconds_per_molecule": elapsed / (iterations * 4)})
    base_seconds = records[0]["seconds_per_molecule"] * 60000
    intervention_seconds = records[1]["seconds_per_molecule"] * 10000
    return {"right_pad_prediction_max_abs_difference": diff, "tolerance": 3e-5,
            "actual_input_token_lengths": [len(a["input_ids"]) for a in audits], "right_padded_length": 512,
            "timings": records, "base_60000_estimated_seconds": base_seconds,
            "intervention_10000_TOTAL_molecule_evaluations_estimated_seconds": intervention_seconds,
            "intervention_budget_is_already_condition_aggregated": True,
            "base_two_hour_cap_contains_2x_measured_estimate": base_seconds * 2 <= 7200,
            "intervention_two_hour_cap_contains_2x_measured_estimate": intervention_seconds * 2 <= 7200,
            "suggested_base_hard_stop_seconds": 7200, "suggested_intervention_hard_stop_seconds": 7200,
            "budget_scope": "encoder forward only; AP/rotation/bootstrap have separate budgets"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--forward-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("threads must be1..4")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=args.threads):
        checks = goldens()
        timing = timed_statistics()
        tokens = token512_spotcheck(args.forward_seconds)
    result = {"status": "PASS_SYNTHETIC_STATISTICS_AND_512_TOKEN_SPOTCHECK", "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "threads": args.threads,
              "checks": checks, "statistics_timing": timing, "token512_spotcheck": tokens,
              "D_or_T_files_read": False, "scientific_forward_count": 0, "D_fit_or_winner_selection_count": 0}
    (OUT / "STATISTICS_ENGINEERING.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
