"""Exact threshold AP and molecule-cluster bootstrap for the frozen chemistry lane.

Scores are float32; accumulators/AP are float64. Exact score equality defines
ties. Sorting is reused for both poles, all labels, and all molecule resamples.
Numba only compiles arithmetic loops; no atom-IID or linear approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from numba import njit, prange

WINNER_ATOL = 8 * np.finfo(np.float64).eps


def tie_groups(ascending_scores):
    scores = np.asarray(ascending_scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("require nonempty finite scalar scores")
    if np.any(scores[1:] < scores[:-1]):
        raise ValueError("scores are not ascending")
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(scores)].astype(np.int64)
    return starts, ends


@njit(parallel=True, cache=False)
def _unweighted_both(labels_sorted, starts, ends):
    n_concepts = labels_sorted.shape[0]
    result = np.empty((2, n_concepts), np.float64)
    for c in prange(n_concepts):
        for pole in range(2):
            cumulative_positive = 0.0
            cumulative_count = 0.0
            numerator = 0.0
            for step in range(len(starts)):
                group = len(starts) - step - 1 if pole == 0 else step
                positive = 0.0
                for i in range(starts[group], ends[group]):
                    positive += labels_sorted[c, i]
                cumulative_count += ends[group] - starts[group]
                cumulative_positive += positive
                if positive > 0:
                    numerator += positive * cumulative_positive / cumulative_count
            result[pole, c] = numerator / cumulative_positive if cumulative_positive > 0 else np.nan
    return result


@njit(parallel=True, cache=False)
def _weighted_cached(labels_sorted, molecule_sorted, starts, ends, molecule_weights, positive_pole):
    reps, concepts = len(molecule_weights), len(labels_sorted)
    result = np.empty((reps, concepts), np.float64)
    for rep in prange(reps):
        for c in range(concepts):
            cumulative_positive = 0.0
            cumulative_count = 0.0
            numerator = 0.0
            for step in range(len(starts)):
                group = len(starts) - step - 1 if positive_pole else step
                positive = 0.0
                count = 0.0
                for i in range(starts[group], ends[group]):
                    weight = molecule_weights[rep, molecule_sorted[i]]
                    count += weight
                    if labels_sorted[c, i]:
                        positive += weight
                cumulative_count += count
                cumulative_positive += positive
                if positive > 0:
                    numerator += positive * cumulative_positive / cumulative_count
            result[rep, c] = numerator / cumulative_positive if cumulative_positive > 0 else np.nan
    return result


@dataclass
class CachedScalar:
    order: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    labels_sorted: np.ndarray
    molecule_sorted: np.ndarray

    @classmethod
    def prepare(cls, scores, labels, molecule_indices):
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels)
        molecules = np.asarray(molecule_indices, dtype=np.int32)
        if labels.ndim == 1:
            labels = labels[:, None]
        if labels.shape[0] != len(scores) or molecules.shape != scores.shape:
            raise ValueError("score, label and molecule row shapes differ")
        if not np.all((labels == 0) | (labels == 1)) or np.any(molecules < 0):
            raise ValueError("labels must be binary and molecule IDs nonnegative")
        order = np.argsort(scores, kind="stable")
        starts, ends = tie_groups(scores[order])
        return cls(order, starts, ends, np.ascontiguousarray(labels[order].T, dtype=np.bool_), molecules[order])

    def both_poles(self):
        return _unweighted_both(self.labels_sorted, self.starts, self.ends)

    def weighted_ap(self, molecule_weights, *, positive_pole=True):
        weights = np.asarray(molecule_weights)
        if weights.ndim == 1:
            weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[1] <= int(self.molecule_sorted.max()):
            raise ValueError("molecule weight shape does not cover all molecules")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("molecule weights must be finite and nonnegative")
        return _weighted_cached(self.labels_sorted, self.molecule_sorted, self.starts, self.ends,
                                np.ascontiguousarray(weights, dtype=np.float64), positive_pole)


def average_precision(scores, labels, *, molecule_indices=None, molecule_weights=None, positive_pole=True):
    labels = np.asarray(labels)
    if molecule_indices is None:
        molecule_indices = np.arange(len(labels), dtype=np.int32)
    cached = CachedScalar.prepare(scores, labels, molecule_indices)
    if molecule_weights is None:
        return cached.both_poles()[0 if positive_pole else 1]
    return cached.weighted_ap(molecule_weights, positive_pole=positive_pole)[0]


def select_winners(score_matrix, labels, *, wall_deadline=None, progress=None):
    """One ascending sort per physical unit, shared by 2 poles and all concepts.

    Unit-major iteration with positive pole first implements the frozen winner
    tie-break, independent of the [all +, all -] numerical candidate encoding.
    """
    scores, labels = np.asarray(score_matrix), np.asarray(labels)
    if scores.dtype != np.float32 or scores.ndim != 2 or scores.shape[1] != 464:
        raise ValueError("locked scores must have float32 shape (atoms,464)")
    if labels.ndim != 2 or labels.shape[0] != len(scores):
        raise ValueError("label matrix differs from score rows")
    best = np.full(labels.shape[1], np.nan, np.float64)
    winner = np.full(labels.shape[1], -1, np.int32)
    dummy_molecule = np.zeros(len(scores), np.int32)
    for unit in range(464):
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            raise TimeoutError("AP selection optional wall deadline reached; no T access")
        candidate = CachedScalar.prepare(scores[:, unit], labels, dummy_molecule).both_poles()
        for pole in range(2):
            values = candidate[pole]
            update = np.isfinite(values) & (~np.isfinite(best) | (values > best + WINNER_ATOL))
            best[update] = values[update]
            winner[update] = unit + 464 * pole
        if progress is not None and (unit % 32 == 31 or unit == 463):
            progress(unit + 1)
    return winner, best


def cluster_weights(n_molecules, n_replicates, seed):
    """One shared PCG64 integer-resampling stream; each row carries all atoms."""
    if n_molecules <= 0 or n_replicates <= 0:
        raise ValueError("positive molecule and replicate counts required")
    rng = np.random.default_rng(seed)
    result = np.empty((n_replicates, n_molecules), np.int32)
    for rep in range(n_replicates):
        result[rep] = np.bincount(rng.integers(0, n_molecules, size=n_molecules), minlength=n_molecules)
    return result


def molecule_label_counts(labels, molecule_indices, n_molecules):
    labels = np.asarray(labels, dtype=np.bool_)
    if labels.ndim == 1:
        labels = labels[:, None]
    molecules = np.asarray(molecule_indices, np.int32)
    positives = np.zeros((n_molecules, labels.shape[1]), np.int32)
    np.add.at(positives, molecules, labels)
    atoms = np.bincount(molecules, minlength=n_molecules).astype(np.int32)
    return positives, atoms


def weighted_prevalence(molecule_positive_counts, molecule_atom_counts, weights):
    weights = np.asarray(weights, np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    denominators = weights @ np.asarray(molecule_atom_counts, np.float64)
    numerator = weights @ np.asarray(molecule_positive_counts, np.float64)
    return np.divide(numerator, denominators[:, None], out=np.full_like(numerator, np.nan),
                     where=denominators[:, None] > 0)


def estimands(ell):
    """ell shape (...,4 networks,10 bases,C); concept medians precede means."""
    ell = np.asarray(ell, np.float64)
    if ell.ndim < 3 or ell.shape[-3:-1] != (4, 10):
        raise ValueError("ell must have network=4,basis=10 before concept axis")
    b = ell[..., :, 0, :] - np.median(ell[..., :, 1:, :], axis=-2)
    theta_per_concept = b[..., 0, :] - np.median(b[..., 1:, :], axis=-2)
    g_per_concept = ell[..., 0, 0, :] - np.median(ell[..., 1:, 0, :], axis=-2)
    return {"Theta": np.mean(theta_per_concept, axis=-1), "G_init": np.mean(g_per_concept, axis=-1),
            "Theta_per_concept": theta_per_concept, "G_init_per_concept": g_per_concept,
            "B_per_network_concept": b}


def fixed_winner_ap(score_matrix, labels, molecule_indices, candidates, weights, *, wall_deadline=None, budget_check=None):
    """Sort each selected scalar once; reuse exact ties through every replicate.

    Group concepts sharing a physical unit/pole. Only one scalar's sorted label
    block exists at a time. Return (replicates,concepts), not all 928 scores.
    """
    candidates = np.asarray(candidates)
    result = np.full((len(weights), len(candidates)), np.nan, np.float64)
    for candidate in sorted(set(int(v) for v in candidates if v >= 0)):
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            raise TimeoutError("cluster bootstrap optional wall deadline reached; no T access")
        if budget_check is not None:
            budget_check()
        columns = np.flatnonzero(candidates == candidate)
        cached = CachedScalar.prepare(score_matrix[:, candidate % 464], labels[:, columns], molecule_indices)
        # A bounded replicate chunk preserves one sort/tie cache and gives the
        # cumulative process-CPU budget a checkpoint during a long scalar job.
        for begin in range(0, len(weights), 64):
            if budget_check is not None:
                budget_check()
            result[begin:begin + 64, columns] = cached.weighted_ap(weights[begin:begin + 64], positive_pole=candidate < 464)
    return result


def bootstrap_summary(point_ell, replicated_ell, *, precision=False):
    point, reps = estimands(point_ell), estimands(replicated_ell)
    # One undefined concept/coordinate makes the joint replicate undefined.
    valid = np.isfinite(replicated_ell).all(axis=(-3, -2, -1))
    invalid_fraction = float(1 - valid.mean())
    reliable = int((~valid).sum()) * 100 <= len(valid)
    summary = {"replicates": len(valid), "defined_replicates": int(valid.sum()),
               "undefined_replicates": int((~valid).sum()), "undefined_fraction": invalid_fraction,
               "conditional_interval_after_undefined_exclusion": bool((~valid).any()),
               "reliable_interval_gate": reliable, "quantile_method": "linear"}
    halfwidths = []
    for name in ("Theta", "G_init"):
        values = reps[name][valid]
        interval = np.quantile(values, [.025, .975], method="linear") if len(values) else np.array([np.nan, np.nan])
        halfwidth = float((interval[1] - interval[0]) / 2)
        halfwidths.append(halfwidth)
        summary[name] = {"point": float(point[name]), "ci95": interval.tolist(), "halfwidth": halfwidth}
    if precision:
        projected = float(max(halfwidths) * 1.25 * np.sqrt(1000 / 10000))
        summary.update({"D_precision_molecules": 1000, "target_T_molecules": 10000,
                        "projection_multiplier": 1.25 * np.sqrt(.1),
                        "projected_T_max_halfwidth": projected, "target_halfwidth": float(np.log(1.10)),
                        "precision_gate": bool(reliable and np.isfinite(projected)
                                               and projected <= np.log(1.10)),
                        "T_confirmation_guaranteed": False})
    else:
        summary["joint_IUT_supported"] = bool(reliable and all(
            np.isfinite(summary[n]["ci95"][0]) and summary[n]["ci95"][0] > 0 for n in ("Theta", "G_init")))
    return summary
