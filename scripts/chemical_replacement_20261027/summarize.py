"""One locked whole-molecule bootstrap for the two-coordinate replacement lane.

--synthetic-check never opens T data. --execute requires completed fit/forward
receipts, and refuses to overwrite any planned summary artifact.
"""
from __future__ import annotations

import time
_IMPORT_WALL = time.monotonic()
# Process CPU includes Python startup and all imports in this invocation.
_IMPORT_CPU = 0.

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import csv
import ctypes
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "results/chemical_replacement_20261027"
DOC = ROOT / "docs/chemical_replacement_20261027"
OUT = LANE / "summary"
COORDINATES = (293, 454)
CONCEPTS = (26, 24)
CONDITIONS = ("D_mean", "M1", "M2", "M1_delta_perm", "M2_delta_perm")
STRATA = ("all_T", "any_target_positive_molecule", "all_target_negative_molecule")
ENDPOINTS = ((0, 115, "primary"), (1, 124, "comparison"), (1, 125, "secondary"))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261028


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    bounds = (np.quantile(values[valid], [.025, .975], method="linear")
              if valid.any() else (None, None))
    return dict(ci95_lower=finite_or_none(bounds[0]) if bounds[0] is not None else None,
                ci95_upper=finite_or_none(bounds[1]) if bounds[1] is not None else None,
                bootstrap_defined=int(valid.sum()),
                bootstrap_undefined=int((~valid).sum()))


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64))
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def r_squared(sse, count, centered_sum, centered_square_sum):
    """Equivalent to recomputing each draw's pooled-atom T mean, stably."""
    adjustment = ratio(np.asarray(centered_sum) ** 2, count)
    sst = np.asarray(centered_square_sum, dtype=np.float64) - adjustment
    scale = np.maximum(np.abs(centered_square_sum), np.abs(adjustment))
    if np.any(sst < -64 * np.finfo(np.float64).eps * np.maximum(scale, 1.)):
        raise ArithmeticError("Pooled centered SST is substantially negative")
    sst = np.maximum(sst, 0.)
    result = 1. - ratio(sse, sst)
    return result


def atom_sums(values, offsets):
    return np.add.reduceat(np.asarray(values, dtype=np.float64), offsets[:-1], axis=0)


def molecular_strata(target_labels, offsets):
    positive = atom_sums(target_labels, offsets) > 0
    return {STRATA[0]: np.ones(len(offsets) - 1, dtype=bool),
            STRATA[1]: positive, STRATA[2]: ~positive}


def activation_stats(actual, predictions, offsets):
    actual = np.asarray(actual, dtype=np.float64)
    center = float(actual.mean(dtype=np.float64))
    centered = actual - center
    return dict(count=np.diff(offsets).astype(np.float64),
                centered_sum=atom_sums(centered, offsets),
                centered_square_sum=atom_sums(centered ** 2, offsets),
                sse=atom_sums((np.asarray(predictions, dtype=np.float64)
                               - actual[:, None]) ** 2, offsets),
                center=center)


class WeightedStatistics:
    """Only a small float64 per-molecule matrix is multiplied by shared draws."""

    def __init__(self, n_molecules):
        self.n = n_molecules
        self.columns = []

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n,) or not np.isfinite(values).all():
            raise ValueError("Invalid per-molecule sufficient statistic")
        self.columns.append(values)
        return len(self.columns) - 1

    def matrix(self):
        return np.ascontiguousarray(np.column_stack(self.columns), dtype=np.float64)


def draw_counts(n_molecules, n_draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                path=None, check=None):
    if n_molecules >= np.iinfo(np.int16).max:
        raise ValueError("int16 draw counts are not adequate")
    if path is None:
        weights = np.empty((n_draws, n_molecules), dtype=np.int16)
    else:
        weights = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16,
                                            shape=(n_draws, n_molecules))
    rng = np.random.Generator(np.random.PCG64(seed))
    for draw in range(n_draws):
        # Exactly N sampled global molecule indices, then counts. Never sample
        # strata, atom rows, conditions, coordinates, or endpoints independently.
        indices = rng.integers(0, n_molecules, size=n_molecules)
        weights[draw] = np.bincount(indices, minlength=n_molecules).astype(np.int16)
        if check is not None and draw % 128 == 0:
            check()
    if isinstance(weights, np.memmap):
        weights.flush()
    if not np.all(np.sum(weights, axis=1, dtype=np.int64) == n_molecules):
        raise AssertionError("Bootstrap draw size mismatch")
    return weights


def weighted_totals(weights, matrix, check=None, block_size=128):
    result = np.empty((len(weights), matrix.shape[1]), dtype=np.float64)
    for start in range(0, len(weights), block_size):
        stop = min(start + block_size, len(weights))
        result[start:stop] = np.asarray(weights[start:stop], dtype=np.float64) @ matrix
        if check is not None:
            check()
    return result


def make_row(kind, coordinate, concept, stratum, condition, metric,
             n_molecules, n_atoms, point, boot=None, **extra):
    record = dict(kind=kind, coordinate=coordinate, concept=concept,
                  stratum=stratum, condition=condition, metric=metric,
                  n_molecules=int(n_molecules), n_atoms=int(n_atoms),
                  estimate=finite_or_none(point), ci95_lower=None, ci95_upper=None,
                  bootstrap_defined=None, bootstrap_undefined=None,
                  uncertainty="point_only")
    if boot is not None:
        record.update(interval(boot), uncertainty="marginal_95_percentile_linear")
    record.update(extra)
    return record


def output_specifications(baseline, conditions, dose_rms, offsets,
                          all_strata, sufficient):
    specs, denominator_cache = [], {}
    n_atoms_per_molecule = np.diff(offsets)
    rms_by_coordinate = []
    for coordinate_axis in range(2):
        delta = np.asarray(conditions[coordinate_axis], dtype=np.float64) - baseline[None]
        rms_by_coordinate.append(np.sqrt(np.mean(delta ** 2, axis=2, dtype=np.float64)))
    for coordinate_axis, endpoint, role in ENDPOINTS:
        delta_q = (np.asarray(conditions[coordinate_axis, :, :, endpoint], dtype=np.float64)
                   - np.asarray(baseline[:, endpoint], dtype=np.float64)[None])
        absolute = np.abs(delta_q)
        for stratum, mask in all_strata[coordinate_axis].items():
            key = (coordinate_axis, stratum)
            if key not in denominator_cache:
                denominator_cache[key] = sufficient.add(mask)
            den = denominator_cache[key]
            n_molecules, n_atoms = int(mask.sum()), int(n_atoms_per_molecule[mask].sum())
            for condition_axis, condition in enumerate(CONDITIONS):
                values = [("mae", absolute[condition_axis]),
                          ("signed_mean_shift", delta_q[condition_axis]),
                          ("mean_normalized199_rms", rms_by_coordinate[coordinate_axis][condition_axis]),
                          ("paired_mae_minus_D_mean", absolute[condition_axis] - absolute[0])]
                if condition_axis in (1, 2):
                    values.append(("paired_mae_minus_own_delta_perm",
                                   absolute[condition_axis] - absolute[condition_axis + 2]))
                for metric, per_molecule in values:
                    num = sufficient.add(np.where(mask, per_molecule, 0.))
                    point = per_molecule[mask].mean(dtype=np.float64) if n_molecules else np.nan
                    specs.append(dict(coordinate=COORDINATES[coordinate_axis],
                                      concept=CONCEPTS[coordinate_axis], stratum=stratum,
                                      condition=condition, metric=metric,
                                      n_molecules=n_molecules, n_atoms=n_atoms, point=point,
                                      num=num, den=den, endpoint=endpoint, endpoint_role=role))
                tail = (np.quantile(absolute[condition_axis, mask], .95, method="linear")
                        if n_molecules else np.nan)
                specs.append(dict(coordinate=COORDINATES[coordinate_axis],
                                  concept=CONCEPTS[coordinate_axis], stratum=stratum,
                                  condition=condition, metric="absolute_error_p95",
                                  n_molecules=n_molecules, n_atoms=n_atoms, point=tail,
                                  endpoint=endpoint, endpoint_role=role))
    # Activation displacement dose is retained as a point-only diagnostic; it is
    # distinct from normalized199 output RMS and is not an extra endpoint test.
    dose_rows = []
    for c in range(2):
        for stratum, mask in all_strata[c].items():
            for k, condition in enumerate(CONDITIONS):
                dose_rows.append(make_row("dose", COORDINATES[c], CONCEPTS[c], stratum,
                    condition, "mean_realized_activation_displacement_rms",
                    mask.sum(), n_atoms_per_molecule[mask].sum(),
                    np.asarray(dose_rms[c, k, mask], dtype=np.float64).mean()
                    if mask.any() else np.nan))
    return specs, dose_rows


def activation_specifications(actual, predictions, offsets, all_strata, sufficient):
    specs = []
    for c in range(2):
        stats = activation_stats(actual[:, c], predictions[:, c], offsets)
        for stratum, mask in all_strata[c].items():
            count_id = sufficient.add(np.where(mask, stats["count"], 0.))
            sum_id = sufficient.add(np.where(mask, stats["centered_sum"], 0.))
            square_id = sufficient.add(np.where(mask, stats["centered_square_sum"], 0.))
            count, csum, square = (stats[key][mask].sum(dtype=np.float64)
                                  for key in ("count", "centered_sum", "centered_square_sum"))
            for k, condition in enumerate(CONDITIONS[:3]):
                sse = stats["sse"][mask, k].sum(dtype=np.float64)
                sse_id = sufficient.add(np.where(mask, stats["sse"][:, k], 0.))
                specs.append(dict(coordinate=COORDINATES[c], concept=CONCEPTS[c],
                    condition=condition, stratum=stratum,
                    n_molecules=int(mask.sum()), n_atoms=int(count),
                    count_id=count_id, sum_id=sum_id, square_id=square_id, sse_id=sse_id,
                    mse=ratio(sse, count), r2=r_squared(sse, count, csum, square)))
    return specs


def descriptive_activation(actual, predictions, targets, bins, mol_idx):
    rows = []
    for c in range(2):
        masks = [(f"target_atom_label_{label}", targets[:, c] == label) for label in (0, 1)]
        masks += [(f"D_fixed_bin_{bin_id}", bins[:, c] == bin_id) for bin_id in range(5)]
        for stratum, mask in masks:
            values = np.asarray(actual[mask, c], dtype=np.float64)
            n_atoms = len(values)
            n_molecules = len(np.unique(mol_idx[mask]))
            sst = np.sum((values - values.mean()) ** 2, dtype=np.float64) if n_atoms else 0.
            for k, condition in enumerate(CONDITIONS[:3]):
                sse = np.sum((predictions[mask, c, k] - values) ** 2, dtype=np.float64)
                for metric, point in (("pooled_atom_mse", ratio(sse, n_atoms)),
                                      ("pooled_atom_r2", 1. - ratio(sse, sst))):
                    rows.append(make_row("activation", COORDINATES[c], CONCEPTS[c],
                        stratum, condition, metric, n_molecules, n_atoms, point))
    return rows


def bin_chemistry(bins, labels, label_ids, mol_idx):
    rows = []
    uncaptured = ~np.any(labels != 0, axis=1)
    for c in range(2):
        for bin_id in range(5):
            mask = bins[:, c] == bin_id
            n_atoms, n_molecules = int(mask.sum()), len(np.unique(mol_idx[mask]))
            hits = np.count_nonzero(labels[mask], axis=0)
            unhit = int(np.count_nonzero(uncaptured[mask]))
            for label_id, count in zip(label_ids, hits):
                rows.append(dict(coordinate=COORDINATES[c], target_concept=CONCEPTS[c],
                    D_fixed_bin=bin_id, dictionary_concept_id=int(label_id),
                    n_atoms=n_atoms, n_molecules=n_molecules,
                    hit_atom_count=int(count),
                    hit_atom_proportion=float(count / n_atoms) if n_atoms else None,
                    uncaptured_dictionary_atom_count=unhit,
                    uncaptured_dictionary_atom_proportion=unhit / n_atoms if n_atoms else None,
                    composition_interpretation="overlapping_atom_labels_not_exclusive_components"))
    return rows


def write_csv_exclusive(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_memory():
    if sys.platform != "win32":
        raise RuntimeError("Resource accounting is locked to the authorized Windows runtime")
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return dict(peak_rss_bytes=int(counters.PeakWorkingSetSize),
                private_commit_bytes=int(counters.PrivateUsage),
                peak_private_commit_bytes=int(counters.PeakPagefileUsage))


def synthetic_check():
    """Hand-computable cluster/paired/SST cases, with no scientific input IO."""
    offsets = np.array([0, 2, 3, 5], dtype=np.int64)
    actual = np.array([0., 2., 4., 6., 8.])
    predictions = np.column_stack([np.full(5, 2.), [0., 2., 3., 5., 9.],
                                   [1., 3., 4., 6., 8.]])
    stats = activation_stats(actual, predictions, offsets)
    np.testing.assert_allclose(stats["sse"].sum(axis=0), [60., 3., 2.], rtol=0, atol=0)
    np.testing.assert_allclose(ratio(stats["sse"].sum(axis=0), 5), [12., .6, .4])
    np.testing.assert_allclose(r_squared(stats["sse"].sum(axis=0), 5,
        stats["centered_sum"].sum(), stats["centered_square_sum"].sum()), [-.5, .925, .95])
    weights = np.array([[2, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int16)
    matrix = np.column_stack([stats["count"], stats["centered_sum"],
                              stats["centered_square_sum"], stats["sse"]])
    totals = weighted_totals(weights, matrix, block_size=2)
    np.testing.assert_allclose(ratio(totals[0, 3:], totals[0, 0]), [10., 1 / 3, 2 / 3])
    np.testing.assert_allclose(r_squared(totals[0, 3:], totals[0, 0],
        totals[0, 1], totals[0, 2]), [-1 / 9, 26 / 27, 25 / 27])
    assert np.isnan(r_squared(totals[1, 3], totals[1, 0], totals[1, 1], totals[1, 2]))
    for draw, count in enumerate(weights):
        explicit_atoms = np.concatenate([np.arange(offsets[m], offsets[m + 1])
                                         for m in np.repeat(np.arange(3), count)])
        y = actual[explicit_atoms]
        sse = np.sum((predictions[explicit_atoms] - y[:, None]) ** 2, axis=0)
        sst = np.sum((y - y.mean()) ** 2)
        np.testing.assert_allclose(totals[draw, 3:], sse)
        np.testing.assert_allclose(r_squared(totals[draw, 3:], totals[draw, 0],
            totals[draw, 1], totals[draw, 2]), 1 - ratio(sse, sst), equal_nan=True)
    strata = molecular_strata(np.array([0, 1, 0, 0, 0]), offsets)
    np.testing.assert_array_equal(strata[STRATA[1]], [True, False, False])
    np.testing.assert_array_equal(strata[STRATA[2]], [False, True, True])
    absent_positive_den = weights @ strata[STRATA[1]].astype(np.float64)
    assert np.isnan(ratio(0., absent_positive_den)[1])
    baseline = np.zeros((3, 199), dtype=np.float32)
    conditions = np.zeros((2, 5, 3, 199), dtype=np.float32)
    conditions[0, 0, :, 115] = [1., -2., 3.]
    conditions[0, 1, :, 115] = [.5, -1., 1.5]
    conditions[0, 3, :, 115] = [1.5, -.5, 3.]
    sufficient = WeightedStatistics(3)
    specs, _ = output_specifications(baseline, conditions, np.zeros((2, 5, 3)),
        offsets, [strata, strata], sufficient)
    boot_totals = weighted_totals(weights, sufficient.matrix())
    lookup = {(s["coordinate"], s["endpoint"], s["stratum"], s["condition"], s["metric"]): s
              for s in specs}
    def output(condition, metric):
        return lookup[(293, 115, STRATA[0], condition, metric)]
    np.testing.assert_allclose(output("D_mean", "mae")["point"], 2.)
    np.testing.assert_allclose(output("D_mean", "signed_mean_shift")["point"], 2 / 3)
    np.testing.assert_allclose(output("D_mean", "mean_normalized199_rms")["point"], 2 / math.sqrt(199))
    np.testing.assert_allclose(output("D_mean", "absolute_error_p95")["point"], 2.9)
    np.testing.assert_allclose(output("M1", "paired_mae_minus_D_mean")["point"], -1.)
    np.testing.assert_allclose(output("M1", "paired_mae_minus_own_delta_perm")["point"], -2 / 3)
    sample = output("D_mean", "mae")
    np.testing.assert_allclose(ratio(boot_totals[:, sample["num"]],
                                    boot_totals[:, sample["den"]]), [5 / 3, 2., 1.])
    ci = interval(np.array([0., 10., 20., 30., np.nan]))
    np.testing.assert_allclose([ci["ci95_lower"], ci["ci95_upper"]], [.75, 29.25])
    assert ci["bootstrap_defined"] == 4 and ci["bootstrap_undefined"] == 1
    first = draw_counts(3, n_draws=7, seed=BOOTSTRAP_SEED)
    np.testing.assert_array_equal(first, draw_counts(3, n_draws=7, seed=BOOTSTRAP_SEED))
    labels = np.array([[1, 1], [0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    chemical = bin_chemistry(np.zeros((5, 2), dtype=np.int8), labels,
                            np.array([24, 26]), np.array([0, 0, 1, 2, 2]))
    assert len(chemical) == 20
    assert chemical[0]["hit_atom_count"] == 2
    assert chemical[0]["uncaptured_dictionary_atom_count"] == 2
    assert chemical[2]["n_atoms"] == 0 and chemical[2]["hit_atom_proportion"] is None
    return dict(status="PASS_SYNTHETIC_HAND_CALCULATION", opened_T_inputs=False,
                cases=["atom_pooled_sse_mse_r2", "same_draw_all_atoms_and_conditions",
                       "draw_specific_T_mean_sst", "constant_draw_null_r2",
                       "global_draw_molecular_stratum_denominators",
                       "paired_condition_comparisons", "199_output_rms", "linear_tail95",
                       "linear_marginal_interval_and_undefined_count",
                       "PCG64_count_replay", "overlapping_and_uncaptured_dictionary"])


def execute(args):
    # Completion receipts, fixed interfaces, and prior fit resource use are read
    # before opening arrays or creating scientific summary output.
    fit_receipt_path = Path(args.fit_receipt)
    fit_ledger_path = Path(args.fit_ledger)
    forward_receipt_path = Path(args.forward_receipt)
    fit_receipt = read_json(fit_receipt_path)
    forward_receipt = read_json(forward_receipt_path)
    if forward_receipt.get("status") != "COMPLETE_FORWARD_REPLACEMENT":
        raise RuntimeError("Forward completion receipt is absent or incomplete")
    np.testing.assert_array_equal(forward_receipt["coordinate_ids"], COORDINATES)
    np.testing.assert_array_equal(forward_receipt["condition_names"], CONDITIONS)
    if fit_receipt.get("status") != "PASS_ALL_T_FROZEN_PREDICTIONS":
        raise RuntimeError("Fit completion receipt is absent or incomplete")
    np.testing.assert_array_equal(fit_receipt["coordinate_ids"], COORDINATES)
    np.testing.assert_array_equal(fit_receipt["concept_ids"], CONCEPTS)
    np.testing.assert_array_equal(fit_receipt["conditions"], CONDITIONS[:3])
    lock = read_json(DOC / "RESOURCE_LOCK.json")
    limits = lock["fit_and_summary_combined"]
    protocol_sha = sha256(DOC / "PROTOCOL.md")
    if fit_receipt["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Frozen fit protocol binding differs from current protocol")
    fit_ledger = read_json(fit_ledger_path)
    if fit_ledger.get("status") != "PASS":
        raise RuntimeError("Fit resource ledger is absent or incomplete")
    fit_wall = float(fit_ledger["wall_seconds"])
    fit_cpu = float(fit_ledger["process_cpu_seconds"])
    peaks = dict(peak_rss_bytes=0, private_commit_bytes=0, peak_private_commit_bytes=0)

    def check():
        memory = process_memory()
        for key, value in memory.items():
            peaks[key] = max(peaks[key], value)
        wall = fit_wall + time.monotonic() - _IMPORT_WALL
        cpu = fit_cpu + time.process_time() - _IMPORT_CPU
        if wall > limits["wall_seconds_cap"] or cpu > limits["process_cpu_seconds_cap"]:
            raise TimeoutError("Locked combined fit/summary time budget exceeded")
        if peaks["peak_rss_bytes"] > limits["peak_rss_bytes_cap"]:
            raise MemoryError("Locked summary peak RSS cap exceeded")
        if peaks["peak_private_commit_bytes"] > limits["private_commit_bytes_cap"]:
            raise MemoryError("Locked summary private commit cap exceeded")

    if (fit_ledger["memory"]["peak_rss_bytes"] > limits["peak_rss_bytes_cap"]
            or fit_ledger["memory"]["peak_private_commit_bytes"] > limits["private_commit_bytes_cap"]):
        raise MemoryError("Fit peak memory exceeded the locked shared-stage cap")
    check()
    planned = ["bootstrap_weights.npy", "OUTPUT_METRICS.csv", "ACTIVATION_METRICS.csv",
               "BIN_CHEMISTRY.csv", "figure_data.csv", "SUMMARY.json"]
    if any((OUT / name).exists() for name in planned):
        raise FileExistsError("Summary artifacts already exist; no automatic overwrite/rerun")
    n_molecules, n_atoms = 10000, 246722
    with np.load(LANE / "fit/T_activations.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("coordinate_ids", "concept_ids", "actual",
            "predictions", "target_labels", "mol_offsets", "mol_idx", "labels_all", "label_ids", "bins")}
    np.testing.assert_array_equal(arrays["coordinate_ids"], COORDINATES)
    np.testing.assert_array_equal(arrays["concept_ids"], CONCEPTS)
    expected = {"actual": (n_atoms, 2), "predictions": (n_atoms, 2, 3),
                "target_labels": (n_atoms, 2), "mol_offsets": (n_molecules + 1,),
                "mol_idx": (n_atoms,), "labels_all": (n_atoms, 91),
                "label_ids": (91,), "bins": (n_atoms, 2)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape: {arrays[key].shape}")
    offsets = arrays["mol_offsets"]
    if offsets[0] != 0 or offsets[-1] != n_atoms or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid locked molecule offsets")
    if len(set(arrays["label_ids"].tolist())) != 91:
        raise ValueError("The complete 91-concept dictionary must have unique IDs")
    for key in ("target_labels", "labels_all"):
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError(f"Non-binary {key}")
    if not np.isin(arrays["bins"], np.arange(5)).all():
        raise ValueError("Invalid D-fixed activation bins")
    for key in ("actual", "predictions"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Non-finite {key}")
    baseline = np.load(LANE / "forward/baseline.npy", mmap_mode="r", allow_pickle=False)
    conditions = np.load(LANE / "forward/conditions.npy", mmap_mode="r", allow_pickle=False)
    dose_rms = np.load(LANE / "forward/dose_rms.npy", mmap_mode="r", allow_pickle=False)
    if (baseline.shape, conditions.shape, dose_rms.shape) != (
            (n_molecules, 199), (2, 5, n_molecules, 199), (2, 5, n_molecules)):
        raise ValueError("Unexpected locked forward array shapes")
    for array in (baseline, conditions, dose_rms):
        if not np.isfinite(array).all():
            raise ValueError("Non-finite forward results")
    check()
    sufficient = WeightedStatistics(n_molecules)
    all_strata = [molecular_strata(arrays["target_labels"][:, c], offsets) for c in range(2)]
    output_specs, dose_rows = output_specifications(baseline, conditions, dose_rms, offsets,
                                                    all_strata, sufficient)
    activation_specs = activation_specifications(arrays["actual"], arrays["predictions"],
                                                offsets, all_strata, sufficient)
    matrix = sufficient.matrix()
    del sufficient
    check()
    OUT.mkdir(parents=True, exist_ok=True)
    weights = draw_counts(n_molecules, path=OUT / "bootstrap_weights.npy", check=check)
    with threadpool_limits(limits=4):
        totals = weighted_totals(weights, matrix, check=check)
        actual_threadpools = threadpool_info()
    output_rows = []
    for spec in output_specs:
        spec = dict(spec)
        num, den = spec.pop("num", None), spec.pop("den", None)
        boot = ratio(totals[:, num], totals[:, den]) if num is not None else None
        output_rows.append(make_row("output", boot=boot, **spec))
    activation_rows = []
    for spec in activation_specs:
        spec = dict(spec)
        sse = totals[:, spec.pop("sse_id")]
        count = totals[:, spec.pop("count_id")]
        csum = totals[:, spec.pop("sum_id")]
        square = totals[:, spec.pop("square_id")]
        mse, r2 = spec.pop("mse"), spec.pop("r2")
        activation_rows.append(make_row("activation", metric="pooled_atom_mse", point=mse,
                                         boot=ratio(sse, count), **spec))
        activation_rows.append(make_row("activation", metric="pooled_atom_r2", point=r2,
                                         boot=r_squared(sse, count, csum, square), **spec))
    activation_rows.extend(descriptive_activation(arrays["actual"], arrays["predictions"],
        arrays["target_labels"], arrays["bins"], arrays["mol_idx"]))
    chemistry_rows = bin_chemistry(arrays["bins"], arrays["labels_all"],
                                    arrays["label_ids"], arrays["mol_idx"])
    write_csv_exclusive(OUT / "OUTPUT_METRICS.csv", output_rows)
    write_csv_exclusive(OUT / "ACTIVATION_METRICS.csv", activation_rows)
    write_csv_exclusive(OUT / "BIN_CHEMISTRY.csv", chemistry_rows)
    write_csv_exclusive(OUT / "figure_data.csv", output_rows + activation_rows + dose_rows)
    check()
    denominators = {f"j{COORDINATES[c]}": {
        stratum: dict(n_molecules=int(mask.sum()),
                      n_atoms=int(np.diff(offsets)[mask].sum()))
        for stratum, mask in all_strata[c].items()} for c in range(2)}
    hashes = {name: sha256(OUT / name) for name in planned if name != "SUMMARY.json"}
    check()
    summary = dict(schema="chemical-replacement-summary/1",
        status="COMPLETE_SUMMARY_PENDING_ROOT_SCIENTIFIC_ACCEPTANCE",
        created_utc=datetime.now(timezone.utc).isoformat(),
        coordinate_ids=list(COORDINATES), concept_ids=list(CONCEPTS),
        condition_names=list(CONDITIONS),
        endpoints=[dict(coordinate=COORDINATES[c], q=q, role=role,
                        anonymous_normalized_output=True) for c, q, role in ENDPOINTS],
        denominators=denominators,
        D_fixed_bin_atom_counts={f"j{COORDINATES[c]}":
            [int(np.count_nonzero(arrays["bins"][:, c] == b)) for b in range(5)] for c in range(2)},
        D_fixed_bin_note="Four D linear percentile cutpoints are retained including ties; bins may be highly unequal or empty, never asserted equal-size quintiles",
        output_metric_rows=len(output_rows), activation_metric_rows=len(activation_rows),
        bin_chemistry_rows=len(chemistry_rows), figure_data_rows=len(output_rows + activation_rows + dose_rows),
        bootstrap=dict(replicates=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, generator="NumPy PCG64",
            algorithm="Each draw samples 10000 global molecule indices with replacement using rng.integers; counts reused for all coordinates, conditions, endpoints and strata",
            weight_file="bootstrap_weights.npy", weight_dtype="int16", weight_shape=[2000, 10000],
            matrix_columns=int(matrix.shape[1]), reduction_dtype="float64",
            interval="marginal 95% percentile, numpy.quantile method=linear; finite draws retained and undefined draw counts explicit",
            unit="whole molecule including all atoms and conditions", simultaneous_coverage=False,
            p_values_computed=False, endpoint_search_performed=False,
            activation_sst="Pooled atom SST recomputes corresponding stratum/draw T mean via centered sum-of-squares sufficient statistics; zero SST yields null R2",
            point_only=["absolute_error_p95", "target-atom-label activation groups", "D-fixed activation bins", "bin chemistry", "realized activation displacement dose"]),
        definitions=dict(mae="mean across molecules of abs(condition_q-fresh_baseline_q)",
            absolute_error_p95="linear 95th percentile across molecules of abs(condition_q-fresh_baseline_q)",
            signed_mean_shift="mean across molecules of condition_q-fresh_baseline_q",
            mean_normalized199_rms="mean across molecules of sqrt(mean across all199 normalized output dimensions of squared condition-fresh_baseline shift))",
            paired_mae_minus_D_mean="condition MAE minus D_mean MAE using paired molecule draw; negative favors condition",
            paired_mae_minus_own_delta_perm="M1 or M2 MAE minus its own delta-permuted condition MAE using paired molecule draw; negative favors unpermuted",
            molecular_positive="at least one target-label-positive atom; activation metrics include every atom of included molecules",
            molecular_negative="no target-label-positive atoms; activation metrics include every atom of included molecules",
            bin_composition="all91 overlapping atom-level labels; hit proportions need not sum to one; uncaptured means no dictionary label hits on the atom; empty bin proportions null"),
        interpretation_limits=["New post hoc analysis on previously analyzed T; not preregistered or independent confirmation",
            "Anonymous normalized outputs have no asserted descriptor identities",
            "Delta permutation preserves prescribed within-molecule displacement multiset/dose, changes direction and replacement margins; not a strict distribution-matched null",
            "No equivalence or complete-fidelity claim; marginal intervals are not simultaneous intervals"],
        input_bindings=dict(protocol_sha256=protocol_sha,
            resource_lock_sha256=sha256(DOC / "RESOURCE_LOCK.json"),
            summarize_program_sha256=sha256(Path(__file__)),
            fit_receipt_path=str(fit_receipt_path.resolve()), fit_receipt_sha256=sha256(fit_receipt_path),
            fit_resource_ledger_path=str(fit_ledger_path.resolve()),
            fit_resource_ledger_sha256=sha256(fit_ledger_path),
            T_activations_sha256_from_fit_receipt=fit_receipt.get("T_activations_sha256"),
            D_FIT_sha256_from_fit_receipt=fit_receipt.get("D_FIT_sha256"),
            forward_receipt_path=str(forward_receipt_path.resolve()),
            forward_receipt_sha256=sha256(forward_receipt_path),
            identity_validation="Reuses completed fit/forward identity receipts; no old-cache or checkpoint rehash/rerun"),
        output_sha256=hashes,
        resource_ledger=dict(fit_wall_seconds=fit_wall, fit_process_cpu_seconds=fit_cpu,
            summary_wall_seconds=time.monotonic() - _IMPORT_WALL,
            summary_process_cpu_seconds=time.process_time() - _IMPORT_CPU,
            combined_wall_seconds=fit_wall + time.monotonic() - _IMPORT_WALL,
            combined_process_cpu_seconds=fit_cpu + time.process_time() - _IMPORT_CPU,
            combined_peak_rss_bytes=max(fit_ledger["memory"]["peak_rss_bytes"], peaks["peak_rss_bytes"]),
            combined_peak_private_commit_bytes=max(fit_ledger["memory"]["peak_private_commit_bytes"], peaks["peak_private_commit_bytes"]),
            **peaks, thread_limit=4, actual_threadpools=actual_threadpools,
            source_resource_lock="fit_and_summary_combined"),
        synthetic_hand_calculation=dict(status="PASS_SYNTHETIC_HAND_CALCULATION_PREVIOUSLY_RUN",
            opened_T_inputs=False, evidence="docs/chemical_replacement_20261027/SUMMARY_IMPLEMENTATION.md",
            rerun_during_scientific_summary=False))
    write_json_exclusive(OUT / "SUMMARY.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "denominators", "output_metric_rows",
        "activation_metric_rows", "bin_chemistry_rows", "resource_ledger")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--fit-receipt", default=str(LANE / "fit/T_ACTIVATIONS_RECEIPT.json"))
    parser.add_argument("--fit-ledger", default=str(LANE / "fit/EXECUTION_all.json"))
    parser.add_argument("--forward-receipt", default=str(LANE / "forward/FORWARD_RECEIPT.json"))
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), ensure_ascii=False))
    else:
        execute(args)


if __name__ == "__main__":
    main()
