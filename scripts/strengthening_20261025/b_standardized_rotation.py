"""Post hoc D-standardized rotations; old artifacts are read-only.

Run: .venv_chem/Scripts/python.exe -B scripts/strengthening_20261025/b_standardized_rotation.py
Only CPU cap may be changed by supervisor via the new CPU_BUDGET.json.
"""
from __future__ import annotations
import time
START_CPU = time.process_time()
START_WALL = time.perf_counter()
START_EPOCH = time.time()
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
import csv
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/strengthening_20261025/standardized_rotation"
OLD_D = ROOT / "results/revision_20261017/chemistry_D"
OLD_T = ROOT / "results/revision_20261017/chemistry_T"
BASE = ROOT / "data/revision_20261017/chemistry"
PROTOCOL = ROOT / "docs/strengthening_20261025/B_STANDARDIZED_ROTATION_PROTOCOL.md"
sys.path.insert(0, str(ROOT / "scripts/revision_20261017"))
from chem_statistics import (select_winners, fixed_winner_ap, molecule_label_counts,
                            weighted_prevalence, bootstrap_summary, estimands)

NETWORKS = ("trained", "init21000", "init21001", "init21002")
BASES = ("native",) + tuple(f"Q{i}" for i in range(30000, 30009))
MEM_CAP = 2 * 1024 ** 3
CONFIG = {
    "analysis": "POST_HOC_D_STANDARDIZED_POST_GELU_ROTATION",
    "networks": list(NETWORKS), "bases": list(BASES), "concepts": 71,
    "units": 464, "candidates_per_basis": 928, "replicates": 2000,
    "bootstrap_weights_seed": 42017, "bootstrap_weights": "reuse unchanged original count weights",
    "normalization": "float64(a-mu_D)/safe_sd_D -> float32 -> matmul Q_float32 -> score_float32",
    "near_constant_threshold": 1e-12, "near_constant_denominator": 1.0,
    "moments": "reuse per-network D float64 population moments of saved float32 activations; ddof0",
    "score_ties": "exact float32 equality; no epsilon merging",
    "winner_AP_atol": float(8 * np.finfo(np.float64).eps),
    "winner_tie_order": "unit ascending then positive pole", "threads": 4,
    "native": "all original D selection and T point/bootstrap evidence reused",
    "T_reselection": False, "model_forward_calls": 0, "new_random_weights": 0,
    "memory_cap_bytes": MEM_CAP,
}


def utc(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def memory():
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not fn(get_process(), ctypes.byref(counters), counters.cb):
        raise RuntimeError("cannot measure working set")
    return {"rss_bytes": int(counters.WorkingSetSize), "peak_rss_bytes": int(counters.PeakWorkingSetSize)}


class Ledger:
    def __init__(self):
        self.path = OUT / "CPU_LEDGER.json"
        if self.path.exists():
            self.data = read_json(self.path)
            if self.data["config"] != CONFIG or self.data["script_sha256"] != sha(Path(__file__)):
                raise RuntimeError("resume source/config drift")
            self.prior_cpu = self.data["cpu_seconds"]
            self.prior_wall = self.data["wall_seconds"]
        else:
            self.prior_cpu = self.prior_wall = 0.0
            self.data = {"config": CONFIG, "script_sha256": sha(Path(__file__)),
                         "protocol_sha256": sha(PROTOCOL), "started_utc": utc(START_EPOCH),
                         "inputs": {}, "units": [], "runs": [], "model_forward_calls": 0,
                         "new_random_weights": 0, "T_scientific_reads_before_D_STD_lock": 0}
        self.data["runs"].append({"started_utc": utc(START_EPOCH), "pid": os.getpid(),
            "command": f'"{sys.executable}" -B "{Path(__file__).resolve()}"', "cwd": str(Path.cwd())})
        self.data["status"] = "RUNNING"
        self.data["python"] = sys.version
        self.data["numpy"] = np.__version__
        self.data["BLAS"] = threadpool_info()
        self.last_write = 0.0
        self.active = None
        self.persist()

    def cpu(self):
        return self.prior_cpu + time.process_time() - START_CPU

    def persist(self):
        self.data.update(cpu_seconds=self.cpu(), wall_seconds=self.prior_wall + time.perf_counter() - START_WALL,
                         updated_utc=utc(), memory=memory())
        if self.active is not None:
            self.data["active_unit"] = {**self.active,
                "cpu_seconds": time.process_time() - self.active["start_cpu"],
                "wall_seconds": time.perf_counter() - self.active["start_wall"]}
        dump(self.path, self.data)
        self.last_write = time.perf_counter()

    def check(self, *_):
        budget = read_json(OUT / "CPU_BUDGET.json")
        cap = float(budget["cpu_cap_seconds"])
        self.data["current_cpu_cap_seconds"] = cap
        if self.cpu() >= cap - 10:
            raise TimeoutError(f"B cumulative CPU reserve reached: {self.cpu():.3f}/{cap}")
        if memory()["rss_bytes"] > MEM_CAP:
            raise MemoryError("B working set exceeds 2GiB stop line")
        if time.perf_counter() - self.last_write >= 10:
            self.persist()

    def begin(self, phase, network, basis):
        self.active = {"phase": phase, "network": network, "basis": basis,
                       "start_cpu": time.process_time(), "start_wall": time.perf_counter(), "started_utc": utc()}
        self.persist()

    def end(self, **extra):
        record = {**self.active, "cpu_seconds": time.process_time() - self.active["start_cpu"],
                  "wall_seconds": time.perf_counter() - self.active["start_wall"],
                  "finished_utc": utc(), **memory(), **extra}
        self.data["units"].append(record)
        self.active = None
        self.data.pop("active_unit", None)
        self.persist()
        print(json.dumps(record), flush=True)
        return record

    def small_input(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key in self.data["inputs"]:
            return self.data["inputs"][key]
        identity = {"path": key, "bytes": path.stat().st_size, "sha256": sha(path), "identity_method": "actual SHA256 once"}
        if expected is not None and identity["sha256"] != expected:
            raise ValueError(f"input identity drift: {path}")
        self.data["inputs"][key] = identity
        return identity


def close_array(array):
    if isinstance(array, np.memmap):
        array._mmap.close()


def cache_array(dataset, network, ledger):
    directory = BASE / f"activations_{dataset}" / network
    receipt_path = directory / "CACHE_RECEIPT.json"
    ledger.small_input(receipt_path)
    receipt = read_json(receipt_path)
    if receipt["status"] != f"PASS_{dataset}_EXTRACTION":
        raise ValueError("old activation receipt is not PASS")
    path = directory / "atom_a.npy"
    expected = receipt["files"]["atom_a.npy"]
    stat = path.stat()
    if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"old {dataset} cache size/mtime drift {network}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = ((124530 if dataset == "D" else 246722), 464)
    if array.dtype != np.float32 or array.shape != expected_shape:
        raise ValueError("activation shape/dtype drift")
    ledger.data["inputs"][str(path)] = {"path": str(path), **expected,
        "identity_method": "reuse original closeout SHA; current size,mtime,shape,dtype verified; no rehash"}
    return array


def load_labels(dataset, C, ledger):
    directory = BASE / ("development" if dataset == "D" else "test")
    receipt = read_json(BASE / "corpus/BUILD_RECEIPT.json") if dataset == "D" else read_json(directory / "T_DATA_RECEIPT.json")
    labels_path = directory / f"labels_{dataset}.npz"
    atoms_path = directory / f"atoms_{dataset}.npz"
    expected_map = receipt["outputs"] if dataset == "D" else receipt["outputs"]
    for path in (labels_path, atoms_path):
        expected = expected_map[path.relative_to(ROOT).as_posix()]
        ledger.small_input(path, expected["sha256"])
    with np.load(labels_path, allow_pickle=False) as source:
        all_labels, ids = source["labels"], source["concept_ids"]
    columns = [int(np.flatnonzero(ids == c)[0]) for c in C]
    labels = np.ascontiguousarray(all_labels[:, columns], dtype=np.bool_)
    del all_labels
    with np.load(atoms_path, allow_pickle=False) as source:
        molecules, offsets = source["mol_idx"], source["mol_offsets"]
    nm = 5000 if dataset == "D" else 10000
    na = 124530 if dataset == "D" else 246722
    if labels.shape != (na, 71) or len(offsets) != nm + 1 or offsets[-1] != na:
        raise ValueError("label/offset dimensions differ")
    if not np.array_equal(molecules, np.repeat(np.arange(nm), np.diff(offsets))):
        raise ValueError("molecule-cluster row order differs")
    return labels, molecules


def standardize_rotate(native, mu, sd, q, ledger, name):
    scratch = np.lib.format.open_memmap(OUT / "rotation_scratch.npy", mode="w+", dtype=np.float32, shape=native.shape)
    for begin in range(0, len(native), 2048):
        ledger.check()
        block = (np.asarray(native[begin:begin + 2048], np.float64) - mu) / sd
        scratch[begin:begin + 2048] = block.astype(np.float32) @ q
    scratch.flush()
    return scratch


def engineering_golden(ledger):
    path = OUT / "STANDARDIZATION_ENGINEERING.json"
    if path.exists():
        if read_json(path)["status"] != "PASS":
            raise RuntimeError("existing new-standardization engineering failure")
        return
    # Finite hand-defined fixture for the new arithmetic only. Old six statistics goldens are reused.
    a = np.arange(6 * 464, dtype=np.float32).reshape(6, 464) / np.float32(17)
    mu = np.linspace(-2, 3, 464, dtype=np.float64)
    raw_sd = np.linspace(.01, 4, 464, dtype=np.float64)
    raw_sd[:2] = [0.0, 1e-13]
    sd = np.where(raw_sd <= 1e-12, 1.0, raw_sd)
    q = np.eye(464, dtype=np.float32)
    q[[2, 3]] = q[[3, 2]]
    expected = np.empty_like(a)
    for row in range(6):
        for col in range(464):
            expected[row, col] = np.float32((float(a[row, col]) - mu[col]) / sd[col])
    expected[:, [2, 3]] = expected[:, [3, 2]]
    observed = standardize_rotate(a, mu, sd, q, ledger, "golden")
    checks = {"chunk_float64_standardize_float32_Q32_matches_scalar_reference": bool(np.array_equal(observed, expected)),
              "nearconstant_denominator_one_and_retains_464": bool(np.array_equal(sd[:2], [1., 1.]) and observed.shape[1] == 464)}
    close_array(observed)
    del observed
    old = OLD_D / "STATISTICS_ENGINEERING.json"
    ledger.small_input(old)
    old_evidence = read_json(old)
    if old_evidence["status"] != "PASS_SYNTHETIC_STATISTICS_AND_512_TOKEN_SPOTCHECK":
        raise RuntimeError("old statistics engineering evidence absent")
    dump(path, {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                "old_six_statistics_goldens": "reused unchanged, not rerun", "old_evidence_path": str(old),
                "old_statistical_check_count": len(old_evidence["checks"]), "new_model_forwards": 0})
    if not all(checks.values()):
        raise AssertionError("new standardization arithmetic golden failed")


def run(ledger):
    old_winners = OLD_D / "D_FINAL_WINNERS.npz"
    ledger.small_input(old_winners)
    with np.load(old_winners, allow_pickle=False) as source:
        C = source["concept_ids"].copy()
        candidates = source["candidate_indices"].copy()
        ap_D = source["AP_D"].copy()
        if source["networks"].tolist() != list(NETWORKS) or source["bases"].tolist() != list(BASES):
            raise ValueError("original network/basis order differs")
    if len(C) != 71 or candidates.shape != (4, 10, 71):
        raise ValueError("original 71 concept/winner shape differs")
    ledger.small_input(BASE / "corpus/BUILD_RECEIPT.json")
    ledger.small_input(ROOT / "scripts/revision_20261017/chem_statistics.py")
    mus, sds, safe_sds, near_constants = [], [], [], []
    for network in NETWORKS:
        path = BASE / "activations_D" / network / "D_moments.npz"
        receipt = read_json(path.parent / "CACHE_RECEIPT.json")
        if receipt["moment_SD_ddof"] != 0 or receipt["moment_and_column_norm_dtype"] != "float64":
            raise ValueError("old moments provenance differs")
        ledger.small_input(path, receipt["files"][path.name]["sha256"])
        with np.load(path, allow_pickle=False) as source:
            mu, sd = source["atom_mean"], source["atom_sd"]
        if mu.dtype != np.float64 or sd.dtype != np.float64 or mu.shape != (464,) or sd.shape != (464,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd < 0):
            raise ValueError("moment shape/dtype/value mismatch")
        near = sd <= 1e-12
        mus.append(mu); sds.append(sd); safe_sds.append(np.where(near, 1.0, sd)); near_constants.append(near)
    mus, sds, safe_sds, near_constants = map(np.asarray, (mus, sds, safe_sds, near_constants))
    params_path = OUT / "D_STD_PARAMETERS.npz"
    if not params_path.exists():
        np.savez(params_path, atom_mean=mus, atom_sd=sds, safe_sd=safe_sds, nearconstant=near_constants,
                 networks=np.array(NETWORKS), ddof=np.array(0))
    dump(OUT / "NEAR_CONSTANTS.json", {network: {
        "n": int(near_constants[mi].sum()), "unit_indices": np.flatnonzero(near_constants[mi]).tolist(),
        "means": mus[mi, near_constants[mi]].tolist(), "original_sd": sds[mi, near_constants[mi]].tolist(),
        "replacement_denominator": 1.0} for mi, network in enumerate(NETWORKS)})
    qs = []
    for basis in BASES[1:]:
        path = OLD_D / "rotations" / f"{basis}.npy"
        ledger.small_input(path, read_json(path.with_suffix(".json"))["sha256"])
        q = np.load(path, allow_pickle=False)
        if q.shape != (464, 464) or q.dtype != np.float64:
            raise ValueError("saved Q dtype/shape differs")
        qs.append(q.astype(np.float32))
    engineering_golden(ledger)
    d_labels, d_molecules = load_labels("D", C, ledger)
    for mi, network in enumerate(NETWORKS):
        native = None
        for bi, basis in enumerate(BASES[1:], 1):
            checkpoint = OUT / "D" / f"{network}_{basis}.npz"
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    candidates[mi, bi], ap_D[mi, bi] = saved["candidate_indices"], saved["AP_D"]
                continue
            ledger.check()
            if native is None:
                native = cache_array("D", network, ledger)
            ledger.begin("D_standardize_rotate_and_all928_selection", network, basis)
            scores = standardize_rotate(native, mus[mi], safe_sds[mi], qs[bi - 1], ledger, basis)
            candidate, ap = select_winners(scores, d_labels, progress=ledger.check)
            if np.any((candidate < 0) | (candidate >= 928)) or not np.isfinite(ap).all():
                raise ValueError("D standardized winner/AP missing or invalid")
            candidates[mi, bi], ap_D[mi, bi] = candidate, ap
            np.savez(checkpoint, candidate_indices=candidate, AP_D=ap, concept_ids=C)
            close_array(scores); del scores
            record = ledger.end(candidates_per_concept=928, concepts=71, T_scientific_reads=0)
            dump(checkpoint.with_suffix(".json"), record)
            if not (OUT / "FIRST_D_COST_PROBE.json").exists():
                historical = read_json(OLD_D / "CPU_LEDGER.json")["timings"]
                prior_t = [v["cpu_seconds"] for v in historical if v["phase"] in
                           ("Tfixed_trained_Q30000_rotation", "Tfixed_trained_Q30000_2000_cluster_bootstrap")]
                probe = {"first_actual_D_unit": record, "projected_36_D_cpu_seconds": record["cpu_seconds"] * 36,
                         "original_T_first_unit_cpu_seconds": sum(prior_t),
                         "projected_36_T_from_original_T_seconds": sum(prior_t) * 36,
                         "projected_36D_36T_total_seconds": (record["cpu_seconds"] + sum(prior_t)) * 36,
                         "limitation": "first D unit includes first production compilation; old raw T timing is only planning, new T may differ"}
                dump(OUT / "FIRST_D_COST_PROBE.json", probe)
                print(json.dumps({"FIRST_D_COST_PROBE": probe}), flush=True)
        if native is not None:
            close_array(native); del native
        gc.collect()
    del d_labels, d_molecules
    winners_path = OUT / "D_STD_WINNERS.npz"
    if not winners_path.exists():
        np.savez(winners_path, candidate_indices=candidates, AP_D=ap_D, concept_ids=C,
                 networks=np.array(NETWORKS), bases=np.array(BASES))
    lock_path = OUT / "D_STD_OUTPUT_LOCK.json"
    if not lock_path.exists():
        dump(lock_path, {"status": "D_STD_OUTPUT_LOCKED", "utc": utc(), "config": CONFIG,
            "new_D_rotated_units": 36, "original_native_units_reused": 4,
            "concept_ids": C.tolist(), "winner_sha256": sha(winners_path),
            "parameter_sha256": sha(params_path), "protocol_sha256": ledger.data["protocol_sha256"],
            "script_sha256": ledger.data["script_sha256"], "inputs": dict(ledger.data["inputs"]),
            "T_new_score_or_label_reads_before_lock": 0})
    lock = read_json(lock_path)
    if lock["status"] != "D_STD_OUTPUT_LOCKED" or lock["config"] != CONFIG:
        raise RuntimeError("new D output lock invalid")
    ledger.data["D_STD_lock_status"] = lock["status"]
    ledger.persist()

    # T begins only after all 36 D rotations and the new output lock are durable.
    ledger.small_input(BASE / "test/T_DATA_RECEIPT.json")
    labels, molecules = load_labels("T", C, ledger)
    weights_path = OLD_T / "T_BOOTSTRAP_WEIGHTS.npy"
    weight_receipt = read_json(weights_path.with_suffix(".json"))
    ledger.small_input(weights_path, weight_receipt["sha256"])
    weights_original = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights_original.shape != (2000, 10000) or weights_original.dtype != np.int32 or not np.all(weights_original.sum(1) == 10000) or weight_receipt["seed"] != 42017:
        raise ValueError("old shared weight schema/seed differs")
    weights = np.vstack((np.ones((1, 10000), np.int32), weights_original))
    close_array(weights_original); del weights_original
    positive_counts, atom_counts = molecule_label_counts(labels, molecules, 10000)
    prevalence = weighted_prevalence(positive_counts, atom_counts, weights)
    ell = np.full((2001, 4, 10, 71), np.nan, np.float64)
    point_AP = np.full((4, 10, 71), np.nan, np.float64)
    original_reps = OLD_T / "T_FIXED_REPLICATES.npz"
    ledger.small_input(original_reps)
    with np.load(original_reps, allow_pickle=False) as saved:
        if not np.array_equal(saved["concept_ids"], C):
            raise ValueError("native T concept order differs")
        ell[0, :, 0, :] = saved["point_ell"][:, 0, :]
        old_replicated = saved["replicated_ell"]
        ell[1:, :, 0, :] = old_replicated[:, :, 0, :]
        del old_replicated
        point_AP[:, 0, :] = saved["point_AP"][:, 0, :]
        if not np.array_equal(saved["point_prevalence"], prevalence[0]):
            raise ValueError("recomputed prevalence differs from unchanged old T")
    native_reuse_path = OUT / "NATIVE_REUSED.npz"
    if not native_reuse_path.exists():
        np.savez(native_reuse_path, point_ell=ell[0, :, 0], replicated_ell=ell[1:, :, 0],
                 point_AP=point_AP[:, 0], point_prevalence=prevalence[0], concept_ids=C)
    for mi, network in enumerate(NETWORKS):
        native = None
        for bi, basis in enumerate(BASES[1:], 1):
            checkpoint = OUT / "T" / f"{network}_{basis}.npz"
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    ell[:, mi, bi], point_AP[mi, bi] = saved["ell"], saved["AP"][0]
                continue
            ledger.check()
            if native is None:
                native = cache_array("T", network, ledger)
            ledger.begin("T_standardize_rotate_fixed_winner_2000bootstrap", network, basis)
            scores = standardize_rotate(native, mus[mi], safe_sds[mi], qs[bi - 1], ledger, basis)
            ap = fixed_winner_ap(scores, labels, molecules, candidates[mi, bi], weights, budget_check=ledger.check)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.log(ap / prevalence)
            ell[:, mi, bi], point_AP[mi, bi] = values, ap[0]
            np.savez(checkpoint, ell=values, AP=ap, concept_ids=C, candidate_indices=candidates[mi, bi])
            close_array(scores); del scores, ap, values
            record = ledger.end(replicates=2000, concepts=71, D_winners_reselected=False)
            dump(checkpoint.with_suffix(".json"), record)
            if not (OUT / "FIRST_T_COST_PROBE.json").exists():
                d_units = [v for v in ledger.data["units"] if v["phase"].startswith("D_")]
                probe = {"first_actual_T_unit": record,
                    "actual_36D_cpu_seconds": sum(v["cpu_seconds"] for v in d_units),
                    "projected_36T_cpu_seconds": 36 * record["cpu_seconds"],
                    "projected_complete_total_cpu_seconds": ledger.cpu() + 35 * record["cpu_seconds"],
                    "limitation": "network sparsity, selected units and ties can change subsequent actual costs"}
                dump(OUT / "FIRST_T_COST_PROBE.json", probe)
                print(json.dumps({"FIRST_T_COST_PROBE": probe}), flush=True)
        if native is not None:
            close_array(native); del native
        gc.collect()
    ledger.check()
    ledger.begin("final_estimands_and_output", "all", "all")
    summary = bootstrap_summary(ell[0], ell[1:])
    point = estimands(ell[0])
    old_summary_path = OLD_T / "T_FIXED_SUMMARY.json"
    ledger.small_input(old_summary_path)
    old_summary = read_json(old_summary_path)
    if float(point["G_init"]) != old_summary["G_init"]["point"]:
        raise ValueError("reused native G_init point differs from original")
    summary["G_init"] = old_summary["G_init"]
    summary.update(status="POST_HOC_STANDARDIZED_ROTATION_COMPLETE", config=CONFIG,
        concept_ids=C.tolist(), Theta_raw_historical=old_summary["Theta"],
        G_init_reuse="original point and interval unchanged; no new native AP or bootstrap calculation",
        native_AP_and_ell_reused=True, new_independent_confirmation=False,
        D_STD_OUTPUT_LOCK=str(lock_path), new_D_rotated_units=36, new_T_rotated_units=36,
        difference_CI_or_new_comparison_significance=False,
        interpretation="D coordinate scaling sensitivity, not whitening/pure direction; no component attribution; not H4 pre-GELU J2 validation")
    np.savez(OUT / "B_T_FIXED_REPLICATES.npz", point_ell=ell[0], replicated_ell=ell[1:],
             point_AP=point_AP, point_prevalence=prevalence[0], concept_ids=C,
             Theta_per_concept=point["Theta_per_concept"], G_init_per_concept=point["G_init_per_concept"],
             B_per_network_concept=point["B_per_network_concept"])
    with (OUT / "PER_CONCEPT.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["concept_id", "Theta_std", "G_init_reused"] + [f"B_{n}" for n in NETWORKS])
        for ci, concept in enumerate(C):
            writer.writerow([int(concept), point["Theta_per_concept"][ci], point["G_init_per_concept"][ci]] + point["B_per_network_concept"][:, ci].tolist())
    dump(OUT / "B_SUMMARY.json", summary)
    ledger.end(defined_replicates=summary["defined_replicates"], undefined_replicates=summary["undefined_replicates"])
    ledger.data["status"] = "COMPLETED"
    ledger.data["final_summary"] = {"Theta_std": summary["Theta"], "Theta_raw_historical": old_summary["Theta"], "G_init_reused": summary["G_init"]}
    print(json.dumps(ledger.data["final_summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-cap", type=float, default=10790)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "D").mkdir(exist_ok=True)
    (OUT / "T").mkdir(exist_ok=True)
    budget_path = OUT / "CPU_BUDGET.json"
    if not budget_path.exists():
        dump(budget_path, {"cpu_cap_seconds": args.cpu_cap, "authority": "initial Supervisor allocation; only Supervisor may release more shared A+B CPU", "utc": utc()})
    set_num_threads(4)
    with threadpool_limits(limits=4):
        ledger = Ledger()
        try:
            run(ledger)
        except BaseException as error:
            ledger.data["status"] = "PARTIAL_BUDGET_STOP" if isinstance(error, TimeoutError) else "PARTIAL_FAILED"
            ledger.data["failure"] = {"utc": utc(), "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            ledger.persist()
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise
        finally:
            ledger.data["runs"][-1].update(ended_utc=utc(), cpu_seconds=time.process_time() - START_CPU,
                                          wall_seconds=time.perf_counter() - START_WALL)
            ledger.persist()


if __name__ == "__main__":
    main()
