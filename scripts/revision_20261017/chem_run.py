"""D-only runner: bounded four-network extraction, D precision, final D winners.

No CLI stage opens sealed/T files. Actual D reads/forwards/statistics require a
root-owned D_RUN_LOCK binding this READY and the accepted data BUILD_RECEIPT.
T execution belongs to a later runner with an independent D_OUTPUT_LOCK check.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from numba import set_num_threads
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from chem_model import (ChembertaMTR, PRIMARY_LAYER, REPOSITORY, REVISION,
                        TOKENIZER_SPECIAL_IDS, batch_from_audits)
from chem_statistics import (bootstrap_summary, cluster_weights, fixed_winner_ap,
                             molecule_label_counts, select_winners, weighted_prevalence)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/revision_20261017"
BASE = ROOT / "data/revision_20261017/chemistry"
D_DIR = BASE / "development"
ACT = BASE / "activations_D"
OUT = ROOT / "results/revision_20261017/chemistry_D"
READY = OUT / "D_RUNNER_READY.json"
DEFAULT_LOCK = DOCS / "D_RUN_LOCK.json"
NETWORKS = ("trained", "init21000", "init21001", "init21002")
BASES = ("native",) + tuple(f"Q{seed}" for seed in range(30000, 30009))
CONFIG = {
    "schema": "chemistry-D-runner/1", "networks": list(NETWORKS), "bases": list(BASES),
    "initialization_seeds": [21000, 21001, 21002], "rotation_seeds": list(range(30000, 30009)),
    "layer_zero_based": 1, "native_or_rotated_score_dtype": "float32", "Q_saved_dtype": "float64",
    "Q_algorithm": "NumPy default_rng PCG64 Gaussian/float64QR; Q columns multiplied by sign(R diagonal); zero sign=+1",
    "AP_accumulator_dtype": "float64", "score_ties": "exact equality; no epsilon score merging",
    "winner_AP_absolute_tolerance": float(8 * np.finfo(np.float64).eps),
    "winner_tie_break": "unit ascending, positive pole first", "candidate_encoding": "+a_j=j;-a_j=464+j",
    "AP": "distinct-threshold precision weighted by positive recall increments; not trapezoid",
    "bootstrap": "shared PCG64 integers molecule-resample count weights; no atom-IID or approximation",
    "D_fit_molecules": 4000, "D_precision_molecules": 1000, "D_precision_replicates": 500,
    "D_precision_seed": 42016, "T_replicates_not_run_here": 2000, "T_seed_not_run_here": 42017,
    "quantile_method": "linear", "D_halfwidth": "(q.975-q.025)/2 per estimand; max of Theta/G_init times1.25sqrt(.1)",
    "D_target_halfwidth": float(np.log(1.10)), "undefined_replicate_max_fraction": .01,
    "encoder_and_intervention_total_cpu_seconds": 4 * 3600,
    "AP_rotation_clusterbootstrap_total_cpu_seconds": 12 * 3600,
    "budget_unit": "cumulative process CPU seconds, separate wall time; shared across D and any future T",
    "threads": 4, "torch_interop_threads": 1, "batch_size": 4,
    "memory_working_set_stop_bytes": 2 * 1024**3,
    "max_atom_rows_D": 250000, "activation_bytes_per_network_max": 250000 * 464 * 4,
    "retained_basis_matrices": "one temporary rotated score matrix; no 40-matrix expansion",
    "intervention_budget_total_molecule_evaluations": 10000,
    "T_access": "none; later T stage must separately require root D_OUTPUT_LOCK",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only owned local result/cache receipts are replaced, never source inputs/locks.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_D_path(path):
    path = Path(path).resolve()
    if BASE.resolve() not in path.parents or "sealed" in (p.lower() for p in path.parts):
        raise ValueError("D runner refuses any path outside chemistry D/metadata or inside sealed")
    if "_t." in path.name.lower() or path.name.lower() in {"t.tsv", "anchors_t.jsonl.gz"}:
        raise ValueError("D runner refuses T input")
    return path


def memory_usage():
    if os.name != "nt":
        return {"rss_bytes": None, "peak_rss_bytes": None}

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = MemoryCounters(); counters.cb = ctypes.sizeof(counters)
    # ctypes otherwise defaults HANDLE return/arguments to a 32-bit int, which
    # truncates/sign-extends the pseudo handle incorrectly on 64-bit Windows.
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    process = current_process()
    if not get_memory(process, ctypes.byref(counters), counters.cb):
        raise RuntimeError("cannot measure Windows process memory")
    return {"rss_bytes": int(counters.WorkingSetSize), "peak_rss_bytes": int(counters.PeakWorkingSetSize)}


def check_memory():
    value = memory_usage()
    if value["rss_bytes"] is not None and value["rss_bytes"] > CONFIG["memory_working_set_stop_bytes"]:
        raise MemoryError("2GiB process working-set budget reached; owned checkpoints retained")
    return value


class Budget:
    """Persistent additive ledger; no restart resets the approved CPU budgets."""

    def __init__(self, binding):
        self.path = OUT / "CPU_LEDGER.json"
        self.binding = binding
        if self.path.exists():
            self.record = read_json(self.path)
            if self.record["binding"] != binding:
                raise ValueError("CPU ledger belongs to another locked identity")
        else:
            self.record = {"binding": binding, "encoder_cpu_seconds": 0.0, "statistics_cpu_seconds": 0.0,
                           "encoder_wall_seconds": 0.0, "statistics_wall_seconds": 0.0,
                           "timings": [], "T_scientific_reads": 0, "model_forward_molecule_evaluations": 0}
        if self.record.get("active_phase"):
            prior = self.record.pop("active_phase")
            self.record[f"{prior['group']}_cpu_seconds"] += prior["cpu_seconds"]
            self.record[f"{prior['group']}_wall_seconds"] += prior["wall_seconds"]
            self.record["timings"].append({**prior, "status": "RECOVERED_UNCLOSED_PHASE_LAST_DURABLE_CPU_CHECKPOINT"})
            dump(self.path, self.record)
        self.active = None
        self.last_persisted = time.monotonic()

    def check(self):
        if self.active:
            group, cpu, wall, label = self.active
            limit = CONFIG["encoder_and_intervention_total_cpu_seconds" if group == "encoder"
                           else "AP_rotation_clusterbootstrap_total_cpu_seconds"]
            if self.record[f"{group}_cpu_seconds"] + time.process_time() - cpu >= limit:
                raise TimeoutError(f"approved cumulative {group} CPU hard stop reached")
            if time.monotonic() - self.last_persisted >= 10:
                self.record["active_phase"] = {"group": group, "phase": label,
                    "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - wall,
                    "checkpoint_utc": utc(), "hard_process_kill_may_lose_up_to_one_checkpoint_interval": True}
                dump(self.path, self.record)
                self.last_persisted = time.monotonic()
        check_memory()

    @contextmanager
    def track(self, group, label):
        if self.active:
            raise RuntimeError("nested budget phases are forbidden")
        self.active = (group, time.process_time(), time.monotonic(), label)
        started = utc()
        status = "PASS"
        try:
            self.check()
            yield
            self.check()
        except BaseException:
            status = "INTERRUPTED_OR_FAILED"
            raise
        finally:
            _, cpu, wall, _ = self.active
            cpu_elapsed, wall_elapsed = time.process_time() - cpu, time.monotonic() - wall
            self.record[f"{group}_cpu_seconds"] += cpu_elapsed
            self.record[f"{group}_wall_seconds"] += wall_elapsed
            self.record.pop("active_phase", None)
            try:
                memory = memory_usage()
            except RuntimeError as error:
                memory = {"memory_measurement_error": str(error)}
            self.record["timings"].append({"phase": label, "group": group, "started_utc": started,
                "finished_utc": utc(), "cpu_seconds": cpu_elapsed, "wall_seconds": wall_elapsed,
                "status": status, **memory})
            self.record["updated_utc"] = utc()
            dump(self.path, self.record)
            self.active = None


def numerical_identity():
    files = [Path(__file__), Path(__file__).with_name("chem_statistics.py"), Path(__file__).with_name("chem_model.py"),
             Path(__file__).with_name("test_chem_statistics.py"), DOCS / "CHEMISTRY_CONFIRMATION_PROTOCOL.md",
             DOCS / "CHEMISTRY_DATA_PROTOCOL.md", ROOT / "assets/chemberta_mtr_model/config.json",
             ROOT / "assets/chemberta_mtr_model/provenance.json", OUT / "STATISTICS_ENGINEERING.json",
             OUT / "CPU_BUDGET_PROBE.json", OUT / "CPU_BUDGET_PLAN.json", OUT / "RUNNER_ENGINEERING.json"]
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}


def write_ready():
    engineering = read_json(OUT / "STATISTICS_ENGINEERING.json")
    if engineering["status"] != "PASS_SYNTHETIC_STATISTICS_AND_512_TOKEN_SPOTCHECK":
        raise ValueError("statistics and512-token synthetic gate missing")
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "torch", "transformers", "tokenizers", "rdkit", "numba", "llvmlite", "threadpoolctl")}
    value = {"status": "READY_AWAIT_ROOT_D_RUN_LOCK_AND_D_GO", "created_utc": utc(), "config": CONFIG,
             "files": numerical_identity(), "versions": versions, "python_executable": sys.executable,
             "lock_contract": {"default_path": str(DEFAULT_LOCK), "required_status": "D_RUN_LOCKED",
                "required_fields": ["runner_ready_sha256", "data_build_receipt_sha256"],
                "meaning": "root authorizes D only; no D_OUTPUT_LOCK is created by this runner"},
             "D_reads_or_model_forwards_or_D_winner_selection": 0,
             "commands_after_root_D_GO": [f".venv_chem/Scripts/python.exe -B scripts/revision_20261017/chem_run.py {stage} --lock docs/revision_20261017/D_RUN_LOCK.json"
                                          for stage in ("extract-D", "precision-D", "final-D")],
             "handoff": "root owns finite input references, mechanism selection, D_OUTPUT_LOCK and any T runner",
             "cache_schema": {
                 "root": str(ACT), "networks": list(NETWORKS),
                 "atom_a.npy": "float32[n_D_atoms,464]; D molecule order then canonical RDKit atom index",
                 "cls_a.npy": "float32[5000,464]; position0 in D molecule order",
                 "normalized_head.npy": "float32[5000,199]; anonymous normalized coordinates",
                 "D_moments.npz": "float64[464] each: atom_mean,atom_mean_abs,atom_sd,cls_mean,cls_sd,output_dense_column_norm,residual_dose; population SD ddof0",
                 "D_FINAL_WINNERS.npz": "candidate_indices int32[4,10,C], AP_D float64[4,10,C], concept_ids,networks,bases",
                 "D_PRECISION.json": "both estimands,500 shared cluster replicates,undefined fraction,projected halfwidth,precision_gate",
             },
             "known_budget_limit": "CPU caps include future T work; D usage is additive, not reset by each command",
             "memory": {"process_working_set_stop_bytes": CONFIG["memory_working_set_stop_bytes"],
                        "max_disk_native_four_network_atom_arrays_bytes": 4 * CONFIG["activation_bytes_per_network_max"],
                        "temporary_rotated_array_max_bytes": CONFIG["activation_bytes_per_network_max"],
                        "algorithm": "native mmap per network, one basis scratch mmap, one sorted scalar and concept-label block, 500x1000 molecule weights"}}
    dump(READY, value)
    return {"status": value["status"], "ready_path": str(READY), "runner_ready_sha256": sha(READY)}


def require_lock(path):
    path = Path(path).resolve()
    lock = read_json(path)
    if lock.get("status") != "D_RUN_LOCKED":
        raise ValueError("root D_RUN_LOCK status is not D_RUN_LOCKED")
    if lock.get("runner_ready_sha256") != sha(READY):
        raise ValueError("D_RUN_LOCK does not bind current runner READY")
    ready = read_json(READY)
    if ready["config"] != CONFIG or ready["files"] != numerical_identity():
        raise ValueError("runner/protocol/numerical implementation drift after READY")
    versions = {p: importlib.metadata.version(p) for p in ready["versions"]}
    if ready["versions"] != versions:
        raise ValueError("runtime versions drifted after READY")
    receipt_path = require_D_path(BASE / "corpus/BUILD_RECEIPT.json")
    if lock.get("data_build_receipt_sha256") != sha(receipt_path):
        raise ValueError("D_RUN_LOCK does not bind accepted data BUILD_RECEIPT")
    receipt = read_json(receipt_path)
    if receipt["status"] != "PASS_DATA_BUILD" or receipt["n_D"] != 5000:
        raise ValueError("data build or D size gate failed")
    return {"D_RUN_LOCK_sha256": sha(path), "runner_ready_sha256": sha(READY),
            "data_build_receipt_sha256": sha(receipt_path)}


def load_D(binding):
    receipt = read_json(BASE / "corpus/BUILD_RECEIPT.json")
    files = [D_DIR / name for name in ("D.tsv", "anchors_D.jsonl.gz", "labels_D.npz", "atoms_D.npz", "coverage.json", "panel.json")]
    for path in files:
        require_D_path(path)
        expected = receipt["outputs"][path.relative_to(ROOT).as_posix()]
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"accepted D artifact identity drift: {path.name}")
    with (D_DIR / "D.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    with gzip.open(D_DIR / "anchors_D.jsonl.gz", "rt", encoding="utf-8") as stream:
        anchors = [json.loads(line) for line in stream]
    with np.load(D_DIR / "labels_D.npz", allow_pickle=False) as source:
        all_labels, original_ids = source["labels"], source["concept_ids"]
    with np.load(D_DIR / "atoms_D.npz", allow_pickle=False) as source:
        molecules, atom_indices, offsets = source["mol_idx"], source["atom_idx"], source["mol_offsets"]
    coverage, panel = read_json(D_DIR / "coverage.json"), read_json(D_DIR / "panel.json")
    if coverage["coverage_gate"] != "PASS" or coverage["C_ids"] != panel["C_ids"]:
        raise ValueError("D concept coverage gate failed or C differs")
    concept_ids = np.asarray(coverage["C_ids"], np.int32)
    columns = [int(np.flatnonzero(original_ids == c)[0]) for c in concept_ids]
    labels = np.ascontiguousarray(all_labels[:, columns], dtype=np.bool_)
    if len(rows) != 5000 or len(anchors) != 5000 or len(offsets) != 5001 or offsets[-1] != len(labels):
        raise ValueError("D molecule/atom/anchor dimensions differ")
    if not np.array_equal(molecules, np.repeat(np.arange(5000), np.diff(offsets))):
        raise ValueError("atom rows are not D.tsv then graph atom order")
    if not np.array_equal(atom_indices, np.concatenate([np.arange(n) for n in np.diff(offsets)])):
        raise ValueError("graph atom index row ordering differs")
    audits = []
    for mi, (row, anchor) in enumerate(zip(rows, anchors)):
        if int(row["rank"]) != mi or anchor["rank"] != mi or int(row["cid"]) != anchor["cid"]:
            raise ValueError("D/anchor molecule identity order differs")
        if hashlib.sha256(row["smiles"].encode("utf-8")).hexdigest() != row["canonical_smiles_sha256"]:
            raise ValueError("saved exact canonical input identity differs")
        if row["canonical_smiles_sha256"] != anchor["canonical_smiles_sha256"]:
            raise ValueError("D/anchor canonical identity differs")
        fingerprint = hashlib.sha256(json.dumps(anchor["input_ids"], separators=(",", ":")).encode("ascii")).hexdigest()
        if fingerprint != anchor["input_ids_sha256"]:
            raise ValueError("anchor input IDs fingerprint differs")
        n_atoms = int(offsets[mi + 1] - offsets[mi])
        if [a["graph_atom_id"] for a in anchor["atoms"]] != list(range(n_atoms)):
            raise ValueError("anchor atom order differs from label graph indices")
        audits.append({**anchor, "accepted": anchor["gate"] == "PASS_POSITIONAL_ELEMENT_ANCHOR",
                       "tokenizer": {"repository": REPOSITORY, "revision": REVISION},
                       "special_tokens_mask": [int(i in TOKENIZER_SPECIAL_IDS) for i in anchor["input_ids"]],
                       "graph_atom_count": n_atoms})
    dump(OUT / "D_INPUT_ACCEPTANCE.json", {"status": "PASS_D_INPUT_IDENTITY", "binding": binding,
        "n_molecules": 5000, "n_atoms": len(labels), "concept_ids": concept_ids.tolist(),
        "n_concepts": len(concept_ids), "max_tokens": max(len(a["input_ids"]) for a in audits),
        "Dfit_atoms": int(offsets[4000]), "Dprecision_atoms": int(offsets[-1] - offsets[4000]),
        "T_file_reads": 0})
    return {"rows": rows, "audits": audits, "labels": labels, "molecules": molecules,
            "offsets": offsets, "concept_ids": concept_ids, "panel": panel, "binding": binding}


def array_file(path, shape, dtype, *, resume):
    if resume:
        value = np.load(path, mmap_mode="r+", allow_pickle=False)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError("partial cache array shape/dtype drift")
        return value
    if path.exists():
        raise ValueError("array exists without a matching progress receipt; inspect before repair")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def extract_D(data, budget):
    ACT.mkdir(parents=True, exist_ok=True)
    for name in NETWORKS:
        destination = ACT / name
        destination.mkdir(exist_ok=True)
        progress_path, receipt_path = destination / "PROGRESS.json", destination / "CACHE_RECEIPT.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["binding"] != data["binding"] or receipt["status"] != "PASS_D_EXTRACTION":
                raise ValueError("completed D cache belongs to another lock")
            continue
        progress = read_json(progress_path) if progress_path.exists() else {"binding": data["binding"], "completed_molecules": 0}
        if progress["binding"] != data["binding"]:
            raise ValueError("partial D cache belongs to another lock")
        completed = progress["completed_molecules"]
        resume = progress_path.exists()
        atoms = array_file(destination / "atom_a.npy", (len(data["labels"]), 464), np.float32, resume=resume)
        cls = array_file(destination / "cls_a.npy", (5000, 464), np.float32, resume=resume)
        outputs = array_file(destination / "normalized_head.npy", (5000, 199), np.float32, resume=resume)
        dump(progress_path, progress)
        with budget.track("encoder", f"extract_D_{name}"):
            model = ChembertaMTR.from_official_checkpoint() if name == "trained" else ChembertaMTR.initialized_from_config(int(name[4:]))
            for begin in range(completed, 5000, CONFIG["batch_size"]):
                budget.check()
                end = min(begin + CONFIG["batch_size"], 5000)
                batch = batch_from_audits(data["audits"][begin:end])
                result = model(batch)
                for local, mi in enumerate(range(begin, end)):
                    # Explicit graph-order indexing: do not silently substitute boolean token order.
                    positions = [a["anchor_token_index"] for a in data["audits"][mi]["atoms"]]
                    atoms[data["offsets"][mi]:data["offsets"][mi + 1]] = result.activations[local, positions].numpy()
                cls[begin:end] = result.activations[:, 0].numpy()
                outputs[begin:end] = result.normalized_predictions.numpy()
                budget.record["model_forward_molecule_evaluations"] += end - begin
                # Flush the bounded batch before advancing durable progress.
                atoms.flush(); cls.flush(); outputs.flush()
                progress.update(completed_molecules=end, updated_utc=utc())
                dump(progress_path, progress)
                if end % 200 == 0 or end == 5000:
                    print(json.dumps({"phase": "extract_D", "network": name, "completed_molecules": end}), flush=True)
            n, total, squares, absolute = 0, np.zeros(464), np.zeros(464), np.zeros(464)
            for begin in range(0, len(atoms), 4096):
                budget.check()
                block = np.asarray(atoms[begin:begin + 4096], np.float64)
                n += len(block); total += block.sum(0); squares += np.square(block).sum(0); absolute += np.abs(block).sum(0)
            mean = total / n
            sd = np.sqrt(np.maximum(0, squares / n - mean * mean))
            cls64 = np.asarray(cls, np.float64)
            # Matching statistics use float64 from the frozen float32 weights,
            # rather than rounding a previously computed float32 column norm.
            column_norm = model.roberta.encoder.layer[PRIMARY_LAYER].output.dense.weight.to(torch.float64).norm(dim=0).numpy()
            np.savez(destination / "D_moments.npz", atom_mean=mean, atom_mean_abs=absolute / n, atom_sd=sd,
                     cls_mean=cls64.mean(0), cls_sd=cls64.std(0),
                     output_dense_column_norm=column_norm, residual_dose=sd * column_norm)
            receipt = {"status": "PASS_D_EXTRACTION", "binding": data["binding"], "network": name,
                "initialization_seed": None if name == "trained" else int(name[4:]), "n_molecules": 5000,
                "n_atoms": len(atoms), "dtype": "float32", "layer_zero_based": PRIMARY_LAYER,
                "atom_order": "D.tsv order then graph_atom_id; positions read independently from saved mapper audit",
                "atom_shape": list(atoms.shape), "cls_shape": list(cls.shape), "head_shape": list(outputs.shape),
                "moment_SD_ddof": 0, "moment_and_column_norm_dtype": "float64", "normalization": "normalized anonymous head",
                "files": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns, "sha256": sha(p)} for p in
                    [destination / "atom_a.npy", destination / "cls_a.npy", destination / "normalized_head.npy", destination / "D_moments.npz"]},
                "completed_utc": utc(), "T_reads": 0, **memory_usage()}
            dump(receipt_path, receipt)
            del model, atoms, cls, outputs, cls64
            gc.collect()


def require_cache(data, name):
    receipt = read_json(ACT / name / "CACHE_RECEIPT.json")
    if receipt["status"] != "PASS_D_EXTRACTION" or receipt["binding"] != data["binding"]:
        raise ValueError("missing or mismatched D cache")
    # Large cache hashes were computed once on closeout; cheap size/shape checks
    # reuse that evidence here, instead of hashing four full arrays at every basis.
    path = ACT / name / "atom_a.npy"
    identity = receipt["files"]["atom_a.npy"]
    if path.stat().st_size != identity["bytes"] or path.stat().st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("cached atom array size drift")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (len(data["labels"]), 464):
        raise ValueError("D cache atom shape/dtype drift")
    return value


def basis_matrix(native, basis, budget):
    if basis == "native":
        return native
    rotations = OUT / "rotations"
    rotations.mkdir(exist_ok=True)
    seed = int(basis[1:]); path = rotations / f"Q{seed}.npy"
    if not path.exists():
        q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((464, 464)))
        q *= np.where(np.diag(r) < 0, -1.0, 1.0)
        np.save(path, q)
        dump(rotations / f"Q{seed}.json", {"seed": seed, "dtype": "float64", "shape": [464, 464],
            "sha256": sha(path), "algorithm": CONFIG["Q_algorithm"], "all_networks_share_this_Q": True})
    else:
        receipt = read_json(rotations / f"Q{seed}.json")
        if receipt["sha256"] != sha(path):
            raise ValueError("saved common rotation identity drift")
    q32 = np.load(path, allow_pickle=False).astype(np.float32)
    scratch_path = ACT / "rotation_scratch.npy"
    scratch = np.lib.format.open_memmap(scratch_path, mode="w+", dtype=np.float32, shape=native.shape)
    for begin in range(0, len(native), 2048):
        budget.check()
        scratch[begin:begin + 2048] = native[begin:begin + 2048] @ q32
    scratch.flush()
    return scratch


def choose_stage(data, budget, stage):
    destination = OUT / stage
    destination.mkdir(exist_ok=True)
    boundary = int(data["offsets"][4000])
    labels = data["labels"][:boundary] if stage == "Dfit" else data["labels"]
    candidates = np.full((4, 10, len(data["concept_ids"])), -1, np.int32)
    aps = np.full(candidates.shape, np.nan, np.float64)
    for mi, network in enumerate(NETWORKS):
        native = require_cache(data, network)
        for bi, basis in enumerate(BASES):
            file = destination / f"{network}_{basis}.npz"
            receipt_path = file.with_suffix(".json")
            if file.exists() and receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt["binding"] != data["binding"] or receipt["stage"] != stage or receipt["sha256"] != sha(file):
                    raise ValueError("completed winner stage identity drift")
                with np.load(file, allow_pickle=False) as saved:
                    candidates[mi, bi], aps[mi, bi] = saved["candidate_indices"], saved["AP"]
                continue
            with budget.track("statistics", f"{stage}_{network}_{basis}_rotation"):
                scores = basis_matrix(native, basis, budget)
            with budget.track("statistics", f"{stage}_{network}_{basis}_all928_AP"):
                candidate, ap = select_winners(scores[:len(labels)], labels,
                    progress=lambda unit: budget.check())
                np.savez(file, candidate_indices=candidate, AP=ap, concept_ids=data["concept_ids"])
                dump(receipt_path, {"status": "PASS_ALL928_SELECTION", "binding": data["binding"], "stage": stage,
                    "network": network, "basis": basis, "n_atoms": len(labels), "n_molecules": 4000 if stage == "Dfit" else 5000,
                    "candidates_per_concept": 928, "concept_ids": data["concept_ids"].tolist(), "sha256": sha(file),
                    "missing_positive_winner_count": int((candidate < 0).sum()), "T_scientific_reads": 0})
                candidates[mi, bi], aps[mi, bi] = candidate, ap
            del scores
            print(json.dumps({"phase": stage, "network": network, "basis": basis, "status": "complete"}), flush=True)
        del native
    target = OUT / ("D_FIT_WINNERS.npz" if stage == "Dfit" else "D_FINAL_WINNERS.npz")
    np.savez(target, candidate_indices=candidates, AP_D=aps, concept_ids=data["concept_ids"],
             networks=np.array(NETWORKS), bases=np.array(BASES))
    dump(target.with_suffix(".json"), {"status": "PASS_D_FIXED_WINNERS_DEVELOPMENT_ONLY", "binding": data["binding"],
        "stage": stage, "array_shape": list(candidates.shape), "networks": list(NETWORKS), "bases": list(BASES),
        "candidate_encoding": CONFIG["candidate_encoding"], "concept_ids": data["concept_ids"].tolist(),
        "sha256": sha(target), "T_access_authorized": False, "root_D_OUTPUT_LOCK_required": True})
    return candidates


def precision_D(data, budget):
    winners = choose_stage(data, budget, "Dfit")
    boundary = int(data["offsets"][4000])
    labels = data["labels"][boundary:]
    molecules = data["molecules"][boundary:] - 4000
    with budget.track("statistics", "Dprecision_shared_molecule_weights_and_prevalence"):
        weights = cluster_weights(1000, 500, 42016)
        # Add point sample first, then the500 shared molecule bootstrap samples.
        all_weights = np.vstack((np.ones((1, 1000), np.int32), weights))
        positives, atom_counts = molecule_label_counts(labels, molecules, 1000)
        pi = weighted_prevalence(positives, atom_counts, all_weights)
    ell = np.full((501, 4, 10, len(data["concept_ids"])), np.nan, np.float64)
    destination = OUT / "Dprecision"
    destination.mkdir(exist_ok=True)
    for mi, network in enumerate(NETWORKS):
        native = require_cache(data, network)
        for bi, basis in enumerate(BASES):
            path = destination / f"{network}_{basis}.npz"
            receipt_path = path.with_suffix(".json")
            if path.exists() and receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt["binding"] != data["binding"] or receipt["sha256"] != sha(path):
                    raise ValueError("precision cache identity drift")
                with np.load(path, allow_pickle=False) as saved:
                    ell[:, mi, bi] = saved["ell"]
                continue
            with budget.track("statistics", f"Dprecision_{network}_{basis}_rotation"):
                scores = basis_matrix(native, basis, budget)
            with budget.track("statistics", f"Dprecision_{network}_{basis}_500_molecule_bootstrap"):
                ap = fixed_winner_ap(scores[boundary:], labels, molecules, winners[mi, bi], all_weights, budget_check=budget.check)
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.log(ap / pi)
                ell[:, mi, bi] = values
                np.savez(path, ell=values)
                dump(receipt_path, {"binding": data["binding"], "sha256": sha(path), "replicates": 500,
                     "point_plus_replicates": 501, "sorted_once_per_selected_scalar": True, "T_reads": 0})
            del scores
            print(json.dumps({"phase": "Dprecision", "network": network, "basis": basis, "status": "complete"}), flush=True)
        del native
    np.savez(OUT / "D_PRECISION_REPLICATES.npz", point_ell=ell[0], replicated_ell=ell[1:], concept_ids=data["concept_ids"])
    with budget.track("statistics", "Dprecision_joint_estimands_and_percentile_summary"):
        summary = bootstrap_summary(ell[0], ell[1:], precision=True)
    summary.update({"status": "PASS_D_PRECISION_DEVELOPMENT_ONLY" if summary["precision_gate"] else "FAIL_D_PRECISION_STOP_T",
                    "binding": data["binding"], "concept_ids": data["concept_ids"].tolist(),
                    "Dfit": 4000, "Dprecision": 1000, "all_C_from_full_D": True,
                    "T_unsealed": False, "T_event_gate_evaluated": False,
                    "Dprecision_is_development_data": True, "completed_utc": utc()})
    dump(OUT / "D_PRECISION.json", safe_json(summary))
    return safe_json(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("ready", "extract-D", "precision-D", "final-D"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1); set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.stage == "ready":
            result = write_ready()
        else:
            binding = require_lock(args.lock)
            data = load_D(binding)
            budget = Budget(binding)
            dump(OUT / "RUNTIME.json", {"started_utc": utc(), "threads": 4,
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                "BLAS": threadpool_info(), "memory": memory_usage(), "config": CONFIG})
            try:
                if args.stage == "extract-D":
                    extract_D(data, budget)
                    result = {"status": "PASS_D_EXTRACTION", "activation_root": str(ACT), "T_reads": 0}
                elif args.stage == "precision-D":
                    result = precision_D(data, budget)
                else:
                    precision = read_json(OUT / "D_PRECISION.json")
                    if precision.get("binding") != binding or not precision.get("precision_gate"):
                        raise ValueError("final D selection requires the matching passing D precision gate")
                    choose_stage(data, budget, "Dfull")
                    result = {"status": "PASS_FINAL_D_WINNERS_DEVELOPMENT_ONLY", "precision_gate": precision["precision_gate"],
                        "winner_file": str(OUT / "D_FINAL_WINNERS.npz"), "D_OUTPUT_LOCK_created": False,
                        "root_next_step": "finite input D readout and mechanism/control freezing; root judges precision gate; T remains sealed"}
                    dump(OUT / "D_RUN_HANDOFF.json", {**result, "binding": binding, "activation_root": str(ACT),
                        "networks": list(NETWORKS), "bases": list(BASES), "concept_ids": data["concept_ids"].tolist(),
                        "atom_cache_dtype": "float32", "atom_cache_shape": [len(data["labels"]), 464],
                        "head_cache_shape": [5000, 199], "CLS_cache_shape": [5000, 464],
                        "D_input_dir": str(D_DIR), "panel_ids": data["panel"]["panel_ids"],
                        "mechanism_candidate_ids": data["panel"]["mechanism_candidate_ids"], "CPU_ledger": budget.record})
            except BaseException as error:
                dump(OUT / "D_FAILURE.json", {"stage": args.stage, "time_utc": utc(), "binding": binding,
                    "error_type": type(error).__name__, "message": str(error), "T_scientific_reads": 0,
                    "resume_rule": "same locked identities only; no change to seeds/C/replicates or limits"})
                raise
        print(json.dumps(safe_json(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
