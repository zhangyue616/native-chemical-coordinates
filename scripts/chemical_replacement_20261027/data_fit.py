"""D-only continuous replacement fitting and frozen all-T prediction; no forward.

The old finite-input implementation supplies only the unchanged graph keys.
This module never invokes old scientific entry points or writes their outputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OLD_SCRIPTS = ROOT / "scripts/revision_20261017"
if str(OLD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(OLD_SCRIPTS))
from chem_input_reference import input_keys

BASE = ROOT / "data/revision_20261017/chemistry"
FIT = ROOT / "results/chemical_replacement_20261027/fit"
DOCS = ROOT / "docs/chemical_replacement_20261027"
COORDINATE_IDS = (293, 454)
CONCEPT_IDS = (26, 24)
MODEL_NAMES = ("D_mean", "M1", "M2")
ALPHA = 20.0
QUANTILES = (0.2, 0.4, 0.6, 0.8)
REPOSITORY = "DeepChem/ChemBERTa-77M-MTR"
REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"
SPECIAL_IDS = (0, 11, 12, 13, 14, 591, 592)
_SPLITS = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_new(path, value):
    """No replacement of completed or partial artifacts is implicit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def memory_usage():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    if os.name != "nt":
        return {"rss_bytes": None, "peak_rss_bytes": None, "private_commit_bytes": None, "peak_private_commit_bytes": None}
    value = Counters(); value.cb = ctypes.sizeof(value)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    measure.restype = ctypes.c_int
    if not measure(current(), ctypes.byref(value), value.cb):
        raise RuntimeError("Cannot measure Windows process memory")
    return {"rss_bytes": int(value.WorkingSetSize), "peak_rss_bytes": int(value.PeakWorkingSetSize),
            "private_commit_bytes": int(value.PagefileUsage), "peak_private_commit_bytes": int(value.PeakPagefileUsage)}


def metadata(path, expected=None):
    path = Path(path); stat = path.stat()
    if expected is not None and stat.st_size != expected["bytes"]:
        raise ValueError(f"Previously accepted input byte size changed: {path}")
    result = {"path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if expected is not None:
        result["sha256_from_existing_receipt"] = expected["sha256"]
        if "mtime_ns" in expected and stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Previously accepted cache mtime changed: {path}")
    return result


def load_split(split):
    """Read-only original identity/order loader, cached once per process.

    Return rows/audits and original all91 atom labels, graph ordering arrays,
    plus read-only mmap atom_a and cached_head. No file is created here.
    """
    split = split.upper()
    if split not in ("D", "T"):
        raise ValueError("split must be D or T")
    if split in _SPLITS:
        return _SPLITS[split]
    is_d = split == "D"
    nm, na, rank_start = (5000, 124530, 0) if is_d else (10000, 246722, 5000)
    directory = BASE / ("development" if is_d else "test")
    original = directory if is_d else BASE / "sealed"
    data_receipt_path = BASE / "corpus/BUILD_RECEIPT.json" if is_d else directory / "T_DATA_RECEIPT.json"
    data_receipt = read_json(data_receipt_path)
    if not is_d and data_receipt.get("status") != "PASS_T_DATA_BUILD":
        raise ValueError("Accepted T data receipt missing")
    inputs = {}
    for path in (original / f"{split}.tsv", original / f"anchors_{split}.jsonl.gz",
                 directory / f"labels_{split}.npz", directory / f"atoms_{split}.npz"):
        rel = path.relative_to(ROOT).as_posix()
        expected = data_receipt.get("T_inputs", {}).get(rel) or data_receipt["outputs"][rel]
        inputs[rel] = metadata(path, expected)
    with (original / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    with gzip.open(original / f"anchors_{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
        anchors = [json.loads(line) for line in stream]
    with np.load(directory / f"labels_{split}.npz", allow_pickle=False) as source:
        labels_all, label_ids = source["labels"].copy(), source["concept_ids"].copy()
    with np.load(directory / f"atoms_{split}.npz", allow_pickle=False) as source:
        mol_idx, atom_idx, mol_offsets = source["mol_idx"].copy(), source["atom_idx"].copy(), source["mol_offsets"].copy()
    if labels_all.shape != (na, 91) or label_ids.shape != (91,) or len(np.unique(label_ids)) != 91:
        raise ValueError(f"{split} original all91 label shape/identity differs")
    if not np.isin(labels_all, (0, 1)).all() or any(c not in label_ids for c in CONCEPT_IDS):
        raise ValueError(f"{split} atom labels invalid or target concept missing")
    if len(rows) != nm or len(anchors) != nm or mol_offsets.shape != (nm + 1,) or int(mol_offsets[0]) != 0 or int(mol_offsets[-1]) != na:
        raise ValueError(f"{split} molecule/atom/anchor dimensions differ")
    counts = np.diff(mol_offsets)
    if np.any(counts <= 0) or not np.array_equal(mol_idx, np.repeat(np.arange(nm), counts)):
        raise ValueError(f"{split} molecule atom ordering differs")
    if not np.array_equal(atom_idx, np.concatenate([np.arange(n) for n in counts])):
        raise ValueError(f"{split} graph atom order differs")
    audits = []
    for mi, (row, anchor) in enumerate(zip(rows, anchors)):
        if int(row["rank"]) != rank_start + mi or anchor["rank"] != rank_start + mi or int(row["cid"]) != anchor["cid"]:
            raise ValueError(f"{split} rank/CID identity differs at {mi}")
        if hashlib.sha256(row["smiles"].encode("utf-8")).hexdigest() != row["canonical_smiles_sha256"]:
            raise ValueError(f"{split} exact canonical identity differs at {mi}")
        if row["canonical_smiles_sha256"] != anchor["canonical_smiles_sha256"]:
            raise ValueError(f"{split} anchor canonical identity differs at {mi}")
        fingerprint = hashlib.sha256(json.dumps(anchor["input_ids"], separators=(",", ":")).encode("ascii")).hexdigest()
        if fingerprint != anchor["input_ids_sha256"]:
            raise ValueError(f"{split} input-ID fingerprint differs at {mi}")
        n = int(counts[mi]); positions = [a["anchor_token_index"] for a in anchor["atoms"]]
        if [a["graph_atom_id"] for a in anchor["atoms"]] != list(range(n)) or len(set(positions)) != n:
            raise ValueError(f"{split} atom-anchor bijection differs at {mi}")
        if anchor["gate"] != "PASS_POSITIONAL_ELEMENT_ANCHOR" or not all(a["anchor_offset_verified"] for a in anchor["atoms"]):
            raise ValueError(f"{split} accepted anchor gate differs at {mi}")
        ids = anchor["input_ids"]; mask = anchor["attention_mask"]
        if len(ids) != len(mask) or not ids or ids[0] != 12 or not all(x == 1 for x in mask):
            raise ValueError(f"{split} unpadded encoding differs at {mi}")
        if any(not isinstance(p, int) or not 0 <= p < len(ids) or ids[p] in SPECIAL_IDS for p in positions):
            raise ValueError(f"{split} invalid/special atom anchor at {mi}")
        audits.append({**anchor, "accepted": True, "tokenizer": {"repository": REPOSITORY, "revision": REVISION},
                       "special_tokens_mask": [int(i in SPECIAL_IDS) for i in ids], "graph_atom_count": n})
    cache_dir = BASE / f"activations_{split}/trained"
    cache_receipt_path = cache_dir / "CACHE_RECEIPT.json"
    cache = read_json(cache_receipt_path)
    if cache.get("status") != f"PASS_{split}_EXTRACTION" or cache.get("network") != "trained" or cache.get("layer_zero_based") != 1:
        raise ValueError(f"{split} trained post-GELU cache identity differs")
    cache_files = {}
    for name, shape in (("atom_a.npy", (na, 464)), ("normalized_head.npy", (nm, 199))):
        path = cache_dir / name
        cache_files[name] = metadata(path, cache["files"][name])
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != np.float32 or value.flags.writeable:
            raise ValueError(f"{split} {name} shape/dtype/read-only differs")
        cache_files[name].update(shape=list(value.shape), dtype=str(value.dtype))
        if name == "atom_a.npy":
            atom_a = value
        else:
            cached_head = value
    input_metadata = {"split": split, "status": "PASS_METADATA_SHAPE_GRAPH_ANCHOR_IDENTITY", "n_molecules": nm, "n_atoms": na,
        "label_ids": label_ids.tolist(), "max_tokens": max(len(a["input_ids"]) for a in audits), "inputs": inputs,
        "cache_files": cache_files, "cache_binding": cache["binding"], "T_binding": cache.get("T_binding"),
        "source_data_receipt": metadata(data_receipt_path), "source_cache_receipt": metadata(cache_receipt_path),
        "verification_scope": "once per process: bytes/accepted-cache mtime, NPY header, all molecule/atom/anchor identities; existing large scientific files not rehashed"}
    result = dict(rows=rows, audits=audits, labels_all=labels_all, label_ids=label_ids, mol_offsets=mol_offsets,
                  mol_idx=mol_idx, atom_idx=atom_idx, atom_a=atom_a, cached_head=cached_head, input_metadata=input_metadata)
    _SPLITS[split] = result
    return result


def graph_keys(data):
    tokens = [[a["input_ids"][p["anchor_token_index"]] for p in a["atoms"]] for a in data["audits"]]
    return input_keys([row["smiles"] for row in data["rows"]], tokens)


def fit_continuous(rows, activation, labels, *, coordinate, concept_id, molecules=None):
    """Independent continuous r0/r1 shrinkage; labels affect M1 only."""
    y = np.asarray(activation, dtype=np.float64)
    label = np.asarray(labels)
    if y.shape != (len(rows),) or label.shape != y.shape or not len(rows) or not np.isfinite(y).all() or not np.isin(label, (0, 1)).all():
        raise ValueError("Invalid continuous fitting dimensions/values/labels")
    negative, positive = label == 0, label == 1
    if not negative.any() or not positive.any():
        raise ValueError("M1 requires nonempty D positive and negative atom groups")
    mean = float(np.sum(y, dtype=np.float64) / len(y))
    counts0, counts1, parents = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0]), {}
    for row, value in zip(rows, y):
        r0, r1 = row["r0"], row["r1"]
        if r1 in parents and parents[r1] != r0:
            raise ValueError("r1 does not determine exactly one r0 parent")
        parents[r1] = r0
        counts0[r0][0] += 1; counts0[r0][1] += float(value)
        counts1[r1][0] += 1; counts1[r1][1] += float(value)
    table0 = {g: float((s + ALPHA * mean) / (n + ALPHA)) for g, (n, s) in counts0.items()}
    table1 = {g: float((s + ALPHA * table0[parents[g]]) / (n + ALPHA)) for g, (n, s) in counts1.items()}
    cuts = np.quantile(y, QUANTILES, method="linear")
    bins = np.searchsorted(cuts, y, side="right")
    dcounts = {"atoms": len(y), "negative_atoms": int(negative.sum()), "positive_atoms": int(positive.sum()),
               "r0_categories": len(table0), "r1_categories": len(table1), "D_bin_atoms": np.bincount(bins, minlength=5).tolist()}
    if molecules is not None:
        dcounts.update(molecules=int(np.unique(molecules).size), positive_molecules=int(np.unique(molecules[positive]).size))
    return {"coordinate": int(coordinate), "concept_id": int(concept_id), "mean": mean,
        "m1_negative": float(np.sum(y[negative], dtype=np.float64) / negative.sum()),
        "m1_positive": float(np.sum(y[positive], dtype=np.float64) / positive.sum()),
        "table_r0": table0, "table_r1": table1, "counts_r0": dict(counts0), "counts_r1": dict(counts1), "parents_r1": parents,
        "bin_cuts": cuts.tolist(), "alpha": ALPHA, "Dcounts": dcounts, "quantiles": list(QUANTILES),
        "quantile_method": "linear", "bin_side": "right", "bin_count_fixed": 5,
        "M2_inputs": ["exact_r0", "exact_r1"], "M2_fallback": ["r1", "r0", "global"],
        "M2_no_inputs": ["T_actual_activation", "activation_bin", "output_q", "target_label", "token"]}


def predict_continuous(model, rows, labels):
    """D_mean/M1/M2 targets; T activation is intentionally not an argument."""
    labels = np.asarray(labels)
    if labels.shape != (len(rows),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Invalid prediction target-label dimensions/values")
    prediction = np.empty((len(rows), 3), dtype=np.float64)
    prediction[:, 0] = model["mean"]
    prediction[:, 1] = np.where(labels, model["m1_positive"], model["m1_negative"])
    unseen_r0 = unseen_r1 = used_r0 = used_global = 0
    for i, row in enumerate(rows):
        r0_known, r1_known = row["r0"] in model["table_r0"], row["r1"] in model["table_r1"]
        unseen_r0 += not r0_known; unseen_r1 += not r1_known
        if r1_known:
            value = model["table_r1"][row["r1"]]
        elif r0_known:
            value = model["table_r0"][row["r0"]]; used_r0 += 1
        else:
            value = model["mean"]; used_global += 1
        prediction[i, 2] = value
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite replacement predictions")
    n = len(rows)
    return prediction, {"atoms": n, "unseen_counts": {"r0": int(unseen_r0), "r1": int(unseen_r1)},
        "unseen_fractions": {"r0": unseen_r0 / n, "r1": unseen_r1 / n},
        "actual_fallback_counts": {"r1": n - used_r0 - used_global, "r0": used_r0, "global": used_global},
        "actual_fallback_fractions": {"r1": (n - used_r0 - used_global) / n, "r0": used_r0 / n, "global": used_global / n}}


def synthetic_check():
    rows = [{"token": "x", "r0": a, "r1": b} for a, b in (("A", "a"), ("A", "a"), ("B", "b"), ("B", "c"))]
    model = fit_continuous(rows, [0, 2, 6, 8], [0, 1, 0, 1], coordinate=293, concept_id=26)
    expected0 = 82 / 22
    expected1 = (2 + 20 * expected0) / 22
    assert model["mean"] == 4 and abs(model["table_r0"]["A"] - expected0) < 1e-14
    assert abs(model["table_r1"]["a"] - expected1) < 1e-14
    test = [{"token": "z", "r0": a, "r1": b} for a, b in (("A", "a"), ("A", "unseen"), ("unseen", "new"))]
    p, report = predict_continuous(model, test, [0, 1, 0])
    assert np.array_equal(p[:, 1], [3, 5, 3])
    assert np.allclose(p[:, 2], [expected1, expected0, 4], rtol=0, atol=1e-14)
    assert report["actual_fallback_counts"] == {"r1": 1, "r0": 1, "global": 1}
    # Different held-out actual values alter only descriptive bins; prediction
    # receives keys/labels alone and token changes likewise cannot affect M2.
    before = p.copy()
    bins_a = np.searchsorted(model["bin_cuts"], [-100, 0, 100], side="right")
    bins_b = np.searchsorted(model["bin_cuts"], [100, 100, -100], side="right")
    after, _ = predict_continuous(model, [{**r, "token": "other"} for r in test], [0, 1, 0])
    assert not np.array_equal(bins_a, bins_b) and np.array_equal(before, after)
    constant = fit_continuous(rows, [1, 1, 1, 1], [0, 1, 0, 1], coordinate=293, concept_id=26)
    assert constant["bin_cuts"] == [1, 1, 1, 1] and constant["Dcounts"]["D_bin_atoms"] == [0, 0, 0, 0, 4]
    return {"status": "PASS_SYNTHETIC_HAND_CALCULATIONS", "checks": ["continuous_r0_r1_alpha20_shrinkage", "M1_atom_label_means",
        "r1_r0_global_fallback", "prediction_independent_of_heldout_actual_and_token", "all_four_duplicate_cutpoints_retained"],
        "real_D_or_T_reads": 0, "model_forwards": 0}


class Budget:
    def __init__(self):
        self.wall, self.cpu = time.perf_counter(), time.process_time()
        self.caps = read_json(DOCS / "RESOURCE_LOCK.json")["fit_and_summary_combined"]
        self.stages = []

    def check(self):
        m = memory_usage()
        if time.perf_counter() - self.wall > self.caps["wall_seconds_cap"] or time.process_time() - self.cpu > self.caps["process_cpu_seconds_cap"]:
            raise RuntimeError("Locked fit/summary time cap reached; no automatic expansion")
        for key, cap in (("peak_rss_bytes", "peak_rss_bytes_cap"), ("peak_private_commit_bytes", "private_commit_bytes_cap")):
            if m[key] is not None and m[key] > self.caps[cap]:
                raise MemoryError(f"Locked memory cap reached: {key}")
        return m

    def stage(self, name, start_wall, start_cpu):
        self.stages.append({"stage": name, "wall_seconds": time.perf_counter() - start_wall,
                            "process_cpu_seconds": time.process_time() - start_cpu, "memory": self.check()})

    def receipt(self):
        return {"wall_seconds": time.perf_counter() - self.wall, "process_cpu_seconds": time.process_time() - self.cpu,
                "memory": memory_usage(), "stages": self.stages, "model_forwards": 0, "model_loads": 0, "network_calls": 0,
                "T_model_fitting": 0, "threads": 4, "resource_lock_scope": "fit_and_summary_combined; summary adds this fit consumption"}


def fit_D(budget):
    if (FIT / "D_FIT.json").exists() or (FIT / "D_FIT_LOCK.json").exists():
        raise ValueError("D fit already exists; use apply-T or inspect existing freeze, never refit implicitly")
    start_w, start_c = time.perf_counter(), time.process_time()
    data = load_split("D"); keys = graph_keys(data); models = {}
    for coordinate, concept in zip(COORDINATE_IDS, CONCEPT_IDS):
        column = int(np.flatnonzero(data["label_ids"] == concept)[0])
        models[f"j{coordinate}"] = fit_continuous(keys, data["atom_a"][:, coordinate], data["labels_all"][:, column],
            coordinate=coordinate, concept_id=concept, molecules=data["mol_idx"])
        budget.check()
    implementation = sha(Path(__file__))
    result = {"schema": "chemical-continuous-D-fit/1", "status": "D_ONLY_MODELS_FITTED", "created_utc": utc(),
        "coordinate_ids": list(COORDINATE_IDS), "concept_ids": list(CONCEPT_IDS), "conditions": list(MODEL_NAMES), "models": models,
        "protocol_sha256": sha(DOCS / "PROTOCOL.md"), "resource_lock_sha256": sha(DOCS / "RESOURCE_LOCK.json"),
        "implementation_sha256": implementation, "graph_keys_implementation_sha256": sha(OLD_SCRIPTS / "chem_input_reference.py"),
        "input_metadata_D": data["input_metadata"], "T_reads_before_D_freeze": 0, "model_forwards": 0,
        "fit_definition": "float64 D-only continuous activation means; r0 shrinks to global and r1 to r0 with alpha20"}
    dump_new(FIT / "D_FIT.json", result)
    lock = {"schema": "chemical-continuous-D-fit-lock/1", "status": "D_FIT_FROZEN_BEFORE_T_APPLY", "created_utc": utc(),
        "D_FIT_sha256": sha(FIT / "D_FIT.json"), "protocol_sha256": result["protocol_sha256"],
        "resource_lock_sha256": result["resource_lock_sha256"], "implementation_sha256": implementation,
        "T_reads_before_D_freeze": 0, "refitting_on_T_allowed": False}
    dump_new(FIT / "D_FIT_LOCK.json", lock)
    budget.stage("D_identity_keys_continuous_fit_and_freeze", start_w, start_c)
    print(json.dumps({"stage": "D_FIT_FROZEN", "D_FIT_sha256": lock["D_FIT_sha256"],
                      "Dcounts": {k: v["Dcounts"] for k, v in models.items()}}), flush=True)
    return result, lock


def require_frozen_fit():
    lock = read_json(FIT / "D_FIT_LOCK.json")
    checks = {"D_FIT_sha256": FIT / "D_FIT.json", "protocol_sha256": DOCS / "PROTOCOL.md",
              "resource_lock_sha256": DOCS / "RESOURCE_LOCK.json", "implementation_sha256": Path(__file__)}
    if lock.get("status") != "D_FIT_FROZEN_BEFORE_T_APPLY" or any(lock[k] != sha(p) for k, p in checks.items()):
        raise ValueError("D frozen fit/program/protocol/resource identity differs")
    return read_json(FIT / "D_FIT.json"), lock


def apply_T(budget):
    target = FIT / "T_activations.npz"
    if target.exists() or (FIT / "T_ACTIVATIONS_RECEIPT.json").exists():
        raise ValueError("T prediction artifact exists; no automatic replacement or repeat")
    start_w, start_c = time.perf_counter(), time.process_time()
    fit, lock = require_frozen_fit()
    data = load_split("T"); keys = graph_keys(data)
    if data["label_ids"].tolist() != fit["input_metadata_D"]["label_ids"]:
        raise ValueError("T and D original all91 label dictionaries differ")
    labels = data["labels_all"][:, [int(np.flatnonzero(data["label_ids"] == c)[0]) for c in CONCEPT_IDS]].copy()
    predictions = np.empty((len(keys), 2, 3), dtype=np.float64); reports = {}
    for ci, coordinate in enumerate(COORDINATE_IDS):
        predictions[:, ci], reports[f"j{coordinate}"] = predict_continuous(fit["models"][f"j{coordinate}"], keys, labels[:, ci])
        budget.check()
    # Actual activations are accessed only after all replacement targets exist.
    actual = np.asarray(data["atom_a"][:, COORDINATE_IDS], dtype=np.float32).copy()
    if actual.shape != (246722, 2) or not np.isfinite(actual).all():
        raise ValueError("T selected actual activation shape/finite gate failed")
    bins = np.empty(actual.shape, dtype=np.int8)
    for ci, coordinate in enumerate(COORDINATE_IDS):
        bins[:, ci] = np.searchsorted(fit["models"][f"j{coordinate}"]["bin_cuts"], actual[:, ci], side="right").astype(np.int8)
        reports[f"j{coordinate}"]["T_Dfixed_bin_atom_counts"] = np.bincount(bins[:, ci], minlength=5).tolist()
    if labels.shape != (246722, 2) or predictions.shape != (246722, 2, 3) or not np.isfinite(predictions).all():
        raise ValueError("T prediction dimensions/finite gate failed")
    with target.open("xb") as stream:
        np.savez_compressed(stream, coordinate_ids=np.asarray(COORDINATE_IDS, np.int32), concept_ids=np.asarray(CONCEPT_IDS, np.int32),
            conditions=np.asarray(MODEL_NAMES), model_names=np.asarray(MODEL_NAMES), actual=actual, predictions=predictions,
            target_labels=labels, mol_offsets=data["mol_offsets"], mol_idx=data["mol_idx"], atom_idx=data["atom_idx"],
            labels_all=data["labels_all"], label_ids=data["label_ids"], bins=bins)
    receipt = {"schema": "chemical-continuous-T-predictions/1", "status": "PASS_ALL_T_FROZEN_PREDICTIONS", "created_utc": utc(),
        "T_activations_sha256": sha(target), "D_FIT_sha256": lock["D_FIT_sha256"],
        "protocol_sha256": lock["protocol_sha256"], "implementation_sha256": lock["implementation_sha256"],
        "shape": {"actual": list(actual.shape), "predictions": list(predictions.shape), "target_labels": list(labels.shape),
                  "labels_all": list(data["labels_all"].shape), "mol_offsets": list(data["mol_offsets"].shape)},
        "dtype": {"actual": str(actual.dtype), "predictions": str(predictions.dtype), "bins": str(bins.dtype)},
        "coordinate_ids": list(COORDINATE_IDS), "concept_ids": list(CONCEPT_IDS), "conditions": list(MODEL_NAMES),
        "fallback_reports": reports, "input_metadata_T": data["input_metadata"],
        "prediction_inputs": "D-frozen model and T exact graph keys; M1 additionally uses existing target atom labels",
        "T_actual_use": "read only after prediction: actual output array and D-fixed descriptive bins; never input to prediction",
        "model_forwards": 0, "T_refitting": False}
    dump_new(FIT / "T_ACTIVATIONS_RECEIPT.json", receipt)
    budget.stage("T_identity_keys_Dfrozen_apply_and_artifact", start_w, start_c)
    print(json.dumps({"stage": receipt["status"], "T_activations_sha256": receipt["T_activations_sha256"],
                      "fallback_reports": reports}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("synthetic-check", "fit-D", "apply-T", "all"))
    args = parser.parse_args(); FIT.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    with threadpool_limits(limits=4):
        try:
            if args.stage in ("synthetic-check", "all"):
                report = synthetic_check(); dump_new(FIT / "SYNTHETIC_CHECKS.json", report); print(json.dumps(report), flush=True)
            if args.stage in ("fit-D", "all"):
                fit_D(budget)
            if args.stage in ("apply-T", "all"):
                apply_T(budget)
            dump_new(FIT / f"EXECUTION_{args.stage.replace('-', '_')}.json", {"status": "PASS", "created_utc": utc(),
                     "stage": args.stage, **budget.receipt()})
        except BaseException as error:
            failure = FIT / f"FAILURE_{time.time_ns()}.json"
            dump_new(failure, {"status": "FAILED", "stage": args.stage, "created_utc": utc(), "error_type": type(error).__name__,
                              "message": str(error), **budget.receipt()})
            raise


if __name__ == "__main__":
    main()
