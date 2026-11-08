"""Pilot3-A implementation. Model work requires an explicit frozen stage release.

Imports, synthetic selftest, and metadata preparation never load a model.
Only this script and results/pilot3a are Pilot3-A analysis implementation outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
DIM = 1280
HIDDEN = 320
SEED = 2026101603
SPLITS = {"D_fit": 64, "D_select": 32, "C_cal": 16, "T_confirm": 64}
MAX_FORWARDS = 2048
MAX_PLANNED_FORWARDS = 760
MAX_MODEL_SECONDS = 3600.0
RIDGE_LAMBDA = 10.0
DATA_SHA256 = "ab28dd0e372e26b1530c806bfe994f7bc4b172da91fb90f38d02777dd7d405f7"
MANIFEST_SHA256 = "38cd7cc376809543ca511bde39e6b01cfcc64a881264959b3369e8a87faf6c6b"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def readjson(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return json.load(f)


def writejson(path, value, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)


def load_records(path):
    """Check the accepted analysis schema; do not rebuild the acquisition graph."""
    with Path(path).open(encoding="utf-8-sig") as f:
        records = [json.loads(line) for line in f if line.strip()]
    seen_ids, seen_components = set(), set()
    for p in records:
        if p["split"] not in SPLITS:
            raise ValueError("Unexpected analysis split")
        if p["id"] in seen_ids or p["component_key"] in seen_components:
            raise ValueError("Duplicate analysis protein/component")
        seen_ids.add(p["id"])
        seen_components.add(p["component_key"])
        sequence = p["sequence"]
        if not 64 <= len(sequence) <= 256 or set(sequence) - set(AA):
            raise ValueError("Unexpected frozen sequence")
        qs = []
        for field, label in [("h_site", "HELIX"), ("s_site", "STRAND")]:
            site = p[field]
            q = int(site["q_0based"])
            if site["label"] != label or not 7 <= q < len(sequence) - 7:
                raise ValueError("Unexpected frozen site")
            if sequence[q] != "A" or site["center_aa"] != "A":
                raise ValueError("Center must be alanine")
            if site["future_token_index_0based"] != q + 1:
                raise ValueError("Frozen BOS offset mismatch")
            qs.append(q)
        if qs[0] == qs[1]:
            raise ValueError("Paired sites must be distinct")
    if Counter(p["split"] for p in records) != Counter(SPLITS):
        raise ValueError("Frozen split sizes differ from 64/32/16/64")
    return records


def flatten(records, split):
    """Preserve frozen component order, HELIX then STRAND within component."""
    result = []
    for p in records:
        if p["split"] != split:
            continue
        for field, label, target in [("h_site", "HELIX", 1.0),
                                     ("s_site", "STRAND", 0.0)]:
            q = int(p[field]["q_0based"])
            result.append({"id": p["id"], "component": p["component_key"],
                           "split": split, "sequence": p["sequence"],
                           "q": q, "label": label, "y": target,
                           "case_id": f"{p['id']}:{label}:{q}"})
    return result


def control_features(sequence, q):
    """60 visible AA proportions, q/(L-1), its square, L/256, local availability."""
    length = len(sequence)
    if not 0 <= q < length or set(sequence) - set(AA):
        raise ValueError("Invalid feature sequence/center")
    result = []
    for radius in [None, 7, 15]:
        lo = 0 if radius is None else max(0, q - radius)
        hi = length if radius is None else min(length, q + radius + 1)
        visible = sequence[lo:q] + sequence[q + 1:hi]
        counts = Counter(visible)
        result.extend(counts[a] / len(visible) for a in AA)
    relative = q / (length - 1)
    visible15 = min(q, 15) + min(length - q - 1, 15)
    result.extend([relative, relative ** 2, length / 256,
                   visible15 / 30])
    x = np.asarray(result, dtype=np.float64)
    if x.shape != (64,) or not np.isfinite(x).all():
        raise RuntimeError("Control feature dimension/value mismatch")
    return x


def feature_names():
    return [f"{scope}_{a}" for scope in ["global", "r7", "r15"] for a in AA] + [
        "q_over_Lminus1", "relative_q_squared", "length_over_256", "available_r15_over30"]


def features(cases):
    return np.stack([control_features(c["sequence"], c["q"]) for c in cases])


def fit_scaler(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    scale = np.where(sd > 0, sd, 1.0)
    return {"mean": mean, "scale": scale, "constant": sd == 0}


def transform(x, scaler):
    return (np.asarray(x, dtype=np.float64) - scaler["mean"]) / scaler["scale"]


def ridge_fit(x, y, penalty=RIDGE_LAMBDA):
    """Fixed ridge with unpenalized intercept, primal or equivalent dual solve."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or penalty <= 0:
        raise ValueError("Invalid ridge shape/penalty")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    if x.shape[1] <= x.shape[0]:
        beta = np.linalg.solve(xc.T @ xc + penalty * np.eye(x.shape[1]), xc.T @ yc)
    else:
        beta = xc.T @ np.linalg.solve(xc @ xc.T + penalty * np.eye(len(x)), yc)
    return {"coef": beta, "intercept": ym - xm @ beta, "penalty": float(penalty)}


def ridge_predict(model, x):
    return np.asarray(x, dtype=np.float64) @ model["coef"] + model["intercept"]


def all_unit_ridges(x_fit, z_fit, y_fit, x_select, z_select, penalty=RIDGE_LAMBDA):
    """Fit all X+single-z ridges through one base factorization.

    Centering here makes this identical to ridge_fit on every augmented matrix.
    z_fit/z_select are already scaled using D_fit statistics only.
    """
    x, z, y = map(lambda a: np.asarray(a, dtype=np.float64), (x_fit, z_fit, y_fit))
    xm, zm, ym = x.mean(0), z.mean(0), y.mean(0)
    xc, zc, yc = x - xm, z - zm, y - ym
    a = xc.T @ xc + penalty * np.eye(x.shape[1])
    v = np.linalg.solve(a, xc.T @ zc)
    beta0 = np.linalg.solve(a, xc.T @ yc)
    denominator = (zc * zc).sum(0) + penalty - (zc * (xc @ v)).sum(0)
    if np.any(denominator <= 0):
        raise RuntimeError("Nonpositive augmented ridge denominator")
    gamma = (zc.T @ yc - (xc.T @ zc).T @ beta0) / denominator
    beta = beta0[:, None] - v * gamma[None, :]
    intercept = ym - xm @ beta - zm * gamma
    predictions = np.asarray(x_select) @ beta + np.asarray(z_select) * gamma + intercept
    return {"coef_x": beta, "coef_z": gamma, "intercept": intercept,
            "predictions": predictions, "penalty": float(penalty)}


def brier(y, prediction):
    return (np.clip(prediction, 0.0, 1.0) - np.asarray(y)) ** 2


def component_means(values, cases):
    values = np.asarray(values, dtype=np.float64)
    groups = []
    for i in range(0, len(cases), 2):
        pair = cases[i:i + 2]
        if len(pair) != 2 or pair[0]["component"] != pair[1]["component"]:
            raise ValueError("Expected exactly two consecutive sites per component")
        groups.append(values[i:i + 2].mean(axis=0))
    return np.asarray(groups)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


ALLOWED_GROUPS = {frozenset(a) for a in AA} | {frozenset(a) for a in [
    "AILMFPVWY", "STNQ", "AGST", "DEKRH", "DE", "KRH", "FWY", "PG"]}


def validate_hypothesis(response):
    if response.get("status") != "valid":
        return {"status": "NA", "reason": "unknown_or_nonvalid_response"}
    if response.get("operator") not in ["all", "any"]:
        return {"status": "NA", "reason": "invalid_operator"}
    clauses = response.get("clauses")
    if not isinstance(clauses, list) or len(clauses) not in [1, 2]:
        return {"status": "NA", "reason": "invalid_clause_count"}
    count = response.get("development_candidate_count")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 32:
        return {"status": "NA", "reason": "invalid_development_candidate_count"}
    if not isinstance(response.get("explanation_zh"), str) or not response["explanation_zh"].strip():
        return {"status": "NA", "reason": "missing_explanation"}
    offsets = set()
    for clause in clauses:
        offset, aa_set = clause.get("offset"), clause.get("aa_set")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset == 0 or not -7 <= offset <= 7:
            return {"status": "NA", "reason": "invalid_offset"}
        if offset in offsets:
            return {"status": "NA", "reason": "duplicate_offsets"}
        offsets.add(offset)
        if not isinstance(aa_set, str) or len(set(aa_set)) != len(aa_set) or frozenset(aa_set) not in ALLOWED_GROUPS:
            return {"status": "NA", "reason": "invalid_aa_set"}
    return {"status": "VALID", "operator": response["operator"], "clauses": clauses,
            "explanation_zh": response["explanation_zh"], "development_candidate_count": count}


def h_value(hypothesis, sequence, q):
    values = [sequence[q + c["offset"]] in c["aa_set"] for c in hypothesis["clauses"]]
    return float(all(values) if hypothesis["operator"] == "all" else any(values))


def swap(sequence, q, pair):
    a, b = [q + int(offset) for offset in pair]
    changed = list(sequence)
    changed[a], changed[b] = changed[b], changed[a]
    return "".join(changed)


def swap_geometry(pair):
    a, b = pair
    d = abs(b - a)
    distance_bin = "1-3" if d <= 3 else "4-7" if d <= 7 else "8-14"
    side = "LL" if b < 0 else "RR" if a > 0 else "LR"
    return distance_bin, side


def edit_invariants(case, c_sequence, n_sequence):
    p_sequence, q = case["sequence"], case["q"]
    for sequence in [c_sequence, n_sequence]:
        if len(sequence) != len(p_sequence) or sequence == p_sequence or sequence[q] != p_sequence[q]:
            raise RuntimeError("Edit length/center/change invariant failed")
        if not np.array_equal(control_features(sequence, q), control_features(p_sequence, q)):
            raise RuntimeError("Edit changed locked composition/position features")


def compile_edits(cases, hypothesis, max_components):
    if hypothesis["status"] != "VALID":
        return []
    offsets = list(range(-7, 0)) + list(range(1, 8))
    results = []
    for start in range(0, len(cases), 2):
        options = []
        for case in cases[start:start + 2]:
            sequence, q = case["sequence"], case["q"]
            if not h_value(hypothesis, sequence, q):
                continue
            by_kind = {False: {}, True: {}}
            for a_index, a in enumerate(offsets):
                for b in offsets[a_index + 1:]:
                    if sequence[q + a] == sequence[q + b]:
                        continue
                    pair = (a, b)
                    edited = swap(sequence, q, pair)
                    kind, geometry = bool(h_value(hypothesis, edited, q)), swap_geometry(pair)
                    by_kind[kind].setdefault(geometry, []).append((pair, edited))
            for geometry in set(by_kind[False]) & set(by_kind[True]):
                for cp, cs in by_kind[False][geometry]:
                    for npair, ns in by_kind[True][geometry]:
                        key_tuple = (q, cp[0], cp[1], npair[0], npair[1])
                        key = digest(f"pilot3a-edit:{case['split']}:{case['id']}:{q}:"
                                     f"{cp[0]}:{cp[1]}:{npair[0]}:{npair[1]}:{SEED}")
                        options.append((key, key_tuple, case, cp, cs, npair, ns, geometry))
        if not options:
            continue
        key, _, case, cp, cs, npair, ns, geometry = min(options, key=lambda o: (o[0], o[1]))
        edit_invariants(case, cs, ns)
        results.append({"case": case, "P": case["sequence"], "C": cs, "N": ns,
                        "C_swap_offsets": list(cp), "N_swap_offsets": list(npair),
                        "geometry": list(geometry), "selection_sha256": key})
        if len(results) == max_components:
            break
    return results


def log_readout(logits20):
    values = np.asarray(logits20, dtype=np.float64)
    a = AA.index("A")
    rest = np.delete(values, a, axis=-1)
    top = np.max(rest, axis=-1)
    return values[..., a] - (top + np.log(np.exp(rest - top[..., None]).sum(axis=-1)))


def float32_ulp(value):
    return np.abs(np.spacing(np.asarray(value, dtype=np.float32))).astype(np.float64)


def replace_native_scalar(output, token_q, unit, value):
    """The shared live/synthetic primitive changes only one scalar in one row."""
    if len(output.shape) != 3 or output.shape[0] != 1:
        raise ValueError("Native scalar replacement requires batch1 rank-three input")
    if not 0 <= token_q < output.shape[1] or not 0 <= unit < output.shape[2] or not math.isfinite(value):
        raise ValueError("Invalid native scalar replacement coordinates/value")
    edited = output.copy() if isinstance(output, np.ndarray) else output.clone()
    edited[0, token_q, unit] = value
    changed = edited != output
    count = int(np.count_nonzero(changed)) if isinstance(output, np.ndarray) else int(changed.sum())
    expected = int(bool(output[0, token_q, unit] != edited[0, token_q, unit]))
    if count != expected:
        raise RuntimeError("Native patch changed more than its requested scalar")
    return edited


def case_with_sequence(edit, condition):
    case = dict(edit["case"])
    case["sequence"] = edit[condition]
    case["case_id"] = case["case_id"] + ":" + condition
    return case


def choose_control(deltas, norms, unit, eligible):
    doses = np.abs(np.asarray(deltas, dtype=np.float64)) * norms[None, :]
    positive = doses[:, unit][doses[:, unit] > 0]
    if not len(positive):
        return {"status": "NA", "reason": "no_positive_target_dose", "unit": None}
    target_median = float(np.median(positive))
    candidates = []
    for k in np.flatnonzero(eligible):
        if k == unit:
            continue
        candidate_positive = doses[:, k][doses[:, k] > 0]
        if len(candidate_positive):
            score = abs(math.log(float(np.median(candidate_positive)) / target_median))
            candidates.append((score, int(k)))
    if not candidates:
        return {"status": "NA", "reason": "no_positive_control_candidate", "unit": None}
    _, control = min(candidates)
    cp = doses[:, control][doses[:, control] > 0]
    lo, hi = max(float(positive.min()), float(cp.min())), min(float(positive.max()), float(cp.max()))
    result = {"status": "SELECTED", "unit": control, "target_positive_median": target_median,
              "control_positive_median": float(np.median(cp)),
              "target_positive_range": [float(positive.min()), float(positive.max())],
              "control_positive_range": [float(cp.min()), float(cp.max())]}
    if lo > hi:
        result.update(common_support_status="NO_INTERSECTION", common_support=None, bin_edges=[])
    else:
        inside = np.concatenate([positive[(positive >= lo) & (positive <= hi)], cp[(cp >= lo) & (cp <= hi)]])
        edges = np.unique(np.quantile(inside, [0, 0.25, 0.5, 0.75, 1])).tolist()
        result.update(common_support_status="AVAILABLE", common_support=[lo, hi], bin_edges=edges)
    return result


def dose_comparability(target, control, rule):
    if control is None or rule.get("common_support_status") != "AVAILABLE":
        return {"dose_stratum_comparable": False, "reason": "no_common_support_or_control"}
    lo, hi = rule["common_support"]
    inside = lo <= target <= hi and lo <= control <= hi
    edges = rule["bin_edges"]
    target_bin = int(np.searchsorted(edges[1:-1], target, side="right")) if inside else None
    control_bin = int(np.searchsorted(edges[1:-1], control, side="right")) if inside else None
    return {"dose_stratum_comparable": bool(inside and target_bin == control_bin),
            "both_within_common_support": bool(inside), "target_bin": target_bin, "control_bin": control_bin}


class Ledger:
    """Each attempted batch1 forward is durably charged before inference."""
    def __init__(self, out, stage):
        self.path = Path(out) / "FORWARD_LEDGER.json"
        self.stage = stage
        self.state = readjson(self.path) if self.path.exists() else {
            "schema": 1, "sequence_forwards": 0, "model_cpu_seconds": 0.0,
            "model_forward_wall_seconds": 0.0,
            "events": [], "created_utc": now()}
        if any(e["status"] == "RUNNING" for e in self.state["events"]):
            raise RuntimeError("Unresolved forward event; supervisor inspection required")

    def begin(self, purpose, case_id):
        if self.state["sequence_forwards"] + 1 > min(MAX_FORWARDS, MAX_PLANNED_FORWARDS):
            raise RuntimeError("760 planned sequence-forward budget exhausted; no retries authorized")
        prior = [e.get("model_cpu_seconds", 0.0) for e in self.state["events"]]
        next_reserve = 2 * max(prior, default=0.0)
        if self.state["model_cpu_seconds"] + next_reserve >= MAX_MODEL_SECONDS:
            raise RuntimeError("3600-second model budget exhausted")
        event = {"index": self.state["sequence_forwards"] + 1,
                 "stage": self.stage, "purpose": purpose, "case_id": case_id,
                 "batch_size": 1, "status": "RUNNING", "started_utc": now()}
        self.state["sequence_forwards"] += 1
        self.state["events"].append(event)
        writejson(self.path, self.state)
        return event

    def end(self, event, elapsed_cpu, elapsed_wall, error=None):
        event.update(status="FAILED" if error else "COMPLETE", model_cpu_seconds=float(elapsed_cpu),
                     model_forward_wall_seconds=float(elapsed_wall),
                     finished_utc=now())
        if error is not None:
            event["error_type"] = type(error).__name__
        self.state["model_cpu_seconds"] += float(elapsed_cpu)
        self.state["model_forward_wall_seconds"] += float(elapsed_wall)
        writejson(self.path, self.state)
        if self.state["model_cpu_seconds"] > MAX_MODEL_SECONDS:
            raise RuntimeError("Model budget crossed in the last charged forward; stop")


def require_initial_lock(args):
    """The supervisor creates this lock after freezing the protocol/code."""
    path = Path(args.out) / "INITIAL_LOCK.json"
    if not path.exists():
        raise RuntimeError("No INITIAL_LOCK; all model loading and forwards are prohibited")
    lock = readjson(path)
    if lock.get("status") != "FROZEN" or lock.get("D_only_authorized") is not True:
        raise RuntimeError("Frozen explicit D-only authorization is required")
    for key in ["protocol", "script", "data"]:
        if filehash(lock[f"{key}_path"]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"Analysis binding drift: {key}")
    if "prompt_path" in lock and filehash(lock["prompt_path"]) != lock["prompt_sha256"]:
        raise RuntimeError("Hypothesis-prompt binding drift")
    if Path(lock["script_path"]).resolve() != Path(__file__).resolve():
        raise RuntimeError("INITIAL_LOCK names a different implementation")
    if Path(lock["data_path"]).resolve() != Path(args.data).resolve():
        raise RuntimeError("INITIAL_LOCK names a different data input")
    if lock["data_sha256"] != DATA_SHA256:
        raise RuntimeError("Unexpected accepted data SHA")
    if lock.get("selected_manifest_sha256") != MANIFEST_SHA256:
        raise RuntimeError("Unexpected accepted selected-manifest SHA")
    identity = lock["model_identity"]
    if Path(identity["path"]).resolve() != Path(args.model).resolve():
        raise RuntimeError("INITIAL_LOCK names a different local model")
    for name, expected in identity["files_sha256"].items():
        if filehash(Path(args.model) / name) != expected:
            raise RuntimeError(f"Model/tokenizer binding drift: {name}")
    return lock


def check_bindings(bindings, required):
    if set(required) - set(bindings):
        raise RuntimeError(f"Missing lock bindings: {sorted(set(required) - set(bindings))}")
    for key, binding in bindings.items():
        if filehash(binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"Stage binding drift: {key}")


def require_hypothesis_lock(args):
    initial = require_initial_lock(args)
    out = Path(args.out)
    lock = readjson(out / "HYPOTHESIS_LOCK.json")
    if lock.get("status") != "FROZEN" or lock.get("allow_D_development_and_C_calibration") is not True:
        raise RuntimeError("Explicit HYPOTHESIS_LOCK development/calibration release is required")
    if lock["initial_lock_sha256"] != filehash(out / "INITIAL_LOCK.json"):
        raise RuntimeError("Initial-lock binding drift in H lock")
    required = ["hypothesis", "D_selected", "D_native", "D_fits", "D_cases"]
    check_bindings(lock["bindings"], required)
    return initial, lock


def require_confirmation_lock(args):
    initial, hypothesis = require_hypothesis_lock(args)
    out = Path(args.out)
    lock = readjson(out / "CONFIRMATION_LOCK.json")
    if lock.get("status") != "FROZEN" or lock.get("T_authorized") is not True:
        raise RuntimeError("Explicit CONFIRMATION_LOCK T release is required")
    if lock["initial_lock_sha256"] != filehash(out / "INITIAL_LOCK.json"):
        raise RuntimeError("Initial-lock binding drift in confirmation lock")
    required = ["hypothesis_lock", "D_selected", "D_fits", "H_fits", "H_development",
                "D_development", "D_edit_native", "D_edit_cases", "C_calibration", "D_patch_diagnostics", "T_cases"]
    check_bindings(lock["bindings"], required)
    calibration = readjson(out / "C_CALIBRATION.json")
    if calibration["status"] != "CALIBRATION_PASS":
        raise RuntimeError("Calibration engineering failure prohibits T")
    return initial, hypothesis, lock


class NativeRunner:
    """Read-only local ESM2, block[2] intermediate post-GELU, batch1 only."""
    def __init__(self, model_path, ledger, release):
        if not release or release.get("status") != "FROZEN":
            raise RuntimeError("A verified stage release is required before model loading")
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        self.torch, self.ledger = torch, ledger
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = EsmForMaskedLM.from_pretrained(model_path, local_files_only=True).cpu().float().eval()
        c = self.model.config
        if (c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.hidden_act) != (320, DIM, 6, "gelu"):
            raise RuntimeError("Unexpected ESM2 8M architecture")
        self.block = self.model.esm.encoder.layer[2]
        self.aa_ids = self.tokenizer.convert_tokens_to_ids(list(AA))
        if len(set(self.aa_ids)) != 20:
            raise RuntimeError("AA tokenizer map is not one-to-one")

    def column_norms(self):
        return self.block.output.dense.weight.detach().float().norm(dim=0).cpu().numpy()

    def w2_column(self, unit):
        return self.block.output.dense.weight[:, unit].detach().float().cpu().numpy()

    def run(self, case, purpose, patch=None):
        if isinstance(case, (list, tuple)):
            raise ValueError("Runner accepts exactly one case; batch size is fixed at 1")
        sequence, q = case["sequence"], int(case["q"])
        tokens = self.tokenizer([sequence], padding=False, return_tensors="pt", truncation=False)
        expected = self.tokenizer.convert_tokens_to_ids(list(sequence))
        if int(tokens["attention_mask"].sum()) != len(sequence) + 2:
            raise RuntimeError("Tokenizer length/BOS alignment mismatch")
        if tokens["input_ids"][0, 1:len(sequence) + 1].tolist() != expected:
            raise RuntimeError("Tokenizer residue alignment mismatch")
        if not 0 <= q < len(sequence):
            raise ValueError("Mask q outside sequence")
        token_q = q + 1
        tokens["input_ids"][0, token_q] = self.tokenizer.mask_token_id
        if int((tokens["input_ids"] == self.tokenizer.mask_token_id).sum()) != 1:
            raise RuntimeError("Exactly one center mask is required")
        if patch is not None:
            unit, value = int(patch["unit"]), float(patch["value"])
            if not 0 <= unit < DIM or not math.isfinite(value):
                raise ValueError("Invalid native scalar patch")
        captured = {}

        def intermediate_hook(module, inputs, output):
            if output.shape != (1, len(sequence) + 2, DIM):
                raise RuntimeError("Intermediate shape mismatch")
            captured["z"] = output[0, token_q].detach().clone()
            if patch is None:
                return None
            return replace_native_scalar(output, token_q, unit, value)

        def dense_hook(module, inputs, output):
            if output.shape != (1, len(sequence) + 2, HIDDEN):
                raise RuntimeError("fc2 shape mismatch")
            captured["fc2"] = output[0, token_q].detach().clone()

        handles = [self.block.intermediate.register_forward_hook(intermediate_hook),
                   self.block.output.dense.register_forward_hook(dense_hook)]
        event = self.ledger.begin(purpose, case.get("case_id", f"{case['id']}:{q}"))
        error, elapsed_cpu, elapsed_wall = None, 0.0, 0.0
        try:
            with self.torch.inference_mode():
                started_wall, started_cpu = time.perf_counter(), time.process_time()
                try:
                    output = self.model(**tokens)
                finally:
                    elapsed_cpu = time.process_time() - started_cpu
                    elapsed_wall = time.perf_counter() - started_wall
                logits = output.logits[0, token_q]
                result = {"z": captured["z"].float().cpu().numpy(),
                          "fc2": captured["fc2"].float().cpu().numpy(),
                          "logits": logits.float().cpu().numpy(),
                          "logits20": logits[self.aa_ids].float().cpu().numpy()}
                if any(not np.isfinite(a).all() for a in result.values()):
                    raise RuntimeError("Nonfinite model output")
                return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            for handle in handles:
                handle.remove()
            self.ledger.end(event, elapsed_cpu, elapsed_wall, error)


def natural_extract(runner, cases, purpose):
    values = {k: [] for k in ["z", "logits20"]}
    for i, case in enumerate(cases):
        row = runner.run(case, purpose)
        for key in values:
            values[key].append(row[key])
        if (i + 1) % 32 == 0 or i + 1 == len(cases):
            print(json.dumps({"stage": purpose, "completed_sites": i + 1,
                              "total_sites": len(cases),
                              "cumulative_forwards": runner.ledger.state["sequence_forwards"]}), flush=True)
    return {key: np.stack(rows) for key, rows in values.items()}


def command_discover(args):
    out = Path(args.out)
    if any((out / name).exists() for name in ["D_STARTED.json", "D_NATIVE.npz", "D_SELECTED.json"]):
        raise RuntimeError("D artifacts already exist; refusing repeated natural forwards")
    release = require_initial_lock(args)
    records = load_records(args.data)
    cases_fit, cases_select = flatten(records, "D_fit"), flatten(records, "D_select")
    cases = cases_fit + cases_select
    writejson(out / "D_STARTED.json", {"created_utc": now(), "expected_forwards": 192,
              "initial_lock_sha256": filehash(out / "INITIAL_LOCK.json"),
              "allowed_splits": ["D_fit", "D_select"]}, exclusive=True)
    ledger = Ledger(out, "D")
    runner = NativeRunner(args.model, ledger, release)
    natural = natural_extract(runner, cases, "D_natural")
    write_npz(out / "D_NATIVE.npz", **natural,
              case_ids=np.array([c["case_id"] for c in cases]),
              split=np.array([c["split"] for c in cases]))
    nfit = len(cases_fit)
    xfit, xselect = features(cases_fit), features(cases_select)
    xscaler = fit_scaler(xfit)
    xf, xs = transform(xfit, xscaler), transform(xselect, xscaler)
    zfit, zselect = natural["z"][:nfit].astype(np.float64), natural["z"][nfit:].astype(np.float64)
    zscaler = fit_scaler(zfit)
    zf, zs = transform(zfit, zscaler), transform(zselect, zscaler)
    yfit, yselect = np.array([c["y"] for c in cases_fit]), np.array([c["y"] for c in cases_select])
    base = ridge_fit(xf, yfit)
    base_prediction = ridge_predict(base, xs)
    base_loss = brier(yselect, base_prediction)
    allmodels = all_unit_ridges(xf, zf, yfit, xs, zs)
    component_gains = component_means(base_loss[:, None] - brier(yselect[:, None], allmodels["predictions"]),
                                      cases_select)
    scores = component_gains.mean(0)
    raw_z_sd = zfit.std(0, ddof=0)
    eligible = raw_z_sd > 1e-12
    if not eligible.any():
        raise RuntimeError("No numerically nonconstant native candidate")
    ranking = sorted(np.flatnonzero(eligible).tolist(), key=lambda j: (-float(scores[j]), j))
    unit = ranking[0]
    label_selected = ridge_fit(np.column_stack([xf, zf[:, unit]]), yfit)
    fullrep = ridge_fit(np.column_stack([xf, zf]), yfit)
    full_prediction = ridge_predict(fullrep, np.column_stack([xs, zs]))
    write_npz(out / "D_FITS.npz", x_mean=xscaler["mean"], x_scale=xscaler["scale"],
              z_mean=zscaler["mean"], z_scale=zscaler["scale"], raw_z_sd=raw_z_sd,
              unit_component_brier_improvement=component_gains,
              unit_mean_brier_improvement=scores, eligible_units=eligible,
              label_base_coef=base["coef"], label_base_intercept=base["intercept"],
              label_selected_coef=label_selected["coef"], label_selected_intercept=label_selected["intercept"],
              fullrep_coef=fullrep["coef"], fullrep_intercept=fullrep["intercept"])
    selected = {"status": "D_EXTRACTED", "created_utc": now(),
                "initial_lock_sha256": filehash(out / "INITIAL_LOCK.json"),
                "selected_unit": int(unit), "candidate_dimension": DIM,
                "eligible_units": int(eligible.sum()), "candidate_sd_epsilon": 1e-12,
                "selection": "largest D_select component-mean clipped-Brier improvement; unit-index tie-break",
                "selected_D_fit_mean": float(zscaler["mean"][unit]),
                "selected_D_fit_sd": float(raw_z_sd[unit]),
                "D_select_base_Brier": float(base_loss.mean()),
                "D_select_selected_Brier_improvement": float(scores[unit]),
                "D_select_fullrep_Brier_improvement": float((base_loss - brier(yselect, full_prediction)).mean()),
                "ranking": [{"unit": j, "mean_Brier_improvement": float(scores[j])} for j in ranking],
                "excluded_constant_units": np.flatnonzero(~eligible).tolist(),
                "label_base_model": jsonable(base), "label_selected_model": jsonable(label_selected),
                "model_forwards_this_stage": 192,
                "model_cpu_seconds_cumulative": ledger.state["model_cpu_seconds"],
                "model_forward_wall_seconds_cumulative": ledger.state["model_forward_wall_seconds"],
                "next_stage_requires": "fresh single hypothesis and HYPOTHESIS_LOCK; no automatic C/T"}
    writejson(out / "D_SELECTED.json", selected, exclusive=True)
    writejson(out / "D_CASES.json", {"cases": cases}, exclusive=True)
    z_baseline = ridge_fit(xf, zf[:, unit])
    all_x = np.vstack([xf, xs])
    z_control_prediction = ridge_predict(z_baseline, all_x)
    all_z_std = np.concatenate([zf[:, unit], zs[:, unit]])
    packet_rows = []
    for i, case in enumerate(cases):
        q, sequence = case["q"], case["sequence"]
        packet_rows.append({**case, "length": len(sequence),
                            "masked_window_r7": sequence[q - 7:q] + "[MASK]" + sequence[q + 1:q + 8],
                            "z_raw": float(natural["z"][i, unit]), "z_standardized": float(all_z_std[i]),
                            "X_prediction_z_standardized": float(z_control_prediction[i]),
                            "X_residual_z_standardized": float(all_z_std[i] - z_control_prediction[i])})
    writejson(out / "H_PACKET.json", {"status": "D_ONLY_PACKET", "selected_unit": int(unit),
              "protocol_path": release["protocol_path"], "protocol_sha256": release["protocol_sha256"],
              "D_fit_n": 128, "D_select_n": 64, "rows": packet_rows,
              "hypothesis_schema": {"status": ["valid", "unknown"], "operator": ["all", "any"],
                  "clauses": "one or two distinct nonzero offsets -7..7 and allowed aa_set membership",
                  "explanation_zh": "Chinese explanation", "development_candidate_count": "integer 1..32 (unknown may be 0)"},
              "candidate_budget": 32, "model_forwards_for_explanation": 0,
              "D_fits_path": str((out / "D_FITS.npz").resolve()),
              "forbidden": ["C/T activations", "PLM calls", "prompt retry", "new hypothesis after response"]}, exclusive=True)
    print(json.dumps({k: v for k, v in selected.items() if k not in ["ranking", "label_base_model", "label_selected_model"]}))


def command_develop(args):
    out = Path(args.out)
    if any((out / name).exists() for name in ["DEVELOP_STARTED.json", "D_DEVELOPMENT.json", "D_EDIT_NATIVE.npz"]):
        raise RuntimeError("Development artifacts exist; refusing repeated forwards")
    initial, hlock = require_hypothesis_lock(args)
    selected = readjson(out / "D_SELECTED.json")
    cases = readjson(out / "D_CASES.json")["cases"]
    cases_fit, cases_select = cases[:128], cases[128:]
    unit = selected["selected_unit"]
    native = np.load(out / "D_NATIVE.npz")
    fitted = np.load(out / "D_FITS.npz")
    z = (native["z"][:, unit] - fitted["z_mean"][unit]) / fitted["z_scale"][unit]
    x = (features(cases) - fitted["x_mean"]) / fitted["x_scale"]
    hypothesis = validate_hypothesis(readjson(hlock["bindings"]["hypothesis"]["path"]))
    h = np.zeros(len(cases), dtype=np.float64)
    if hypothesis["status"] == "VALID":
        h = np.array([h_value(hypothesis, c["sequence"], c["q"]) for c in cases])
        if h[:128].std(ddof=0) == 0:
            hypothesis.update(status="NA", reason="D_fit_H_constant")
    h_mean, h_scale = float(h[:128].mean()), float(h[:128].std(ddof=0))
    if h_scale == 0:
        h_scale = 1.0
    hs = (h - h_mean) / h_scale
    base = ridge_fit(x[:128], z[:128])
    augmented = ridge_fit(np.column_stack([x[:128], hs[:128]]), z[:128]) if hypothesis["status"] == "VALID" else None
    s_h = int(np.sign(augmented["coef"][-1])) if augmented is not None and augmented["coef"][-1] != 0 else None
    write_npz(out / "H_FITS.npz", base_coef=base["coef"], base_intercept=base["intercept"],
              augmented_coef=np.array([]) if augmented is None else augmented["coef"],
              augmented_intercept=np.array(0.0) if augmented is None else augmented["intercept"],
              h_mean=np.array(h_mean), h_scale=np.array(h_scale), D_h=h)
    hdevelopment = {"status": hypothesis["status"], "hypothesis": hypothesis, "s_H": s_h,
                    "h_mean_D_fit": h_mean, "h_scale_D_fit": h_scale,
                    "X_z_model": jsonable(base), "X_H_z_model": jsonable(augmented),
                    "D_select_MSE_improvement": None if augmented is None else float(np.mean(
                        (z[128:] - ridge_predict(base, x[128:])) ** 2 -
                        (z[128:] - ridge_predict(augmented, np.column_stack([x[128:], hs[128:]]))) ** 2))}
    writejson(out / "H_DEVELOPMENT.json", hdevelopment, exclusive=True)
    edits = compile_edits(cases_select, hypothesis, 16)
    writejson(out / "D_EDIT_CASES.json", {"split": "D_select", "edits": edits}, exclusive=True)
    writejson(out / "DEVELOP_STARTED.json", {"created_utc": now(), "expected_baseline_forwards": 2 * len(edits),
              "hypothesis_lock_sha256": filehash(out / "HYPOTHESIS_LOCK.json")}, exclusive=True)
    if edits:
        ledger = Ledger(out, "D_edit_baselines")
        runner = NativeRunner(args.model, ledger, initial)
        edit_z, edit_logits20 = [], []
        for edit in edits:
            rows = [runner.run(case_with_sequence(edit, cond), f"D_edit_{cond}_baseline") for cond in ["C", "N"]]
            edit_z.append(np.stack([row["z"] for row in rows]))
            edit_logits20.append(np.stack([row["logits20"] for row in rows]))
        edit_z, edit_logits20 = np.stack(edit_z), np.stack(edit_logits20)
        lookup = {c["case_id"]: i for i, c in enumerate(cases)}
        p_indices = np.array([lookup[edit["case"]["case_id"]] for edit in edits])
        p_z = native["z"][p_indices]
        p_logits20 = native["logits20"][p_indices]
        norms = runner.column_norms().astype(np.float64)
        control = choose_control(p_z - edit_z[:, 0], norms, unit, fitted["eligible_units"])
    else:
        edit_z = np.empty((0, 2, DIM), dtype=np.float32)
        edit_logits20 = np.empty((0, 2, 20), dtype=np.float32)
        p_z = np.empty((0, DIM), dtype=np.float32)
        p_logits20 = np.empty((0, 20), dtype=np.float32)
        norms = np.zeros(DIM, dtype=np.float64)
        control = {"status": "NA", "reason": "no_compilable_D_edits", "unit": None}
    write_npz(out / "D_EDIT_NATIVE.npz", z=edit_z, logits20=edit_logits20,
              p_z=p_z, p_logits20=p_logits20, w2_norms=norms)
    development = {"status": "D_DEVELOPED", "created_utc": now(), "selected_unit": unit,
                   "H_status": hypothesis["status"], "s_H": s_h, "D_edit_components": len(edits),
                   "control": control, "D_edit_baseline_forwards": len(edits) * 2,
                   "next_stage": "calibrate; C natural and fixed calibration, then D patches"}
    writejson(out / "D_DEVELOPMENT.json", development, exclusive=True)
    print(json.dumps(development))


def patch_diagnostics(runner, edits, cached, unit, control_rule, s_h, z_sd,
                      epsilon_l, epsilon_z, purpose, s_use=None):
    rows = []
    control = control_rule.get("unit")
    norms = cached["w2_norms"]
    for i, edit in enumerate(edits):
        p_z, c_z, n_z = cached["p_z"][i], cached["z"][i, 0], cached["z"][i, 1]
        p_l = float(log_readout(cached["p_logits20"][i]))
        c_l = float(log_readout(cached["logits20"][i, 0]))
        n_l = float(log_readout(cached["logits20"][i, 1]))
        p, c = case_with_sequence(edit, "P"), case_with_sequence(edit, "C")
        p_sham = runner.run(p, f"{purpose}_P_self_sham", {"unit": unit, "value": float(p_z[unit])})
        c_sham = runner.run(c, f"{purpose}_C_self_sham", {"unit": unit, "value": float(c_z[unit])})
        c_target = runner.run(c, f"{purpose}_C_from_P_target", {"unit": unit, "value": float(p_z[unit])})
        p_target = runner.run(p, f"{purpose}_P_from_C_target", {"unit": unit, "value": float(c_z[unit])})
        delta = float(p_z[unit]) - float(c_z[unit])
        r = float(log_readout(c_target["logits20"])) - c_l
        d = p_l - float(log_readout(p_target["logits20"]))
        p_sham_l = float(log_readout(p_sham["logits20"])) - p_l
        c_sham_l = float(log_readout(c_sham["logits20"])) - c_l
        target_dose = abs(delta) * float(norms[unit])
        control_results = None
        control_dose = None
        if control is not None:
            c_control = runner.run(c, f"{purpose}_C_from_P_control", {"unit": control, "value": float(p_z[control])})
            p_control = runner.run(p, f"{purpose}_P_from_C_control", {"unit": control, "value": float(c_z[control])})
            cd = float(p_z[control]) - float(c_z[control])
            control_dose = abs(cd) * float(norms[control])
            control_results = {"unit": control, "delta_z": cd, "dose": control_dose,
                               "R": float(log_readout(c_control["logits20"])) - c_l,
                               "D": p_l - float(log_readout(p_control["logits20"]))}
        row = {"case_id": edit["case"]["case_id"], "component": edit["case"]["component"],
               "q": edit["case"]["q"], "selected_unit": unit,
               "zP": float(p_z[unit]), "zC": float(c_z[unit]), "zN": float(n_z[unit]),
               "a_PminusC": None if s_h is None else s_h * delta / z_sd,
               "a_NminusC": None if s_h is None else s_h * (float(n_z[unit]) - float(c_z[unit])) / z_sd,
               "z_PminusN_standardized": (float(p_z[unit]) - float(n_z[unit])) / z_sd,
               "z_PminusN_raw": float(p_z[unit]) - float(n_z[unit]),
               "L_P": p_l, "L_C": c_l, "L_N": n_l, "L_PminusN": p_l - n_l,
               "delta_z": delta, "dose": target_dose, "R": r, "D": d,
               "P_sham_L": p_sham_l, "C_sham_L": c_sham_l,
               "R_above_epsilon_L": bool(abs(r) > epsilon_l),
               "D_above_epsilon_L": bool(abs(d) > epsilon_l),
               "R_above_own_sham": bool(abs(r) > abs(c_sham_l)),
               "D_above_own_sham": bool(abs(d) > abs(p_sham_l)),
               "delta_above_epsilon_z": bool(abs(delta) > epsilon_z),
               "control": control_results,
               **dose_comparability(target_dose, control_dose, control_rule),
               "s_use": s_use, "R_expected_direction": None if s_use is None else bool(r * s_use > 0),
               "D_expected_direction": None if s_use is None else bool(d * s_use > 0)}
        rows.append(row)
    return rows


def downstream_direction(rows, epsilon_l, epsilon_z):
    directions = {}
    for key in ["R", "D"]:
        values = [row[key] * row["delta_z"] for row in rows
                  if abs(row["delta_z"]) > epsilon_z and abs(row[key]) > epsilon_l]
        positive, negative = sum(v > 0 for v in values), sum(v < 0 for v in values)
        sign = None
        if len(values) >= 4:
            if positive / len(values) >= 0.75:
                sign = 1
            elif negative / len(values) >= 0.75:
                sign = -1
        directions[key] = {"discernible_n": len(values), "positive_product_n": positive,
                           "negative_product_n": negative, "direction": sign}
    sign = directions["R"]["direction"]
    return {"s_g": sign if sign is not None and sign == directions["D"]["direction"] else None,
            "directions": directions, "rule": "both directions n>=4, >=75% same product sign, signs agree"}


def command_calibrate(args):
    out = Path(args.out)
    if any((out / name).exists() for name in ["CALIBRATION_STARTED.json", "C_NATIVE.npz", "C_CALIBRATION.json"]):
        raise RuntimeError("Calibration artifacts exist; refusing repeated forwards")
    initial, _ = require_hypothesis_lock(args)
    development = readjson(out / "D_DEVELOPMENT.json")
    hdevelopment = readjson(out / "H_DEVELOPMENT.json")
    selected = readjson(out / "D_SELECTED.json")
    fitted = np.load(out / "D_FITS.npz")
    dnative = np.load(out / "D_NATIVE.npz")
    unit = selected["selected_unit"]
    dmin, dmax = float(dnative["z"][:128, unit].min()), float(dnative["z"][:128, unit].max())
    cases = flatten(load_records(args.data), "C_cal")
    writejson(out / "CALIBRATION_STARTED.json", {"created_utc": now(),
              "natural_forwards": 32, "calibration_extra_forwards": 24,
              "D_development_sha256": filehash(out / "D_DEVELOPMENT.json")}, exclusive=True)
    ledger = Ledger(out, "C_cal_and_D_patches")
    runner = NativeRunner(args.model, ledger, initial)
    cvalues = {key: [] for key in ["z", "logits20", "logits", "fc2"]}
    for case in cases:
        result = runner.run(case, "C_natural")
        for key in cvalues:
            cvalues[key].append(result[key])
    cvalues = {key: np.stack(rows) for key, rows in cvalues.items()}
    write_npz(out / "C_NATIVE.npz", **cvalues, case_ids=np.array([c["case_id"] for c in cases]))
    check_rows, epsilon_l, epsilon_z = [], 0.0, 0.0
    all_engineering_pass = True
    cal_arrays = {key: [] for key in ["repeat_logits", "sham_logits", "clamp_logits", "clamp_fc2_actual_delta", "clamp_fc2_expected_delta"]}
    for ordinal in range(8):
        index = 2 * ordinal + (0 if ordinal % 2 == 0 else 1)
        case = cases[index]
        base = {key: values[index] for key, values in cvalues.items()}
        repeat = runner.run(case, "C_repeat_batch1")
        sham = runner.run(case, "C_self_and_zero_delta", {"unit": unit, "value": float(base["z"][unit])})
        receiver = float(base["z"][unit])
        clamp_value = min([dmin, dmax], key=lambda v: (-abs(v - receiver), v))
        clamp = runner.run(case, "C_natural_support_nonzero_clamp", {"unit": unit, "value": clamp_value})
        delta = clamp_value - float(clamp["z"][unit])
        actual = clamp["fc2"] - base["fc2"]
        expected = runner.w2_column(unit) * delta
        torch = runner.torch
        fc2_pass = bool(torch.allclose(torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-5, rtol=1e-5))
        repeat_pass = bool(torch.allclose(torch.from_numpy(repeat["logits"]), torch.from_numpy(base["logits"]), atol=1e-5, rtol=1e-5))
        sham_pass = bool(torch.allclose(torch.from_numpy(sham["logits"]), torch.from_numpy(base["logits"]), atol=1e-5, rtol=1e-5))
        passed = fc2_pass and repeat_pass and sham_pass and delta != 0
        all_engineering_pass &= passed
        base_l = float(log_readout(base["logits20"]))
        repeat_l_error = abs(float(log_readout(repeat["logits20"])) - base_l)
        sham_l_error = abs(float(log_readout(sham["logits20"])) - base_l)
        z_error = max(abs(float(repeat["z"][unit]) - receiver), abs(float(sham["z"][unit]) - receiver))
        logits_ulp = max(float(float32_ulp(r["logits20"]).max()) for r in [base, repeat, sham])
        z_ulp = max(float(float32_ulp(r["z"][unit])) for r in [base, repeat, sham])
        epsilon_l = max(epsilon_l, repeat_l_error, sham_l_error, 2 * logits_ulp)
        epsilon_z = max(epsilon_z, z_error, 2 * z_ulp)
        check_rows.append({"ordinal_0based": ordinal, "case_id": case["case_id"], "label": case["label"],
                           "fc2_identity_pass": fc2_pass, "repeat_logits_pass": repeat_pass,
                           "self_zero_delta_logits_pass": sham_pass, "nonzero_clamp": delta != 0,
                           "clamp_value": clamp_value, "actual_delta_z": delta,
                           "fc2_max_abs_identity_error": float(np.abs(actual - expected).max()),
                           "repeat_L_error": repeat_l_error, "self_L_error": sham_l_error,
                           "selected_z_repeat_self_error": z_error, "logits20_max_float32_ULP": logits_ulp,
                           "selected_z_max_float32_ULP": z_ulp,
                           "clamp_L_change": float(log_readout(clamp["logits20"])) - base_l})
        for key, value in [("repeat_logits", repeat["logits"]), ("sham_logits", sham["logits"]),
                           ("clamp_logits", clamp["logits"]), ("clamp_fc2_actual_delta", actual),
                           ("clamp_fc2_expected_delta", expected)]:
            cal_arrays[key].append(value)
    write_npz(out / "C_ENGINEERING_ARRAYS.npz", **{k: np.stack(v) for k, v in cal_arrays.items()})
    calibration = {"status": "CALIBRATION_PASS" if all_engineering_pass else "CALIBRATION_ENGINEERING_FAIL",
                   "created_utc": now(), "selected_unit": unit, "epsilon_L": epsilon_l, "epsilon_z": epsilon_z,
                   "epsilon_interpretation": "observed/float32 resolution reference; not a strict error bound or scientific minimum effect",
                   "engineering_atol": 1e-5, "engineering_rtol": 1e-5,
                   "natural_cases": 32, "calibration_cases": 8, "calibration_extra_forwards": 24,
                   "checks": check_rows}
    writejson(out / "C_CALIBRATION.json", calibration, exclusive=True)
    if not all_engineering_pass:
        print(json.dumps(calibration))
        return
    edits = readjson(out / "D_EDIT_CASES.json")["edits"]
    cached = np.load(out / "D_EDIT_NATIVE.npz")
    rows = patch_diagnostics(runner, edits, cached, unit, development["control"], hdevelopment["s_H"],
                             selected["selected_D_fit_sd"], epsilon_l, epsilon_z, "D_edit")
    direction = downstream_direction(rows, epsilon_l, epsilon_z)
    s_h, s_g = hdevelopment["s_H"], direction["s_g"]
    direction["s_use"] = s_h * s_g if s_h is not None and s_g is not None else None
    result = {"status": "D_PATCHES_COMPLETE", "selected_unit": unit, "control": development["control"],
              "epsilon_L": epsilon_l, "epsilon_z": epsilon_z, "s_H": s_h,
              **direction, "rows": rows, "model_forwards_cumulative": ledger.state["sequence_forwards"],
              "model_cpu_seconds_cumulative": ledger.state["model_cpu_seconds"],
              "model_forward_wall_seconds_cumulative": ledger.state["model_forward_wall_seconds"]}
    writejson(out / "D_PATCH_DIAGNOSTICS.json", result, exclusive=True)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


def artifact_binding(path):
    return {"path": str(Path(path).resolve()), "sha256": filehash(path)}


def command_compile_confirmation(args):
    out = Path(args.out)
    require_hypothesis_lock(args)
    if any((out / name).exists() for name in ["T_STARTED.json", "T_NATIVE.npz", "T_CASES.json"]):
        raise RuntimeError("T artifacts or precompiled cases exist; cannot create another confirmation version")
    calibration = readjson(out / "C_CALIBRATION.json")
    if calibration["status"] != "CALIBRATION_PASS":
        raise RuntimeError("CALIBRATION_ENGINEERING_FAIL prohibits confirmation compilation")
    hdevelopment = readjson(out / "H_DEVELOPMENT.json")
    development = readjson(out / "D_DEVELOPMENT.json")
    patches = readjson(out / "D_PATCH_DIAGNOSTICS.json")
    natural = flatten(load_records(args.data), "T_confirm")
    edits = compile_edits(natural, hdevelopment["hypothesis"], 32)
    per_edit = 8 if development["control"].get("unit") is not None else 6
    ledger = readjson(out / "FORWARD_LEDGER.json")
    if any(e["status"] != "COMPLETE" for e in ledger["events"]):
        raise RuntimeError("An incomplete or failed model attempt requires supervisor inspection")
    t_forwards = len(natural) + per_edit * len(edits)
    if ledger["sequence_forwards"] + t_forwards > MAX_PLANNED_FORWARDS:
        raise RuntimeError("Compiled plan exceeds the authorized 760-forward plan")
    payload = {"status": "T_INPUTS_COMPILED_NO_MODEL", "created_utc": now(),
               "natural_cases": natural, "edits": edits, "natural_forwards": len(natural),
               "edit_forwards": per_edit * len(edits), "planned_T_forwards": t_forwards,
               "planned_total_forwards": ledger["sequence_forwards"] + t_forwards,
               "selected_unit": development["selected_unit"], "control": development["control"],
               "s_H": hdevelopment["s_H"], "s_g": patches["s_g"], "s_use": patches["s_use"],
               "epsilon_L": calibration["epsilon_L"], "epsilon_z": calibration["epsilon_z"],
               "component_count": 64, "edit_component_count": len(edits)}
    writejson(out / "T_CASES.json", payload, exclusive=True)
    names = {"hypothesis_lock": "HYPOTHESIS_LOCK.json", "D_selected": "D_SELECTED.json",
             "D_fits": "D_FITS.npz", "H_fits": "H_FITS.npz", "H_development": "H_DEVELOPMENT.json",
             "D_development": "D_DEVELOPMENT.json", "D_edit_native": "D_EDIT_NATIVE.npz",
             "D_edit_cases": "D_EDIT_CASES.json", "C_calibration": "C_CALIBRATION.json",
             "D_patch_diagnostics": "D_PATCH_DIAGNOSTICS.json", "T_cases": "T_CASES.json"}
    writejson(out / "CONFIRMATION_PAYLOAD.json", {"status": "READY_FOR_SUPERVISOR_LOCK",
              "initial_lock_sha256": filehash(out / "INITIAL_LOCK.json"),
              "bindings": {key: artifact_binding(out / name) for key, name in names.items()},
              "planned_T_forwards": t_forwards, "planned_total_forwards": payload["planned_total_forwards"],
              "warning": "This payload is not authorization; root must create CONFIRMATION_LOCK with T_authorized=true."}, exclusive=True)
    print(json.dumps({k: v for k, v in payload.items() if k not in ["natural_cases", "edits", "control"]}))


def paired_effect(values, endpoint):
    if values is None:
        return {"status": "NA", "reason": "no_valid_H", "p_one_sided_t_reference": 1.0,
                "component_differences": None, "mean": None, "median": None,
                "bootstrap_mean_95_percentile_CI": None}
    from scipy.stats import t
    values = np.asarray(values, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    if values.ndim != 1 or not finite:
        return {"status": "NA", "reason": "nonfinite_or_invalid_component_effects",
                "p_one_sided_t_reference": 1.0, "component_differences": None,
                "mean": None, "median": None, "bootstrap_mean_95_percentile_CI": None}
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    rng = np.random.default_rng(SEED + endpoint)
    boot = values[rng.integers(0, n, size=(5000, n))].mean(1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    valid_test = n > 1 and math.isfinite(sd) and sd > 0
    statistic = mean / (sd / math.sqrt(n)) if valid_test else None
    pvalue = float(t.sf(statistic, df=n - 1)) if valid_test else 1.0
    return {"status": "TESTABLE" if valid_test else "NA", "reason": None if valid_test else "zero_or_nonfinite_variance",
            "n_components": n, "mean": mean, "median": float(np.median(values)),
            "sd_sample": sd, "t_statistic": statistic, "df": n - 1,
            "p_one_sided_t_reference": pvalue, "component_differences": values.tolist(),
            "bootstrap_mean_95_percentile_CI": ci,
            "bootstrap_replicates": 5000, "bootstrap_seed": SEED + endpoint,
            "inference_scope": "approximate component paired-t reference; bootstrap interval is estimation only, not simultaneous"}


def holm_two(pvalues):
    if len(pvalues) != 2:
        raise ValueError("The Holm denominator is fixed at exactly two primary endpoints")
    values = [float(p) if p is not None and math.isfinite(p) else 1.0 for p in pvalues]
    order = sorted(range(2), key=lambda i: (values[i], i))
    adjusted, running = [1.0, 1.0], 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (2 - rank) * values[i]))
        adjusted[i] = running
    return {"p_adjusted": adjusted, "reject_at_alpha_0_05": [p <= 0.05 for p in adjusted],
            "denominator": 2, "alpha": 0.05}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()), "sd_population": float(values.std(ddof=0)),
            "min": float(values.min()), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)), "max": float(values.max())}


def command_confirm(args):
    out = Path(args.out)
    if any((out / name).exists() for name in ["T_STARTED.json", "T_NATIVE.npz", "T_SUMMARY.json"]):
        raise RuntimeError("T artifacts exist; refusing any repeated confirmation forwards")
    initial, _, _ = require_confirmation_lock(args)
    compiled = readjson(out / "T_CASES.json")
    natural, edits = compiled["natural_cases"], compiled["edits"]
    if natural != flatten(load_records(args.data), "T_confirm") or len(edits) > 32:
        raise RuntimeError("Frozen T cases differ from the accepted data/quantity")
    selected = readjson(out / "D_SELECTED.json")
    fitted = np.load(out / "D_FITS.npz")
    hfitted = np.load(out / "H_FITS.npz")
    hdevelopment = readjson(out / "H_DEVELOPMENT.json")
    development = readjson(out / "D_DEVELOPMENT.json")
    unit = selected["selected_unit"]
    writejson(out / "T_STARTED.json", {"created_utc": now(), "confirmation_lock_sha256": filehash(out / "CONFIRMATION_LOCK.json"),
              "expected_forwards": compiled["planned_T_forwards"]}, exclusive=True)
    ledger = Ledger(out, "T_confirm")
    runner = NativeRunner(args.model, ledger, initial)
    native = natural_extract(runner, natural, "T_natural")
    write_npz(out / "T_NATIVE.npz", **native, case_ids=np.array([c["case_id"] for c in natural]))
    x = (features(natural) - fitted["x_mean"]) / fitted["x_scale"]
    z = (native["z"].astype(np.float64) - fitted["z_mean"]) / fitted["z_scale"]
    y = np.array([c["y"] for c in natural])
    pred_base = x @ fitted["label_base_coef"] + fitted["label_base_intercept"]
    pred_native = np.column_stack([x, z[:, unit]]) @ fitted["label_selected_coef"] + fitted["label_selected_intercept"]
    pred_full = np.column_stack([x, z]) @ fitted["fullrep_coef"] + fitted["fullrep_intercept"]
    loss_base, loss_native, loss_full = [brier(y, p) for p in [pred_base, pred_native, pred_full]]
    primary1 = paired_effect(component_means(loss_base - loss_native, natural), 1)
    pred_z_base = x @ hfitted["base_coef"] + hfitted["base_intercept"]
    loss_z_base = (z[:, unit] - pred_z_base) ** 2
    h, pred_z_h, loss_z_h = np.array([]), np.array([]), np.array([])
    if hdevelopment["status"] == "VALID":
        h = np.array([h_value(hdevelopment["hypothesis"], c["sequence"], c["q"]) for c in natural])
        hs = (h - hfitted["h_mean"]) / hfitted["h_scale"]
        pred_z_h = np.column_stack([x, hs]) @ hfitted["augmented_coef"] + hfitted["augmented_intercept"]
        loss_z_h = (z[:, unit] - pred_z_h) ** 2
        primary2 = paired_effect(component_means(loss_z_base - loss_z_h, natural), 2)
    else:
        primary2 = paired_effect(None, 2)
    multiplicity = holm_two([primary1["p_one_sided_t_reference"], primary2["p_one_sided_t_reference"]])
    for i, primary in enumerate([primary1, primary2]):
        primary["Holm_adjusted_p"] = multiplicity["p_adjusted"][i]
        primary["predictive_increment_supported"] = bool(primary["status"] == "TESTABLE" and multiplicity["reject_at_alpha_0_05"][i])
    write_npz(out / "T_PREDICTIONS.npz", y=y, z_selected_standardized=z[:, unit], h=h,
              label_prediction_base_raw=pred_base, label_prediction_native_raw=pred_native,
              label_prediction_fullrep_raw=pred_full, label_loss_base=loss_base,
              label_loss_native=loss_native, label_loss_fullrep=loss_full,
              z_prediction_base=pred_z_base, z_prediction_H=pred_z_h,
              z_loss_base=loss_z_base, z_loss_H=loss_z_h)
    class_distributions = {}
    for label in ["HELIX", "STRAND"]:
        mask = np.array([c["label"] == label for c in natural])
        class_distributions[label] = {"z_raw": distribution(native["z"][mask, unit]),
            "z_standardized": distribution(z[mask, unit]),
            "label_base_clipped_prediction": distribution(np.clip(pred_base[mask], 0, 1)),
            "label_native_clipped_prediction": distribution(np.clip(pred_native[mask], 0, 1)),
            "label_fullrep_clipped_prediction": distribution(np.clip(pred_full[mask], 0, 1)),
            "X_prediction_z_standardized": distribution(pred_z_base[mask]),
            "X_H_prediction_z_standardized": distribution(pred_z_h[mask]) if len(pred_z_h) else None}
    lookup = {c["case_id"]: i for i, c in enumerate(natural)}
    edit_z, edit_logits20 = [], []
    for edit in edits:
        row = [runner.run(case_with_sequence(edit, cond), f"T_edit_{cond}_baseline") for cond in ["C", "N"]]
        edit_z.append(np.stack([r["z"] for r in row]))
        edit_logits20.append(np.stack([r["logits20"] for r in row]))
    indices = np.array([lookup[e["case"]["case_id"]] for e in edits], dtype=np.int64)
    dedit = np.load(out / "D_EDIT_NATIVE.npz")
    cached = {"z": np.stack(edit_z) if edits else np.empty((0, 2, DIM), dtype=np.float32),
              "logits20": np.stack(edit_logits20) if edits else np.empty((0, 2, 20), dtype=np.float32),
              "p_z": native["z"][indices], "p_logits20": native["logits20"][indices],
              "w2_norms": dedit["w2_norms"]}
    write_npz(out / "T_EDIT_NATIVE.npz", **cached)
    rows = patch_diagnostics(runner, edits, cached, unit, development["control"], compiled["s_H"],
                             selected["selected_D_fit_sd"], compiled["epsilon_L"], compiled["epsilon_z"],
                             "T_edit", compiled["s_use"])
    auxiliary = {"status": "MEASURED" if rows else "NA_NO_COMPILABLE_CASES", "rows": rows,
                 "n_components": len(rows), "s_H": compiled["s_H"], "s_g": compiled["s_g"], "s_use": compiled["s_use"],
                 "dose_stratum_comparable_n": sum(r["dose_stratum_comparable"] for r in rows),
                 "R_expected_direction_n": None if compiled["s_use"] is None else sum(r["R_expected_direction"] for r in rows),
                 "D_expected_direction_n": None if compiled["s_use"] is None else sum(r["D_expected_direction"] for r in rows)}
    writejson(out / "T_PATCH_DIAGNOSTICS.json", auxiliary, exclusive=True)
    summary = {"status": "CONFIRMATION_COMPLETE_PENDING_SUPERVISOR_ACCEPTANCE", "created_utc": now(),
               "selected_unit": unit, "natural_components": 64, "natural_sites": 128,
               "primary1_conditional_label_information": primary1,
               "primary2_H_natural_prediction_information": primary2, "multiplicity": multiplicity,
               "actual_losses": {"label_Brier_X": float(loss_base.mean()), "label_Brier_X_z": float(loss_native.mean()),
                   "label_Brier_X_fullrep_diagnostic": float(loss_full.mean()),
                   "z_MSE_X": float(loss_z_base.mean()), "z_MSE_X_H": float(loss_z_h.mean()) if len(loss_z_h) else None},
               "fullrep_component_Brier_improvement_diagnostic": component_means(loss_base - loss_full, natural).tolist(),
               "class_prediction_distributions": class_distributions,
               "auxiliary": {k: v for k, v in auxiliary.items() if k != "rows"},
               "model_forwards_cumulative": ledger.state["sequence_forwards"],
               "model_cpu_seconds_cumulative": ledger.state["model_cpu_seconds"],
               "model_forward_wall_seconds_cumulative": ledger.state["model_forward_wall_seconds"],
               "interpretation_limit": "predictive increments only; no biological minimum-effect, mechanism, necessity, or Agent-superiority claim"}
    writejson(out / "T_SUMMARY.json", summary, exclusive=True)
    print(json.dumps({"status": summary["status"], "selected_unit": unit,
                      "primary_support": [primary1["predictive_increment_supported"], primary2["predictive_increment_supported"]],
                      "Holm_p": multiplicity["p_adjusted"], "forwards": ledger.state["sequence_forwards"],
                      "model_cpu_seconds": ledger.state["model_cpu_seconds"],
                      "model_forward_wall_seconds": ledger.state["model_forward_wall_seconds"]}))


def command_selftest(args):
    """Synthetic numerical and shape checks only; torch/transformers never imported."""
    rng = np.random.default_rng(SEED)
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 4
    q = 23
    x = control_features(sequence, q)
    changed_center = sequence[:q] + "A" + sequence[q + 1:]
    np.testing.assert_allclose(x, control_features(changed_center, q), rtol=0, atol=0)
    assert x.shape == (64,) and len(feature_names()) == 64
    np.testing.assert_allclose(x[:60].reshape(3, 20).sum(1), np.ones(3))
    for boundary in [0, 7, len(sequence) - 8, len(sequence) - 1]:
        assert np.isfinite(control_features(sequence, boundary)).all()
    toy = np.column_stack([rng.normal(size=48), np.ones(48)])
    scaler = fit_scaler(toy)
    assert scaler["scale"][1] == 1 and scaler["constant"][1]
    np.testing.assert_allclose(transform(toy, scaler)[:, 1], 0)
    xf, xs = rng.normal(size=(48, 9)), rng.normal(size=(12, 9))
    zf, zs, y = rng.normal(size=(48, 7)), rng.normal(size=(12, 7)), rng.normal(size=48)
    augmented = all_unit_ridges(xf, zf, y, xs, zs)
    for j in range(zf.shape[1]):
        fit = ridge_fit(np.column_stack([xf, zf[:, j]]), y)
        brute = ridge_predict(fit, np.column_stack([xs, zs[:, j]]))
        np.testing.assert_allclose(augmented["predictions"][:, j], brute, rtol=1e-11, atol=1e-11)
    wide = rng.normal(size=(12, 30))
    yw = rng.normal(size=12)
    dual = ridge_fit(wide, yw)
    wc, yc = wide - wide.mean(0), yw - yw.mean()
    primal = np.linalg.solve(wc.T @ wc + RIDGE_LAMBDA * np.eye(30), wc.T @ yc)
    np.testing.assert_allclose(dual["coef"], primal, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(brier(np.array([0, 1]), np.array([-1, 2])), [0, 0])
    cases = [{"component": c} for c in ["a", "a", "b", "b"]]
    np.testing.assert_allclose(component_means([1, 3, 5, 7], cases), [2, 6])
    hresponse = {"status": "valid", "operator": "all", "clauses": [{"offset": -1, "aa_set": "DE"}],
                 "explanation_zh": "中心前一位是酸性氨基酸", "development_candidate_count": 1}
    hypothesis = validate_hypothesis(hresponse)
    assert hypothesis["status"] == "VALID"
    assert validate_hypothesis({**hresponse, "clauses": [{"offset": 0, "aa_set": "DE"}]})["status"] == "NA"
    toy_sequence = list("A" * 80)
    for qtoy, amino in [(29, "D"), (33, "G"), (35, "V"), (49, "E"), (53, "S"), (55, "L")]:
        toy_sequence[qtoy] = amino
    toy_sequence = "".join(toy_sequence)
    toy_cases = [{"id": "synthetic", "component": "synthetic_component", "split": "D_select",
                  "q": qtoy, "sequence": toy_sequence, "label": label, "y": target,
                  "case_id": f"synthetic:{label}:{qtoy}"}
                 for qtoy, label, target in [(30, "HELIX", 1), (50, "STRAND", 0)]]
    compiled = compile_edits(toy_cases, hypothesis, 1)
    assert len(compiled) == 1
    e = compiled[0]
    assert h_value(hypothesis, e["P"], e["case"]["q"]) == 1
    assert h_value(hypothesis, e["C"], e["case"]["q"]) == 0
    assert h_value(hypothesis, e["N"], e["case"]["q"]) == 1
    assert swap_geometry(e["C_swap_offsets"]) == swap_geometry(e["N_swap_offsets"])
    assert compiled == compile_edits(toy_cases, hypothesis, 1)
    native = rng.normal(size=(1, 6, 7)).astype(np.float32)
    before = native.copy()
    replacement = float(native[0, 2, 4]) + 0.5
    patched = replace_native_scalar(native, 2, 4, replacement)
    assert np.array_equal(native, before) and np.count_nonzero(patched != native) == 1
    assert np.array_equal(replace_native_scalar(native, 2, 4, float(native[0, 2, 4])), native)
    w2 = rng.normal(size=(5, 7)).astype(np.float32)
    actual = patched[0, 2] @ w2.T - native[0, 2] @ w2.T
    expected = (patched[0, 2, 4] - native[0, 2, 4]) * w2[:, 4]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    rule = choose_control(np.array([[1, 2, 2], [2, 4, 4]]), np.array([1, 0.5, 0.5]), 0, np.ones(3, dtype=bool))
    assert rule["unit"] == 1 and rule["common_support_status"] == "AVAILABLE"
    assert dose_comparability(1, 1, rule)["dose_stratum_comparable"]
    assert not dose_comparability(0, 1, rule)["dose_stratum_comparable"]
    directional = downstream_direction([{"delta_z": 1, "R": 2, "D": 3}] * 4, 1e-6, 1e-6)
    assert directional["s_g"] == 1
    assert downstream_direction([{"delta_z": 1, "R": 2, "D": -3}] * 4, 1e-6, 1e-6)["s_g"] is None
    effect = paired_effect([1, 2, 3, 4], 1)
    from scipy.stats import t
    np.testing.assert_allclose(effect["p_one_sided_t_reference"], t.sf(2.5 / (np.std([1, 2, 3, 4], ddof=1) / 2), 3))
    assert paired_effect([1, 1, 1, 1], 1)["status"] == "NA"
    assert paired_effect(None, 2)["p_one_sided_t_reference"] == 1
    np.testing.assert_allclose(holm_two([0.01, 0.04])["p_adjusted"], [0.02, 0.04])
    np.testing.assert_allclose(holm_two([0.03, 1])["p_adjusted"], [0.06, 1])
    guard_args = argparse.Namespace(out=Path(args.out) / "__missing_lock_check__")
    for callback in [lambda: require_initial_lock(guard_args), lambda: NativeRunner(None, None, None)]:
        try:
            callback()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing-lock/model-load guard failed")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Synthetic selftest unexpectedly imported model libraries")
    print(json.dumps({"status": "SYNTHETIC_PASS", "model_loads": 0, "model_forwards": 0,
                      "checks": ["center-excluded X", "feature shape/proportions/boundaries",
                                 "constant-column scaling", "augmented ridge identity",
                                 "primal-dual ridge identity", "Brier clipping", "component averaging",
                                 "finite H validity", "edit geometry/composition/H/determinism",
                                 "single-scalar and fc2 delta algebra", "dose support/tie-break",
                                 "D direction formation", "paired-t/bootstrap/Holm2", "locked-command model guard"]}))


def command_prepare(args):
    records = load_records(args.data)
    summary = {"status": "FRAMEWORK_PREPARED_NO_MODEL_RELEASE", "created_utc": now(),
               "split_components": dict(Counter(p["split"] for p in records)),
               "split_sites": {s: len(flatten(records, s)) for s in SPLITS},
               "feature_count": len(feature_names()), "feature_names": feature_names(),
               "model_loads": 0, "model_forwards": 0,
               "note": "Formal analysis protocol and explicit stage lock are required before model loading."}
    writejson(Path(args.out) / "FRAMEWORK_PREPARED.json", summary)
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"selftest": command_selftest, "prepare": command_prepare, "discover": command_discover,
                "develop": command_develop, "calibrate": command_calibrate,
                "compile-confirmation": command_compile_confirmation, "confirm": command_confirm}
    parser.add_argument("command", choices=list(commands))
    parser.add_argument("--data", type=Path, default=ROOT / "data/pilot3a/proteins.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/pilot3a")
    parser.add_argument("--model", type=Path, default=ROOT / "assets/esm2_8m")
    args = parser.parse_args()
    if args.out.resolve() != (ROOT / "results/pilot3a").resolve():
        raise RuntimeError("Analysis writes are restricted to results/pilot3a")
    started = time.perf_counter()
    error = None
    try:
        commands[args.command](args)
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        receipt = {"command": args.command, "finished_utc": now(), "status": "FAILED" if error else "COMPLETE",
                   "error_type": error, "total_wall_seconds": time.perf_counter() - started}
        receipt_path = Path(args.out) / "COMMAND_RECEIPTS.json"
        receipts = readjson(receipt_path) if receipt_path.exists() else {"commands": []}
        receipts["commands"].append(receipt)
        writejson(receipt_path, receipts)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
