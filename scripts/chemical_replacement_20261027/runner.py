"""Locked two-coordinate chemical replacement; explicit run, never auto-resume.

The synthetic check uses only constructed tensors.  The run command consumes a
frozen D fit and executes the original consecutive four-molecule T geometry.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

START_WALL = time.monotonic()
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/chemical_replacement_20261027"
FIT = ROOT / "results/chemical_replacement_20261027/fit"
OUT = ROOT / "results/chemical_replacement_20261027/forward"
sys.path.insert(0, str(ROOT / "scripts/revision_20261017"))
from chem_model import ChembertaMTR, PRIMARY_LAYER, batch_from_audits

COORDINATES = (293, 454)
CONDITIONS = ("D_mean", "M1", "M2", "M1_delta_perm", "M2_delta_perm")
REPLAY_STARTS = (0, 4000, 8380, 9996)
SCOPE_STARTS = (0, 8380)
ATOL, RTOL = 2e-5, 2e-6


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c = Counters(); c.cb = ctypes.sizeof(c)
    getp = ctypes.windll.kernel32.GetCurrentProcess; getp.restype = ctypes.c_void_p
    getm = ctypes.windll.psapi.GetProcessMemoryInfo
    getm.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]; getm.restype = ctypes.c_int
    if not getm(getp(), ctypes.byref(c), c.cb):
        raise RuntimeError("GetProcessMemoryInfo failed")
    return {"rss_bytes": int(c.WorkingSetSize), "peak_rss_bytes": int(c.PeakWorkingSetSize),
            "private_commit_bytes": int(c.PagefileUsage), "peak_private_commit_bytes": int(c.PeakPagefileUsage)}


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.record = {"status": "RUNNING", "started_utc": utc(), "attempted_forward_molecule_evaluations": 0,
            "completed_forward_molecule_evaluations": 0,
            "attempted_by_category": {k: 0 for k in ("condition", "baseline", "replay", "scope")},
            "completed_by_category": {k: 0 for k in ("condition", "baseline", "replay", "scope")},
            "preflight_molecule_evaluations_separate": 16, "limits": limits, "current": None}

    def check(self):
        self.record.update(wall_seconds=time.monotonic() - START_WALL, process_cpu_seconds=time.process_time(),
                           memory=memory(), updated_utc=utc())
        r, m = self.record, self.record["memory"]
        if r["wall_seconds"] > self.limits["wall_seconds_cap"]:
            raise TimeoutError("Locked forward wall budget reached")
        if r["process_cpu_seconds"] > self.limits["process_cpu_seconds_cap"]:
            raise TimeoutError("Locked forward process CPU budget reached")
        if m["peak_rss_bytes"] > self.limits["peak_rss_bytes_cap"]:
            raise MemoryError("Locked peak RSS budget reached")
        if m["peak_private_commit_bytes"] > self.limits["private_commit_bytes_cap"]:
            raise MemoryError("Locked private-commit budget reached")

    def before(self, category, begin, coordinate=None, condition=None):
        self.record["current"] = {"category": category, "T_begin": begin,
                                  "coordinate": coordinate, "condition": condition}
        self.check()
        if self.record["attempted_forward_molecule_evaluations"] + 4 > self.limits["total_molecule_evaluation_cap"]:
            raise RuntimeError("Locked molecule-evaluation cap would be exceeded")
        # Count an attempted actual forward before invoking it, including failures.
        self.record["attempted_forward_molecule_evaluations"] += 4
        self.record["attempted_by_category"][category] += 4

    def after(self, category):
        self.record["completed_forward_molecule_evaluations"] += 4
        self.record["completed_by_category"][category] += 4
        self.check()

    def save(self):
        self.record.update(wall_seconds=time.monotonic() - START_WALL, process_cpu_seconds=time.process_time(),
                           memory=memory(), updated_utc=utc())
        write(OUT / "CPU_FORWARD_LEDGER.json", self.record)


class CoordinateReplacementHook:
    """Read actual incoming activations, replace only explicit graph-order anchors."""
    def __init__(self, coordinate, positions, targets, *, permutations=None, capture=False):
        self.coordinate, self.positions, self.targets = int(coordinate), positions, targets
        self.permutations, self.capture = permutations, capture
        self.original = self.applied = None
        self.rms = self.nonzero_counts = None
        self.diagnostics = []
        self.called = False

    def __call__(self, module, inputs, activation):
        if self.called:
            raise RuntimeError("Coordinate hook called more than once")
        self.called = True
        if activation.ndim != 3 or activation.shape[-1] != 464 or len(self.positions) != len(activation):
            raise ValueError("Unexpected real post-GELU activation geometry")
        if not torch.isfinite(activation).all():
            raise ArithmeticError("Nonfinite incoming real activation")
        changed = activation.clone()
        if self.capture:
            self.original = activation.detach().clone()
        rms, nonzero = [], []
        for row, positions in enumerate(self.positions):
            if not positions or len(set(positions)) != len(positions):
                raise ValueError("Empty or nonbijective graph-order atom anchors")
            original = activation[row, positions, self.coordinate].detach().numpy().astype(np.float64)
            target = np.asarray(self.targets[row], dtype=np.float64)
            if target.shape != original.shape or not np.isfinite(target).all():
                raise ValueError("Invalid graph-order replacement target")
            delta = target - original
            intended = delta
            identity = effective_identity = constant = None
            if self.permutations is not None:
                permutation = self.permutations[row]
                if not np.array_equal(np.sort(permutation), np.arange(len(delta))):
                    raise ValueError("Permutation is not a bijection")
                intended = delta[permutation]
                before, after = np.sort(delta), np.sort(intended)
                if not np.array_equal(before, after):
                    raise ArithmeticError("Permutation changed the float64 delta multiset")
                # Canonical summation order makes the algebraic L2 check exact.
                if np.square(before).sum(dtype=np.float64) != np.square(after).sum(dtype=np.float64):
                    raise ArithmeticError("Permutation changed canonical float64 delta L2")
                identity = bool(np.array_equal(permutation, np.arange(len(delta))))
                effective_identity = bool(np.array_equal(delta, intended))
                constant = bool(np.all(delta == delta[0]))
                final = original + intended
            else:
                final = target
            changed[row, positions, self.coordinate] = torch.as_tensor(final, dtype=activation.dtype, device=activation.device)
            realized = changed[row, positions, self.coordinate].detach().numpy().astype(np.float64) - original
            error = np.abs(realized - intended)
            ratio = error / (ATOL + RTOL * np.abs(intended))
            worst = int(np.argmax(ratio))
            if float(ratio[worst]) > 1:
                raise ArithmeticError(json.dumps({"error": "float32 realized delta exceeds locked atol/rtol",
                    "row": row, "graph_atom_id": worst, "coordinate": self.coordinate,
                    "intended_delta": float(intended[worst]), "realized_delta": float(realized[worst]),
                    "absolute_error": float(error[worst]), "normalized_tolerance_ratio": float(ratio[worst])}))
            rms.append(float(np.sqrt(np.square(realized).mean(dtype=np.float64))))
            nonzero.append(int(np.count_nonzero(realized)))
            self.diagnostics.append({"float32_delta_max_abs_error": float(error.max()),
                "float32_delta_max_tolerance_ratio": float(ratio.max()), "permutation_identity": identity,
                "delta_rearrangement_identity": effective_identity, "constant_delta": constant,
                "nonzero_intended_delta_atoms": int(np.count_nonzero(intended)),
                "float64_sorted_multiset_exact": True, "float64_canonical_L2_exact": True,
                "unpermuted_delta_L2": float(np.linalg.norm(delta)),
                "permuted_delta_L2": float(np.linalg.norm(intended)),
                "realized_delta_L2": float(np.linalg.norm(realized))})
        if not torch.isfinite(changed).all():
            raise ArithmeticError("Nonfinite applied activation")
        self.rms = np.asarray(rms, dtype=np.float64)
        self.nonzero_counts = np.asarray(nonzero, dtype=np.int32)
        if self.capture:
            self.applied = changed.detach().clone()
        return changed


def actual_forward(model, batch, budget, category, begin, *, coordinate=None, condition=None, hook=None):
    budget.before(category, begin, coordinate, condition)
    handle = None
    try:
        if hook is not None:
            handle = model.roberta.encoder.layer[PRIMARY_LAYER].intermediate.register_forward_hook(hook)
        result = model(batch, layer_index=PRIMARY_LAYER)
        budget.after(category)
        if not torch.isfinite(result.normalized_predictions).all() or not torch.isfinite(result.activations).all():
            raise ArithmeticError("Nonfinite model output or captured activation")
        if hook is not None and not hook.called:
            raise RuntimeError("Real post-GELU hook did not execute")
        return result
    finally:
        if handle is not None:
            handle.remove()


def update_max(record, difference, actual, cached, *, molecules, endpoints, label):
    flat = int(np.argmax(difference)); row, column = np.unravel_index(flat, difference.shape)
    value = float(difference[row, column])
    if record.get("max_abs") is None or value > record["max_abs"]:
        record.update(max_abs=value, T_molecule_index=int(molecules[row]), endpoint=int(endpoints[column]),
                      actual=float(actual[row, column]), cached=float(cached[row, column]), label=label)


def bind_inputs():
    protocol_sha, resource_sha = sha(DOC / "PROTOCOL.md"), sha(DOC / "RESOURCE_LOCK.json")
    model_sha, data_sha = sha(FIT / "D_FIT.json"), sha(Path(__file__).with_name("data_fit.py"))
    lock, receipt = read(FIT / "D_FIT_LOCK.json"), read(FIT / "T_ACTIVATIONS_RECEIPT.json")
    if lock.get("status") != "D_FIT_FROZEN_BEFORE_T_APPLY":
        raise ValueError("D fit is not frozen before T apply")
    for key, expected in (("D_FIT_sha256", model_sha), ("protocol_sha256", protocol_sha),
                          ("resource_lock_sha256", resource_sha), ("implementation_sha256", data_sha)):
        if lock.get(key) != expected:
            raise ValueError(f"Frozen D fit binding mismatch: {key}")
    if receipt.get("D_FIT_sha256") != model_sha:
        raise ValueError("T predictions do not bind the frozen D model")
    prediction_sha = sha(FIT / "T_activations.npz")
    if receipt.get("T_activations_sha256") != prediction_sha:
        raise ValueError("T prediction artifact differs from its receipt")
    return {"schema": "chemical-replacement-forward-binding/1", "created_utc": utc(),
        "protocol_sha256": protocol_sha, "resource_lock_sha256": resource_sha,
        "runner_sha256": sha(Path(__file__)), "data_fit_sha256": data_sha,
        "D_FIT_sha256": model_sha, "D_FIT_LOCK_sha256": sha(FIT / "D_FIT_LOCK.json"),
        "T_activations_sha256": prediction_sha, "T_ACTIVATIONS_RECEIPT_sha256": sha(FIT / "T_ACTIVATIONS_RECEIPT.json"),
        "coordinate_ids": list(COORDINATES), "condition_names": list(CONDITIONS)}


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Forward directory contains prior artifacts; no automatic overwrite or resume")
    OUT.mkdir(parents=True, exist_ok=True)
    resource = read(DOC / "RESOURCE_LOCK.json")
    budget = Budget(resource["forward_stage"])
    baseline = conditions = doses = None
    validation = {"cache_all199": {}, "cache_selected_q": {str(q): {} for q in (115, 124, 125)},
        "cache_atom_coordinates": {str(j): {} for j in COORDINATES}, "replay": [], "scope": [],
        "float32_delta_max_abs_error": 0., "float32_delta_max_tolerance_ratio": 0.,
        "float32_delta_L2_max_abs_error": 0., "float64_permutation_L2_raw_sum_max_abs_difference": 0.,
        "float64_permutation_sorted_multiset_exact": True, "float64_permutation_canonical_L2_exact": True}
    try:
        binding = bind_inputs(); write(OUT / "RUN_BINDING.json", binding)
        from data_fit import load_split
        data = load_split("T")
        offsets = np.asarray(data["mol_offsets"])
        if len(data["audits"]) != 10000 or offsets.shape != (10001,) or offsets[-1] != 246722:
            raise ValueError("Frozen T dimensions differ")
        with np.load(FIT / "T_activations.npz", allow_pickle=False) as z:
            predictions = z["predictions"].copy()
            if z["coordinate_ids"].tolist() != list(COORDINATES) or z["model_names"].tolist() != list(CONDITIONS[:3]):
                raise ValueError("Prediction coordinate/model order differs")
            if not np.array_equal(z["mol_offsets"], offsets):
                raise ValueError("Prediction atom offsets differ")
        if predictions.shape != (246722, 2, 3) or predictions.dtype != np.float64 or not np.isfinite(predictions).all():
            raise ValueError("Invalid frozen prediction targets")
        baseline = np.lib.format.open_memmap(OUT / "baseline.npy", mode="w+", dtype=np.float32, shape=(10000, 199))
        conditions = np.lib.format.open_memmap(OUT / "conditions.npy", mode="w+", dtype=np.float32, shape=(2, 5, 10000, 199))
        doses = np.lib.format.open_memmap(OUT / "dose_rms.npy", mode="w+", dtype=np.float64, shape=(2, 5, 10000))
        baseline[:] = np.nan; conditions[:] = np.nan; doses[:] = np.nan
        identity = np.zeros((2, 10000), dtype=np.bool_)
        rearrangement_identity = np.zeros((2, 2, 10000), dtype=np.bool_)
        constant_delta = np.zeros((2, 2, 10000), dtype=np.bool_)
        nonzero = np.zeros((2, 5, 10000), dtype=np.int32)
        intended_nonzero = np.zeros((2, 2, 10000), dtype=np.int32)
        atom_counts = np.diff(offsets).astype(np.int32)
        with threadpool_limits(limits=4):
            model = ChembertaMTR.from_official_checkpoint()
            write(OUT / "RUNTIME.json", {"python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
                "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "threadpools": threadpool_info(), "model_receipt": model.load_receipt, "memory": memory()})
            for begin in range(0, 10000, 4):
                end = begin + 4; audits = data["audits"][begin:end]
                batch = batch_from_audits(audits)
                positions = [[a["anchor_token_index"] for a in audit["atoms"]] for audit in audits]
                for row, mi in enumerate(range(begin, end)):
                    if [a["graph_atom_id"] for a in audits[row]["atoms"]] != list(range(int(atom_counts[mi]))):
                        raise ValueError("Actual graph-order anchor sequence differs")
                    if len(positions[row]) != int(atom_counts[mi]) or not batch.atom_mask[row, positions[row]].all():
                        raise ValueError("Actual atom-anchor bijection differs")
                original = actual_forward(model, batch, budget, "baseline", begin)
                fresh = original.normalized_predictions.numpy()
                cached = np.asarray(data["cached_head"][begin:end])
                difference = np.abs(fresh.astype(np.float64) - cached.astype(np.float64))
                update_max(validation["cache_all199"], difference, fresh, cached,
                           molecules=range(begin, end), endpoints=range(199), label="q")
                if float(difference.max()) > 2e-5:
                    raise ArithmeticError(json.dumps({"gate": "fresh_cache_all199", **validation["cache_all199"]}))
                for q in (115, 124, 125):
                    update_max(validation["cache_selected_q"][str(q)], difference[:, q:q+1], fresh[:, q:q+1], cached[:, q:q+1],
                               molecules=range(begin, end), endpoints=[q], label="q")
                    if float(difference[:, q].max()) > 2e-6:
                        raise ArithmeticError(json.dumps({"gate": "fresh_cache_selected_q", **validation["cache_selected_q"][str(q)]}))
                for row, mi in enumerate(range(begin, end)):
                    for j in COORDINATES:
                        actual = original.activations[row, positions[row], j].numpy().astype(np.float64)
                        cached_a = np.asarray(data["atom_a"][offsets[mi]:offsets[mi+1], j], dtype=np.float64)
                        error = np.abs(actual - cached_a); ai = int(np.argmax(error)); value = float(error[ai])
                        item = validation["cache_atom_coordinates"][str(j)]
                        if item.get("max_abs") is None or value > item["max_abs"]:
                            item.update(max_abs=value, T_molecule_index=mi, graph_atom_id=ai,
                                global_atom_index=int(offsets[mi]) + ai, anchor_token_index=positions[row][ai],
                                actual=float(actual[ai]), cached=float(cached_a[ai]))
                        if value > 2e-5:
                            raise ArithmeticError(json.dumps({"gate": "fresh_cache_atom_coordinate", "coordinate": j, **item}))
                baseline[begin:end] = fresh
                for ci, j in enumerate(COORDINATES):
                    permutations = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261027, j, mi]))).permutation(int(atom_counts[mi])) for mi in range(begin, end)]
                    identity[ci, begin:end] = [np.array_equal(p, np.arange(len(p))) for p in permutations]
                    targets = [[predictions[offsets[mi]:offsets[mi+1], ci, k] for mi in range(begin, end)] for k in range(3)]
                    if begin in REPLAY_STARTS:
                        exact_targets = [original.activations[row, positions[row], j].numpy().astype(np.float64) for row in range(4)]
                        hook = CoordinateReplacementHook(j, positions, exact_targets, capture=True)
                        result = actual_forward(model, batch, budget, "replay", begin, coordinate=j, condition="no_op", hook=hook)
                        check = {"T_begin": begin, "coordinate": j,
                            "output_bitwise_exact": bool(torch.equal(result.normalized_predictions, original.normalized_predictions)),
                            "activation_bitwise_exact": bool(torch.equal(hook.original, hook.applied))}
                        validation["replay"].append(check)
                        if not check["output_bitwise_exact"] or not check["activation_bitwise_exact"]:
                            raise ArithmeticError(json.dumps({"gate": "no_op_exact_replay", **check}))
                        del result, hook
                    if begin in SCOPE_STARTS:
                        hook = CoordinateReplacementHook(j, positions, targets[1], capture=True)
                        result = actual_forward(model, batch, budget, "scope", begin, coordinate=j, condition="M1_scope", hook=hook)
                        allowed = torch.zeros_like(hook.original, dtype=torch.bool)
                        for row, p in enumerate(positions):
                            allowed[row, p, j] = True
                        check = {"T_begin": begin, "coordinate": j,
                            "other_coordinates_and_nonanchors_bitwise_exact": bool(torch.equal(hook.original[~allowed], hook.applied[~allowed])),
                            "real_post_GELU_applied_capture_exact": bool(torch.equal(result.activations, hook.applied))}
                        validation["scope"].append(check)
                        if not all(check[k] for k in ("other_coordinates_and_nonanchors_bitwise_exact", "real_post_GELU_applied_capture_exact")):
                            raise ArithmeticError(json.dumps({"gate": "selected_atom_coordinate_scope", **check}))
                        del result, hook, allowed
                    for ki, name in enumerate(CONDITIONS):
                        target_kind = ki if ki < 3 else ki - 2
                        hook = CoordinateReplacementHook(j, positions, targets[target_kind], permutations=permutations if ki >= 3 else None)
                        result = actual_forward(model, batch, budget, "condition", begin, coordinate=j, condition=name, hook=hook)
                        conditions[ci, ki, begin:end] = result.normalized_predictions.numpy()
                        doses[ci, ki, begin:end] = hook.rms; nonzero[ci, ki, begin:end] = hook.nonzero_counts
                        for row, detail in enumerate(hook.diagnostics):
                            validation["float32_delta_max_abs_error"] = max(validation["float32_delta_max_abs_error"], detail["float32_delta_max_abs_error"])
                            validation["float32_delta_max_tolerance_ratio"] = max(validation["float32_delta_max_tolerance_ratio"], detail["float32_delta_max_tolerance_ratio"])
                            validation["float32_delta_L2_max_abs_error"] = max(validation["float32_delta_L2_max_abs_error"], abs(detail["realized_delta_L2"] - detail["permuted_delta_L2"]))
                            validation["float64_permutation_L2_raw_sum_max_abs_difference"] = max(validation["float64_permutation_L2_raw_sum_max_abs_difference"], abs(detail["unpermuted_delta_L2"] - detail["permuted_delta_L2"]))
                            if ki >= 3:
                                rearrangement_identity[ci, ki-3, begin+row] = detail["delta_rearrangement_identity"]
                                constant_delta[ci, ki-3, begin+row] = detail["constant_delta"]
                                intended_nonzero[ci, ki-3, begin+row] = detail["nonzero_intended_delta_atoms"]
                        del result, hook
                del original
                baseline.flush(); conditions.flush(); doses.flush()
                write(OUT / "PROGRESS.json", {"status": "RUNNING", "completed_molecules": end,
                    "condition_molecule_evaluations": 10 * end, "last_complete_batch_start": begin,
                    "attempted_forward_molecule_evaluations": budget.record["attempted_forward_molecule_evaluations"], "updated_utc": utc()})
                budget.save()
                if end % 200 == 0:
                    print(json.dumps({"stage": "forward", "completed_T_molecules": end,
                        "evaluations": budget.record["completed_forward_molecule_evaluations"],
                        "wall_seconds": budget.record["wall_seconds"], "CPU_seconds": budget.record["process_cpu_seconds"]}), flush=True)
        expected = {"condition": 100000, "baseline": 10000, "replay": 32, "scope": 16}
        if budget.record["completed_by_category"] != expected or budget.record["attempted_by_category"] != expected:
            raise RuntimeError("Actual forward category counts differ from the locked design")
        if not np.isfinite(baseline).all() or not np.isfinite(conditions).all() or not np.isfinite(doses).all():
            raise ArithmeticError("Completed forward output has missing or nonfinite entries")
        np.savez(OUT / "permutation_metadata.npz", coordinate_ids=np.array(COORDINATES), model_names=np.array(["M1", "M2"]),
            atom_counts=atom_counts, permutation_identity=identity, delta_rearrangement_identity=rearrangement_identity,
            constant_delta=constant_delta, nonzero_realized_shift_atom_counts=nonzero,
            nonzero_intended_permuted_delta_atom_counts=intended_nonzero)
        write(OUT / "PERMUTATION_METADATA.json", {"seed_rule": "PCG64(SeedSequence([20261027,coordinate,T_molecule_index]))",
            "M1_M2_shared_permutation": True, "identity_permutation_molecules_by_coordinate": identity.sum(1).tolist(),
            "nontrivial_delta_rearrangement_molecules_coordinate_by_M1_M2": (~rearrangement_identity).sum(2).tolist(),
            "constant_delta_molecules_coordinate_by_M1_M2": constant_delta.sum(2).tolist(),
            "single_atom_molecules": int((atom_counts == 1).sum()), "degenerate_permutations_retained": True,
            "float64_sorted_multiset_exact": True, "float64_canonical_L2_exact": True,
            "float32_realization_atol": ATOL, "float32_realization_rtol": RTOL,
            "interpretation": "preserves within-molecule intended delta multiset/L2; changes direction and replacement activation marginal; diagnostic, not a strict distribution-matched null"})
        write(OUT / "VALIDATION.json", validation)
        write(OUT / "PROGRESS.json", {"status": "COMPLETE", "completed_molecules": 10000,
            "condition_molecule_evaluations": 100000, "last_complete_batch_start": 9996,
            "attempted_forward_molecule_evaluations": budget.record["attempted_forward_molecule_evaluations"], "updated_utc": utc()})
        outputs = [OUT / n for n in ("baseline.npy", "conditions.npy", "dose_rms.npy", "permutation_metadata.npz",
            "PERMUTATION_METADATA.json", "VALIDATION.json", "RUN_BINDING.json", "PROGRESS.json", "RUNTIME.json")]
        identities = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in outputs}
        budget.check(); budget.record["status"] = "COMPLETE"; budget.save()
        ledger_path = OUT / "CPU_FORWARD_LEDGER.json"
        identities[ledger_path.name] = {"bytes": ledger_path.stat().st_size, "sha256": sha(ledger_path)}
        receipt = {"status": "COMPLETE_FORWARD_REPLACEMENT", "completed_utc": utc(), "binding": binding,
            "coordinate_ids": list(COORDINATES), "condition_names": list(CONDITIONS),
            "shapes": {"baseline": [10000, 199], "conditions": [2, 5, 10000, 199], "dose_rms": [2, 5, 10000]},
            "forward_counts": expected, "total_forward_molecule_evaluations": 110048,
            "files": identities,
            "ledger": "CPU_FORWARD_LEDGER.json", "validation": "VALIDATION.json"}
        write(OUT / "FORWARD_RECEIPT.json", receipt)
        print(json.dumps({"status": receipt["status"], "forward_counts": expected, "wall_seconds": budget.record["wall_seconds"],
            "CPU_seconds": budget.record["process_cpu_seconds"]}), flush=True)
    except BaseException as error:
        for a in (baseline, conditions, doses):
            if a is not None:
                a.flush()
        budget.record["status"] = "FAILED_RETAINED"; budget.save()
        write(OUT / "VALIDATION_AT_FAILURE.json", validation)
        write(OUT / ("FAILURE_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
            {"created_utc": utc(), "error_type": type(error).__name__, "message": str(error),
             "current": budget.record["current"], "ledger": budget.record})
        raise


def synthetic_check():
    generator = torch.Generator().manual_seed(20261027)
    activation = torch.randn((2, 8, 464), generator=generator)
    positions = [[5, 1, 3], [4]]  # Deliberately not token order; includes a singleton.
    target = [np.array([.1, -.2, .3]), np.array([-.5])]
    checks = {}
    for j in COORDINATES:
        hook = CoordinateReplacementHook(j, positions, target, capture=True)
        applied = hook(None, None, activation)
        allowed = torch.zeros_like(activation, dtype=torch.bool)
        for row, p in enumerate(positions):
            allowed[row, p, j] = True
        checks[f"j{j}_scope"] = bool(torch.equal(activation[~allowed], applied[~allowed]))
        checks[f"j{j}_graph_order_target"] = all(torch.equal(applied[row, p, j], torch.tensor(target[row], dtype=torch.float32)) for row, p in enumerate(positions))
        exact = [activation[row, p, j].numpy().astype(np.float64) for row, p in enumerate(positions)]
        no_op = CoordinateReplacementHook(j, positions, exact, capture=True)
        checks[f"j{j}_no_op_tensor_exact"] = bool(torch.equal(no_op(None, None, activation), activation))
        perm = CoordinateReplacementHook(j, positions, target, permutations=[np.array([2, 0, 1]), np.array([0])], capture=True)
        perm(None, None, activation)
        checks[f"j{j}_permutation_multiset_L2"] = all(d["float64_sorted_multiset_exact"] and d["float64_canonical_L2_exact"] for d in perm.diagnostics)
        checks[f"j{j}_singleton_retained"] = perm.diagnostics[1]["permutation_identity"] and perm.diagnostics[1]["delta_rearrangement_identity"]
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    return {"status": "PASS_SYNTHETIC_HOOK_CHECK", "checks": checks, "model_forwards": 0, "T_reads": 0}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--synthetic-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if args.synthetic_check:
        print(json.dumps(synthetic_check()), flush=True)
    else:
        run()


if __name__ == "__main__":
    main()
