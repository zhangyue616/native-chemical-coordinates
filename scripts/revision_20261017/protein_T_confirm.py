"""Locked short-D to long-T confirmation; prepare/selftest never load a PLM.

Reuse the frozen KRR feature/prediction/inference helpers and the original
center-single-MASK post-GELU NativeRunner. No fitting or scientific retries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_20261017/protein_T"
FRESH = ROOT / "data/revision_20261017/protein_fresh"
D = ROOT / "results/revision_20261017/protein_D"
PROTOCOL = ROOT / "docs/revision_20261017/PROTEIN_T_CONFIRM_PROTOCOL.md"
SCIENCE = ROOT / "docs/revision_20261017/PROTEIN_KRR_PROTOCOL.md"
INITIAL = ROOT / "results/pilot3a/INITIAL_LOCK.json"
MODEL = ROOT / "assets/esm2_8m"
TARGET, FORWARDS, THREADS = 1304, 2608, 4
MAX_MODEL_CPU = MAX_READOUT_CPU = 3600.0
PURPOSE = "protein_long_T_center_single_MASK"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


krr = module_at("_protein_T_frozen_krr", Path(__file__).with_name("protein_krr.py"))
p3 = module_at("_protein_T_native_p3", ROOT / "scripts/pilot3a.py")
np = krr.np
CONFIG = {"schema": "protein-long-T/1", "selected_pairs": TARGET, "cases": FORWARDS,
    "model": "existing accepted ESM2 8M only", "unit": 1246,
    "observation": "third block (0-based 2) post-GELU at the single masked center token q+1",
    "sequence_length": [257, 512], "special_tokens": 2, "maximum_token_length": 514,
    "pair_order": "sealed canonical q ascending", "sequence_forwards": FORWARDS,
    "per_case_forward": "one case, one single center MASK; two different masks for each protein pair",
    "threads": THREADS, "model_CPU_seconds": MAX_MODEL_CPU, "readout_CPU_seconds": MAX_READOUT_CPU,
    "new_fits": 0, "new_units_or_initializations": 0, "HTTP_requests": 0,
    "scientific_retries": 0, "partial_confirmation": False,
    "readouts": list(krr.MODELS), "primary_family": krr.CONFIG["primary_family"],
    "multiplicity": "Holm fixed four, alpha .05", "inference": "frozen group t-reference helper",
    "precision_report": "actual 98.75% t-reference CI half-width <=.01 per endpoint; no guarantee",
    "full_Z_policy": "cache only; only z1246 enters the six frozen D readouts"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def runtime():
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "packages": {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "threadpoolctl", "torch", "transformers")}}


def input_paths():
    return {"script": Path(__file__), "test": Path(__file__).with_name("test_protein_T_confirm.py"),
        "protocol": PROTOCOL, "scientific_contract": SCIENCE,
        "frozen_KRR": ROOT / "scripts/revision_20261017/protein_krr.py",
        "frozen_native_runner": ROOT / "scripts/pilot3a.py",
        "fresh_sealed": FRESH / "SEALED_FRESH_CASES.json", "fresh_acceptance": FRESH / "FRESH_DATA_ACCEPTANCE.json",
        "fresh_result": FRESH / "FRESH_DATA_RESULT.json", "fresh_groups": FRESH / "OLD_SELECTED_FINAL_GROUPS.json",
        "D_final_models": D / "FINAL_MODELS.npz", "D_development": D / "D_DEVELOPMENT.json",
        "D_precision": D / "PRECISION_PLAN.json", "D_acceptance": D / "D_ARTIFACT_CHECK.json",
        "accepted_model_identity": INITIAL, "prior_model_acceptance": ROOT / "results/pilot3a/ACCEPTANCE.json",
        "model_config": MODEL / "config.json", "selftest": OUT / "SYNTHETIC_CHECKS.json"}


def validate_token_capacity(config, maximum_tokens=514):
    if ((config.get("hidden_size"), config.get("intermediate_size"), config.get("num_hidden_layers"),
         config.get("hidden_act")) != (320, 1280, 6, "gelu")
            or config.get("position_embedding_type") != "rotary"
            or config.get("max_position_embeddings", 0) < maximum_tokens):
        raise RuntimeError("Accepted model architecture/token window does not support the long contract")


def validate_case(case):
    sequence, q = case["sequence"], case["q"]
    if (not isinstance(sequence, str) or not 257 <= len(sequence) <= 512
            or set(sequence) - set(krr.AA) or type(q) is not int or not 7 <= q < len(sequence) - 7
            or sequence[q] != "A" or case["token_q_0based"] != q + 1):
        raise ValueError("Long T sequence, center or token contract failed")


class LongNativeRunner(p3.NativeRunner):
    """Only new wrapper accepts long cases; the frozen runner is unmodified."""
    def __init__(self, model_path, ledger, release):
        if not release or release.get("status") != "T_OUTPUT_GO":
            raise RuntimeError("Verified T_OUTPUT_GO is required before the single model load")
        super().__init__(model_path, ledger, {"status": "FROZEN"})
        validate_token_capacity(self.model.config.to_dict())

    def run(self, case, purpose=PURPOSE, patch=None):
        validate_case(case)
        if patch is not None or purpose != PURPOSE:
            raise ValueError("Only natural single-MASK T observation is authorized")
        return super().run(case, purpose, patch=None)


def sealed_inputs():
    sealed = read(FRESH / "SEALED_FRESH_CASES.json")
    if (sealed["status"] != "INPUTS_SEALED_NO_MODEL_OUTPUT" or sealed["T_OUTPUT_authorized"] is not False
            or len(sealed["cases"]) != FORWARDS or len(sealed["pairs"]) != TARGET):
        raise RuntimeError("Expected the accepted sealed 1304-pair/2608-case long T input")
    return sealed


def prepare():
    if any((OUT / name).exists() for name in ("T_STARTED.json", "FORWARD_LEDGER.json", "T_NATIVE.npz", "T_SUMMARY.json")):
        raise RuntimeError("A prior T attempt prohibits preparation/resume")
    acceptance = read(FRESH / "FRESH_DATA_ACCEPTANCE.json")
    if acceptance["status"] != "FRESH_DATA_ACCEPTANCE_PASS_NO_MODEL_OUTPUT":
        raise RuntimeError("Accepted source/data gate required; do not reconstruct it")
    sealed = sealed_inputs()
    development, precision = read(D / "D_DEVELOPMENT.json"), read(D / "PRECISION_PLAN.json")
    if (development["status"] != "D_DEVELOPMENT_COMPLETE_NO_NEW_T" or development["real_fits_completed"] != 102
            or development["groups"] != 421 or precision["authorized_target_within_cap"] != TARGET
            or acceptance["D_groups_unchanged"] is not True or acceptance["selected_components"] != TARGET):
        raise RuntimeError("Frozen D/precision/fresh-group entry is inconsistent")
    validate_token_capacity(read(MODEL / "config.json"))
    test = read(OUT / "SYNTHETIC_CHECKS.json")
    if (test["status"] != "PASS" or test["script_sha256"] != sha(__file__)
            or test["test_sha256"] != sha(Path(__file__).with_name("test_protein_T_confirm.py"))):
        raise RuntimeError("Current finite synthetic checks required")
    identity = read(INITIAL)["model_identity"]
    if Path(identity["path"]).resolve() != MODEL.resolve():
        raise RuntimeError("Only the prior accepted local model identity is allowed")
    stats = {name: {"bytes": (MODEL / name).stat().st_size, "mtime_ns": (MODEL / name).stat().st_mtime_ns,
                   "accepted_sha256": value} for name, value in identity["files_sha256"].items()}
    ready = {"status": "PROTEIN_T_READY_NO_MODEL_OUTPUT", "created_utc": utc(), "config": CONFIG,
        "bindings": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)} for key, path in input_paths().items()},
        "model_identity": identity, "model_file_stats": stats, "runtime": runtime(),
        "selected_D_lambda": {m: development["models"][m]["selected_lambda"] for m in krr.MODELS},
        "D_block_scales": development["final_block_mean_diagonals"],
        "maximum_observed_tokens": max(len(c["sequence"]) + 2 for c in sealed["cases"]),
        "new_model_loads": 0, "new_real_PLM_forwards": 0, "T_predictions_read": 0,
        "source_selector_Dgroup_acceptance_repeated": False,
        "awaiting": "root T_OUTPUT_GO with ready_sha256, T_model_forward_authorized=true, T_readout_authorized=true, maximum_forwards=2608 and timestamp after READY"}
    write(OUT / "T_RUNNER_READY.json", ready)
    print(json.dumps({"status": ready["status"], "ready_sha256": sha(OUT / "T_RUNNER_READY.json")}))


def require_lock(path):
    lock, ready = read(path), read(OUT / "T_RUNNER_READY.json")
    if (lock.get("status") != "T_OUTPUT_GO" or lock.get("T_model_forward_authorized") is not True
            or lock.get("T_readout_authorized") is not True or lock.get("maximum_forwards") != FORWARDS
            or lock.get("new_fits_authorized") is not False
            or lock.get("ready_sha256") != sha(OUT / "T_RUNNER_READY.json") or ready["config"] != CONFIG):
        raise RuntimeError("Root-owned T output/forward/readout lock required")
    locked_at, ready_at = datetime.fromisoformat(lock["created_utc"]), datetime.fromisoformat(ready["created_utc"])
    if locked_at.utcoffset() is None or locked_at <= ready_at:
        raise RuntimeError("T lock must be prospectively issued after READY")
    for key, value in ready["bindings"].items():
        if sha(ROOT / value["path"]) != value["sha256"]:
            raise RuntimeError(f"T input/dependency drift: {key}")
    if runtime() != ready["runtime"]:
        raise RuntimeError("Python/dependency version identity changed")
    for name, value in ready["model_file_stats"].items():
        path_model = MODEL / name
        if path_model.stat().st_size != value["bytes"] or path_model.stat().st_mtime_ns != value["mtime_ns"]:
            raise RuntimeError("Accepted model/tokenizer stat drift")
    return lock


class CPUWatchdog:
    """Terminate this scientific process on its CPU cap; never restart it."""
    def __init__(self, phase, maximum, on_stop):
        self.phase, self.maximum, self.on_stop = phase, maximum, on_stop
        self.started = time.process_time()
        self.done = threading.Event()

    def __enter__(self):
        def watch():
            while not self.done.wait(.05):
                elapsed = time.process_time() - self.started
                if elapsed >= self.maximum:
                    try:
                        self.on_stop(elapsed)
                    finally:
                        os._exit(124)
        self.thread = threading.Thread(target=watch, name=f"protein-T-{self.phase}-CPU-limit", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()


class ForwardLedger:
    def __init__(self, cases):
        self.path, self.cases = OUT / "FORWARD_LEDGER.json", cases
        self.mutex = threading.RLock()
        self.started = time.process_time()
        self.state = {"status": "RUNNING", "created_utc": utc(), "sequence_forwards": 0,
            "completed_forwards": 0, "planned_forwards": FORWARDS, "maximum_forwards": FORWARDS,
            "maximum_model_CPU_seconds": MAX_MODEL_CPU, "model_cpu_seconds": 0.0,
            "model_forward_wall_seconds": 0.0, "model_phase_CPU_seconds": 0.0, "events": []}
        write(self.path, self.state, exclusive=True)

    def save(self):
        self.state["model_phase_CPU_seconds"] = time.process_time() - self.started
        write(self.path, self.state)

    def budget_stop(self, elapsed):
        with self.mutex:
            self.state.update(status="MODEL_CPU_HARD_STOP_NO_RETRY", model_phase_CPU_seconds=elapsed)
            write(self.path, self.state)
            write(OUT / "STOP_RECEIPT.json", {"status": self.state["status"], "created_utc": utc(),
                "model_CPU_seconds": elapsed, "no_partial_confirmation": True}, exclusive=True)

    def begin(self, purpose, case_id):
        with self.mutex:
            n = self.state["sequence_forwards"]
            if (n >= FORWARDS or n >= len(self.cases) or purpose != PURPOSE
                    or self.cases[n]["case_id"] != case_id):
                raise RuntimeError("Fixed forward count/order/purpose violated; no scientific retries")
            prior_max = max((r.get("model_cpu_seconds", 0) for r in self.state["events"]), default=0)
            if time.process_time() - self.started + 2 * prior_max >= MAX_MODEL_CPU:
                raise RuntimeError("Model CPU reserve exhausted before next charged forward")
            event = {"index": n + 1, "case_id": case_id, "q": self.cases[n]["q"],
                "token_q_0based": self.cases[n]["q"] + 1, "batch_size": 1, "purpose": purpose,
                "status": "RUNNING", "started_utc": utc()}
            self.state["sequence_forwards"] += 1
            self.state["events"].append(event)
            self.save()
            return event

    def end(self, event, elapsed_cpu, elapsed_wall, error=None):
        with self.mutex:
            event.update(status="FAILED" if error else "COMPLETE", finished_utc=utc(),
                model_cpu_seconds=float(elapsed_cpu), model_forward_wall_seconds=float(elapsed_wall))
            if error:
                event["exception_class"] = type(error).__name__
            else:
                self.state["completed_forwards"] += 1
            self.state["model_cpu_seconds"] += float(elapsed_cpu)
            self.state["model_forward_wall_seconds"] += float(elapsed_wall)
            self.save()
            if self.state["model_phase_CPU_seconds"] >= MAX_MODEL_CPU:
                raise RuntimeError("Model CPU cap crossed; stop without a retry")


def extraction(cases, lock):
    ledger = ForwardLedger(cases)
    vectors = []
    with CPUWatchdog("model", MAX_MODEL_CPU, ledger.budget_stop):
        cpu, wall = time.process_time(), time.perf_counter()
        runner = LongNativeRunner(str(MODEL), ledger, lock)
        write(OUT / "MODEL_LOAD.json", {"created_utc": utc(), "model_load_attempts": 1,
            "CPU_seconds": time.process_time()-cpu, "wall_seconds": time.perf_counter()-wall,
            "model_identity": read(INITIAL)["model_identity"]}, exclusive=True)
        try:
            for i, case in enumerate(cases):
                vectors.append(runner.run(case)["z"])
                if (i+1) % 64 == 0 or i+1 == len(cases):
                    print(json.dumps({"phase": "protein_T_model", "completed_cases": i+1,
                        "planned_cases": FORWARDS, "model_CPU_seconds": time.process_time()-ledger.started}), flush=True)
            if ledger.state["sequence_forwards"] != FORWARDS or ledger.state["completed_forwards"] != FORWARDS:
                raise RuntimeError("Incomplete extraction cannot enter confirmation")
            z = np.stack(vectors)
            if z.shape != (FORWARDS, 1280) or not np.isfinite(z).all():
                raise RuntimeError("Native full-vector cache shape/finite check failed")
            np.savez_compressed(OUT / "T_NATIVE.npz", z=z, case_ids=np.asarray([c["case_id"] for c in cases]))
            ledger.state["status"] = "COMPLETE_NO_RETRY"
            ledger.save()
            return z[:, 1246].astype(np.float64)
        except BaseException:
            if vectors:
                np.savez_compressed(OUT / "PARTIAL_NATIVE_NO_CONFIRMATION.npz", z=np.stack(vectors),
                    case_ids=np.asarray([c["case_id"] for c in cases[:len(vectors)]]))
            ledger.state["status"] = "FAILED_NO_RETRY"
            ledger.save()
            raise


def inference_from_losses(losses, groups):
    contributions = krr.paired_contributions(losses)
    group_ids, grouped = krr.group_average(contributions, groups)
    summary = krr.future_primary_summary(contributions, groups)
    for endpoint in summary["endpoints"]:
        ci = endpoint["CI_98_75_Bonferroni"]
        half = (ci[1]-ci[0])/2 if ci is not None else None
        endpoint.update(actual_simultaneous_CI_half_width=half,
            actual_half_width_le_0_01=bool(half is not None and half <= .01))
    return contributions, group_ids, grouped, summary


def readout(sealed, j):
    from threadpoolctl import threadpool_limits
    state = {"status": "RUNNING", "created_utc": utc(), "maximum_CPU_seconds": MAX_READOUT_CPU,
        "CPU_seconds": 0.0, "real_fits": 0, "predictions_completed": [], "model_forwards": 0}
    started = time.process_time()
    write(OUT / "READOUT_LEDGER.json", state, exclusive=True)
    def budget_stop(elapsed):
        state.update(status="READOUT_CPU_HARD_STOP_NO_RETRY", CPU_seconds=elapsed)
        write(OUT / "READOUT_LEDGER.json", state)
        write(OUT / "STOP_RECEIPT.json", {"status": state["status"], "created_utc": utc(),
            "readout_CPU_seconds": elapsed, "no_partial_confirmation": True}, exclusive=True)
    with CPUWatchdog("readout", MAX_READOUT_CPU, budget_stop), threadpool_limits(limits=THREADS):
        try:
            pair_inputs = krr.pair_data(sealed["cases"], sealed["pairs"], j)
            y = np.asarray([p["y_a"] for p in sealed["pairs"]], dtype=np.float64)
            groups = np.asarray([p["group"] for p in sealed["pairs"]])
            output, losses = {"y_a": y, "groups": groups, "j1246": j}, {}
            for model in krr.MODELS:
                score, probability = krr.predict_saved(pair_inputs, model, D / "FINAL_MODELS.npz")
                losses[model] = (probability - y)**2
                output.update({f"{model}_score": score, f"{model}_p_a": probability,
                               f"{model}_loss": losses[model]})
                state["predictions_completed"].append(model)
                state["CPU_seconds"] = time.process_time()-started
                write(OUT / "READOUT_LEDGER.json", state)
            contributions, group_ids, grouped, summary = inference_from_losses(losses, groups)
            output.update(pair_contributions=contributions, group_ids=group_ids, group_contributions=grouped,
                endpoint_order=np.asarray(krr.ENDPOINTS), case_ids=np.asarray([c["case_id"] for c in sealed["cases"]]))
            if any(not np.isfinite(value).all() for value in output.values() if np.asarray(value).dtype.kind in "fci"):
                raise RuntimeError("Nonfinite confirmation artifact")
            np.savez_compressed(OUT / "T_PREDICTIONS.npz", **output)
            state.update(status="COMPLETE_NO_REFIT", CPU_seconds=time.process_time()-started)
            write(OUT / "READOUT_LEDGER.json", state)
            if state["CPU_seconds"] >= MAX_READOUT_CPU:
                raise RuntimeError("Readout CPU cap crossed")
            return {"status": "PROTEIN_LONG_T_CONFIRMATION_COMPLETE", "created_utc": utc(),
                "pairs": TARGET, "cases": FORWARDS, "groups": len(group_ids), "primary": summary,
                "new_fits": 0, "secondary_H_regression": "not run", "full_Z_readout": "not run",
                "transport": "short D (64-256) to long T (257-512)",
                "precision": "Actual simultaneous half-widths reported; short-D planning does not guarantee long-T precision",
                "limitations": "Group t-reference approximation; known ID/exact-sequence/Pfam isolation does not rule out residual homology or pretraining overlap",
                "readout_CPU_seconds": state["CPU_seconds"]}
        except BaseException:
            state.update(status="FAILED_NO_RETRY", CPU_seconds=time.process_time()-started)
            write(OUT / "READOUT_LEDGER.json", state)
            raise


def run(lock_path):
    lock = require_lock(lock_path)
    if any((OUT / name).exists() for name in ("T_STARTED.json", "FORWARD_LEDGER.json", "T_NATIVE.npz", "T_SUMMARY.json")):
        raise RuntimeError("Existing T attempt forbids any automatic retry/resume")
    sealed = sealed_inputs()
    write(OUT / "T_STARTED.json", {"created_utc": utc(), "T_output_lock_sha256": sha(lock_path),
        "planned_model_loads": 1, "planned_sequence_forwards": FORWARDS, "new_fits": 0}, exclusive=True)
    try:
        j = extraction(sealed["cases"], lock)
        result = readout(sealed, j)
        result.update(model_phase_CPU_seconds=read(OUT / "FORWARD_LEDGER.json")["model_phase_CPU_seconds"],
            sequence_forwards=FORWARDS, native_cache_sha256=sha(OUT / "T_NATIVE.npz"),
            predictions_sha256=sha(OUT / "T_PREDICTIONS.npz"), T_output_lock_sha256=sha(lock_path))
        write(OUT / "T_SUMMARY.json", result, exclusive=True)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        if not (OUT / "STOP_RECEIPT.json").exists():
            write(OUT / "STOP_RECEIPT.json", {"status": "T_FAILED_NO_RETRY_NO_PARTIAL_CONFIRMATION", "created_utc": utc(),
                "exception_class": type(exc).__name__, "automatic_retry_authorized": False}, exclusive=True)
        raise


def selftest():
    import unittest
    import test_protein_T_confirm
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_protein_T_confirm))
    write(OUT / "SYNTHETIC_CHECKS.json", {"status": "PASS" if result.wasSuccessful() else "FAIL", "created_utc": utc(),
        "tests_run": result.testsRun, "script_sha256": sha(__file__),
        "test_sha256": sha(Path(__file__).with_name("test_protein_T_confirm.py")),
        "real_model_loads": 0, "real_PLM_forwards": 0, "new_fits": 0,
        "scope": "Finite synthetic wrapper/mask/count/CPU/array-prediction/fixed-family checks; no scientific input/activation/prediction reads"})
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selftest", "prepare", "run"))
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if args.command == "selftest": selftest()
    elif args.command == "prepare": prepare()
    elif args.lock is None: parser.error("run requires --lock <root T_OUTPUT_LOCK.json>")
    else: run(args.lock)


if __name__ == "__main__":
    main()
