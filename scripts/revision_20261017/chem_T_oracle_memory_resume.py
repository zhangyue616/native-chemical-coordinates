"""Post-output engineering adapter: release oracle-unused verified T inputs.

The original T runner, READY, science lock, AP kernels and CPU/memory caps remain
unchanged. This adapter must receive a separate one-time root execution GO.
"""
from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
from pathlib import Path

from numba import set_num_threads
import torch
from threadpoolctl import threadpool_limits

import chem_T_run as original

ENGINEERING = original.OUT / "oracle_memory_engineering"
READY = ENGINEERING / "ADAPTER_READY.json"
DROPPED_KEYS = ("rows", "audits")


def data_dependencies():
    """Read source structure only; never open scientific D/T arrays or inputs."""
    result = {}
    for function in (original.oracle_T, original.event_gate, original.require_cache):
        tree = ast.parse(inspect.getsource(function))
        keys = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == "data" and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
        if keys.intersection(DROPPED_KEYS):
            raise ValueError("original oracle directly depends on an input proposed for release")
        result[function.__name__] = sorted(keys)
    return result


def release_unused_inputs(data):
    removed = {key: key in data for key in DROPPED_KEYS}
    for key in DROPPED_KEYS:
        data.pop(key, None)
    gc.collect()
    return removed


def identities():
    files = [Path(__file__), original.READY, Path(original.__file__)]
    return {p.relative_to(original.ROOT).as_posix(): {
        "bytes": p.stat().st_size, "sha256": original.sha(p)} for p in files}


def prepare():
    # This is a syntax/key dependency check only, not another AP/model test.
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    dependencies = data_dependencies()
    required = set().union(*(set(v) for v in dependencies.values()))
    synthetic = {key: object() for key in required | set(DROPPED_KEYS)}
    kept = {key: synthetic[key] for key in required}
    removed = release_unused_inputs(synthetic)
    if not all(removed.values()) or any(synthetic[k] is not v for k, v in kept.items()):
        raise AssertionError("release changed an oracle-required input")
    record = {
        "status": "ENGINEERING_ADAPTER_READY_AWAIT_ROOT_ONCE_GO",
        "prepared_utc": original.utc(), "files": identities(),
        "original_T_ready_sha256": original.sha(original.READY),
        "original_oracle_dependencies": dependencies,
        "drop_only_after_original_load_T_validation": list(DROPPED_KEYS),
        "synthetic_syntax_and_required_key_identity": "PASS",
        "real_T_inputs_or_arrays_read_during_prepare": 0,
        "model_forward_calls": 0, "selection_or_bootstrap_calls": 0,
        "science_changes": "none; original oracle_T and its exact kernels/40bases/928candidates remain in use",
        "limits": {"threads": 4, "RSS_bytes": 2147483648,
                   "shared_statistics_CPU_seconds": 43200},
        "context": "pure input-memory release after fixed T outputs were seen; not a new science lock",
        "run_command": ".venv_chem/Scripts/python.exe -B scripts/revision_20261017/chem_T_oracle_memory_resume.py run",
    }
    original.dump(READY, record)
    return {"status": record["status"], "path": str(READY),
            "adapter_sha256": original.sha(Path(__file__)),
            "adapter_ready_sha256": original.sha(READY), "dependencies": dependencies}


def run(lock):
    ready = original.read_json(READY)
    if ready["files"] != identities() or ready["original_oracle_dependencies"] != data_dependencies():
        raise ValueError("engineering adapter or original frozen runner changed after review")
    binding, T_binding = original.require_T_lock(lock)
    budget = original.Budget(binding)
    receipt = {"status": "ENGINEERING_ORACLE_ADAPTER_STARTED", "started_utc": original.utc(),
        "adapter_sha256": original.sha(Path(__file__)), "adapter_ready_sha256": original.sha(READY),
        "original_T_ready_sha256": original.sha(original.READY), "binding": T_binding,
        "post_output_engineering_fix_only": True, "T_winners_C_dtype_replicates_changed": False}
    original.dump(ENGINEERING / "EXECUTION.json", receipt)
    try:
        with budget.track("statistics", "T_oracle_verified_input_load_and_unused_input_release"):
            data = original.load_T(binding, T_binding)
            budget.record["T_scientific_reads"] += 1
            receipt["before_input_release"] = original.memory_usage()
            receipt["removed_keys"] = release_unused_inputs(data)
            receipt["after_input_release"] = original.memory_usage()
        result = original.oracle_T(data, budget)
        receipt.update(status="PASS_ENGINEERING_ORACLE_MEMORY_RESUME", completed_utc=original.utc(),
                       oracle_result=result, CPU_ledger=budget.record)
        original.dump(ENGINEERING / "EXECUTION.json", receipt)
        return result
    except BaseException as error:
        receipt.update(status="FAIL_ENGINEERING_ORACLE_MEMORY_RESUME_PRESERVED", failed_utc=original.utc(),
                       error_type=type(error).__name__, message=str(error), CPU_ledger=budget.record)
        original.dump(ENGINEERING / "EXECUTION.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--lock", type=Path, default=original.DEFAULT_LOCK)
    args = parser.parse_args()
    ENGINEERING.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        result = prepare()
    else:
        torch.set_num_threads(4); torch.set_num_interop_threads(1); set_num_threads(4)
        with threadpool_limits(limits=4):
            result = run(args.lock)
    print(json.dumps(original.safe_json(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
