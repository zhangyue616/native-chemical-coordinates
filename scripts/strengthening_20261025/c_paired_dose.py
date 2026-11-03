"""Post hoc paired scalar-dose experiment; all writes stay in the new lane."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/revision_20261017"))
from chem_model import (ChembertaMTR, PRIMARY_LAYER, REPOSITORY, REVISION,
                        TOKENIZER_SPECIAL_IDS, batch_from_audits)

BASE = ROOT / "data/revision_20261017/chemistry"
OLD = ROOT / "results/revision_20261017/chemistry_T/secondary/mechanism"
PLAN = ROOT / "results/revision_20261017/chemistry_D/secondary/MECHANISM_D_PLAN.json"
OUT = ROOT / "results/strengthening_20261025/paired_dose"
DOC = ROOT / "docs/strengthening_20261025"
SCALES = np.asarray([.25, .5, 1.], np.float64)
SALT = b"eighth-dose-pair-v1-20261025:"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Budget:
    def __init__(self, stage):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "CPU_FORWARD_LEDGER.json"
        self.record = read(self.path) if self.path.exists() else {
            "CPU_limit_seconds": 3600, "molecule_evaluation_limit": 24000,
            "cumulative_CPU_seconds": 0., "molecule_evaluations": 0, "sessions": []}
        self.prior = self.record["cumulative_CPU_seconds"]
        self.stage, self.wall = stage, time.monotonic()
        self.record["active_stage"] = stage
        self.record["current_started_utc"] = utc()
        self.check()
        self.save()

    def check(self):
        if self.prior + time.process_time() >= 3600:
            raise TimeoutError("New C cumulative 3600 process-CPU seconds reached")
        if self.record["molecule_evaluations"] > 24000:
            raise RuntimeError("New C molecule evaluation cap exceeded")

    def reserve(self, n):
        self.check()
        if self.record["molecule_evaluations"] + n > 24000:
            raise RuntimeError("New C molecule evaluation cap would be exceeded")
        self.record["molecule_evaluations"] += n

    def save(self):
        self.record["cumulative_CPU_seconds"] = self.prior + time.process_time()
        self.record["updated_utc"] = utc()
        write(self.path, self.record)

    def finish(self, status):
        self.record.pop("active_stage", None)
        self.record["sessions"].append({"stage": self.stage, "status": status,
            "CPU_seconds": time.process_time(), "wall_seconds": time.monotonic() - self.wall})
        self.save()


def accepted_array(directory, name):
    receipt = read(directory / "CACHE_RECEIPT.json")
    path, meta = directory / name, receipt["files"][name]
    if path.stat().st_size != meta["bytes"] or path.stat().st_mtime_ns != meta["mtime_ns"]:
        raise ValueError("Accepted cache metadata changed: " + str(path))
    return np.load(path, mmap_mode="r", allow_pickle=False), meta


def input_rows():
    with (BASE / "sealed/T.tsv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 10000 or [int(r["rank"]) for r in rows] != list(range(5000, 15000)):
        raise ValueError("Frozen T row order changed")
    return rows


def prepare(budget):
    if (OUT / "READY.json").exists():
        raise RuntimeError("Preparation already frozen; do not replace it")
    plan, rows = read(PLAN), input_rows()
    if [o["concept_id"] for o in plan["objects"]] != [24, 25, 26]:
        raise ValueError("Fixed three-object order changed")
    native, native_meta = accepted_array(BASE / "activations_T/trained", "atom_a.npy")
    with np.load(BASE / "test/atoms_T.npz", allow_pickle=False) as z:
        offsets = z["mol_offsets"].copy()
    moments, moment_meta = accepted_array(BASE / "activations_D/trained", "D_moments.npz")
    norms = moments["output_dense_column_norm"].copy()
    means = moments["atom_mean"].copy()
    moments.close()
    manifests = []
    for obj in plan["objects"]:
        budget.check()
        cid = obj["concept_id"]
        old_path = OLD / f"MECHANISM_{cid}.npz"
        with np.load(old_path, allow_pickle=False) as z:
            old_indices, pos = z["T_molecule_indices"], z["positive"]
        if len(old_indices) != 400 or pos.sum() != 200 or len(set(old_indices.tolist())) != 400:
            raise ValueError("Original 200+200 subset differs")
        key = lambda i: (hashlib.sha256(SALT + str(int(rows[int(i)]["cid"])).encode("ascii")).digest(),
                         int(rows[int(i)]["cid"]), int(i))
        positive = sorted(old_indices[pos].tolist(), key=key)
        negative = sorted(old_indices[~pos].tolist(), key=key)
        indices = np.column_stack((positive, negative)).reshape(-1).astype(np.int32)
        units = np.asarray([obj["unit"], *obj["control_units"]], np.int32)
        mu64 = np.asarray([obj["atom_mean"], *obj["control_atom_means"]], np.float64)
        if len(set(units.tolist())) != 6 or not np.allclose(mu64, means[units], rtol=0, atol=1e-14):
            raise ValueError("D units/means differ")
        mu = mu64.astype(np.float32)
        full = np.empty((400, 6), np.float64)
        for k, mi in enumerate(indices):
            a = np.asarray(native[offsets[mi]:offsets[mi + 1], units], np.float64)
            full[k] = np.sqrt(np.mean((mu.astype(np.float64) - a) ** 2, axis=0)) * norms[units]
        tau = full.reshape(200, 2, 6).min(axis=(1, 2))
        target = SCALES[:, None] * np.repeat(tau, 2)[None, :]
        alpha = np.divide(target[:, :, None], full[None, :, :], out=np.zeros((3, 400, 6), np.float64), where=full[None, :, :] != 0)
        if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
            raise ValueError("Frozen alpha is not in [0,1]")
        if np.any((full == 0) & (target[-1, :, None] != 0)):
            raise ValueError("Zero full-dose has nonzero target")
        parameter_path = OUT / f"PARAMETERS_{cid}.npz"
        np.savez(parameter_path, T_molecule_indices=indices, units=units, means_D_float64=mu64,
                 means_model_float32=mu, outgoing_norms=norms[units], full_dose=full,
                 tau=tau, target=target, alpha=alpha, scales=SCALES)
        limiters = []
        for pair in range(200):
            ties = np.argwhere(full.reshape(200, 2, 6)[pair] == tau[pair])
            limiters.append({"pair": pair, "positive_T_index": int(indices[2 * pair]),
                "negative_T_index": int(indices[2 * pair + 1]), "tau": float(tau[pair]),
                "limiters": [{"stratum": "positive" if side == 0 else "negative",
                    "T_index": int(indices[2 * pair + side]), "CID": int(rows[indices[2 * pair + side]]["cid"]),
                    "unit": int(units[col]), "coordinate_role": "selected" if col == 0 else f"control{col}"}
                    for side, col in ties]})
        write(OUT / f"PAIR_LIMITERS_{cid}.json", limiters)
        manifests.append({"concept_id": cid, "q": obj["anonymous_output_q"], "units": units.tolist(),
            "parameter_file": parameter_path.name, "parameter_sha256": sha(parameter_path),
            "source_subset_sha256": sha(old_path), "pairs": 200, "true_zero_tau_pairs": int((tau == 0).sum()),
            "tau_below_1e_4_fraction": float(np.mean(tau < 1e-4)),
            "tau_quantiles": np.quantile(tau, [0, .01, .25, .5, .75, .99, 1]).tolist(),
            "alpha_quantiles_by_scale": [np.quantile(a, [0, .01, .25, .5, .75, .99, 1]).tolist() for a in alpha]})
    ready = {"status": "PAIRS_TAU_ALPHA_FROZEN_BEFORE_NEW_FORWARD", "created_utc": utc(),
        "design_sha256": sha(DOC / "DESIGN_DECISION.md"), "protocol_sha256": sha(DOC / "C_PAIRED_DOSE_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)), "D_plan_sha256": sha(PLAN), "T_rows_sha256": sha(BASE / "sealed/T.tsv"),
        "accepted_T_native_identity": native_meta, "accepted_D_moments_identity": moment_meta,
        "objects": manifests, "pair_salt": SALT.decode(), "bootstrap_seed": 20261025,
        "bootstrap_replicates": 2000, "primary_scale": 1., "scales": SCALES.tolist(),
        "new_model_evaluations_before_freeze": budget.record["molecule_evaluations"],
        "near_zero_pairs_deleted": False, "post_hoc": True}
    write(OUT / "READY.json", ready)
    return ready


class PerMoleculeHook:
    def __init__(self, mask, unit, mean, alpha, norm):
        self.mask, self.unit, self.mean, self.alpha, self.norm = mask, int(unit), float(mean), alpha, float(norm)
        self.original = self.applied = self.dose = None

    def __call__(self, module, inputs, activation):
        if self.mask.shape != activation.shape[:2] or activation.shape[-1] != 464:
            raise ValueError("Actual hook geometry differs")
        original = activation[..., self.unit]
        a64 = self.alpha.to(dtype=torch.float64, device=activation.device)[:, None]
        changed = activation.clone()
        interpolated = (original.to(torch.float64) + a64 * (self.mean - original.to(torch.float64))).to(activation.dtype)
        changed[..., self.unit] = torch.where(self.mask, interpolated, original)
        difference = changed[..., self.unit].to(torch.float64) - original.to(torch.float64)
        self.dose = torch.sqrt(difference.square().sum(1) / self.mask.sum(1)) * self.norm
        self.original, self.applied = activation.detach().clone(), changed.detach().clone()
        return changed


def actual_forward(model, batch, params, column, alpha, budget):
    budget.reserve(len(batch.input_ids))
    hook = PerMoleculeHook(batch.scope("atoms"), params["units"][column], params["means_model_float32"][column],
                          torch.as_tensor(alpha, dtype=torch.float64), params["outgoing_norms"][column])
    handle = model.roberta.encoder.layer[PRIMARY_LAYER].intermediate.register_forward_hook(hook)
    try:
        result = model(batch, layer_index=PRIMARY_LAYER)
    finally:
        handle.remove()
    if hook.dose is None:
        raise RuntimeError("Post-GELU hook did not execute")
    return result, hook


def selected_audits(wanted):
    audits = {}
    with gzip.open(BASE / "sealed/anchors_T.jsonl.gz", "rt", encoding="utf-8") as stream:
        for mi, line in enumerate(stream):
            if mi not in wanted:
                continue
            a = json.loads(line)
            audits[mi] = {**a, "accepted": a["gate"] == "PASS_POSITIONAL_ELEMENT_ANCHOR",
                "tokenizer": {"repository": REPOSITORY, "revision": REVISION},
                "special_tokens_mask": [int(x in TOKENIZER_SPECIAL_IDS) for x in a["input_ids"]],
                "graph_atom_count": len(a["atoms"])}
    if set(audits) != set(wanted):
        raise ValueError("Required existing anchors missing")
    return audits


def checked_dose(actual, target):
    error = np.abs(np.asarray(actual, np.float64) - target)
    tolerance = 1e-6 * (1 + float(np.max(target)))
    if not np.isfinite(error).all() or float(error.max()) > tolerance:
        raise ArithmeticError(f"Realized dose error {error.max()} exceeds {tolerance}")
    return error


def run(budget):
    ready = read(OUT / "READY.json")
    script_sha = sha(Path(__file__))
    amendment = read(OUT / "engineering_recovery/AMENDMENT.json") if (OUT / "engineering_recovery/AMENDMENT.json").exists() else {}
    geometry_dir = OUT / "geometry_recovery"
    geometry_amendment = read(geometry_dir / "AMENDMENT.json") if (geometry_dir / "AMENDMENT.json").exists() else {}
    accepted_code = (ready["script_sha256"] == script_sha or
                     (amendment.get("original_script_sha256") == ready["script_sha256"] and
                      amendment.get("revised_script_sha256") == script_sha) or
                     (geometry_amendment.get("parent_script_sha256") == amendment.get("revised_script_sha256") and
                      geometry_amendment.get("revised_script_sha256") == script_sha))
    if not accepted_code or ready["protocol_sha256"] != sha(DOC / "C_PAIRED_DOSE_PROTOCOL.md"):
        raise ValueError("Code/protocol changed after alpha freeze; preserve and explain before retry")
    restored = None
    if geometry_amendment:
        restoration = read(geometry_dir / "RESTORATION.json")
        if restoration["status"] != "ALL18_RESTORED_NUMERICALLY" or sha(geometry_dir / "RESTORED_ARRAYS.npz") != restoration["restored_arrays_sha256"]:
            raise ValueError("One-case numerical restoration differs")
        with np.load(geometry_dir / "RESTORED_ARRAYS.npz", allow_pickle=False) as z:
            restored = {key: z[key].copy() for key in z.files}
    parameters = []
    for item in ready["objects"]:
        path = OUT / item["parameter_file"]
        if sha(path) != item["parameter_sha256"]:
            raise ValueError("Frozen alpha parameters changed")
        with np.load(path, allow_pickle=False) as z:
            parameters.append({key: z[key].copy() for key in z.files})
    audits = selected_audits({int(i) for p in parameters for i in p["T_molecule_indices"]})
    cached_head, _ = accepted_array(BASE / "activations_T/trained", "normalized_head.npy")
    model = ChembertaMTR.from_official_checkpoint()
    engineering_path = OUT / "ENGINEERING_CHECK.json"
    if engineering_path.exists():
        if read(engineering_path)["status"] != "PASS":
            raise AssertionError("Existing engineering failure must be addressed explicitly")
    else:
        checks = {}
        p = parameters[0]
        ix = p["T_molecule_indices"][:4]
        batch = batch_from_audits([audits[int(i)] for i in ix])
        budget.reserve(4)
        baseline = model(batch, layer_index=PRIMARY_LAYER)
        zero, hook = actual_forward(model, batch, p, 0, np.zeros(4), budget)
        checks["alpha0_output_exact"] = bool(torch.equal(zero.normalized_predictions, baseline.normalized_predictions))
        checks["alpha0_activation_exact"] = bool(torch.equal(hook.original, hook.applied))
        result, hook = actual_forward(model, batch, p, 0, p["alpha"][2, :4, 0], budget)
        allowed = torch.zeros_like(hook.original, dtype=torch.bool)
        allowed[..., int(p["units"][0])] = batch.atom_mask
        checks["only_selected_atom_coordinate_changed"] = bool(torch.equal(hook.original[~allowed], hook.applied[~allowed]))
        checks["intermediate_is_real_post_GELU"] = bool(torch.equal(hook.applied, result.activations))
        error = checked_dose(hook.dose.numpy(), p["target"][2, :4])
        checks["dose_absolute_error_max"] = float(error.max())
        checks["baseline_vs_accepted_head_max_abs"] = float(np.max(np.abs(baseline.normalized_predictions.numpy().astype(np.float64) - cached_head[ix])))
        checks["baseline_within_existing_tolerance_1e_5"] = checks["baseline_vs_accepted_head_max_abs"] <= 1e-5
        if not all(checks[k] for k in ("alpha0_output_exact", "alpha0_activation_exact", "only_selected_atom_coordinate_changed",
                                      "intermediate_is_real_post_GELU", "baseline_within_existing_tolerance_1e_5")):
            write(engineering_path, {"status": "FAIL", "checks": checks})
            raise AssertionError("C small-batch real-forward precondition failed")
        write(engineering_path, {"status": "PASS", "checks": checks, "new_evaluations": 12})
    budget.save()
    reports = []
    rng = np.random.default_rng(20261025)
    weights = np.asarray([np.bincount(rng.integers(0, 200, size=200), minlength=200) for _ in range(2000)], np.int32)
    np.save(OUT / "PAIR_BOOTSTRAP_WEIGHTS.npy", weights)
    old_summary = read(OLD / "MECHANISM_T_RESULTS.json")
    for oi, (item, p) in enumerate(zip(ready["objects"], parameters)):
        cid, q = item["concept_id"], item["q"]
        dest = OUT / f"concept_{cid}"
        dest.mkdir(exist_ok=True)
        if (dest / "SUMMARY.json").exists():
            reports.append(read(dest / "SUMMARY.json"))
            continue
        indices = p["T_molecule_indices"]
        if (dest / "BASELINE.npy").exists():
            new_base = np.load(dest / "BASELINE.npy", allow_pickle=False)
        else:
            new_base = np.empty((400, 199), np.float32)
            for begin in range(0, 400, 4):
                selected = indices[begin:begin + 4]
                batch = batch_from_audits([audits[int(i)] for i in selected])
                budget.reserve(len(selected))
                new_base[begin:begin + 4] = model(batch, layer_index=PRIMARY_LAYER).normalized_predictions.numpy()
            np.save(dest / "BASELINE.npy", new_base)
        if new_base.shape != (400, 199) or new_base.dtype != np.float32 or not np.isfinite(new_base).all():
            raise ValueError("Saved same-batch baseline is incomplete")
        effective_base = new_base.copy()
        if cid == 26 and restored is not None:
            if int(indices[int(restored["row"])]) != int(restored["T_index"]):
                raise ValueError("Restored row does not match frozen pair order")
            effective_base[int(restored["row"])] = restored["baseline"]
        baseline_error = float(np.max(np.abs(effective_base.astype(np.float64) - cached_head[indices])))
        baseline_target_error = float(np.max(np.abs(effective_base[:, q].astype(np.float64) - cached_head[indices, q])))
        if baseline_target_error > 1e-5:
            raise ArithmeticError("Frozen target q baseline differs above existing absolute tolerance")
        all_delta, all_dose = np.empty((3, 400, 6), np.float64), np.empty((3, 400, 6), np.float64)
        geometry_trace = []
        for si, scale in enumerate(SCALES):
            for column in range(6):
                condition_path = dest / f"s{scale:g}_unit{int(p['units'][column])}.npy"
                if condition_path.exists():
                    arr = np.load(condition_path, allow_pickle=False)
                    if arr.shape != (400, 2) or not np.isfinite(arr).all():
                        raise ValueError("Saved intervention condition incomplete")
                else:
                    arr = np.empty((400, 2), np.float64)
                    for begin in range(0, 400, 4):
                        selected = indices[begin:begin + 4]
                        batch = batch_from_audits([audits[int(i)] for i in selected])
                        result, hook = actual_forward(model, batch, p, column, p["alpha"][si, begin:begin + 4, column], budget)
                        arr[begin:begin + 4, 0] = result.normalized_predictions[:, q].numpy().astype(np.float64) - new_base[begin:begin + 4, q].astype(np.float64)
                        arr[begin:begin + 4, 1] = hook.dose.numpy()
                    np.save(condition_path, arr)
                if cid == 26 and restored is not None:
                    # Keep every original condition file. Only the integrated result
                    # uses the single molecule's consistently restored geometry.
                    row = int(restored["row"])
                    before = arr[row].copy()
                    arr = arr.copy()
                    arr[row] = restored["responses"][si, column]
                    geometry_trace.append({"s": float(scale), "unit": int(p["units"][column]),
                        "original_file": condition_path.name, "before": before.tolist(), "after": arr[row].tolist(),
                        "before_dose_error": float(abs(before[1]-p["target"][si,row])),
                        "after_dose_error": float(abs(arr[row,1]-p["target"][si,row]))})
                all_delta[si, :, column], all_dose[si, :, column] = arr[:, 0], arr[:, 1]
                checked_dose(arr[:, 1], p["target"][si])
                budget.save()
                print(json.dumps({"concept": cid, "s": float(scale), "column": column, "CPU": budget.record["cumulative_CPU_seconds"]}), flush=True)
        if geometry_trace:
            write(geometry_dir / "APPLIED_TRACE.json", {"row": int(restored["row"]), "T_index": int(restored["T_index"]),
                "baseline_original": new_base[int(restored["row"])].tolist(),
                "baseline_restored": effective_base[int(restored["row"])].tolist(), "conditions": geometry_trace})
        paired = np.abs(all_delta[:, 0::2]) - np.abs(all_delta[:, 1::2])
        contrasts = paired.mean(1)
        H = contrasts[:, 0] - np.median(contrasts[:, 1:], axis=1)
        boot_components = weights.astype(np.float64) @ paired[2] / 200.
        H_reps = boot_components[:, 0] - np.median(boot_components[:, 1:], axis=1)
        interval = np.quantile(H_reps, [1 / 120, 119 / 120], method="linear")
        target = np.broadcast_to(p["target"][:, :, None], all_dose.shape)
        errors = np.abs(all_dose - target)
        relative = errors[target > 1e-8] / target[target > 1e-8]
        np.savez(dest / "RESULT_ARRAYS.npz", T_molecule_indices=indices, units=p["units"],
                 deltas=all_delta, realized_doses=all_dose, targets=target,
                 paired_absolute_contrasts=paired, H_by_scale=H, H_pair1_replicates=H_reps)
        report = {"concept_id": cid, "q": q, "units": p["units"].tolist(), "pairs": 200,
            "H_pair1": float(H[2]), "H_pair1_CI98_3333333333": interval.tolist(),
            "H_by_scale_descriptive": H.tolist(), "scales": SCALES.tolist(),
            "mean_absolute_response_positive": np.abs(all_delta[:, 0::2]).mean(1).tolist(),
            "mean_absolute_response_negative": np.abs(all_delta[:, 1::2]).mean(1).tolist(),
            "signed_mean_response_positive": all_delta[:, 0::2].mean(1).tolist(),
            "signed_mean_response_negative": all_delta[:, 1::2].mean(1).tolist(),
            "dose_max_absolute_error": float(errors.max()), "dose_median_absolute_error": float(np.median(errors)),
            "dose_error_absolute_tolerance": 1e-6 * (1 + float(target.max())),
            "dose_target_gt_1e_8_relative_error_max": float(relative.max()) if len(relative) else None,
            "dose_target_gt_1e_8_relative_error_median": float(np.median(relative)) if len(relative) else None,
            "baseline_vs_old_all199_max_abs": baseline_error, "baseline_vs_old_target_q_max_abs": baseline_target_error,
            "defined_bootstrap_replicates": int(np.isfinite(H_reps).sum()),
            "old_H_raw_historical_only": old_summary["objects"][oi]["effect"]["H"],
            "old_H_interval_historical_only": old_summary["objects"][oi]["effect"]["H_interval"]["ci"],
            "parameter_diagnostics": item, "status": "COMPLETE",
            "numeric_geometry_amendment": "geometry_recovery/AMENDMENT.json" if geometry_trace else None,
            "dose_gate_scope": "each scale x coordinate condition separately, original fixed tolerance"}
        reports.append(report)
        write(dest / "SUMMARY.json", report)
    summary = {"status": "PAIRED_DOSE_COMPLETE", "objects": reports, "post_hoc": True,
        "primary_family": "3 H_pair(s=1), each 98.3333333333% percentile; not joint with old H or rho",
        "bootstrap": {"seed": 20261025, "replicates": 2000, "cluster": "fixed artificial positive/negative pair"},
        "interpretation": "conditional on fixed pairs and min attainable scalar dose; not matched perturbation vectors, biochemical mechanism, mediation fraction or equivalence",
        "forward_evaluations": budget.record["molecule_evaluations"]}
    write(OUT / "C_SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    budget = Budget(args.stage)
    try:
        with threadpool_limits(limits=4):
            result = prepare(budget) if args.stage == "prepare" else run(budget)
        budget.finish("COMPLETE")
        print(json.dumps({"status": result["status"], "CPU": budget.record["cumulative_CPU_seconds"],
                          "forward_evaluations": budget.record["molecule_evaluations"]}), flush=True)
    except BaseException as error:
        write(OUT / ("FAILURE_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"),
              {"stage": args.stage, "error": type(error).__name__, "message": str(error), "created_utc": utc()})
        budget.finish("FAILED_RETAINED")
        raise


if __name__ == "__main__":
    main()
