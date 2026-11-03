"""Root-locked T finite-input and anonymous-mechanism secondary quantities.

``prepare`` reads implementation/engineering metadata only. Actual stages are
serial callers of the original D/T shared CPU ledger. No table, winner, output,
control, concept or sample size is chosen using T.
"""
from __future__ import annotations
import argparse
import collections
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numba import set_num_threads
import torch
from threadpoolctl import threadpool_limits

from chem_input_reference import input_keys, predict_one
from chem_model import ChembertaMTR, ActivationReplacement, batch_from_audits
from chem_secondary_statistics import panel_brier, output_association, mechanism_effect
from chem_statistics import average_precision
from chem_T_intervention import coordinate_forward
from chem_T_run import require_T_lock, load_T, require_cache
from chem_run import ROOT, DOCS, BASE, Budget, dump, read_json, safe_json, sha, utc

ENGINEERING = ROOT / "results/revision_20261017/engineering"
READY = ENGINEERING / "T_SECONDARY_READY.json"
CHECKS = ENGINEERING / "T_SECONDARY_CHECKS.json"
TROOT = ROOT / "results/revision_20261017/chemistry_T"
OUT = TROOT / "secondary"
DROOT = ROOT / "results/revision_20261017/chemistry_D"
DFREEZE = DROOT / "secondary/D_SECONDARY_FREEZE.json"
ACT = BASE / "activations_T/trained"
DEFAULT_LOCK = DOCS / "D_OUTPUT_LOCK.json"
SALT = b"native-axis-mechanism-20261017-v1:"
CONFIG = {"schema": "chemistry-T-secondary/1", "threads": 4, "batch_size": 4,
    "panel_probabilities_and_Brier": "float64", "descriptive_AP_scores_and_ties": "float32 exact equality",
    "panel_bootstrap": {"replicates": 2000, "seed": 42017, "reuse_primary_weights": True},
    "rho_bootstrap": {"replicates": 2000, "seed": 45018, "population": "all10000 T molecules"},
    "H_bootstrap": {"replicates": 2000, "seed": 45017, "stratified": True},
    "mechanism_family_size_fixed": 3, "mechanism_quantiles": [0.05 / 6, 1 - 0.05 / 6],
    "joint_mechanism": "H positive lower bound AND D-sign rho positive lower bound and no undefined rho replicate",
    "six_intervals_joint_95_claimed": False, "mechanism_salt_ascii": SALT.decode("ascii"),
    "mechanism_per_stratum_min": 50, "mechanism_per_stratum_max": 200,
    "conditions_per_object": 8, "new_molecule_evaluation_cap": 10000,
    "structural_zero": {"first_T_molecules": 400, "continuous_batch_size": 4, "max_abs_tolerance": 1e-5},
    "shared_encoder_CPU_cap_seconds": 14400, "shared_statistics_CPU_cap_seconds": 43200,
    "extra_sampling_or_object_replacement": False}


def write(path, value):
    dump(path, safe_json(value))


def own_identity():
    files = [Path(__file__), Path(__file__).with_name("chem_T_intervention.py"),
        Path(__file__).with_name("test_chem_T_secondary.py"),
        *[Path(__file__).with_name(n) for n in ("chem_D_secondary.py", "chem_input_reference.py",
           "chem_secondary_statistics.py", "chem_mechanism_plan.py", "chem_model.py", "chem_T_run.py")],
        DOCS / "CHEMISTRY_CONFIRMATION_PROTOCOL.md", DOCS / "CHEMISTRY_T_OPERATIONAL_SUPPLEMENT.md",
        CHECKS, ENGINEERING / "SECONDARY_STATISTICS_CHECKS.json", DROOT / "D_RUNNER_READY.json"]
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}


def prepare():
    if DEFAULT_LOCK.exists():
        raise ValueError("existing root D_OUTPUT_LOCK cannot be rebound by prepare")
    if read_json(CHECKS).get("status") != "PASS_SYNTHETIC_T_SECONDARY":
        raise ValueError("new hook/caller synthetic checks not PASS")
    if read_json(ENGINEERING / "SECONDARY_STATISTICS_CHECKS.json").get("status") != "PASS_SYNTHETIC_ONLY":
        raise ValueError("existing six pure-statistics checks not PASS")
    value = {"status": "READY_T_SECONDARY_AWAIT_ROOT_D_OUTPUT_LOCK", "created_utc": utc(),
        "config": CONFIG, "files": own_identity(), "runtime_versions": read_json(DROOT / "D_RUNNER_READY.json")["versions"],
        "actual_D_or_T_scientific_reads": 0, "actual_T_forwards": 0,
        "lock_fields": ["T_secondary_ready_sha256", "D_secondary_freeze_sha256"],
        "parent_lock": "chem_T_run.require_T_lock including explicit T_GO, D precision and data/winner/runner READY hashes",
        "future_inputs": {"D_secondary_freeze": str(DFREEZE.relative_to(ROOT)),
             "trained_cache": str(ACT.relative_to(ROOT)),
             "weights": str((TROOT / "T_BOOTSTRAP_WEIGHTS.npy").relative_to(ROOT))},
        "commands_after_root_T_GO": [f".venv_chem/Scripts/python.exe -B scripts/revision_20261017/chem_T_secondary.py {s} --lock docs/revision_20261017/D_OUTPUT_LOCK.json" for s in ("input-T", "mechanism-T")]}
    write(READY, value)
    return {"status": value["status"], "path": str(READY), "T_secondary_ready_sha256": sha(READY)}


def require_secondary_lock(path):
    lock = read_json(path)
    if lock.get("status") != "D_OUTPUT_LOCKED" or lock.get("T_scientific_processing_authorized") is not True:
        raise ValueError("root D_OUTPUT_LOCK and explicit T science GO required")
    if lock.get("T_secondary_ready_sha256") != sha(READY):
        raise ValueError("T secondary READY binding differs")
    ready = read_json(READY)
    if ready["config"] != CONFIG or ready["files"] != own_identity():
        raise ValueError("T secondary implementation/protocol drift after READY")
    binding, T_binding = require_T_lock(path)
    if lock.get("D_secondary_freeze_sha256") != sha(DFREEZE):
        raise ValueError("D secondary freeze binding differs")
    freeze = read_json(DFREEZE)
    if freeze["status"] != "D_SECONDARY_FROZEN_AWAIT_ROOT_D_OUTPUT_LOCK" or freeze["binding"] != binding:
        raise ValueError("D secondary freeze belongs to another run")
    if freeze["D_final_winners_sha256"] != lock["D_final_winners_sha256"]:
        raise ValueError("D secondary/final winner identity differs")
    tables = []
    for item in freeze["readout_tables"]:
        p = (ROOT / item["file"]).resolve()
        if DFREEZE.parent.resolve() not in p.parents or sha(p) != item["sha256"]:
            raise ValueError("frozen D readout table identity/path differs")
        table = read_json(p)
        if table["D_binding"] != binding or table["D_final_winners_sha256"] != lock["D_final_winners_sha256"]:
            raise ValueError("D readout table binding differs")
        if table["concept_id"] != item["concept_id"] or table["D_selected_candidate"] != item["candidate"]:
            raise ValueError("D table concept/candidate join differs")
        tables.append((item, table))
    if [item["concept_id"] for item, _ in tables] != freeze["panel_ids"] or not 0 < len(tables) <= 12:
        raise ValueError("frozen panel table coverage/order differs")
    plan_path = (ROOT / freeze["mechanism_plan"]["file"]).resolve()
    if DFREEZE.parent.resolve() not in plan_path.parents or sha(plan_path) != freeze["mechanism_plan"]["sha256"]:
        raise ValueError("D mechanism plan identity/path differs")
    plan = read_json(plan_path)
    if plan["binding"] != binding or plan["D_final_winners_sha256"] != lock["D_final_winners_sha256"]:
        raise ValueError("D mechanism plan binding differs")
    if len(plan["objects"]) > 3 or [o["concept_id"] for o in plan["objects"]] != plan["predetermined_concepts"]:
        raise ValueError("D mechanism object identity differs")
    original_panel = read_json(BASE / "development/panel.json")
    if freeze["panel_ids"] != original_panel["panel_ids"] or plan["predetermined_concepts"] != original_panel["mechanism_candidate_ids"]:
        raise ValueError("D secondary changed the originally frozen panel/mechanism candidates")
    return binding, T_binding, freeze, tables, plan


def identity(data, freeze):
    return {"T_binding": data["T_binding"], "T_secondary_ready_sha256": sha(READY),
        "D_secondary_freeze_sha256": sha(DFREEZE), "T_data_receipt_sha256": sha(BASE / "test/T_DATA_RECEIPT.json"),
        "D_final_winners_sha256": freeze["D_final_winners_sha256"]}


def trained_cache(data, name):
    receipt = read_json(ACT / "CACHE_RECEIPT.json")
    if receipt["status"] != "PASS_T_EXTRACTION" or receipt["binding"] != data["binding"] or receipt["T_binding"] != data["T_binding"]:
        raise ValueError("trained T cache binding differs")
    target = ACT / name
    evidence = receipt["files"][name]
    if target.stat().st_size != evidence["bytes"] or target.stat().st_mtime_ns != evidence["mtime_ns"]:
        raise ValueError("accepted trained T cache changed")
    value = np.load(target, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.float32 or (name == "normalized_head.npy" and value.shape != (10000, 199)):
        raise ValueError("trained T cache shape/dtype differs")
    return value


def original_occurrences(data):
    path = BASE / "test/occurrence_counts_T.npz"
    evidence = read_json(BASE / "test/T_DATA_RECEIPT.json")["outputs"][path.relative_to(ROOT).as_posix()]
    if path.stat().st_size != evidence["bytes"] or sha(path) != evidence["sha256"]:
        raise ValueError("T occurrence identity differs")
    with np.load(path, allow_pickle=False) as saved:
        counts, ids, smarts = saved["counts"], saved["concept_ids"].tolist(), saved["is_smarts"]
    if counts.shape != (10000, 91):
        raise ValueError("T original occurrence geometry differs")
    return counts, ids, smarts


def frozen_weights(data):
    path = TROOT / "T_BOOTSTRAP_WEIGHTS.npy"
    receipt = read_json(path.with_suffix(".json"))
    if receipt["binding"] != data["T_binding"] or receipt["seed"] != 42017 or receipt["sha256"] != sha(path):
        raise ValueError("primary shared T bootstrap weights differ; never regenerate here")
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.int32 or weights.shape != (2000, 10000) or not np.all(weights.sum(1) == 10000):
        raise ValueError("shared T weights schema differs")
    return weights


def input_T(data, freeze, tables, budget):
    destination = OUT / "input"; destination.mkdir(parents=True, exist_ok=True)
    binding = identity(data, freeze)
    result_path = destination / "INPUT_T_RESULTS.json"
    if result_path.exists():
        saved = read_json(result_path)
        if saved["identity"] != binding:
            raise ValueError("completed input-T identity differs")
        return saved
    with budget.track("statistics", "T_frozen_input_predictions_paired_Brier"):
        native = require_cache(data, "trained")
        token_ids = [[a["input_ids"][atom["anchor_token_index"]] for atom in a["atoms"]] for a in data["audits"]]
        keys = input_keys([r["smiles"] for r in data["rows"]], token_ids)
        panel_ids = freeze["panel_ids"]
        columns = [data["concept_ids"].tolist().index(cid) for cid in panel_ids]
        labels = np.ascontiguousarray(data["labels"][:, columns])
        path, progress_path = destination / "predictions_T.npy", destination / "PROGRESS.json"
        shape = (len(labels), len(panel_ids), 4)
        if progress_path.exists():
            progress = read_json(progress_path)
            if progress["identity"] != binding:
                raise ValueError("partial input-T identity differs")
            prediction = np.load(path, mmap_mode="r+", allow_pickle=False)
        else:
            if path.exists():
                raise ValueError("orphan input predictions retained; inspect before recovery")
            prediction = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
            progress = {"identity": binding, "completed_concepts": [], "concept_reports": []}
            write(progress_path, progress)
        if prediction.shape != shape or prediction.dtype != np.float64:
            raise ValueError("prediction array schema differs")
        if progress["completed_concepts"] != panel_ids[:len(progress["completed_concepts"])]:
            raise ValueError("input prediction checkpoint is not a completed frozen-panel prefix")
        for column, (item, table) in enumerate(tables):
            if item["concept_id"] in progress["completed_concepts"]:
                continue
            budget.check()
            candidate = item["candidate"]
            winner_column = data["concept_ids"].tolist().index(item["concept_id"])
            if not 0 <= candidate < 928 or candidate != int(data["winners"][0, 0, winner_column]):
                raise ValueError("invalid frozen native candidate")
            scalar = np.asarray(native[:, candidate % 464]) * np.float32(1 if candidate < 464 else -1)
            p, coverage = predict_one(table, keys, scalar)
            prediction[:, column] = p; prediction.flush()
            ap = [float(average_precision(p[:, k].astype(np.float32), labels[:, column])[0]) for k in range(4)]
            bin_counts = np.bincount(np.searchsorted(table["bin_cuts"], scalar.astype(np.float64), side=table["bin_side"]), minlength=len(table["bin_cuts"]) + 1)
            progress["concept_reports"].append({"concept_id": item["concept_id"], "candidate": candidate,
                "D_table": item, "D_category_counts": table["category_counts"],
                "D_observed_bin_count": table["observed_bin_count"], "T_bin_atom_counts": bin_counts.tolist(),
                "coverage": coverage, "descriptive_AP_float32_probability_scores": ap,
                "AP_columns": ["token", "radius0", "radius1", "radius1_plus_fixed_native_j"]})
            progress["completed_concepts"].append(item["concept_id"]); write(progress_path, progress)
        weights = frozen_weights(data)
        statistics = panel_brier(labels, prediction, data["molecules"], weights)
        replicated = statistics.pop("replicated_panel")
        bootstrap_path = destination / "PANEL_BRIER_BOOTSTRAP.npy"; np.save(bootstrap_path, replicated)
        result = {"status": "INPUT_T_SECONDARY_COMPLETE", "identity": binding,
            "panel_ids": panel_ids, "concepts": progress["concept_reports"], "statistics": statistics,
            "prediction_schema": {"shape": list(shape), "dtype": "float64", "axis": "T atoms,panel order,token/r0/L/Lj"},
            "AP_conversion": "probabilities converted float64 to float32 solely for descriptive AP exact ties",
            "D_tables_refit": False, "T_tuning": False, "new_model_evaluations": 0,
            "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in (path, bootstrap_path)}}
        write(result_path, result)
    return result


def mechanism_subset(rows, counts):
    values = np.asarray(counts)
    if values.shape != (len(rows),) or np.any(values < 0):
        raise ValueError("SMARTS occurrence values required")
    ordered = sorted(range(len(rows)), key=lambda i: (hashlib.sha256(SALT + str(rows[i]["cid"]).encode("ascii")).digest(), int(rows[i]["cid"])))
    positive = [i for i in ordered if values[i] > 0][:200]
    negative = [i for i in ordered if values[i] == 0][:200]
    return positive + negative, np.r_[np.ones(len(positive), bool), np.zeros(len(negative), bool)]


def reserve_evaluations(budget, n):
    budget.check()
    total = budget.record.get("T_secondary_molecule_evaluations_reserved", 0) + n
    if total > 10000:
        raise RuntimeError("10,000 new molecule evaluation cap reached; no extension")
    budget.record["T_secondary_molecule_evaluations_reserved"] = total
    budget.record["model_forward_molecule_evaluations"] += n
    # Reserve before forward: a crash never makes an uncheckpointed call free.
    write(budget.path, budget.record)


def condition_array(destination, name, n, binding):
    path, progress_path = destination / f"{name}.npy", destination / f"{name}.json"
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress["identity"] != binding:
            raise ValueError("intervention condition identity differs")
        value = np.load(path, mmap_mode="r+", allow_pickle=False)
        if progress.get("status") == "PASS" and progress["sha256"] != sha(path):
            raise ValueError("completed intervention condition changed")
    else:
        if path.exists():
            raise ValueError("orphan condition array retained; bounded recovery needed")
        value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(n, 2))
        progress = {"identity": binding, "completed_molecules": 0, "status": "STARTED"}
        write(progress_path, progress)
    if value.shape != (n, 2) or value.dtype != np.float64 or not 0 <= progress["completed_molecules"] <= n:
        raise ValueError("condition checkpoint geometry differs")
    return value, progress, path, progress_path


def final_layer_zero(model, data, baseline, budget, binding):
    destination = OUT / "mechanism"; destination.mkdir(parents=True, exist_ok=True)
    contract = {**binding, "check": CONFIG["structural_zero"], "baseline": "cached trained full-T normalized outputs"}
    errors, progress, path, progress_path = condition_array(destination, "FINAL_LAYER_ZERO", 400, contract)
    with budget.track("encoder", "T_final_layer_atom_all_FFN_zero_same_batch4_geometry"):
        for begin in range(progress["completed_molecules"], 400, 4):
            end = min(begin + 4, 400)
            batch = batch_from_audits(data["audits"][begin:end])
            reserve_evaluations(budget, end - begin)
            changed = model(batch, layer_index=2, intervention=ActivationReplacement(
                torch.zeros((*batch.input_ids.shape, 464), dtype=torch.float32), batch.scope("atoms")))
            difference = np.max(np.abs(changed.normalized_predictions.numpy().astype(np.float64) - baseline[begin:end].astype(np.float64)), axis=1)
            errors[begin:end, 0] = difference; errors[begin:end, 1] = batch.input_ids.shape[1]
            errors.flush(); progress.update(completed_molecules=end); write(progress_path, progress)
        maximum = float(errors[:, 0].max())
        progress.update(status="PASS" if maximum <= 1e-5 else "FAIL", sha256=sha(path),
            max_abs_error=maximum, tolerance=1e-5, completed_molecules=400,
            same_continuous_T_batch4_and_padding_geometry=True)
        write(progress_path, progress)
    if maximum > 1e-5:
        raise RuntimeError("final-layer structural-zero engineering gate failed; no mechanism claim")
    return progress


def condition_definitions(obj):
    units, means = [obj["unit"]] + obj["control_units"], [obj["atom_mean"]] + obj["control_atom_means"]
    if len(units) != 6 or len(set(units)) != 6:
        raise ValueError("locked selected/five control native units differ")
    definitions = [{"name": "selected_atom_full", "unit": units[0], "mean": means[0], "alpha": 1.0, "scope": "atoms"}]
    definitions.extend({"name": f"control{k}_atom_full", "unit": units[k], "mean": means[k], "alpha": 1.0, "scope": "atoms"} for k in range(1, 6))
    definitions.extend([{"name": "selected_atom_half", "unit": units[0], "mean": means[0], "alpha": .5, "scope": "atoms"},
        {"name": "selected_CLS_full", "unit": units[0], "mean": obj["cls_mean"], "alpha": 1.0, "scope": "cls"}])
    return definitions


def interventions(model, data, obj, indices, baseline, budget, binding):
    destination = OUT / "mechanism" / f"concept_{obj['concept_id']}"; destination.mkdir(parents=True, exist_ok=True)
    definitions = condition_definitions(obj)
    deltas, doses = np.empty((len(indices), 8), np.float64), np.empty((len(indices), 8), np.float64)
    for ci, condition in enumerate(definitions):
        contract = {**binding, "T_molecule_indices": indices, "concept_id": obj["concept_id"],
                    "anonymous_output_q": obj["anonymous_output_q"], "condition": condition}
        saved, progress, path, progress_path = condition_array(destination, condition["name"], len(indices), contract)
        with budget.track("encoder", f"T_mechanism_{obj['concept_id']}_{condition['name']}"):
            for begin in range(progress["completed_molecules"], len(indices), 4):
                selected = indices[begin:begin + 4]
                batch = batch_from_audits([data["audits"][i] for i in selected])
                reserve_evaluations(budget, len(selected))
                result, hook = coordinate_forward(model, batch, unit=condition["unit"], native_mean=condition["mean"],
                    alpha=condition["alpha"], scope=condition["scope"])
                q = obj["anonymous_output_q"]
                saved[begin:begin + len(selected), 0] = result.normalized_predictions[:, q].numpy().astype(np.float64) - baseline[selected, q].astype(np.float64)
                saved[begin:begin + len(selected), 1] = hook.dose.numpy().astype(np.float64)
                saved.flush(); progress.update(completed_molecules=begin + len(selected)); write(progress_path, progress)
            progress.update(status="PASS", sha256=sha(path)); write(progress_path, progress)
        deltas[:, ci], doses[:, ci] = saved[:, 0], saved[:, 1]
    return deltas, doses, definitions


def dose_report(doses, positive, obj, cls_D_dose):
    targets = [obj["D_residual_dose"], *obj["control_D_residual_doses"], .5 * obj["D_residual_dose"], cls_D_dose]
    reports = []
    for column, target in enumerate(targets):
        row = {"condition_index": column, "D_matching_dose": target}
        for name, mask in (("positive", positive), ("negative", ~positive)):
            values = doses[mask, column]
            row[name] = {"mean": float(values.mean()), "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max()),
                "mean_over_D_dose": float(values.mean() / target) if abs(target) > 1e-12 else None}
        reports.append(row)
    return reports


def mechanism_supported(effect, association):
    return bool(effect["positive_lower_bound"] and association["same_direction_positive_lower_bound"]
                and association["oriented_interval"]["undefined_replicates"] == 0)


def mechanism_T(data, freeze, plan, budget):
    destination = OUT / "mechanism"; destination.mkdir(parents=True, exist_ok=True)
    binding = identity(data, freeze); result_path = destination / "MECHANISM_T_RESULTS.json"
    if result_path.exists():
        saved = read_json(result_path)
        if saved["identity"] != binding:
            raise ValueError("completed mechanism-T identity differs")
        return saved
    with budget.track("statistics", "T_mechanism_original_occurrence_and_baseline_identity"):
        counts, count_ids, is_smarts = original_occurrences(data)
        baseline = trained_cache(data, "normalized_head.npy")
    reports = []
    for obj in plan["objects"]:
        column = count_ids.index(obj["concept_id"])
        if not is_smarts[column] or np.any(counts[:, column] < 0):
            raise ValueError("mechanism concept must use original complete SMARTS occurrences")
        winner_column = data["concept_ids"].tolist().index(obj["concept_id"])
        if obj["candidate"] != int(data["winners"][0, 0, winner_column]) or obj["unit"] != obj["candidate"] % 464:
            raise ValueError("mechanism selected physical unit differs from final D winner")
        with budget.track("statistics", f"T_mechanism_fixed_hash_subset_{obj['concept_id']}"):
            indices, positive = mechanism_subset(data["rows"], counts[:, column])
        row = {"concept_id": obj["concept_id"], "D_object": obj,
            "n_positive_subset": int(positive.sum()), "n_negative_subset": int((~positive).sum()),
            "T_mechanism_event_gate": bool(positive.sum() >= 50 and (~positive).sum() >= 50),
            "limited_mechanism_IUT_supported": False, "failure_reasons": list(obj["failure_reasons"])}
        association_path = destination / f"ASSOCIATION_{obj['concept_id']}.json"
        if obj["anonymous_output_q"] is not None and obj["D_association_sign"] in (-1, 1):
            if association_path.exists():
                association = read_json(association_path)
                if association["identity"] != binding:
                    raise ValueError("full-T association checkpoint identity differs")
            else:
                with budget.track("statistics", f"T_all10000_output_association_{obj['concept_id']}_2000"):
                    association = output_association(counts[:, column], baseline[:, obj["anonymous_output_q"]],
                        obj["D_association_sign"], budget_check=budget.check)
                    replicated = association.pop("replicated_raw_rho")
                    rp = destination / f"ASSOCIATION_{obj['concept_id']}_BOOTSTRAP.npy"; np.save(rp, replicated)
                    association.update(identity=binding, bootstrap_sha256=sha(rp)); write(association_path, association)
            row["full_T_output_association"] = association
        else:
            row["failure_reasons"].append("D_anonymous_output_or_sign_undefined")
        if not row["T_mechanism_event_gate"]:
            row["failure_reasons"].append("T_mechanism_positive_or_negative_stratum_below50")
        row["T_molecule_indices"] = indices
        reports.append(row)
    with budget.track("encoder", "T_secondary_official_checkpoint_load"):
        model = ChembertaMTR.from_official_checkpoint()
    structural = final_layer_zero(model, data, baseline, budget, binding)
    with budget.track("statistics", "T_secondary_frozen_D_CLS_dose_identity"):
        d_moments_path = BASE / "activations_D/trained/D_moments.npz"
        d_cache = read_json(d_moments_path.parent / "CACHE_RECEIPT.json")
        d_identity = d_cache["files"][d_moments_path.name]
        if d_cache["binding"] != data["binding"] or d_moments_path.stat().st_size != d_identity["bytes"] or d_moments_path.stat().st_mtime_ns != d_identity["mtime_ns"]:
            raise ValueError("accepted D CLS dose identity differs")
        with np.load(d_moments_path, allow_pickle=False) as moments:
            cls_sd = moments["cls_sd"].copy()
    for index, (row, obj) in enumerate(zip(reports, plan["objects"])):
        object_result_path = destination / f"OBJECT_{obj['concept_id']}_RESULT.json"
        if object_result_path.exists():
            saved = read_json(object_result_path)
            if saved["identity"] != binding or saved["D_object"] != obj:
                raise ValueError("completed mechanism object identity differs")
            reports[index] = saved
            continue
        row["identity"] = binding
        if obj["status"] != "D_MECHANISM_OBJECT_LOCKABLE" or not row["T_mechanism_event_gate"]:
            row["status"] = "MECHANISM_D_OR_T_EVENT_GATE_FAILED_NO_REPLACEMENT"
            write(object_result_path, row)
            continue
        indices = row["T_molecule_indices"]
        positive = counts[indices, count_ids.index(obj["concept_id"])] > 0
        deltas, doses, conditions = interventions(model, data, obj, indices, baseline, budget, binding)
        with budget.track("statistics", f"T_mechanism_H_{obj['concept_id']}_2000_stratified"):
            effect = mechanism_effect(deltas, positive, budget_check=budget.check)
            replicated = effect.pop("replicated_H")
            path = destination / f"MECHANISM_{obj['concept_id']}.npz"
            np.savez(path, T_molecule_indices=np.asarray(indices, np.int32), positive=positive,
                     deltas=deltas, doses=doses, replicated_H=replicated)
            row.update(status="MECHANISM_T_COMPLETE", effect=effect, conditions=conditions,
                dose_comparison=dose_report(doses, positive, obj, float(cls_sd[obj["unit"]] * obj["outgoing_norm"])),
                output_file=str(path.relative_to(ROOT)), output_sha256=sha(path),
                limited_mechanism_IUT_supported=mechanism_supported(effect, row["full_T_output_association"]))
        write(object_result_path, row)
    del model; gc.collect()
    result = {"status": "MECHANISM_T_SECONDARY_COMPLETE", "identity": binding, "objects": reports,
        "structural_zero_engineering": structural, "family_size_per_H_or_rho_fixed": 3,
        "six_intervals_joint_95_claimed": False, "no_replacement_or_T_reselection": True,
        "new_evaluations_reserved": budget.record.get("T_secondary_molecule_evaluations_reserved", 0),
        "evaluation_cap": 10000, "baseline_forwards_added": 0,
        "mechanism_population_limit": "frozen model computational sensitivity and chemical conditional association; no biochemical causal claim"}
    write(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "input-T", "mechanism-T"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK); args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); set_num_threads(4)
    try:
        with threadpool_limits(limits=4):
            if args.stage == "prepare":
                result = prepare()
            else:
                binding, T_binding, freeze, tables, plan = require_secondary_lock(args.lock)
                ledger_path = DROOT / "CPU_LEDGER.json"
                if read_json(ledger_path).get("active_phase"):
                    raise RuntimeError("shared CPU ledger is active/unclosed; resolve prior process before serial T stage")
                budget = Budget(binding)
                with budget.track("statistics", f"T_secondary_{args.stage}_locked_input_identity"):
                    data = load_T(binding, T_binding)
                result = input_T(data, freeze, tables, budget) if args.stage == "input-T" else mechanism_T(data, freeze, plan, budget)
            print(json.dumps(safe_json({k: v for k, v in result.items() if k in ("status", "path", "T_secondary_ready_sha256", "new_evaluations_reserved", "evaluation_cap")})), flush=True)
        return 0
    except Exception as exc:
        if args.stage != "prepare":
            write(OUT / "T_SECONDARY_FAILURE.json", {"stage": args.stage, "time_utc": utc(), "reason": str(exc),
                "completed_conditions_retained": True, "budget_not_reset": True})
        print(json.dumps({"status": "STOP", "reason": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
