"""New real-hook/caller checks on synthetic molecules only; no pure-statistics rerun."""
from __future__ import annotations
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from chem_model import ChembertaMTR, NativeMeanIntervention, ActivationReplacement, batch_from_audits
from chem_T_intervention import coordinate_forward
import chem_T_secondary as secondary
from semantic_mapper import SemanticMapper


def run():
    checks = []
    def check(name, passed):
        checks.append({"case": name, "status": "PASS" if bool(passed) else "FAIL"})
    mapper = SemanticMapper()
    audits = [mapper.audit_molecule(s, molecule_id=f"synthetic_{i}", require_canonical=True, max_tokens=512)
              for i, s in enumerate(("CCCCC", "O=C(O)CC(=O)O"))]
    assert all(a["accepted"] for a in audits)
    batch = batch_from_audits(audits)
    model = ChembertaMTR.initialized_from_config(99117)
    unit, mean = 7, .25
    baseline = model(batch)
    original = model(batch, intervention=NativeMeanIntervention((464 + unit,), torch.tensor([-mean]), batch.scope("atoms")))
    full, full_hook = coordinate_forward(model, batch, unit=unit, native_mean=mean, alpha=1., scope="atoms")
    check("alpha1_equals_original_signed_mean_replacement", torch.equal(full.normalized_predictions, original.normalized_predictions)
          and torch.equal(full_hook.applied, original.applied_activations)
          and torch.equal(full_hook.dose, original.dose[:, 0]))
    sham, sham_hook = coordinate_forward(model, batch, unit=unit, native_mean=mean, alpha=0., scope="atoms")
    check("alpha0_exact_sham", torch.equal(sham.normalized_predictions, baseline.normalized_predictions)
          and torch.equal(sham_hook.original, sham_hook.applied) and torch.all(sham_hook.dose == 0))
    half, half_hook = coordinate_forward(model, batch, unit=unit, native_mean=mean, alpha=.5, scope="atoms")
    full_change = full_hook.applied - full_hook.original
    half_change = half_hook.applied - half_hook.original
    check("half_coordinate_change_and_RMSdose_half", torch.allclose(half_change, .5 * full_change, atol=3e-8, rtol=1e-6)
          and torch.allclose(half_hook.dose, .5 * full_hook.dose, atol=3e-8, rtol=1e-6))
    keep = torch.ones(464, dtype=torch.bool); keep[unit] = False
    check("other463_coordinates_and_mask_outside_unchanged", torch.equal(half_hook.original[..., keep], half_hook.applied[..., keep])
          and torch.equal(half_hook.original[~batch.scope("atoms")], half_hook.applied[~batch.scope("atoms")]))
    check("half_forward_changes_actual_prediction", torch.max(torch.abs(half.normalized_predictions - baseline.normalized_predictions)) > 0)
    cls, cls_hook = coordinate_forward(model, batch, unit=unit, native_mean=-.2, alpha=1., scope="cls")
    check("CLS_scope_only_changes_position0_target", torch.equal(cls_hook.original[:, 1:], cls_hook.applied[:, 1:])
          and torch.equal(cls_hook.applied[:, 0, unit], torch.full((2,), -.2)))
    structural = model(batch, layer_index=2, intervention=ActivationReplacement(
        torch.zeros((*batch.input_ids.shape, 464)), batch.scope("atoms")))
    error = float(torch.max(torch.abs(structural.normalized_predictions - baseline.normalized_predictions)))
    check("same_batch_padding_final_layer_all_atom_zero", error <= 1e-5
          and batch.input_ids.shape[1] == max(len(a["input_ids"]) for a in audits))
    obj = {"unit": unit, "candidate": 464 + unit, "atom_mean": mean, "cls_mean": -.2,
           "control_units": [1, 2, 3, 4, 5], "control_atom_means": [.1, .2, .3, .4, .5]}
    conditions = secondary.condition_definitions(obj)
    check("eight_conditions_keep_native_mean_for_negative_pole", len(conditions) == 8
          and conditions[0]["mean"] == mean and conditions[0]["unit"] == unit
          and conditions[6]["alpha"] == .5 and conditions[7]["scope"] == "cls"
          and conditions[7]["mean"] == -.2)
    rows = [{"cid": str(i + 1000)} for i in range(500)]
    counts = np.r_[np.ones(250, np.int32), np.zeros(250, np.int32)]
    selected, positive = secondary.mechanism_subset(rows, counts)
    backwards, bpositive = secondary.mechanism_subset(rows[::-1], counts[::-1])
    check("mechanism_hash_fixed_strata_max200_order_invariant", len(selected) == 400 and positive.sum() == 200
          and [rows[i]["cid"] for i in selected] == [rows[::-1][i]["cid"] for i in backwards]
          and np.array_equal(positive, bpositive))
    association = {"same_direction_positive_lower_bound": True, "oriented_interval": {"undefined_replicates": 0}}
    check("IUT_needs_both_and_zero_undefined", secondary.mechanism_supported({"positive_lower_bound": True}, association)
          and not secondary.mechanism_supported({"positive_lower_bound": False}, association)
          and not secondary.mechanism_supported({"positive_lower_bound": True}, {**association, "same_direction_positive_lower_bound": False})
          and not secondary.mechanism_supported({"positive_lower_bound": True}, {**association, "oriented_interval": {"undefined_replicates": 1}}))
    report = {"status": "PASS_SYNTHETIC_T_SECONDARY" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "created_utc": secondary.utc(), "checks": checks, "passed": sum(c["status"] == "PASS" for c in checks),
        "total": len(checks), "synthetic_model_seed": 99117, "synthetic_molecule_count": 2,
        "synthetic_forward_batches": 7, "actual_D_or_T_scientific_reads": 0, "actual_T_forwards": 0,
        "existing_six_pure_statistics_checks_reused_not_rerun": True,
        "final_layer_structural_max_abs_error": error}
    secondary.write(secondary.CHECKS, report)
    print({"status": report["status"], "passed": report["passed"], "total": report["total"],
           "failures": [c for c in checks if c["status"] == "FAIL"]})
    return 0 if report["status"] == "PASS_SYNTHETIC_T_SECONDARY" else 1


if __name__ == "__main__":
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=4):
        raise SystemExit(run())
