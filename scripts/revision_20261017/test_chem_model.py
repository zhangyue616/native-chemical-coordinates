"""One consolidated synthetic-only engineering acceptance and CPU benchmark.

Run from the project root:
  .venv_chem/Scripts/python.exe scripts/revision_20261017/test_chem_model.py
No PubChem request, scientific split, descriptor label, old activation or fit.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch.nn import functional as F
from rdkit import Chem

from chem_model import (ActivationReplacement, ChembertaMTR, MODEL_DIR,
                        NativeMeanIntervention, PRIMARY_LAYER, ROOT, TokenBatch,
                        batch_from_audits, engineering_identity, signed_features)
from semantic_mapper import SemanticMapper

OUT = ROOT / "results/revision_20261017/engineering"


def maxdiff(a, b):
    return float((a - b).abs().max())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def must_reject(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return True
    raise AssertionError("invalid input was not rejected")


def synthetic_batch(mapper, strings, pad_to=None):
    # Exact engineering fixture inputs only; no population/canonicalization choice.
    audits = [mapper.audit_molecule(s, require_canonical=False) for s in strings]
    for s, audit in zip(strings, audits):
        require(audit["accepted"], f"synthetic fixture rejected: {s}: {audit['exclusion_reasons']}")
    return batch_from_audits(audits, pad_to=pad_to), audits


def acceptance(model, mapper):
    checks = []

    def record(name, **details):
        checks.append({"check": name, "status": "PASS", **details})

    strings = ["CCO", "CC(Cl)Br", "c1ccccc1", "CC(C)C(=O)O"]
    batch, audits = synthetic_batch(mapper, strings)
    base = model(batch)
    require(base.activations.shape == (*batch.input_ids.shape, 464), "wrong activation shape")
    require(base.normalized_predictions.shape == (4, 199), "wrong anonymous head shape")
    require(not any(p.requires_grad for p in model.parameters()), "parameters are not frozen")
    record("checkpoint_identity_and_shapes", parameter_count=sum(p.numel() for p in model.parameters()),
           activations_shape=list(base.activations.shape), anonymous_head_shape=list(base.normalized_predictions.shape),
           load_receipt=model.load_receipt)

    x = base.last_hidden_state[:, 0]
    manual = F.linear(F.relu(F.linear(x, model.regression.dense.weight, model.regression.dense.bias)),
                      model.regression.out_proj.weight, model.regression.out_proj.bias)
    require(torch.equal(manual, base.normalized_predictions), "manual ReLU full head differs")
    record("manual_official_relu_head", max_abs_difference=maxdiff(manual, base.normalized_predictions))

    raw = {}
    dense = model.roberta.encoder.layer[PRIMARY_LAYER].intermediate.dense
    handle = dense.register_forward_hook(lambda m, i, z: raw.update(z=z.clone()))
    actual = {}
    output_dense = model.roberta.encoder.layer[PRIMARY_LAYER].output.dense
    handle2 = output_dense.register_forward_pre_hook(lambda m, i: actual.update(a=i[0].clone()))
    try:
        checked = model(batch)
    finally:
        handle.remove()
        handle2.remove()
    require(torch.equal(F.gelu(raw["z"]), checked.activations), "captured a is not actual GELU(z)")
    require(torch.equal(actual["a"], checked.applied_activations), "replacement is not input to output dense")
    record("actual_post_gelu_same_point", gelu_max_abs_difference=maxdiff(F.gelu(raw["z"]), checked.activations),
           dense_input_max_abs_difference=maxdiff(actual["a"], checked.applied_activations))

    repeated = model(batch)
    require(torch.equal(base.normalized_predictions, repeated.normalized_predictions), "eval output is nondeterministic")
    require(torch.equal(base.activations, repeated.activations), "eval a is nondeterministic")
    record("eval_determinism", output_max_abs_difference=maxdiff(base.normalized_predictions, repeated.normalized_predictions))

    solo, _ = synthetic_batch(mapper, [strings[0]])
    padded, _ = synthetic_batch(mapper, [strings[0]], pad_to=batch.input_ids.shape[1] + 9)
    solo_result, padded_result = model(solo), model(padded)
    n = solo.input_ids.shape[1]
    pad_pred = maxdiff(solo_result.normalized_predictions, padded_result.normalized_predictions)
    batch_pred = maxdiff(solo_result.normalized_predictions[0], base.normalized_predictions[0])
    pad_a = maxdiff(solo_result.activations, padded_result.activations[:, :n])
    batch_a = maxdiff(solo_result.activations[0], base.activations[0, :n])
    require(max(pad_pred, batch_pred, pad_a, batch_a) < 3e-5, "right-pad/batch invariance tolerance exceeded")
    # Roberta generates positions from config padding_idx=1, while tokenizer PAD=0.
    # Right PAD receives increasing positions, but remains excluded as an attention key.
    position_ids = {}
    ph = model.roberta.embeddings.position_embeddings.register_forward_pre_hook(
        lambda m, i: position_ids.update(ids=i[0].clone()))
    try:
        model(padded)
    finally:
        ph.remove()
    require(torch.equal(position_ids["ids"][0, :n], torch.arange(2, n + 2)), "unexpected original Roberta real-token positions")
    record("right_padding_and_batch_invariance_with_original_config", tolerance=3e-5,
           pad_prediction_max_abs_difference=pad_pred, batch_prediction_max_abs_difference=batch_pred,
           pad_activation_max_abs_difference=pad_a, batch_activation_max_abs_difference=batch_a,
           config_padding_idx=model.config.pad_token_id, tokenizer_pad_id=padded.pad_id,
           actual_padded_position_ids=position_ids["ids"][0].tolist(),
           conclusion="original config retained; tested right padding only; no left-padding support")

    atoms, cls, all_tokens = batch.scope("atoms"), batch.scope("cls"), batch.scope("all")
    require(not torch.any(atoms & batch.special_tokens_mask), "atom scope contains special")
    require(int(cls.sum()) == len(strings), "CLS scope differs")
    require(torch.equal(all_tokens, batch.attention_mask.bool()), "all scope differs")
    require(torch.equal(batch.scope("scoped", atoms), atoms), "explicit scope differs")
    bad = batch.atom_mask.clone(); bad[:, 0] = True
    must_reject(lambda: replace(batch, atom_mask=bad).validate())
    bad_scope = torch.ones_like(padded.attention_mask, dtype=torch.bool)
    must_reject(lambda: padded.scope("scoped", bad_scope))
    record("scope_masks_exclude_special_and_pad_from_atoms", atom_counts=atoms.sum(1).tolist(),
           graph_atom_counts=[a["graph_atom_count"] for a in audits], cls_count=int(cls.sum()),
           all_counts=all_tokens.sum(1).tolist(), invalid_atom_and_pad_scopes_rejected=True)

    sham = model(batch, intervention=ActivationReplacement(base.activations.clone(), atoms))
    require(torch.equal(sham.normalized_predictions, base.normalized_predictions), "same-point sham changed head")
    record("sham_original_a", output_max_abs_difference=maxdiff(sham.normalized_predictions, base.normalized_predictions))

    final = model(batch, layer_index=2)
    changed = final.activations + 0.5
    final_atoms = model(batch, layer_index=2, intervention=ActivationReplacement(changed, atoms))
    final_cls = model(batch, layer_index=2, intervention=ActivationReplacement(changed, cls))
    require(torch.equal(final.normalized_predictions, final_atoms.normalized_predictions), "last-layer atom-only structural zero violated")
    require(torch.isfinite(final_cls.normalized_predictions).all(), "CLS check produced nonfinite output")
    record("final_layer_atom_only_structural_zero", atom_output_max_abs_difference=maxdiff(final.normalized_predictions, final_atoms.normalized_predictions),
           cls_output_observed_max_abs_difference=maxdiff(final.normalized_predictions, final_cls.normalized_predictions),
           cls_nonzero_effect_required=False)

    next_attention_inputs = []
    ah = model.roberta.encoder.layer[2].attention.self.register_forward_pre_hook(
        lambda m, i: next_attention_inputs.append(i[0].clone()))
    try:
        model(batch)
        penultimate = model(batch, intervention=ActivationReplacement(base.activations + 0.5, atoms))
    finally:
        ah.remove()
    entering = next_attention_inputs[1] - next_attention_inputs[0]
    require(float(entering[atoms].abs().max()) > 0, "atom intervention did not reach later attention input")
    require(float(entering[cls].abs().max()) == 0, "CLS changed before later attention despite atom-only scope")
    require(torch.isfinite(penultimate.normalized_predictions).all(), "penultimate patch nonfinite")
    record("penultimate_atom_only_has_actual_later_attention_path", later_attention_layers=1,
           atom_change_entering_later_attention=float(entering[atoms].abs().max()),
           cls_change_before_later_attention=float(entering[cls].abs().max()),
           output_observed_max_abs_difference=maxdiff(base.normalized_predictions, penultimate.normalized_predictions),
           minimum_head_effect_required=False)

    # Engineering fixture means are zero; these are NOT D estimates or scientific results.
    intervention = NativeMeanIntervention((0,), torch.tensor([0.0]), atoms)
    ablated = model(batch, intervention=intervention)
    expected_a = base.activations.clone()
    expected_a[..., 0] = torch.where(atoms, 0.0, expected_a[..., 0])
    require(torch.equal(ablated.applied_activations, expected_a), "native mean replacement differs")
    delta = (base.activations[..., 0].square() * atoms).sum(1) / atoms.sum(1)
    expected_dose = delta.sqrt() * model.roberta.encoder.layer[1].output.dense.weight[:, 0].norm()
    require(torch.allclose(ablated.dose[:, 0], expected_dose, atol=1e-7, rtol=1e-6), "dose formula differs")
    positive = model(batch, intervention=NativeMeanIntervention((0,), torch.tensor([0.2]), atoms))
    negative = model(batch, intervention=NativeMeanIntervention((464,), torch.tensor([-0.2]), atoms))
    require(torch.equal(positive.applied_activations, negative.applied_activations), "signed candidate conversion differs")
    require(torch.equal(positive.normalized_predictions, negative.normalized_predictions), "signed physical intervention differs")
    must_reject(lambda: model(batch, intervention=NativeMeanIntervention((0, 464), torch.zeros(2), atoms)))
    record("native_mean_signed_units_and_dose", dose_formula="RMS_selected_tokens(delta_a_j) * norm2(W[:,j])",
           dose_max_abs_difference=maxdiff(ablated.dose[:, 0], expected_dose),
           synthetic_fixture_mean=0.0, discovery_mean_estimation_performed=False,
           mean_effect_strength_required=False, opposite_signs_same_physical_intervention=True)

    features = signed_features(base.activations)
    require(features.shape[-1] == 928, "native candidate count differs")
    generator = torch.Generator().manual_seed(20261017)  # Engineering fixture only, not a scientific seed.
    q, _ = torch.linalg.qr(torch.randn(464, 464, dtype=torch.float64, generator=generator))
    aa = torch.randn(7, 464, dtype=torch.float64, generator=generator)
    w = model.roberta.encoder.layer[1].output.dense.weight.double()
    rotated = signed_features(aa, q)
    lhs, rhs = (aa @ q) @ (q.T @ w.T), aa @ w.T
    require(torch.allclose(lhs, rhs, atol=3e-12, rtol=3e-12), "compensated linear identity differs")
    require(rotated.shape[-1] == 928, "rotated candidate count differs")
    record("signed_candidates_and_compensated_rotation_linear_identity_only", native_candidates=928,
           rotated_candidates=928, compensated_max_abs_difference=maxdiff(lhs, rhs),
           uncompensated_linear_observed_difference=maxdiff((aa @ q) @ w.T, rhs),
           rotated_model_forward_or_intervention_performed=False,
           uncompensated_same_function_claim=False)

    # Verify same-structure initialization without scientific data or training.
    seed = 20261017
    init1 = ChembertaMTR.initialized_from_config(seed)
    init2 = ChembertaMTR.initialized_from_config(seed)
    require(all(torch.equal(v, init2.state_dict()[k]) for k, v in init1.state_dict().items()), "same-seed init differs")
    require(sum(p.numel() for p in init1.parameters()) == sum(p.numel() for p in model.parameters()), "init architecture differs")
    require(not torch.equal(init1.regression.dense.weight, model.regression.dense.weight), "init loaded pretrained head")
    del init1, init2
    record("from_config_initialization_interface", engineering_test_seed=seed,
           scientific_reference_seeds_selected=False, reference_data_forward_count=0,
           interpretation="three future initialization references are descriptive; no exchangeable causal p value")

    state = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
    require(torch.equal(model.norm_mean, state["norm_mean"]), "checkpoint norm mean not used")
    require(torch.equal(model.norm_std, torch.where(state["norm_std"] == 0, 1, state["norm_std"])), "checkpoint norm std not used")
    expected_unnormalized = base.normalized_predictions * state["norm_std"] + state["norm_mean"]
    require(torch.equal(model.unnormalize_predictions(base.normalized_predictions), expected_unnormalized), "unnormalization differs")
    record("checkpoint_norm_precedence_and_explicit_unnormalization", coordinate_count=199,
           checkpoint_zero_std_count=int((state["norm_std"] == 0).sum()), coordinate_names_recovered=False)
    return checks, [{"exact_smiles": s, "heavy_atoms": Chem.MolFromSmiles(s).GetNumHeavyAtoms(),
                     "token_count": len(a["input_ids"]), "atom_anchor_count": a["graph_atom_count"],
                     "identity_lossy": a["identity_lossy"]} for s, a in zip(strings, audits)]


def benchmark(model, mapper, *, seconds):
    groups = {
        "short": ["CCO", "CC(Cl)Br", "c1ccccc1", "CC(C)C(=O)O"] * 2,
        "medium": ["CC(C)CC(=O)NCCc1ccccc1O", "CCCCCCCCCCCCCCCCCCCC",
                   "CC(C)CC(C)CC(C)CC(C)CC(C)C", "CC(=O)NCCOC(=O)NCCO"] * 2,
        "long_approximately_50_heavy_atoms": ["C" * 50, "CC(C)" * 16 + "CC",
                                               "c1ccccc1" * 8, "CC(C)" * 16 + "CC"],
    }
    records = []
    for name, strings in groups.items():
        batch, audits = synthetic_batch(mapper, strings)
        mask = batch.scope("atoms")
        native = NativeMeanIntervention((0,), torch.tensor([0.0]), mask)
        for mode in ("base_signed_atom_features", "one_native_unit_intervention_forward"):
            def operation():
                if mode == "base_signed_atom_features":
                    output = model(batch)
                    return signed_features(output.activations)[mask]
                return model(batch, intervention=native).normalized_predictions

            for _ in range(4):
                operation()
            started = datetime.now(timezone.utc).isoformat()
            tick = time.perf_counter()
            repeats = 0
            timings = []
            while time.perf_counter() - tick < seconds:
                one = time.perf_counter()
                operation()
                timings.append(time.perf_counter() - one)
                repeats += 1
            elapsed = time.perf_counter() - tick
            molecules = len(strings) * repeats
            records.append({
                "fixture_group": name, "mode": mode, "batch_size": len(strings),
                "exact_synthetic_smiles": strings,
                "heavy_atoms_per_molecule": [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in strings],
                "actual_tokens_per_molecule": [len(a["input_ids"]) for a in audits],
                "padded_tokens": batch.input_ids.shape[1], "iterations": repeats,
                "timed_molecules": molecules, "elapsed_wall_seconds": elapsed,
                "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "molecules_per_second": molecules / elapsed,
                "seconds_per_molecule": elapsed / molecules,
                "median_batch_seconds": statistics.median(timings),
                "p95_batch_seconds": sorted(timings)[max(0, math.ceil(len(timings) * .95) - 1)],
                "includes": "frozen CPU forward, same-point capture, requested feature/patch operations",
                "excludes": "network, mapper, sanitation, disk export, real-data latency, OS contention outside this measurement",
                "scientific_data_forward_count": 0,
            })
    base = max(r["seconds_per_molecule"] for r in records if r["mode"] == "base_signed_atom_features")
    single = max(r["seconds_per_molecule"] for r in records if r["mode"] == "one_native_unit_intervention_forward")
    # Engineering extrapolation across all three measured lengths, not one short molecule.
    # Six separate native conditions = selected unit plus five matched controls.
    estimate_base = 60000 * base
    estimate_six = 10000 * 6 * single
    policy = {
        "scope": "planning only; not an empirical upper bound on unseen chemistry or future machine load",
        "measured_length_eligibility": "right-padded batch <=8, max tokens <= measured long padded_tokens; max heavy atoms<=50",
        "base_60000_worst_measured_forward_hours": estimate_base / 3600,
        "intervention_10000_six_separate_conditions_worst_measured_forward_hours": estimate_six / 3600,
        "suggested_base_wall_stop_seconds": max(7200, math.ceil(5 * estimate_base / 1800) * 1800),
        "suggested_intervention_wall_stop_seconds": max(7200, math.ceil(5 * estimate_six / 1800) * 1800),
        "suggested_total_wall_stop_seconds": max(14400, math.ceil(5 * (estimate_base + estimate_six) / 1800) * 1800),
        "extra_runs_included": "six conditions only; no implicit replay, retries, three init references or dose repeats",
        "stop_action": "persist bounded checkpoint/processed count; stop and report; do not silently enlarge budget",
        "required_before_scientific_execution": "controller freezes protocol/D means, sample counts, condition count and real token window; no science executed here",
        "conservative_multiplier": 5,
    }
    return records, policy


def write_report(payload):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CHEM_MODEL_ENGINEERING.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for r in payload["cpu_benchmarks"]:
        rows.append(f"| {r['fixture_group']} | {r['mode']} | {r['batch_size']} | {r['padded_tokens']} | {r['elapsed_wall_seconds']:.3f} | {r['molecules_per_second']:.2f} |")
    cap = payload["cpu_planning"]
    text = "\n".join([
        "# ChemBERTa-MTR 合成工程验收", "", f"状态：{payload['status']}；仅工程验证，科学 D/T、真实标签、旧 activations、训练均为 0。", "",
        "固定官方 HF checkpoint 与 config 的 SHA256 已核；原始官方回归源码及来源见 assets/chemberta_mtr_model/provenance.json。",
        "三层 Roberta、hidden 384、FFN 464；主层 zero-based 1；捕获实际 post-GELU a。实际参数 3,504,055；模型名中的 77M 不表示参数数量。",
        "头为 CLS/dropout/Linear384/ReLU/dropout/Linear199，eval dropout 关闭。默认 normalized 输出；checkpoint norm buffers 优先，显式接口可反标准化。199 坐标仅匿名编号，未恢复其精确 descriptor 名称/顺序。", "",
        "## 工程检查", "",
        *[f"- PASS：{c['check']}" for c in payload["checks"]], "",
        "官方 config pad_token_id=1，实际 tokenizer PAD=0、CLS=12。未改原config；实际验证 right-pad 与 mixed-batch 对有效 token 的 a / CLS head 在 3e-5 容差内等价。内部PAD位置仍递增，只支持右PAD。",
        "最终层 atom-only 为严格输出结构零；CLS-only 实测差异记录但未设最低效应门。倒数第二层证明干预原子进入后续 attention；不把 mean-ablation 强度当作必过。",
        "native ±a 与旋转特征 ±(aQ) 均928候选；只实施 native 干预。旋转仅做补偿线性恒等式检查，不运行 rotated 模型或声称不补偿仍同函数。",
        "均值0/0.2、unit0与QR种子只是可复现工程fixture，不是D均值、科学选择或已冻结研究种子。3 init未来仅属经验参照，不作交换性因果p值。", "",
        "## 实际 CPU 测速", "", f"CPU：{payload['runtime']['cpu']}；torch threads={payload['runtime']['torch_threads']}；interop={payload['runtime']['torch_interop_threads']}。", "",
        "| 合成组 | 操作 | batch | PAD token长 | 实测秒 | 分子/秒 |",
        "|---|---|---:|---:|---:|---:|", *rows, "",
        "测速包括模型forward、真实post-GELU捕获和相应特征/patch；不包括mapping/标准化/网络/磁盘或未来系统争用。短/中/约50 heavy atom 三组各自计时，原始精确SMILES、长度、迭代和UTC起止见JSON。", "",
        "## CPU 硬停止建议", "",
        f"按三组中最慢实测耗时，60k基础forward约 {cap['base_60000_worst_measured_forward_hours']:.3f} 小时；≤10k分子×6个分开native干预条件约 {cap['intervention_10000_six_separate_conditions_worst_measured_forward_hours']:.3f} 小时。",
        f"以5倍余量及最低缓冲，建议基础硬停止 {cap['suggested_base_wall_stop_seconds']/3600:.1f} 小时、干预硬停止 {cap['suggested_intervention_wall_stop_seconds']/3600:.1f} 小时、总硬停止 {cap['suggested_total_wall_stop_seconds']/3600:.1f} 小时。",
        "这些是可执行的停止政策，不是未知真实数据耗时的数学上界；仅适用于已测token窗口、right-padding、batch≤8。若冻结后长度/条件/参考模型次数增加，应由研究操作者重新定预算，不能沿用旧GPU耗时或自动扩跑。到时保存已完成计数并停止。", "",
        "本阶段未读取旧项目根、旧语料/labels/activations，未请求新PubChem，未训练、计算科学AP或更新协议/mapper；科学运行仍待研究操作者冻结。", "",
        "复现：`.venv_chem/Scripts/python.exe scripts/revision_20261017/test_chem_model.py --threads 4 --benchmark-seconds 2`。", "",
    ])
    (OUT / "CHEM_MODEL_ENGINEERING.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or args.benchmark_seconds < 0.5:
        parser.error("threads must be 1..4; benchmark seconds must be >=0.5")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = datetime.now(timezone.utc).isoformat()
    model, mapper = ChembertaMTR.from_official_checkpoint(), SemanticMapper()
    checks, fixtures = acceptance(model, mapper)
    records, policy = benchmark(model, mapper, seconds=args.benchmark_seconds)
    packages = ["torch", "transformers", "numpy", "rdkit", "tokenizers", "huggingface-hub"]
    runtime = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
               "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
               "packages": {p: importlib.metadata.version(p) for p in packages}}
    payload = {"schema": "chemberta-mtr-engineering/1", "status": "PASS_SYNTHETIC_ENGINEERING_ONLY",
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "identity": engineering_identity(), "runtime": runtime, "checks": checks,
               "synthetic_acceptance_fixtures": fixtures, "cpu_benchmarks": records, "cpu_planning": policy,
               "scope_receipt": {"new_pubchem_requests": 0, "scientific_data_forwards": 0,
                                 "training_steps": 0, "old_corpus_labels_activations_read": False,
                                 "legacy_root_read": False, "rotated_model_forwards": 0}}
    write_report(payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks),
                      "report": str(OUT / "CHEM_MODEL_ENGINEERING.md"), "cpu_planning": policy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
