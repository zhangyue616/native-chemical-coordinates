"""Fixed-j protein KRR development. No HTTP, model loader or forward path.

prepare/selftest do not fit real data. develop requires a root-owned D_GO
receipt bound to FINAL_READY.json; it never accepts new T inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_20261017/protein_D"
PROTOCOL = ROOT / "docs/revision_20261017/PROTEIN_KRR_PROTOCOL.md"
DEFAULT_GROUPS = ROOT / "data/revision_20261017/protein_metadata/OLD_SELECTED_UNION_GROUPS.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
OFFSETS = tuple(range(-15, 0)) + tuple(range(1, 16))
UNIT, SEED = 1246, 2026101708
LAMBDAS = (0.0001, 0.001, 0.01, 0.1)
BLOCKS = ("X", "P", "Q", "j")
MODEL_BLOCKS = {"X": ("X",), "Xj": ("X", "j"),
                "S": ("X", "P"), "Sj": ("X", "P", "j"),
                "L": ("X", "P", "Q"), "Lj": BLOCKS}
MODELS = tuple(MODEL_BLOCKS)
ENDPOINTS = ("gX", "gS", "gL", "psi")
CONFIG = {
    "unit": UNIT, "layer": "third block (0-based 2) post-GELU", "model": "existing ESM2 8M",
    "new_forward_count": 0, "new_model_load_count": 0, "development_pairs": 432,
    "models": {k: list(v) for k, v in MODEL_BLOCKS.items()}, "lambda_grid": list(LAMBDAS),
    "kernel_block_scaling": "raw pair kernel / training-pair mean diagonal; zero trace -> zero block",
    "feature_coordinate_scaling": "none; no per-coordinate RMS or centering",
    "fold_rule": f"sorted unique group by SHA256(protein-krr-cv:{{group}}:{SEED}), rank modulo 4",
    "fit_objective": "mean_pair (t-f)^2 + lambda*||f||_RKHS^2; alpha=(K+n_pair*lambda*I)^-1*t",
    "training_weight": "equal pair; no duplicated or mirrored rows",
    "selection_weight": "mean within known group, then equal known groups",
    "selection": "minimum group-mean OOF paired Brier; exact tie larger lambda",
    "real_fits": 102, "maximum_readout_cpu_seconds": 3600,
    "primary_family": {"gX": "greater", "gS": "greater", "gL": "greater", "psi": "two-sided"},
    "family_alpha": 0.05, "multiplicity": "Holm, fixed denominator 4",
    "simultaneous_CI": "two-sided Bonferroni t-reference: each 98.75%, family >=95% approximately",
    "precision": {"half_width": 0.01, "SD_inflation": 1.25, "simultaneous_z": 2.4977,
                  "minimum_target": 512, "maximum_target": 1536},
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_features(sequence, q):
    """Exactly the old X64 definitions, including fixed length/256."""
    if not 0 <= q < len(sequence) or set(sequence) - set(AA):
        raise ValueError("Invalid sequence/center")
    values = []
    for radius in (None, 7, 15):
        lo = 0 if radius is None else max(0, q - radius)
        hi = len(sequence) if radius is None else min(len(sequence), q + radius + 1)
        visible = sequence[lo:q] + sequence[q + 1:hi]
        count = Counter(visible)
        values.extend(count[a] / len(visible) for a in AA)
    relative = q / (len(sequence) - 1)
    values.extend((relative, relative**2, len(sequence) / 256,
                   (min(q, 15) + min(len(sequence) - q - 1, 15)) / 30))
    return np.asarray(values, dtype=np.float64)


def position_categories(sequence, q):
    return np.asarray([AA.index(sequence[q + offset]) if 0 <= q + offset < len(sequence) else 20
                       for offset in OFFSETS], dtype=np.int8)


def fixed_h(sequence, q):
    return float(sequence[q - 1] in "AILMFPVWY" and sequence[q + 2] in "AILMFPVWY")


def build_cases(records):
    cases, pairs = [], []
    for row in records:
        seq = row["sequence"]
        if not 64 <= len(seq) <= 256 or set(seq) - set(AA):
            raise ValueError("Historical short-D sequence invariant failed")
        sites = sorted((int(row[field]["q_0based"]), label, y) for field, label, y in
                       (("h_site", "HELIX", 1.0), ("s_site", "STRAND", 0.0)))
        if sites[0][0] == sites[1][0]:
            raise ValueError("Repeated site within pair")
        start = len(cases)
        for q, label, y in sites:
            if not 7 <= q < len(seq) - 7 or seq[q] != "A":
                raise ValueError("Historical central Alanine/window invariant failed")
            source_site = row["h_site" if y else "s_site"]
            if source_site["label"] != label or source_site["future_token_index_0based"] != q + 1:
                raise ValueError("Historical label/token coordinate invariant failed")
            cases.append({"id": row["id"], "component": row["component_key"], "sequence": seq,
                          "q": q, "label": label, "y": y, "case_id": f"{row['id']}:{label}:{q}",
                          "origin": row["origin"]})
        pairs.append({"id": row["id"], "origin": row["origin"],
                      "prior_component_key": row["component_key"], "a_index": start,
                      "b_index": start + 1, "y_a": sites[0][2],
                      "a_case_id": cases[start]["case_id"], "b_case_id": cases[start + 1]["case_id"]})
    return cases, pairs


def load_metadata():
    records, sources = [], []
    for origin, expected in (("pilot3a", 176), ("pilot4a", 128), ("pilot5a", 128)):
        path = ROOT / "data" / origin / "proteins.jsonl"
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if len(rows) != expected:
            raise ValueError(f"Wrong count in {path}")
        records.extend(dict(row, origin=origin) for row in rows)
        sources.append({"path": str(path), "pairs": expected, "size_bytes": path.stat().st_size})
    if len({r["id"] for r in records}) != 432 or len({r["component_key"] for r in records}) != 432:
        raise ValueError("Old selected pair/accession identities overlap")
    cases, pairs = build_cases(records)
    previous = []
    # P5 D_CASES contains all P3+P4, including P3 C16, which is absent
    # from P3 D_CASES (D_fit/D_select only) and T_CASES.
    for origin, name, field in (("pilot5a", "D_CASES.json", "cases"),
                                ("pilot5a", "T_CASES.json", "natural_cases")):
        previous.extend(read_json(ROOT / "results" / origin / name)[field])
    lookup = {c["case_id"]: c for c in cases}
    if len(previous) != len(cases) or {c["case_id"] for c in previous} != set(lookup):
        raise ValueError("Historical metadata case-id set mismatch")
    for prior in previous:
        if any(prior[key] != lookup[prior["case_id"]][key]
               for key in ("id", "component", "sequence", "q", "label", "y")):
            raise ValueError("Historical sequence/q/label/component mismatch")
    return cases, pairs, sources


def read_groups(path, pairs, field="current_known_union_group"):
    """A future finalgroup file may replace the field/path, without changing code."""
    value = read_json(path)
    rows = value.get("selected_pair_group_mapping", value.get("pair_group_mapping", []))
    lookup = {}
    for row in rows:
        key = (row["origin"], row["id"])
        if key in lookup or not isinstance(row.get(field), str) or not row[field]:
            raise ValueError("Duplicate/missing group mapping")
        lookup[key] = row
    if set(lookup) != {(p["origin"], p["id"]) for p in pairs}:
        raise ValueError("Group file must map each of the 432 selected pairs exactly once")
    groups = []
    for pair in pairs:
        row = lookup[(pair["origin"], pair["id"])]
        if row.get("prior_component_key", pair["prior_component_key"]) != pair["prior_component_key"]:
            raise ValueError("Prior component key drift in group mapping")
        groups.append(row[field])
    return np.asarray(groups)


def fixed_folds(groups):
    groups = np.asarray(groups).astype(str)
    unique = sorted(set(groups), key=lambda g: (hashlib.sha256(
        f"protein-krr-cv:{g}:{SEED}".encode()).hexdigest(), g))
    if len(unique) < 4:
        raise ValueError("At least four groups required")
    assignment = {group: rank % 4 for rank, group in enumerate(unique)}
    return np.asarray([assignment[g] for g in groups], dtype=np.int64)


def group_average(values, groups):
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups).astype(str)
    if values.shape[0] != len(groups) or not np.isfinite(values).all():
        raise ValueError("Nonfinite or unaligned group values")
    unique = np.asarray(sorted(set(groups)))
    return unique, np.asarray([values[groups == g].mean(axis=0) for g in unique])


def native_paths():
    return [ROOT / "results" / origin / name for origin, name in
            (("pilot3a", "D_NATIVE.npz"), ("pilot3a", "C_NATIVE.npz"),
             ("pilot3a", "T_NATIVE.npz"), ("pilot4a", "T_NATIVE.npz"), ("pilot5a", "T_NATIVE.npz"))]


def load_native_j(cases):
    lookup = {}
    for path in native_paths():
        with np.load(path, allow_pickle=False) as arrays:
            ids, z = arrays["case_ids"].astype(str), arrays["z"]
            if z.shape != (len(ids), 1280) or not np.isfinite(z).all():
                raise ValueError("Historical native shape/nonfinite failure")
            for case_id, value in zip(ids, z[:, UNIT]):
                if case_id in lookup:
                    raise ValueError("Duplicate historical native case-id")
                lookup[case_id] = float(value)
    if set(lookup) != {c["case_id"] for c in cases}:
        raise ValueError("Native and explicit metadata case-id sets differ")
    return np.asarray([lookup[c["case_id"]] for c in cases], dtype=np.float64)


def pair_data(cases, pairs, j):
    a, b = np.asarray([p["a_index"] for p in pairs]), np.asarray([p["b_index"] for p in pairs])
    x = np.stack([control_features(c["sequence"], c["q"]) for c in cases])
    pos = np.stack([position_categories(c["sequence"], c["q"]) for c in cases])
    j = np.asarray(j, dtype=np.float64)
    if j.shape != (len(cases),) or not np.isfinite(j).all():
        raise ValueError("j shape/finite failure")
    return {"dx": x[a] - x[b], "pa": pos[a], "pb": pos[b], "dj": j[a] - j[b]}


def match_count(left, right):
    if left.shape[1] != right.shape[1]:
        raise ValueError("Inconsistent position count")
    return np.sum(left[:, None, :] == right[None, :, :], axis=2, dtype=np.int16).astype(np.float64)


def second_order_site(left, right):
    m = match_count(left, right)
    positions = left.shape[1]
    return m * (m - 1) / (positions * (positions - 1))


def raw_blocks(left, right=None):
    right = left if right is None else right
    matches = [match_count(left[a], right[b]) for a, b in
               (("pa", "pa"), ("pa", "pb"), ("pb", "pa"), ("pb", "pb"))]
    denominator = left["pa"].shape[1] * (left["pa"].shape[1] - 1)
    q = [m * (m - 1) / denominator for m in matches]
    return {"X": left["dx"] @ right["dx"].T,
            "P": matches[0] - matches[1] - matches[2] + matches[3],
            "Q": q[0] - q[1] - q[2] + q[3],
            "j": np.outer(left["dj"], right["dj"])}


def block_scales(blocks, train):
    train = np.asarray(train, dtype=np.int64)
    scales = {}
    for name in BLOCKS:
        diagonal = np.diag(blocks[name])[train]
        if not np.isfinite(diagonal).all() or diagonal.min() < -1e-10:
            raise ValueError("Nonfinite/negative block diagonal")
        value = float(np.maximum(diagonal, 0).mean())
        scales[name] = value if value > 0 else 0.0
    return scales


def sum_kernel(blocks, scales, model):
    result = np.zeros_like(blocks["X"], dtype=np.float64)
    for name in MODEL_BLOCKS[model]:
        if scales[name] > 0:
            result += blocks[name] / scales[name]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite kernel")
    return result


def fit_krr(kernel, target, penalty):
    kernel, target = np.asarray(kernel, dtype=np.float64), np.asarray(target, dtype=np.float64)
    n = len(target)
    if kernel.shape != (n, n) or not np.isfinite(kernel).all() or not np.isfinite(target).all():
        raise ValueError("Invalid KRR inputs")
    if not penalty > 0 or not np.allclose(kernel, kernel.T, atol=1e-11, rtol=1e-11):
        raise ValueError("Positive penalty and symmetric kernel required")
    system = kernel.copy()
    system.flat[::n + 1] += n * penalty
    alpha = cho_solve(cho_factor(system, lower=True, check_finite=False), target, check_finite=False)
    residual = float(np.linalg.norm(system @ alpha - target) / max(1.0, np.linalg.norm(target)))
    if not np.isfinite(alpha).all() or residual > 1e-8:
        raise RuntimeError("KRR numerical solve failed")
    return alpha, residual


def probability(scores):
    return np.clip((1 + np.asarray(scores, dtype=np.float64)) / 2, 0, 1)


def select_candidate(candidates):
    return min(candidates, key=lambda item: (item["OOF_group_Brier"], -item["lambda"]))


def paired_contributions(loss):
    gx, gs, gl = loss["X"] - loss["Xj"], loss["S"] - loss["Sj"], loss["L"] - loss["Lj"]
    return np.column_stack((gx, gs, gl, gx - gs))


def precision_plan(contributions, groups):
    unique, grouped = group_average(contributions, groups)
    if len(unique) < 2 or grouped.shape[1:] != (4,):
        raise ValueError("Four endpoints and >=2 groups required")
    sd = grouped.std(axis=0, ddof=1)
    required = np.ceil((2.4977 * 1.25 * sd / 0.01)**2).astype(np.int64)
    maximum = int(required.max())
    feasible = maximum <= 1536
    return {"status": "PRECISION_TARGET_FEASIBLE_PENDING_FRESH_SOURCE" if feasible else "PRECISION_INFEASIBLE_WITHIN_CAP",
            "N_req": maximum, "N_req_by_endpoint": dict(zip(ENDPOINTS, required.tolist())),
            "requested_target": max(512, maximum), "authorized_target_within_cap": max(512, maximum) if feasible else None,
            "maximum_target": 1536, "group_count": len(unique), "pair_count": len(groups),
            "group_SD": dict(zip(ENDPOINTS, sd.tolist())), "group_mean": dict(zip(ENDPOINTS, grouped.mean(axis=0).tolist())),
            "half_width": 0.01, "SD_inflation": 1.25, "simultaneous_z": 2.4977,
            "source_gate": "NOT_EVALUATED: 2074 metadata components are only a pre-sequence/label upper bound",
            "interpretation": "D-selected cross-fit planning only; short-D variance does not guarantee long-T precision or power"}


def future_primary_summary(contributions, groups):
    """Prelocked future-T inference helper; never invoked on D in develop."""
    contributions, groups = np.asarray(contributions, dtype=np.float64), np.asarray(groups).astype(str)
    if contributions.shape != (len(groups), 4):
        raise ValueError("Fixed four aligned endpoint columns required")
    unique = np.asarray(sorted(set(groups)))
    if not len(unique):
        raise ValueError("At least one future group required")
    n, family = len(unique), []
    for index, name in enumerate(ENDPOINTS):
        finite = bool(np.isfinite(contributions[:, index]).all())
        values = (np.asarray([contributions[groups == g, index].mean() for g in unique])
                  if finite else None)
        sd = float(values.std(ddof=1)) if finite and n > 1 else 0.0
        mean = float(values.mean()) if finite else None
        se = sd / math.sqrt(n) if finite else None
        row = {"name": name, "mean": mean, "n_groups": n, "SD": sd, "SE": se,
               "direction": CONFIG["primary_family"][name], "p_for_Holm": 1.0,
               "status": "UNTESTABLE", "CI_95_estimation": None, "CI_98_75_Bonferroni": None}
        if finite and n > 1 and se > 0 and math.isfinite(se):
            statistic = mean / se
            p = float(2 * student_t.sf(abs(statistic), n - 1) if name == "psi" else student_t.sf(statistic, n - 1))
            ci = [float(mean - student_t.ppf(0.975, n - 1) * se), float(mean + student_t.ppf(0.975, n - 1) * se)]
            c = float(student_t.ppf(1 - 0.05 / 8, n - 1))
            row.update(status="TESTABLE", p_for_Holm=p, t=statistic,
                       CI_95_estimation=ci, CI_98_75_Bonferroni=[mean - c * se, mean + c * se])
        family.append(row)
    order = sorted(range(4), key=lambda i: (family[i]["p_for_Holm"], i))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (4 - rank) * family[index]["p_for_Holm"]))
        family[index]["p_Holm"] = running
        family[index]["reject"] = running <= 0.05
    return {"family_size": 4, "alpha": 0.05, "endpoints": family,
            "inference": "Approximate paired group t-reference; residual homology remains a limitation"}


def prepare(groups_path, group_field):
    if (OUT / "D_STARTED.json").exists():
        raise RuntimeError("Real D already started; do not regenerate readiness")
    cases, pairs, sources = load_metadata()
    groups = read_groups(groups_path, pairs, group_field)
    folds = fixed_folds(groups)
    inputs = {"created_utc": utc(), "cases": cases, "pairs": pairs, "groups": groups.tolist(),
              "fold_ids": folds.tolist(), "group_file": str(groups_path), "group_field": group_field,
              "group_file_sha256": sha(groups_path), "metadata_sources": sources,
              "native_sources": [{"path": str(p), "size_bytes": p.stat().st_size} for p in native_paths()],
              "independence": "Known metadata grouping, including bridging from seen new metadata; no distant-homology guarantee"}
    write_json(OUT / "D_INPUT_MANIFEST.json", inputs)
    ready = {"status": "FINAL_READY", "created_utc": utc(), "real_fits_completed": 0,
             "new_model_loads": 0, "new_model_forwards": 0, "new_http_requests": 0,
             "config": CONFIG, "script_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
             "input_manifest_sha256": sha(OUT / "D_INPUT_MANIFEST.json"),
             "group_file_sha256": sha(groups_path), "pairs": len(pairs), "sites": len(cases),
             "known_union_groups": len(set(groups)), "pair_counts_per_fold": np.bincount(folds).tolist(),
             "group_counts_per_fold": [len(set(groups[folds == f])) for f in range(4)],
             "synthetic_check_required": str(OUT / "SELFTEST.json"),
             "awaiting": "root-owned D_GO receipt, then exactly 102 D-only KRR fits"}
    check = read_json(OUT / "SELFTEST.json")
    if check.get("status") != "PASS":
        raise RuntimeError("Passing synthetic SELFTEST required")
    if check.get("script_sha256") != ready["script_sha256"]:
        revision_path = OUT / "GRID_REVISION_CHECK.json"
        revision = read_json(revision_path)
        if (revision.get("status") != "PASS" or revision.get("current_script_sha256") != ready["script_sha256"]
                or revision.get("reused_selftest_sha256") != sha(OUT / "SELFTEST.json")
                or revision.get("new_grid") != list(LAMBDAS)):
            raise RuntimeError("Current-script grid revision receipt required to reuse prior kernel checks")
        ready["grid_revision_check_sha256"] = sha(revision_path)
    if check.get("test_script_sha256") != sha(Path(__file__).with_name("test_protein_krr.py")):
        raise RuntimeError("Synthetic test source changed since SELFTEST")
    ready["selftest_sha256"] = sha(OUT / "SELFTEST.json")
    write_json(OUT / "FINAL_READY.json", ready)
    print(json.dumps({k: ready[k] for k in ("status", "pairs", "sites", "known_union_groups", "pair_counts_per_fold", "group_counts_per_fold")}))


def require_go(path):
    go, ready = read_json(path), read_json(OUT / "FINAL_READY.json")
    if (go.get("status") != "D_GO" or go.get("D_only_authorized") is not True
            or go.get("T_authorized") is not False or go.get("maximum_real_fits") != 102
            or go.get("ready_sha256") != sha(OUT / "FINAL_READY.json")):
        raise RuntimeError("Root D_GO receipt missing or unbound")
    if ready.get("config") != CONFIG or ready.get("status") != "FINAL_READY":
        raise RuntimeError("Ready configuration mismatch")
    for key, p in (("script_sha256", Path(__file__)), ("protocol_sha256", PROTOCOL),
                   ("input_manifest_sha256", OUT / "D_INPUT_MANIFEST.json"), ("selftest_sha256", OUT / "SELFTEST.json")):
        if ready[key] != sha(p):
            raise RuntimeError(f"Ready binding drift: {key}")
    if ("grid_revision_check_sha256" in ready
            and ready["grid_revision_check_sha256"] != sha(OUT / "GRID_REVISION_CHECK.json")):
        raise RuntimeError("Grid revision receipt drift")
    manifest = read_json(OUT / "D_INPUT_MANIFEST.json")
    if ready["group_file_sha256"] != sha(manifest["group_file"]):
        raise RuntimeError("Group mapping changed: prepare and root D_GO again before any real fit")
    return go, ready, manifest


def develop(go_path):
    go, ready, manifest = require_go(go_path)
    if (OUT / "D_STARTED.json").exists():
        raise RuntimeError("D run already started; no automatic restart or grid expansion")
    cases, pairs, _ = load_metadata()
    if cases != manifest["cases"] or pairs != manifest["pairs"]:
        raise RuntimeError("Historical input metadata drift")
    groups = read_groups(manifest["group_file"], pairs, manifest["group_field"])
    folds, y = fixed_folds(groups), np.asarray([p["y_a"] for p in pairs], dtype=np.float64)
    data = pair_data(cases, pairs, load_native_j(cases))
    blocks = raw_blocks(data)
    target, n = 2 * y - 1, len(y)
    write_json(OUT / "D_STARTED.json", {"created_utc": utc(), "D_GO_sha256": sha(go_path),
               "ready_sha256": sha(OUT / "FINAL_READY.json"), "planned_real_fits": 102}, exclusive=True)
    state = {"status": "RUNNING", "started_utc": utc(), "real_fits_started": 0,
             "real_fits_completed": 0, "fit_cpu_seconds": 0.0, "fits": [],
             "new_model_loads": 0, "new_model_forwards": 0, "new_http_requests": 0}

    def fit(kernel, train, model, penalty, fold):
        if state["real_fits_started"] >= 102 or state["fit_cpu_seconds"] >= 3600:
            raise RuntimeError("D readout budget exhausted")
        state["real_fits_started"] += 1
        row = {"model": model, "lambda": penalty, "fold": fold, "training_pairs": len(train), "status": "STARTED"}
        state["fits"].append(row)
        write_json(OUT / "FIT_LEDGER.json", state)
        tick = time.process_time()
        try:
            alpha, residual = fit_krr(kernel[np.ix_(train, train)], target[train], penalty)
            row.update(status="COMPLETE", normalized_residual=residual)
            state["real_fits_completed"] += 1
            return alpha
        finally:
            elapsed = time.process_time() - tick
            row["cpu_seconds"] = elapsed
            state["fit_cpu_seconds"] += elapsed
            write_json(OUT / "FIT_LEDGER.json", state)

    oof, development, final = {"y_a": y, "fold": folds, "group": groups}, {}, {}
    all_indices = np.arange(n)
    try:
        for model in MODELS:
            candidates = []
            for penalty in LAMBDAS:
                scores = np.empty(n)
                for fold in range(4):
                    train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                    scales = block_scales(blocks, train)
                    kernel = sum_kernel(blocks, scales, model)
                    alpha = fit(kernel, train, model, penalty, fold)
                    scores[valid] = kernel[np.ix_(valid, train)] @ alpha
                p = probability(scores)
                losses = (p - y)**2
                _, grouped = group_average(losses, groups)
                key = f"{model}_lambda{penalty:g}"
                oof[f"{key}_score"], oof[f"{key}_p_a"] = scores, p
                candidates.append({"lambda": penalty, "OOF_group_Brier": float(grouped.mean()),
                                   "OOF_pair_Brier": float(losses.mean())})
            selected = select_candidate(candidates)
            development[model] = {"selected_lambda": selected["lambda"], "candidates": candidates}
            oof[f"{model}_selected_p_a"] = oof[f"{model}_lambda{selected['lambda']:g}_p_a"]
        scales = block_scales(blocks, all_indices)
        for model in MODELS:
            penalty = development[model]["selected_lambda"]
            kernel = sum_kernel(blocks, scales, model)
            final[f"{model}_alpha"] = fit(kernel, all_indices, model, penalty, "final")
            final[f"{model}_lambda"] = np.asarray(penalty)
        if state["real_fits_completed"] != 102:
            raise RuntimeError("Incomplete fixed fit budget")
        losses = {model: (oof[f"{model}_selected_p_a"] - y)**2 for model in MODELS}
        contributions = paired_contributions(losses)
        group_ids, grouped = group_average(contributions, groups)
        plan = precision_plan(contributions, groups)
        oof.update(paired_contributions=contributions, group_ids=group_ids, group_contributions=grouped)
        np.savez_compressed(OUT / "D_OOF.npz", **oof)
        final.update({f"scale_{name}": np.asarray(value) for name, value in scales.items()})
        final.update({f"train_{name}": value for name, value in data.items()})
        final.update(train_group=groups, train_y_a=y, model_order=np.asarray(MODELS))
        np.savez_compressed(OUT / "FINAL_MODELS.npz", **final)
        write_json(OUT / "PRECISION_PLAN.json", plan, exclusive=True)
        state["status"] = "COMPLETE"
        write_json(OUT / "FIT_LEDGER.json", state)
        summary = {"status": "D_DEVELOPMENT_COMPLETE_NO_NEW_T", "completed_utc": utc(),
                   "pairs": n, "sites": len(cases), "groups": len(group_ids), "models": development,
                   "final_block_mean_diagonals": scales, "real_fits_completed": 102,
                   "readout_fit_cpu_seconds": state["fit_cpu_seconds"], "precision": plan,
                   "new_model_loads": 0, "new_model_forwards": 0, "new_http_requests": 0,
                   "scientific_scope": "Selected OOF means are developmental, not confirmation; original P1-P5 remain frozen",
                   "secondary_H_regression": "not run; H is a deterministic AND of S; no new H agent or rule selection",
                   "T_authorized": False}
        write_json(OUT / "D_DEVELOPMENT.json", summary, exclusive=True)
        print(json.dumps({"status": summary["status"], "fits": 102, "precision": plan,
                          "selected_lambda": {m: development[m]["selected_lambda"] for m in MODELS}}))
    except Exception as exc:
        state.update(status="FAILED_NO_AUTOMATIC_RETRY", failure_type=type(exc).__name__, failure=str(exc))
        write_json(OUT / "FIT_LEDGER.json", state)
        raise


def predict_saved(pair_inputs, model, final_path=OUT / "FINAL_MODELS.npz"):
    """Array-only prediction for a separately authorized future evaluator."""
    with np.load(final_path, allow_pickle=False) as saved:
        train = {name: saved[f"train_{name}"] for name in ("dx", "pa", "pb", "dj")}
        scales = {name: float(saved[f"scale_{name}"]) for name in BLOCKS}
        kernel = sum_kernel(raw_blocks(pair_inputs, train), scales, model)
        score = kernel @ saved[f"{model}_alpha"]
        return score, probability(score)


def selftest():
    import unittest
    import test_protein_krr
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_protein_krr)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"status": "PASS" if result.wasSuccessful() else "FAIL", "created_utc": utc(),
               "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "script_sha256": sha(__file__), "synthetic_only": True, "real_fits_completed": 0,
               "test_script_sha256": sha(Path(__file__).with_name("test_protein_krr.py")),
               "new_model_loads": 0, "new_model_forwards": 0, "new_http_requests": 0}
    write_json(OUT / "SELFTEST.json", receipt)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selftest", "prepare", "develop"))
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--group-field", default="current_known_union_group")
    parser.add_argument("--d-go", type=Path)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "prepare":
        prepare(args.groups.resolve(), args.group_field)
    else:
        if args.d_go is None:
            parser.error("develop requires --d-go <root-owned receipt.json>")
        develop(args.d_go)


if __name__ == "__main__":
    main()
