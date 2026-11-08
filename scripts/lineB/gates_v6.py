"""Protocol v6 §4-5: joint test of MoLFormer-XL-both-10pct on H4 with a maximum-over-layers statistic.

usage: python -I gates_v6.py <out_root> <concept_dir> <out_json> [--K 199] [--R 9] [--K_hf 39] [--K_orf 9] [--dry]
                              [--freeze_dir DIR] [--token_visible path]
Run once, after audit_v6.py reports complete. For each layer l and network m, T1[m, l] is the mean-rank statistic of
gates_v2.exact_test over all networks at that layer for S, and T2[m, l] the same for exp(Δ), Δ = log S_native −
median_r log S_rotated. J1: each network's statistic is max_l T1[m, l] (ties to the deeper layer); p1 from the rank
of the pretrained model's maximum among the twins' maxima. J2: each network's statistic is T2 at its own J1 layer;
p2 likewise. Effects, intervals, win shares and the twins' leave-one-out references are same-layer quantities at
each network's own J1 layer. Pre-specified secondary: the last layer. Descriptive: J2 maximised freely, per-layer J1
and J2, the twins' statistics by layer, the concept subsets of protocol v4, the HF-initialization twins (J1 only),
the random-feature redraws of the pretrained model (J1 only) and the token-visible concepts.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, os, json, argparse, hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from gates_v2 import exact_test  # noqa: E402
from audit_v6 import LAYERS, LAST, PROTOCOL_DESIGN, network_list, verify_freeze, sha  # noqa: E402

ALPHA, ALPHA_FW = 0.05, 0.05 / 8   # with K = 199, p <= 0.00625 means p = 0.005: first of 200 networks
V5_IDS = os.path.join(ROOT, "lineB", "results_v5", "chemberta", "readable_concept_ids.npy")
SUBSET_SIZES = {"common": 65, "rare": 108, "curated": 91}   # protocol v4
TOKEN_VISIBLE_IDS = [23, 42, 102, 136, 163]                  # protocol v6 §1, computed on the pilot set
DEFAULT_FREEZE_DIR = os.path.join(ROOT, "lineB")
DEFAULT_OUT_ROOT = os.path.join(ROOT, "lineB", "results_v6")


def boot(r, n=2000, seed=0):
    """95% concept-bootstrap interval of the fold (as in gates_v5)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n, len(r)))
    return [float(x) for x in np.exp(np.percentile(np.median(r[idx], axis=1), [2.5, 97.5]))]


def T_stat(X):
    """X: (M, C) values; returns (M,) mean-rank statistics as in gates_v2.exact_test (ties count one half)."""
    M, C = X.shape
    u = np.empty_like(X, dtype=np.float64)
    for i in range(M):
        less = (X < X[i]).sum(axis=0)
        eq = (X == X[i]).sum(axis=0) - 1
        u[i] = (less + 0.5 * eq) / (M - 1)
    return u.mean(axis=1)


def argmax_deep(T):
    """Index of the maximum along axis 1; ties go to the last (deepest) layer."""
    return T.shape[1] - 1 - np.argmax(T[:, ::-1], axis=1)


def delta(nat, rots):
    return np.log(nat) - np.median(np.log(np.vstack(rots)), axis=0)


def outcome(p1, e1, p2, e2, p2f):
    """J2 passes only if it passes at the J1 layer AND by the free maximum over layers (exact under the J2 null)."""
    pass1, pass2 = (p1 <= ALPHA and e1 > 0), (p2 <= ALPHA and e2 > 0 and p2f <= ALPHA)
    if p1 <= ALPHA_FW and p2 <= ALPHA_FW and p2f <= ALPHA_FW and e1 > 0 and e2 > 0:
        return "O1", "supported, both p = 0.005 (first of 200 networks)"
    if pass1 and pass2:
        return "O2", "supported at the uncorrected level only"
    if not pass1:
        return "O3", "J1 fails: not shown to exceed its untrained twins" + (" (J2 passes)" if pass2 else "")
    return "O4", "J1 passes, J2 fails: advantage not shown to sit on native units"


def pattern_text(pat):
    return {(True, True): "passed both tests", (True, False): "passed J1 only", (False, True): "passed J2 only",
            (False, False): "passed neither"}[(pat["J1_pass"], pat["J2_pass"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root"); ap.add_argument("cdir"); ap.add_argument("out_json")
    ap.add_argument("--K", type=int, default=199); ap.add_argument("--R", type=int, default=9)
    ap.add_argument("--K_hf", type=int, default=39); ap.add_argument("--K_orf", type=int, default=9)
    ap.add_argument("--dry", action="store_true", help="dry run: no protocol asserts, no freeze check")
    ap.add_argument("--freeze_dir", default=DEFAULT_FREEZE_DIR)
    ap.add_argument("--token_visible", default=os.path.join(ROOT, "lineB", "results_v6", "token_visible_concepts.json"))
    a = ap.parse_args()
    K, R, K_hf, K_orf = a.K, a.R, a.K_hf, a.K_orf
    base = os.path.join(a.out_root, "molformer")
    rd = np.load(os.path.join(base, "readable_concept_ids.npy"))
    out = {"protocol": "lineB PROTOCOL_v6", "model": "MoLFormer-XL-both-10pct", "alpha": ALPHA, "alpha_v5_level": ALPHA_FW,
           "gates_v6_sha256": sha(os.path.abspath(__file__))}
    print("gates_v6.py sha256", out["gates_v6_sha256"], flush=True)
    rec = json.load(open(os.path.join(base, "run_record.json")))
    fps = json.load(open(os.path.join(base, "fingerprints.json")))
    blind = bool(rec["starts"][0]["args"].get("blind", False))
    rehearse = (not a.dry) or os.path.abspath(a.freeze_dir) != os.path.abspath(DEFAULT_FREEZE_DIR)
    if rehearse:   # the freeze check and the start-consistency loop also run in a dry run with a scratch freeze
        fz = verify_freeze(a.freeze_dir)
        out["freeze"] = {k: v for k, v in fz.items() if k != "entries"}
        s0 = rec["starts"][0]
        for s in rec["starts"]:
            assert s["freeze"]["protocol_line"] == fz["protocol_line"] and s["freeze"]["manifest_sha256"] == fz["manifest_sha256"]
            assert s["script_sha256"] == s0["script_sha256"] and s["input_sha256"] == s0["input_sha256"], "starts differ"
        assert "completed" in rec and rec["completed"]["K"] == K, "audit not complete"
    if not a.dry:
        assert os.path.abspath(a.freeze_dir) == os.path.abspath(DEFAULT_FREEZE_DIR), "confirmatory gates use lineB/ as freeze dir"
        assert os.path.abspath(a.out_root) == os.path.abspath(DEFAULT_OUT_ROOT), "confirmatory gates read lineB/results_v6"
        assert os.path.abspath(a.token_visible) == os.path.abspath(os.path.join(DEFAULT_OUT_ROOT, "token_visible_concepts.json"))
        assert (K, R, K_hf, K_orf) == tuple(PROTOCOL_DESIGN[k] for k in ("K", "R", "K_hf", "K_orf")), "confirmatory K, R, K_hf, K_orf"
        assert os.path.exists(V5_IDS) and np.array_equal(rd, np.load(V5_IDS)) and len(rd) == 173, "concept set differs from v5"
        assert not rec["completed"]["blind"] and not blind
        assert rec["completed"]["N_atoms"] == PROTOCOL_DESIGN["N_atoms"] and rec["completed"]["N_mol"] == PROTOCOL_DESIGN["N_mol"]
        for s in rec["starts"]:
            assert s["args"]["tag"] == "H4" and s["args"]["n_mol"] is None and not s["args"]["blind"] and s["confirmatory"]
        assert not os.path.exists(os.path.join(base, "twins", LAST, f"S_native_{K}.npy")), "more twins than K on disk"
    nets = network_list(K, K_hf, K_orf, blind)
    names = [n for _, n, _, _, _ in nets]
    assert set(fps) == set(names) and len({v["state_dict_sha256_16"] for v in fps.values()}) == len(names), \
        "fingerprints missing or not distinct"
    main_names = ["pretrained"] + [f"twin_{k}" for k in range(K)]
    hf_names = [f"hf_{k}" for k in range(K_hf)]
    orf_names = [n for n in names if n.startswith("orf_")]

    def ld(name, layer, r=None):
        sub = "pretrained" if name == "pretrained" else {"twin": "twins", "hf": "twins_hf", "orf": "pretrained_orf"}[name.split("_")[0]]
        k = None if name == "pretrained" else name.split("_")[1]
        f = ("S_native" if k is None else f"S_native_{k}") if r is None else (f"S_rot_{r}" if k is None else f"S_rot_{k}_{r}")
        return np.load(os.path.join(base, sub, layer, f + ".npy"))

    S = {n: {l: ld(n, l) for l in LAYERS} for n in names}
    D = {n: {l: delta(S[n][l], [ld(n, l, r) for r in range(R)]) for l in LAYERS} for n in main_names}
    for n in main_names:
        sub = "pretrained" if n == "pretrained" else "twins"
        extra = f"S_rot_{R}" if n == "pretrained" else f"S_rot_{n.split('_')[1]}_{R}"
        assert not os.path.exists(os.path.join(base, sub, LAST, extra + ".npy")), "R mismatch"
    X2 = {n: {l: np.exp(D[n][l]) for l in LAYERS} for n in main_names}
    C = len(rd)
    allc = np.arange(C)

    def same_layer(names_, X, cols, layer_of, i):
        """Same-layer log ratio vector of network names_[i] at layer_of[i]: the target (i = 0) against all twins, a
        twin (i >= 1) against the other twins (the target excluded)."""
        l = layer_of[i]
        others = np.vstack([np.log(X[n][l][cols]) for j, n in enumerate(names_) if j != i and j != 0])
        return np.log(X[names_[i]][l][cols]) - np.median(others, axis=0)

    def joint(names_, cols, seed, primary=False):
        """J1 by the maximum over layers; J2 at each network's J1 layer. names_[0] is the target."""
        M = len(names_)
        T1 = np.array([T_stat(np.vstack([S[n][l][cols] for n in names_])) for l in LAYERS]).T   # (M, 12)
        T2 = np.array([T_stat(np.vstack([X2[n][l][cols] for n in names_])) for l in LAYERS]).T
        arg = argmax_deep(T1)
        idx = np.arange(M)
        T1m, T2m = T1[idx, arg], T2[idx, arg]
        p1 = float((1 + np.sum(T1m[1:] >= T1m[0])) / M)
        p2 = float((1 + np.sum(T2m[1:] >= T2m[0])) / M)
        layer_of = [LAYERS[int(i)] for i in arg]
        r1 = same_layer(names_, S, cols, layer_of, 0)
        r2 = same_layer(names_, X2, cols, layer_of, 0)
        E1, E2 = float(np.median(r1)), float(np.median(r2))
        loo1 = [float(np.median(same_layer(names_, S, cols, layer_of, i))) for i in range(1, M)]
        loo2 = [float(np.median(same_layer(names_, X2, cols, layer_of, i))) for i in range(1, M)]

        def win(X, i):   # share of concepts won: the target against all twins; a twin against the other twins
            l = layer_of[i]
            others = np.vstack([X[n][l][cols] for j, n in enumerate(names_) if j != i and j != 0])
            return float(np.mean(X[names_[i]][l][cols] > others.max(axis=0)))
        w1, w2 = win(S, 0), win(X2, 0)
        tw1 = [win(S, i) for i in range(1, M)]
        tw2 = [win(X2, i) for i in range(1, M)]
        rng = lambda v: [float(np.exp(x)) for x in np.percentile(v, [2.5, 97.5])]
        # J2 by the free maximum over layers: exact under the J2 null; required in addition to J2 at the J1 layer
        arg2 = argmax_deep(T2)
        T2f = T2[idx, arg2]
        p2f = float((1 + np.sum(T2f[1:] >= T2f[0])) / M)
        res = dict(K=M - 1, n_concepts=int(len(cols)), layer=layer_of[0],
                   J1=dict(p=p1, T_target=float(T1m[0]), T_twins_max=float(T1m[1:].max()), E=E1, fold=float(np.exp(E1)),
                           fold_ci_conditional_on_layer=boot(r1, seed=seed), win_share=w1,
                           twins_loo_fold_2p5_97p5=rng(loo1), twins_loo_fold_range=[float(np.exp(min(loo1))), float(np.exp(max(loo1)))],
                           twins_win_share_2p5_97p5=[float(x) for x in np.percentile(tw1, [2.5, 97.5])]),
                   J2=dict(p=p2, p_free=p2f, layer_free=LAYERS[int(arg2[0])], free_same_as_J1_layer=bool(arg2[0] == arg[0]),
                           T_target=float(T2m[0]), T_twins_max=float(T2m[1:].max()), E=E2, fold=float(np.exp(E2)),
                           fold_ci_conditional_on_layer=boot(r2, seed=100 + seed), win_share=w2,
                           twins_loo_fold_2p5_97p5=rng(loo2), twins_loo_fold_range=[float(np.exp(min(loo2))), float(np.exp(max(loo2)))],
                           twins_win_share_2p5_97p5=[float(x) for x in np.percentile(tw2, [2.5, 97.5])]),
                   T1_by_layer_target={l: float(T1[0, i]) for i, l in enumerate(LAYERS)},
                   T2_by_layer_target={l: float(T2[0, i]) for i, l in enumerate(LAYERS)},
                   twins_J1_layer_counts={l: int(np.sum(arg[1:] == i)) for i, l in enumerate(LAYERS)},
                   twins_T1_by_layer_median={l: float(np.median(T1[1:, i])) for i, l in enumerate(LAYERS)},
                   twins_T1_by_layer_97p5={l: float(np.percentile(T1[1:, i], 97.5)) for i, l in enumerate(LAYERS)},
                   twins_T2_by_layer_median={l: float(np.median(T2[1:, i])) for i, l in enumerate(LAYERS)},
                   twins_T2_by_layer_97p5={l: float(np.percentile(T2[1:, i], 97.5)) for i, l in enumerate(LAYERS)},
                   pattern=dict(J1_pass=bool(p1 <= ALPHA and E1 > 0), J2_pass=bool(p2 <= ALPHA and E2 > 0 and p2f <= ALPHA)))
        if primary:
            code, label = outcome(p1, E1, p2, E2, p2f)
            res.update(outcome=code, outcome_label=label, supports_uncorrected=code in ("O1", "O2"),
                       supports_v5_level=code == "O1")
        return res

    def fixed_layer(names_, layer, cols, seed):
        """Both tests with every network at one fixed layer (the exact test of protocol v5)."""
        s = [S[n][layer][cols] for n in names_]
        d = [D[n][layer][cols] for n in names_]
        j1 = exact_test(s[0], s[1:])
        j2 = exact_test(np.exp(d[0]), [np.exp(x) for x in d[1:]])
        r1 = np.log(s[0]) - np.median(np.log(np.vstack(s[1:])), axis=0)
        r2 = d[0] - np.median(np.vstack(d[1:]), axis=0)
        assert abs(np.median(r1) - j1["E_median_log_ratio"]) < 1e-12 and abs(np.median(r2) - j2["E_median_log_ratio"]) < 1e-12
        return dict(layer=layer, J1=j1, J2=j2, J1_fold=float(np.exp(j1["E_median_log_ratio"])), J1_fold_ci=boot(r1, seed=seed),
                    J2_fold=float(np.exp(j2["E_median_log_ratio"])), J2_fold_ci=boot(r2, seed=100 + seed),
                    pattern=dict(J1_pass=bool(j1["p"] <= ALPHA and j1["E_median_log_ratio"] > 0),
                                 J2_pass=bool(j2["p"] <= ALPHA and j2["E_median_log_ratio"] > 0)))

    # ---- confirmatory: primary (maximum over layers; J2 at the J1 layer) ----
    out["primary"] = joint(main_names, allc, seed=0, primary=True)
    out["outcome"], out["G6_pass"] = out["primary"]["outcome"], out["primary"]["supports_uncorrected"]
    p = out["primary"]
    print(f"primary: layer {p['layer']}; J1 p={p['J1']['p']:.3f} fold={p['J1']['fold']:.3f} | J2 p={p['J2']['p']:.3f} "
          f"fold={p['J2']['fold']:.3f}, free maximum p={p['J2']['p_free']:.3f} at {p['J2']['layer_free']} -> {p['outcome']} "
          f"({p['outcome_label']})", flush=True)
    # ---- pre-specified secondary: the last layer ----
    out["last_layer"] = fixed_layer(main_names, LAST, allc, seed=1)
    q = out["last_layer"]
    out["disagreement"] = dict(differs=q["pattern"] != p["pattern"], last_layer_text=pattern_text(q["pattern"]),
                               primary_text=pattern_text(p["pattern"]),
                               abstract_clause="; at its last layer, the rule applied to the other models, it "
                                               + pattern_text(q["pattern"]) if q["pattern"] != p["pattern"] else "")
    print(f"last layer: p1={q['J1']['p']:.3f} fold1={q['J1_fold']:.3f} p2={q['J2']['p']:.3f} fold2={q['J2_fold']:.3f} "
          f"-> {out['disagreement']['last_layer_text']} (primary: {out['disagreement']['primary_text']})", flush=True)
    # ---- descriptive: every layer ----
    out["depth_profile_descriptive"] = {}
    for l in LAYERS:
        f = fixed_layer(main_names, l, allc, seed=200 + int(l[1:]))
        out["depth_profile_descriptive"][l] = dict(
            J1_fold=f["J1_fold"], J1_p=f["J1"]["p"], J1_win=f["J1"]["win_fraction"], J2_fold=f["J2_fold"], J2_p=f["J2"]["p"],
            J2_win=f["J2"]["win_fraction"], median_S_pretrained=float(np.median(S["pretrained"][l])),
            median_S_twins=float(np.median([np.median(S[n][l]) for n in main_names[1:]])),
            median_delta_twins=float(np.median([np.median(D[n][l]) for n in main_names[1:]])),
            twins_T1_median=p["twins_T1_by_layer_median"][l], twins_T1_97p5=p["twins_T1_by_layer_97p5"][l],
            twins_T2_median=p["twins_T2_by_layer_median"][l], twins_T2_97p5=p["twins_T2_by_layer_97p5"][l])
    print("J1 by layer:", {l: round(v["J1_fold"], 2) for l, v in out["depth_profile_descriptive"].items()}, flush=True)
    print("J2 by layer:", {l: round(v["J2_fold"], 2) for l, v in out["depth_profile_descriptive"].items()}, flush=True)
    # ---- descriptive: concept subsets of protocol v4 ----
    concepts = json.load(open(os.path.join(a.cdir, "concepts_v1.json"), encoding="utf-8"))
    prev = np.array([concepts[c]["prevalence"] for c in rd])
    src = np.array([concepts[c]["source"] for c in rd])
    subsets = {"common": np.nonzero(prev >= 0.005)[0], "rare": np.nonzero(prev < 0.005)[0],
               "curated": np.nonzero(src != "ertl")[0]}
    if not a.dry:
        assert {k: len(v) for k, v in subsets.items()} == SUBSET_SIZES, "subset sizes differ from protocol v4"
    out["subsets_descriptive"] = {k: joint(main_names, cols, seed=10 + i) for i, (k, cols) in enumerate(subsets.items())}
    for k, v in out["subsets_descriptive"].items():
        print(f"  subset {k}: n={v['n_concepts']} layer {v['layer']} J1 p={v['J1']['p']:.3f} fold={v['J1']['fold']:.3f} "
              f"J2 p={v['J2']['p']:.3f} fold={v['J2']['fold']:.3f}", flush=True)
    # ---- descriptive: twins from the Hugging Face initialization (J1 only) ----
    hn = ["pretrained"] + hf_names
    T1h = np.array([T_stat(np.vstack([S[n][l] for n in hn])) for l in LAYERS]).T
    argh = argmax_deep(T1h)
    T1hm = T1h[np.arange(len(hn)), argh]
    lh = LAYERS[int(argh[0])]
    rh = np.log(S["pretrained"][lh]) - np.median(np.log(np.vstack([S[n][lh] for n in hf_names])), axis=0)
    looh = []   # each hf twin at its own maximum layer against the other hf twins
    for i, n in enumerate(hf_names, start=1):
        l = LAYERS[int(argh[i])]
        looh.append(float(np.median(np.log(S[n][l]) - np.median(np.log(np.vstack([S[m][l] for m in hf_names if m != n])), axis=0))))
    l1 = p["layer"]
    same = exact_test(S["pretrained"][l1], [S[n][l1] for n in hf_names])
    out["hf_init_descriptive"] = dict(
        K=K_hf, J1_max=dict(layer=lh, p=float((1 + np.sum(T1hm[1:] >= T1hm[0])) / len(hn)), fold=float(np.exp(np.median(rh))),
                            fold_ci_conditional_on_layer=boot(rh, seed=50),
                            twins_loo_fold_2p5_97p5=[float(np.exp(x)) for x in np.percentile(looh, [2.5, 97.5])]),
        J1_same_layer_at_primary_layer=dict(layer=l1, p=same["p"], fold=float(np.exp(same["E_median_log_ratio"])),
                                            note="layer chosen from the pretrained model's data; not an exact test"))
    h = out["hf_init_descriptive"]
    print(f"hf-init twins (K={K_hf}): J1 max p={h['J1_max']['p']:.3f} fold={h['J1_max']['fold']:.3f} ({lh}); at {l1}: "
          f"p={same['p']:.3f} fold={h['J1_same_layer_at_primary_layer']['fold']:.3f}", flush=True)
    # ---- descriptive: random-feature redraws of the pretrained model (J1 only, each against the 199 twins) ----
    if orf_names:
        rows = []
        for n in orf_names:
            vn = [n] + main_names[1:]
            T1v = np.array([T_stat(np.vstack([S[m][l] for m in vn])) for l in LAYERS]).T
            argv = argmax_deep(T1v)
            T1vm = T1v[np.arange(len(vn)), argv]
            lv = LAYERS[int(argv[0])]
            rv = np.log(S[n][lv]) - np.median(np.log(np.vstack([S[m][lv] for m in main_names[1:]])), axis=0)
            at = exact_test(S[n][l1], [S[m][l1] for m in main_names[1:]])
            rows.append(dict(variant=n, max_layer=lv, max_p=float((1 + np.sum(T1vm[1:] >= T1vm[0])) / len(vn)),
                             max_fold=float(np.exp(np.median(rv))), at_primary_layer_p=at["p"],
                             at_primary_layer_fold=float(np.exp(at["E_median_log_ratio"]))))
        out["orf_redraw_descriptive"] = dict(
            n=len(rows), primary_layer=l1, variants=rows,
            twins_loo_fold_2p5_97p5=p["J1"]["twins_loo_fold_2p5_97p5"],   # the same 199 twins as in the primary test
            max_p_range=[min(r["max_p"] for r in rows), max(r["max_p"] for r in rows)],
            max_fold_range=[min(r["max_fold"] for r in rows), max(r["max_fold"] for r in rows)],
            at_primary_layer_p_range=[min(r["at_primary_layer_p"] for r in rows), max(r["at_primary_layer_p"] for r in rows)],
            at_primary_layer_fold_range=[min(r["at_primary_layer_fold"] for r in rows), max(r["at_primary_layer_fold"] for r in rows)])
        o = out["orf_redraw_descriptive"]
        print("ORF redraws: max-layer fold", [round(x, 3) for x in o["max_fold_range"]], "p", o["max_p_range"],
              f"| at {l1}: fold", [round(x, 3) for x in o["at_primary_layer_fold_range"]], "p", o["at_primary_layer_p_range"], flush=True)
    # ---- descriptive: token-visible concepts at the primary layer ----
    tvj = json.load(open(a.token_visible, encoding="utf-8"))
    idx = {int(c): i for i, c in enumerate(rd)}
    if not a.dry:
        assert tvj["ids"] == TOKEN_VISIBLE_IDS and all(c in idx for c in TOKEN_VISIBLE_IDS), "token-visible list differs"
    tv = [idx[c] for c in tvj["ids"] if c in idx]
    r0 = np.log(S["pretrained"][l1]) - np.median(np.log(np.vstack([S[n][l1] for n in main_names[1:]])), axis=0)
    shares = {int(r["id"]): r["share_on_tokens_absent_in_chemberta"] for r in tvj["concepts"]}
    out["token_visible_descriptive"] = dict(
        rule=tvj["rule"], threshold=tvj["threshold"], layer=l1,
        concepts=[dict(id=int(rd[i]), name=concepts[int(rd[i])]["name"], share=shares[int(rd[i])], fold=float(np.exp(r0[i]))) for i in tv],
        median_fold_token_visible=float(np.exp(np.median(r0[tv]))) if tv else None,
        median_fold_other=float(np.exp(np.median(np.delete(r0, tv)))))
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("outcome", "G6_pass")}))


if __name__ == "__main__":
    main()
