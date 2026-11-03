"""Shared paths, style and statistics for the line B paper figures and tables.

Every number in a figure is recomputed from the saved lift arrays and checked against the frozen gate JSON
(`check_against`), so the figures cannot drift from the reported results.
"""
import os, sys, json

import numpy as np

ROOT = os.environ.get("NATIVE_UNIT_LEGACY_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
LB = f"{ROOT}/lineB"
OUT = f"{LB}/figures"
CACHE = f"{OUT}/cache"
CDIR = f"{ROOT}/data/derived/lineB/concepts"
CORPUS = f"{ROOT}/data/derived/lineB/corpus"
ACT = f"{ROOT}/data/derived/lineB/activations"
sys.path.insert(0, f"{ROOT}/scripts/lineB")
from gates_v2 import best  # noqa: E402

K, R = 39, 9

NAMES = {"chemberta_mlm": "ChemBERTa-2 MLM", "chemberta_mtr": "ChemBERTa-2 MTR", "chemeleon": "CheMeleon",
         "molclr_gin": "MolCLR", "hu_masking": "Hu masking", "hu_edgepred": "Hu edgepred",
         "hu_contextpred": "Hu contextpred", "hu_infomax": "Hu infomax", "molformer": "MoLFormer"}
GROUP_V3 = {"chemeleon": ["chemeleon"], "molclr": ["molclr_gin"],
            "hu": ["hu_masking", "hu_contextpred", "hu_edgepred", "hu_infomax"],
            "chemberta": ["chemberta_mlm", "chemberta_mtr"]}
GROUP_OF = {m: g for g, ms in GROUP_V3.items() for m in ms}
LAYERS_V4 = {"chemberta": ["L0", "L1", "L2"], "chemeleon": ["mp3", "main"]}
LAYER_LABEL = {"L0": "layer 1", "L1": "layer 2", "L2": "layer 3 (main)", "mp3": "after MP step 3",
               "main": "output (main)"}
SUBSETS = ["all", "common", "rare", "curated"]
SUBSET_LABEL = {"all": "all (173)", "common": "common (65)", "rare": "rare (108)", "curated": "curated (91)"}

# Okabe-Ito, colour-blind safe
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
MODEL_COLOR = {"chemberta_mlm": OI["blue"], "chemberta_mtr": OI["sky"], "chemeleon": OI["vermillion"],
               "molclr_gin": OI["green"], "hu_masking": OI["purple"], "hu_edgepred": OI["purple"],
               "hu_contextpred": OI["purple"], "hu_infomax": OI["purple"], "molformer": OI["orange"]}
SUPPORTING = ("chemberta_mlm", "chemberta_mtr", "chemeleon")
V6 = os.environ.get("V6_RESULTS", f"{LB}/results_v6")   # protocol v6 results (override to rehearse on a dry run)


def setup_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300, "figure.dpi": 150})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    print("saved", name, flush=True)


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def concepts():
    return json.load(open(f"{CDIR}/concepts_v1.json", encoding="utf-8"))


# ---------------- statistics (same definitions as gates_v2.exact_test / gates_v3 / gates_v4) ----------------
def effect(S0, others):
    """E = median over concepts of log S0 - median_k log S_k (natural log)."""
    return float(np.median(np.log(S0) - np.median(np.log(np.vstack(others)), axis=0)))


def did(nat, rots):
    """Delta = log S_native - median_r log S_rot, per concept."""
    return np.log(nat) - np.median(np.log(np.vstack(rots)), axis=0)


def loo_effects(S_list):
    """Null cloud: each competitor in turn against the other K-1."""
    return np.array([effect(S_list[k], [S_list[j] for j in range(len(S_list)) if j != k])
                     for k in range(len(S_list))])


def rank_T(S0, others):
    """T for the target and every competitor (mean per-concept rank), exactly as gates_v2.exact_test."""
    allS = np.vstack([S0] + list(others))
    M = allS.shape[0]
    u = np.empty_like(allS)
    for i in range(M):
        less = (allS < allS[i]).sum(axis=0)
        eq = (allS == allS[i]).sum(axis=0) - 1
        u[i] = (less + 0.5 * eq) / (M - 1)
    return u.mean(axis=1)


def check_against(value, reference, what, tol=1e-9):
    if abs(value - reference) > tol:
        raise AssertionError(f"{what}: recomputed {value} != frozen {reference}")


# ---------------- cached best-S arrays ----------------
def _cached(path, build):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = build()
    np.savez(path, **d)
    return d


def v3_group(group):
    """Week 2 (reserve_A): best S per concept. Keys: twin_nat (K,C), twin_rot (K,R,C), <m>_nat (C,), <m>_rot (R,C)."""
    base = f"{LB}/results_v3/lifts/{group}"

    def build():
        ld = lambda sub, t: np.load(f"{base}/{sub}/{t}.npy")
        d = {"twin_nat": np.stack([best(ld("twins", f"twin_native_{k}")) for k in range(K)]),
             "twin_rot": np.stack([np.stack([best(ld("twins", f"twin_rot_{k}_{r}")) for r in range(R)])
                                   for k in range(K)]),
             "rd": np.load(f"{base}/readable_concept_ids.npy")}
        for m in GROUP_V3[group]:
            d[f"{m}_nat"] = best(ld(m, "pre_native"))
            d[f"{m}_rot"] = np.stack([best(ld(m, f"pre_rot_{r}")) for r in range(R)])
        return d
    return _cached(f"{CACHE}/v3_{group}.npz", build)


def v4_group(group, layer):
    """Week 3 (reserve_B): same keys as v3_group, for one layer."""
    base = f"{LB}/results_v4/lifts/{group}"
    names = GROUP_V3[group]

    def build():
        ld = lambda sub, t: np.load(f"{base}/{layer}/{sub}/{t}.npy")
        d = {"twin_nat": np.stack([best(ld("twins", f"twin_native_{k}")) for k in range(K)]),
             "twin_rot": np.stack([np.stack([best(ld("twins", f"twin_rot_{k}_{r}")) for r in range(R)])
                                   for k in range(K)]),
             "rd": np.load(f"{base}/readable_concept_ids.npy")}
        for m in names:
            d[f"{m}_nat"] = best(ld(m, "pre_native"))
            d[f"{m}_rot"] = np.stack([best(ld(m, f"pre_rot_{r}")) for r in range(R)])
        return d
    return _cached(f"{CACHE}/v4_{group}_{layer}.npz", build)


def joint_effects(d, m, cols=None):
    """J1 and J2 effects for model m plus the leave-one-out twin null clouds."""
    c = slice(None) if cols is None else cols
    tn = [x[c] for x in d["twin_nat"]]
    td = [did(d["twin_nat"][k][c], d["twin_rot"][k][:, c]) for k in range(K)]
    s0 = d[f"{m}_nat"][c]
    d0 = did(s0, d[f"{m}_rot"][:, c])
    e1 = effect(s0, tn)
    e2 = float(np.median(d0 - np.median(np.vstack(td), axis=0)))
    null1 = loo_effects(tn)
    null2 = np.array([float(np.median(td[k] - np.median(np.vstack([td[j] for j in range(K) if j != k]), axis=0)))
                      for k in range(K)])
    T1 = rank_T(s0, tn)
    T2 = rank_T(np.exp(d0), [np.exp(x) for x in td])
    return dict(E1=e1, E2=e2, null1=null1, null2=null2, T1=T1, T2=T2)
