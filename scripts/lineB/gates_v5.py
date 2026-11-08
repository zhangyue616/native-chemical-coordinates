"""Protocol v5 §4-5: joint test with K twins per architecture, Bonferroni over eight models, and comparison with v3.

usage: python -I gates_v5.py <out_root> <out_json> [v3_gates_json]
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, os, json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gates_v2 import exact_test  # noqa: E402
from audit_v5 import GROUPS  # noqa: E402

ALPHA_FW = 0.05 / 8


def ld(base, sub, name):
    return np.load(os.path.join(base, sub, f"{name}.npy"))


def delta(nat, rots):
    return np.log(nat) - np.median(np.log(np.vstack(rots)), axis=0)


def boot(r, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n, len(r)))
    return [float(x) for x in np.exp(np.percentile(np.median(r[idx], axis=1), [2.5, 97.5]))]


def main():
    root, out_json = sys.argv[1:3]
    v3 = json.load(open(sys.argv[3]))["models"] if len(sys.argv) > 3 else None
    out = {"protocol": "lineB PROTOCOL_v5", "alpha_familywise_per_model": ALPHA_FW, "models": {}}
    for group, names in GROUPS.items():
        base = os.path.join(root, group)
        if not os.path.isdir(base):
            continue
        tw = sorted(int(f[len("S_native_"):-4]) for f in os.listdir(os.path.join(base, "twins"))
                    if f.startswith("S_native_") and ".tmp" not in f)
        K = len(tw)
        assert tw == list(range(K)), "twin indices not contiguous"
        R = len([f for f in os.listdir(os.path.join(base, "twins")) if f.startswith("S_rot_0_") and ".tmp" not in f])
        tn = [ld(base, "twins", f"S_native_{k}") for k in range(K)]
        td = [delta(tn[k], [ld(base, "twins", f"S_rot_{k}_{r}") for r in range(R)]) for k in range(K)]
        for i, m in enumerate(names):
            s0 = ld(base, m, "S_native")
            d0 = delta(s0, [ld(base, m, f"S_rot_{r}") for r in range(R)])
            j1 = exact_test(s0, tn)
            j2 = exact_test(np.exp(d0), [np.exp(x) for x in td])
            r1 = np.log(s0) - np.median(np.log(np.vstack(tn)), axis=0)
            r2 = d0 - np.median(np.vstack(td), axis=0)
            assert abs(np.median(r1) - j1["E_median_log_ratio"]) < 1e-12
            assert abs(np.median(r2) - j2["E_median_log_ratio"]) < 1e-12
            pos = j1["E_median_log_ratio"] > 0 and j2["E_median_log_ratio"] > 0
            res = dict(K=K, R=R, n_concepts=int(len(s0)), J1=j1, J2=j2,
                       J1_fold=float(np.exp(j1["E_median_log_ratio"])), J1_fold_ci=boot(r1, seed=i),
                       J2_fold=float(np.exp(j2["E_median_log_ratio"])), J2_fold_ci=boot(r2, seed=100 + i),
                       supports_familywise=bool(pos and j1["p"] <= ALPHA_FW and j2["p"] <= ALPHA_FW),
                       supports_uncorrected=bool(pos and j1["p"] <= 0.05 and j2["p"] <= 0.05))
            if v3:
                res["v3_supports"] = bool(v3[m]["supports_H_joint"])
                res["agrees_with_v3_uncorrected"] = res["supports_uncorrected"] == res["v3_supports"]
            out["models"][m] = res
            print(m, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in
                                 dict(p1=j1["p"], fold1=res["J1_fold"], p2=j2["p"], fold2=res["J2_fold"],
                                      fw=res["supports_familywise"], unc=res["supports_uncorrected"]).items()}),
                  flush=True)
    sup = [m for m, r in out["models"].items() if r["supports_familywise"]]
    out["n_models"] = len(out["models"])
    out["supporting_familywise"] = sup
    out["G5_pass"] = bool(len(sup) >= 2)
    if v3:
        out["agreement_with_v3_uncorrected"] = sum(r["agrees_with_v3_uncorrected"] for r in out["models"].values())
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_models", "supporting_familywise", "G5_pass")}, indent=1))


if __name__ == "__main__":
    main()
