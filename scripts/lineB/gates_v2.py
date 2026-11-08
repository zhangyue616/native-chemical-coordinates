"""Line B protocol v2 §3-§4: exact model-level permutation tests on best-unit-per-concept lift.

usage: python -I gates_v2.py <lift_root> <out_json> <model> [<model> ...]
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, os, json

import numpy as np

K = 39


def best(arr):
    """S(c): max over units and poles. arr: (units, 2, concepts)."""
    return arr.max(axis=(0, 1)).astype(np.float64)


def exact_test(S0, others):
    """S0: (C,) target; others: list of (C,) competitors. Returns dict with T, p, E, win fraction."""
    allS = np.vstack([S0] + list(others))          # (K+1, C)
    M, C = allS.shape
    u = np.empty_like(allS)
    for i in range(M):
        less = (allS < allS[i]).sum(axis=0)
        eq = (allS == allS[i]).sum(axis=0) - 1
        u[i] = (less + 0.5 * eq) / (M - 1)
    T = u.mean(axis=1)
    p = (1 + np.sum(T[1:] >= T[0])) / M
    logS = np.log(allS)
    E = float(np.median(logS[0] - np.median(logS[1:], axis=0)))
    win = float(np.mean(allS[0] > allS[1:].max(axis=0)))
    return dict(T_target=float(T[0]), T_competitors_max=float(T[1:].max()), T_competitors_median=float(np.median(T[1:])),
                p=float(p), E_median_log_ratio=E, win_fraction=win, chance_win=1 / M, n_competitors=M - 1,
                n_concepts=C)


def load(d, t):
    return np.load(os.path.join(d, f"{t}.npy"))


def main():
    root, out_json = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    out = {"protocol": "lineB PROTOCOL_v2", "models": {}}
    for name in names:
        d = os.path.join(root, name)
        r = {}
        twins = [best(load(d, f"twin_native_{k}")) for k in range(K)]
        r["G1"] = exact_test(best(load(d, "pre_native")), twins)
        r["G1"]["pass_"] = bool(r["G1"]["p"] <= 0.05 and r["G1"]["E_median_log_ratio"] > 0)
        r["G2"] = exact_test(best(load(d, "pre_native")), [best(load(d, f"rot_{i}")) for i in range(K)])
        r["G2"]["pass_"] = bool(r["G2"]["p"] <= 0.05 and r["G2"]["E_median_log_ratio"] > 0)
        r["G2_negative_control"] = exact_test(twins[0], [best(load(d, f"twin0rot_{i}")) for i in range(K)])
        r["G2_negative_control"]["valid_"] = bool(r["G2_negative_control"]["p"] > 0.05)
        r["G3"] = exact_test(best(load(d, "pre_resid")), [best(load(d, f"twin_resid_{k}")) for k in range(K)])
        e0, e1 = r["G1"]["E_median_log_ratio"], r["G3"]["E_median_log_ratio"]
        r["G3"]["retained_ratio"] = (e1 / e0) if e0 > 0 else None
        r["G3"]["pass_"] = bool(e0 > 0 and e1 >= 0.5 * e0)
        # descriptive: input partition, pretrained vs first 3 twins
        ip = [best(load(d, f"inp_twin_native_{k}")) for k in range(3)]
        r["input_partition_descriptive"] = exact_test(best(load(d, "inp_pre_native")), ip)
        out["models"][name] = r
        print(name, json.dumps({g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r[g].items()
                                    if k in ("p", "E_median_log_ratio", "win_fraction", "pass_", "valid_",
                                             "retained_ratio")} for g in ("G1", "G2", "G2_negative_control", "G3")}),
              flush=True)
    ms = out["models"].values()
    n = len(out["models"])
    need = 2 if n >= 3 else n
    out["V2_G1_pass"] = sum(r["G1"]["pass_"] for r in ms) >= need
    out["V2_G2_pass_raw"] = sum(r["G2"]["pass_"] for r in ms) >= need
    out["V2_G2_negative_control_valid"] = sum(r["G2_negative_control"]["valid_"] for r in ms) >= need
    out["V2_G2_pass"] = bool(out["V2_G2_pass_raw"] and out["V2_G2_negative_control_valid"])
    out["V2_G3_per_model"] = {k: r["G3"]["pass_"] for k, r in out["models"].items()}
    out["proceed_to_week2"] = bool(out["V2_G1_pass"] and out["V2_G2_pass"])
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("V2_G1_pass", "V2_G2_pass_raw", "V2_G2_negative_control_valid",
                                          "V2_G2_pass", "V2_G3_per_model", "proceed_to_week2")}, indent=1))


if __name__ == "__main__":
    main()
