"""Post hoc M4 (lineB/POSTHOC_M1_M4_PLAN.md): weight-shuffled networks as an alternative null model.

usage: python -I posthoc_m4_shuffled.py
For each of the eight pretrained models, 39 networks in which every floating-point parameter and buffer tensor is
permuted within itself (seeds 2000-2038). S (best lift per concept) is computed on H2 (reserve_A) exactly as in the
main study and compared with the pretrained model and with the cached untrained twins. Resumable: each S is saved.
"""
import os, sys, csv, json, time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figures"))
import models  # noqa: E402
import lift  # noqa: E402
import common as C  # noqa: E402
from gates_v2 import best, exact_test  # noqa: E402

SEEDS = list(range(2000, 2039))
OUT = f"{C.LB}/results_posthoc/m4"
BATCH = 128


def shuffled_state(sd, seed):
    g = torch.Generator().manual_seed(seed)
    new = {}
    for k in sorted(sd):
        v = sd[k]
        if torch.is_floating_point(v) and v.numel() > 1:
            flat = v.reshape(-1)
            new[k] = flat[torch.randperm(flat.numel(), generator=g)].reshape(v.shape).clone()
        else:
            new[k] = v.clone()
    return new


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with open(f"{C.CORPUS}/reserveA_used.tsv", encoding="utf-8") as fh:
        smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
    Lsp = sp.load_npz(f"{C.CDIR}/labels_reserveA.npz").tocsc()
    N = Lsp.shape[0]
    npos = np.asarray(Lsp.sum(0)).ravel()
    summary = {}
    for group, names in C.GROUP_V3.items():
        d = C.v3_group(group)
        rd = d["rd"]
        P = lift.Positives(torch.from_numpy(Lsp[:, rd].toarray().astype(np.uint8)).cuda())
        prev = npos[rd] / N
        act = "gelu" if group == "chemberta" else "relu"
        twins = [d["twin_nat"][k] for k in range(C.K)]
        for m in names:
            os.makedirs(f"{OUT}/{m}", exist_ok=True)
            model, fam = models.load_model(m)
            orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            def S_of():
                Z, _, counts = models.extract(model, fam, smiles, on_gpu=True)
                assert Z.shape[0] == N and counts.sum() == N
                s = best(lift.lift_array_batched(Z, P, prev, batch=BATCH, act=act))
                del Z
                torch.cuda.empty_cache()
                return s

            chk = f"{OUT}/{m}/pretrained_check.npy"
            if not os.path.exists(chk):
                np.save(chk, S_of())
            s_pre = np.load(chk)
            dev = float(np.median(np.abs(np.log(s_pre) - np.log(d[f"{m}_nat"]))))
            assert dev < 1e-3, f"{m}: pipeline check failed, median |log diff| = {dev}"
            shuf = []
            for k, seed in enumerate(SEEDS):
                p = f"{OUT}/{m}/shuffled_{k}.npy"
                if not os.path.exists(p):
                    model.load_state_dict(shuffled_state(orig, seed))
                    np.save(p[:-4] + ".tmp.npy", S_of())
                    os.replace(p[:-4] + ".tmp.npy", p)
                shuf.append(np.load(p))
            model.load_state_dict(orig)
            s0 = d[f"{m}_nat"]
            vs_shuf = exact_test(s0, shuf)
            shuf_vs_twins = [C.effect(s, twins) for s in shuf]
            summary[m] = dict(
                pipeline_check_median_abs_log_diff=dev,
                pretrained_vs_shuffled=dict(p=vs_shuf["p"], fold=float(np.exp(vs_shuf["E_median_log_ratio"])),
                                            concepts_won=vs_shuf["win_fraction"]),
                shuffled_vs_twins_fold=dict(median=float(np.exp(np.median(shuf_vs_twins))),
                                            min=float(np.exp(np.min(shuf_vs_twins))),
                                            max=float(np.exp(np.max(shuf_vs_twins)))),
                pretrained_vs_twins_fold=float(np.exp(C.effect(s0, twins))))
            print(m, json.dumps(summary[m]), f"{time.time() - t0:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()
    json.dump(summary, open(f"{C.LB}/results_posthoc/m4_shuffled.json", "w"), indent=1)
    print("M4 done", flush=True)


if __name__ == "__main__":
    main()
