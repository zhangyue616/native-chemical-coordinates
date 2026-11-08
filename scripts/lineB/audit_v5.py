"""Protocol v5 (M2): best-lift vectors S for the pretrained models, K untrained twins and R rotations each.

usage: python -I audit_v5.py <group> <tsv> <concept_dir> <label_tag> <out_root> [K] [R]
groups: chemeleon | molclr | hu | chemberta.  Defaults K = 199 (seeds 3000-3198), R = 9 (rotation seeds 100-108).
Under <out_root>/<group>/: readable_concept_ids.npy; <model>/pre_native_full.npy (units x 2 x C lift array) and
S_native.npy, S_rot_{r}.npy; twins/S_native_{k}.npy, S_rot_{k}_{r}.npy (best lift per concept, C,).
Every vector is written as soon as it is computed and skipped when present (resumable).
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, os, csv, json, time, hashlib

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402
import lift  # noqa: E402
from audit import rotation  # noqa: E402

GROUPS = {"chemeleon": ["chemeleon"], "molclr": ["molclr_gin"],
          "hu": ["hu_masking", "hu_contextpred", "hu_edgepred", "hu_infomax"],
          "chemberta": ["chemberta_mlm", "chemberta_mtr"]}
TWIN_SEED0, ROT_SEED0, BATCH = 3000, 100, 128


def main():
    group, tsv, cdir, tag, out_root = sys.argv[1:6]
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 199
    R = int(sys.argv[7]) if len(sys.argv) > 7 else 9
    act = "gelu" if group == "chemberta" else "relu"
    base = os.path.join(out_root, group)
    os.makedirs(os.path.join(base, "twins"), exist_ok=True)
    t0 = time.time()
    print(f"{group}: K={K} R={R} act={act} lift.py sha256 "
          f"{hashlib.sha256(open(lift.__file__, 'rb').read()).hexdigest()} models.py sha256 "
          f"{hashlib.sha256(open(models.__file__, 'rb').read()).hexdigest()}", flush=True)
    with open(tsv, encoding="utf-8") as fh:
        smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
    concepts = json.load(open(os.path.join(cdir, "concepts_v1.json"), encoding="utf-8"))
    Lsp = sp.load_npz(os.path.join(cdir, f"labels_{tag}.npz")).tocsc()
    N = Lsp.shape[0]
    pos = np.asarray(Lsp.sum(0)).ravel()
    rd = np.array([c["id"] for c in concepts if c["partition"] == "readable" and pos[c["id"]] > 0])
    print(f"readable concepts with positives: {len(rd)} of 173; atoms {N}", flush=True)
    np.save(os.path.join(base, "readable_concept_ids.npy"), rd)
    P = lift.Positives(torch.from_numpy(Lsp[:, rd].toarray().astype(np.uint8)).cuda())
    prev = pos[rd] / N
    rots = {}

    def rot(D, r):
        if (D, r) not in rots:
            rots[(D, r)] = torch.from_numpy(rotation(D, ROT_SEED0 + r).astype(np.float32)).cuda()
        return rots[(D, r)]

    def path(sub, name):
        return os.path.join(base, sub, f"{name}.npy")

    def save(sub, name, arr):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
        tmp = path(sub, name)[:-4] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path(sub, name))

    def lifts(Z):
        return lift.lift_array_batched(Z, P, prev, batch=BATCH, act=act)

    def run(m, fam, sub, native_name, rot_name, keep_full=False):
        need = [native_name] + [rot_name(r) for r in range(R)]
        if all(os.path.exists(path(sub, n)) for n in need):
            return
        Z, _, counts = models.extract(m, fam, smiles, on_gpu=True)
        assert Z.shape[0] == N and counts.sum() == N, (Z.shape, N)
        if not os.path.exists(path(sub, native_name)):
            full = lifts(Z)
            if keep_full:
                save(sub, "pre_native_full", full.astype(np.float32))
            save(sub, native_name, full.max(axis=(0, 1)).astype(np.float64))
        for r in range(R):
            if not os.path.exists(path(sub, rot_name(r))):
                save(sub, rot_name(r), lifts(Z @ rot(Z.shape[1], r)).max(axis=(0, 1)).astype(np.float64))
        del Z
        torch.cuda.empty_cache()

    for name in GROUPS[group]:
        m, fam = models.load_model(name)
        run(m, fam, name, "S_native", lambda r: f"S_rot_{r}", keep_full=True)
        del m
        print(f"  {name} pretrained done {time.time() - t0:.0f}s", flush=True)
    rep = GROUPS[group][0]
    for k in range(K):
        seed = TWIN_SEED0 + k
        need = [f"S_native_{k}"] + [f"S_rot_{k}_{r}" for r in range(R)]
        if all(os.path.exists(path("twins", n)) for n in need):
            continue
        t_, fam = models.load_model(rep, twin_seed=seed)
        run(t_, fam, "twins", f"S_native_{k}", lambda r, k=k: f"S_rot_{k}_{r}")
        del t_
        if (k + 1) % 10 == 0 or k < 2:
            print(f"  twin {k} (seed {seed}) done {time.time() - t0:.0f}s", flush=True)
    print(f"{group}: complete, K={K}, R={R}, readable {len(rd)}, atoms {N}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
