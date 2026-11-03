"""Line B protocol v1, D5-D6: atom-level activations and lift arrays for one pilot model.

usage: python -I audit.py <model_name> <pilot_tsv> <concept_dir> <act_dir> <lift_dir>
Writes one .npy per lift array into <lift_dir>/<model_name>/ as soon as it is computed and skips arrays that
already exist, so an interrupted run resumes where it stopped. Lift arrays have shape units x 2 poles x concepts;
pole 0 = ReLU(pre), pole 1 = -pre. Tags:
  pre_native, twin_native_{0,1,2}, rot_{0..9} (pretrained, rotated pre), pre_resid, twin_resid_{0,1,2},
  and for GIN models depth_pre_l{0..3}, depth_twin_l{0..3}_{0,1,2}.
Writes fp16 activations of the audited layer to act_dir.
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

N_ROT = 10


def rotation(d, seed):
    g = np.random.default_rng(seed).normal(size=(d, d))
    q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))


def residualize(Z, h):
    """Per-unit OLS residual of Z (N, D) on [1, h] (N,), in place on GPU."""
    hc = h - h.mean()
    b = (hc @ Z) / (hc @ hc)
    Z -= Z.mean(0, keepdim=True)
    Z -= hc[:, None] * b[None, :]
    return Z


def main():
    name, pilot_tsv, cdir, act_dir, lift_root = sys.argv[1:6]
    out_dir = os.path.join(lift_root, name)
    os.makedirs(act_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    lift_sha = hashlib.sha256(open(lift.__file__, "rb").read()).hexdigest()
    print(f"{name}: lift.py sha256 {lift_sha}", flush=True)
    with open(pilot_tsv, encoding="utf-8") as fh:
        smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
    concepts = json.load(open(os.path.join(cdir, "concepts_v1.json"), encoding="utf-8"))
    Lsp = sp.load_npz(os.path.join(cdir, "labels_pilot.npz")).tocsc()
    atoms = np.load(os.path.join(cdir, "atoms_pilot.npz"))
    N, C = Lsp.shape
    L = torch.from_numpy(Lsp.toarray().astype(np.uint8)).cuda()
    prev = np.asarray(Lsp.sum(0)).ravel() / N
    assert np.allclose(prev, [c["prevalence"] for c in concepts]), "prevalence mismatch with concepts_v1.json"
    h = torch.from_numpy(atoms["mol_heavy"].astype(np.float32)).cuda()

    def have(tag):
        return os.path.exists(os.path.join(out_dir, f"{tag}.npy"))

    def save(tag, arr):
        tmp = os.path.join(out_dir, f"{tag}.tmp.npy")
        np.save(tmp, arr.astype(np.float32))
        os.replace(tmp, os.path.join(out_dir, f"{tag}.npy"))
        print(f"  {tag} done {time.time() - t0:.0f}s", flush=True)

    def extract(model, fam, layer=None):
        Z, _, counts = models.extract(model, fam, smiles, layer=layer, on_gpu=True)
        assert Z.shape[0] == N and counts.sum() == N, (Z.shape, N)
        return Z

    gin = not name.startswith("chemeleon")
    # twins
    for k, seed in enumerate(models.TWIN_SEEDS):
        tags = [f"twin_native_{k}", f"twin_resid_{k}"] + ([f"depth_twin_l{l}_{k}" for l in range(4)] if gin else [])
        if all(have(t) for t in tags):
            continue
        t, fam = models.load_model(name, twin_seed=seed)
        if not (have(f"twin_native_{k}") and have(f"twin_resid_{k}")):
            Z = extract(t, fam)
            np.save(os.path.join(act_dir, f"{name}_twin{seed}_pre_f16.npy"), Z.half().cpu().numpy())
            if not have(f"twin_native_{k}"):
                save(f"twin_native_{k}", lift.lift_array(Z, L, prev))
            if not have(f"twin_resid_{k}"):
                save(f"twin_resid_{k}", lift.lift_array(residualize(Z, h), L, prev))
            del Z
        if gin:
            for layer in range(4):
                if not have(f"depth_twin_l{layer}_{k}"):
                    Zl = extract(t, fam, layer)
                    save(f"depth_twin_l{layer}_{k}", lift.lift_array(Zl, L, prev))
                    del Zl
        del t
        torch.cuda.empty_cache()
    # pretrained
    tags = ["pre_native", "pre_resid"] + [f"rot_{r}" for r in range(N_ROT)] + \
           ([f"depth_pre_l{l}" for l in range(4)] if gin else [])
    if not all(have(t) for t in tags):
        m, fam = models.load_model(name)
        if not all(have(t) for t in ["pre_native", "pre_resid"] + [f"rot_{r}" for r in range(N_ROT)]):
            Z = extract(m, fam)
            np.save(os.path.join(act_dir, f"{name}_pretrained_pre_f16.npy"), Z.half().cpu().numpy())
            if not have("pre_native"):
                save("pre_native", lift.lift_array(Z, L, prev))
            D = Z.shape[1]
            for r in range(N_ROT):
                if not have(f"rot_{r}"):
                    R = torch.from_numpy(rotation(D, r).astype(np.float32)).cuda()
                    save(f"rot_{r}", lift.lift_array(Z @ R, L, prev))
                    del R
                    torch.cuda.empty_cache()
            if not have("pre_resid"):
                save("pre_resid", lift.lift_array(residualize(Z, h), L, prev))
            del Z
        if gin:
            for layer in range(4):
                if not have(f"depth_pre_l{layer}"):
                    Zl = extract(m, fam, layer)
                    save(f"depth_pre_l{layer}", lift.lift_array(Zl, L, prev))
                    del Zl
    print(f"{name}: complete, {len(os.listdir(out_dir))} arrays, units-x-2-x-{C}, atoms {N}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
