"""Determinism, rebuild and padding-invariance check for MoLFormer activations (protocol v6 §2; pilot molecules).

usage: python -I molformer_determinism_check.py <out_json> [n_mol=256]
Acceptance criteria (asserted; the JSON records "passed"): for the pretrained model, twin 0 (seed 4000), the first
HF-initialization twin (seed 6000) and the first random-feature redraw (seed 7000): two forward passes on the same
padded batch give bitwise identical FFN pre-activations at all 12 layers; building the network a second time gives an
identical state dict and identical activations; each of the first 32 molecules alone gives the same atom-token
activations as inside the padded batch of n_mol molecules, within 10^-3 absolute and 10^-4 relative to the mean
absolute activation. The redraw must differ from the pretrained model only in the feature-map buffers.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, csv, json, hashlib
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402

LAYERS = [f"L{i}" for i in range(12)]


def fingerprint(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    out_json = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    with open(Path(__file__).resolve().parents[2] / "data/derived/lineB/corpus/pilot_20k.tsv", encoding="utf-8") as fh:
        smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")][:n]
    rep = {}
    sd_pre = None
    for name, kw in (("pretrained", {}), ("twin_seed4000", dict(twin_seed=4000)),
                     ("hf_twin_seed6000", dict(twin_seed=6000, molformer_init="hf")),
                     ("orf_redraw_seed7000", dict(molformer_orf_seed=7000))):
        m, fam = models.load_model("molformer", **kw)
        z1 = models.extract_multi(m, fam, smiles, LAYERS, batch_size=n)
        z2 = models.extract_multi(m, fam, smiles, LAYERS, batch_size=n)
        identical = all(torch.equal(z1[l], z2[l]) for l in LAYERS)
        single = [models.extract_multi(m, fam, [s], LAYERS, batch_size=1) for s in smiles[:32]]
        n_atoms = sum(x["L0"].shape[0] for x in single)
        maxdiff = max(float((torch.cat([x[l] for x in single]) - z1[l][:n_atoms]).abs().max()) for l in LAYERS)
        scale = float(sum(z1[l].abs().mean() for l in LAYERS) / 12)
        r = dict(molecules=n, single_molecules_compared=32, two_passes_identical=bool(identical),
                 single_vs_padded_max_abs_diff=maxdiff, mean_abs_activation=scale, relative_diff=maxdiff / scale,
                 state_dict_sha256_16=fingerprint(m), attn_implementation=m.config._attn_implementation)
        m2, _ = models.load_model("molformer", **kw)
        z3 = models.extract_multi(m2, fam, smiles, LAYERS, batch_size=n)
        r["rebuild_identical"] = bool(fingerprint(m2) == r["state_dict_sha256_16"] and all(torch.equal(z1[l], z3[l]) for l in LAYERS))
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        if name == "pretrained":
            sd_pre = sd
        elif name.startswith("orf"):
            changed = sorted(k for k in sd if not torch.equal(sd[k], sd_pre[k]))
            r["keys_changed_vs_pretrained"] = changed
            r["only_feature_maps_changed"] = bool(changed and all(k.endswith("feature_map.weight") for k in changed))
        rep[name] = r
        print(name, {k: v for k, v in r.items() if k != "keys_changed_vs_pretrained"}, flush=True)
        del m, m2
        torch.cuda.empty_cache()
    rep["versions"] = {k: __import__(k).__version__ for k in ("torch", "transformers", "tokenizers", "rdkit", "numpy")}
    checks = [v for k, v in rep.items() if k != "versions"]
    ok = (all(v["two_passes_identical"] and v["rebuild_identical"] for v in checks)
          and all(v["single_vs_padded_max_abs_diff"] < 1e-3 and v["relative_diff"] < 1e-4 for v in checks)
          and rep["orf_redraw_seed7000"]["only_feature_maps_changed"]
          and len({v["state_dict_sha256_16"] for v in checks}) == len(checks))
    rep["passed"] = bool(ok)
    json.dump(rep, open(out_json, "w"), indent=1)
    assert ok, "determinism check failed"
    print("determinism check passed")


if __name__ == "__main__":
    main()
