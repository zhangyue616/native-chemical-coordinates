"""Post hoc M1 (lineB/POSTHOC_M1_M4_PLAN.md): ablate concept-selective units of ChemBERTa-2 MTR and measure the
effect on its released descriptor-regression head.

usage: python -I posthoc_m1_mtr_ablation.py
Outputs: lineB/results_posthoc/m1_mapping.json, m1_ablation.json, m1_ablation.csv
"""
import os, sys, csv, json, time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figures"))
import models  # noqa: E402
import common as C  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = f"{C.LB}/results_posthoc"
CKPT = f"{C.ROOT}/models/chemberta_77m_mtr"
LAYERS = ["L0", "L1", "L2"]
N_MAP, N_POS, N_NEG, N_RAND = 2000, 500, 500, 20
# Head activation as in the authors' released code (RobertaRegressionHead: dense -> ReLU -> out_proj, first token;
# seyonechithrananda/bert-loves-chemistry, chemberta/utils/roberta_regression.py, commit 9933ac7). The first runs
# chose it by fit (GELU); see lineB/POSTHOC_M1_ADDENDUM.md.
HEAD_ACT = "relu"
MODES = ("all", "atoms")   # ablate at every position, or only at atom-token positions ([CLS] and others untouched)
STATE = {}
DEV = "cuda"


class Head(torch.nn.Module):
    def __init__(self, sd, act):
        super().__init__()
        self.dense = torch.nn.Linear(384, 384)
        self.out_proj = torch.nn.Linear(384, 199)
        self.dense.load_state_dict({"weight": sd["regression.dense.weight"], "bias": sd["regression.dense.bias"]})
        self.out_proj.load_state_dict({"weight": sd["regression.out_proj.weight"],
                                       "bias": sd["regression.out_proj.bias"]})
        self.act = {"tanh": torch.tanh, "relu": torch.relu, "gelu": torch.nn.functional.gelu}[act]

    def forward(self, cls):
        return self.out_proj(self.act(self.dense(cls)))


@torch.no_grad()
def cls_states(model, tok, smiles, hooks=(), bs=256):
    """Last-layer [CLS] state for each molecule; optional forward hooks are registered for the pass.
    STATE["atom_mask"] holds, for the current batch, which token positions are atom tokens."""
    hs = [h() for h in hooks]
    out = []
    try:
        for i in range(0, len(smiles), bs):
            batch = smiles[i:i + bs]
            enc = tok(batch, padding=True, return_tensors="pt").to(DEV)
            mask = torch.zeros(enc["input_ids"].shape, dtype=torch.bool, device=DEV)
            for b, s in enumerate(batch):
                pos = models.chemberta_atom_positions(s)
                assert pos is not None
                mask[b, pos] = True
            STATE["atom_mask"] = mask
            out.append(model(**enc).last_hidden_state[:, 0].float())
    finally:
        for h in hs:
            h.remove()
    return torch.cat(out)


@torch.no_grad()
def token_means(model, tok, smiles, bs=256):
    """Mean pre-activation of every FFN intermediate unit over all non-padding tokens, per layer."""
    store = {}
    hs = [model.encoder.layer[int(l[1:])].intermediate.dense.register_forward_hook(
        lambda mod, inp, o, l=l: store.__setitem__(l, o.detach())) for l in LAYERS]
    sums = {l: torch.zeros(464, device=DEV, dtype=torch.float64) for l in LAYERS}
    n = 0
    try:
        for i in range(0, len(smiles), bs):
            enc = tok(smiles[i:i + bs], padding=True, return_tensors="pt").to(DEV)
            model(**enc)
            mask = enc["attention_mask"].bool()
            n += int(mask.sum())
            for l in LAYERS:
                sums[l] += store[l][mask].double().sum(0)
    finally:
        for h in hs:
            h.remove()
    return {l: (sums[l] / n).float() for l in LAYERS}


def ablation_hook(model, layer, unit, value, mode="all"):
    mod = model.encoder.layer[int(layer[1:])].intermediate.dense

    def fn(m, inp, o):
        o = o.clone()
        if mode == "all":
            o[..., unit] = value
        else:
            m_ = STATE["atom_mask"]
            o[..., unit] = torch.where(m_, torch.full_like(o[..., unit], value), o[..., unit])
        return o
    return lambda: mod.register_forward_hook(fn)


def rdkit_descriptors(smiles):
    names = [n for n, _ in Descriptors._descList]
    X = np.full((len(smiles), len(names)), np.nan)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        for j, (_, f) in enumerate(Descriptors._descList):
            try:
                X[i, j] = f(m)
            except Exception:
                pass
    return names, X


def spearman_matrix(A, B):
    """|Spearman rho| between every column of A (n x p) and every column of B (n x q); NaN-free inputs."""
    ra = np.apply_along_axis(rankdata, 0, A)
    rb = np.apply_along_axis(rankdata, 0, B)
    ra = (ra - ra.mean(0)) / ra.std(0)
    rb = (rb - rb.mean(0)) / rb.std(0)
    return (ra.T @ rb) / A.shape[0]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from transformers import RobertaModel
    model = RobertaModel.from_pretrained(CKPT, add_pooling_layer=False).to(DEV).eval()
    sd = torch.load(f"{CKPT}/pytorch_model.bin", map_location="cpu", weights_only=True)
    norm_mean, norm_std = sd["norm_mean"].numpy(), sd["norm_std"].numpy()
    tok = models.chemberta_tokenizer()
    with open(f"{C.CORPUS}/pilot_20k.tsv", encoding="utf-8") as fh:
        pilot = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
    rng = np.random.default_rng(0)

    # ---------------- step 1: output mapping ----------------
    map_idx = rng.choice(len(pilot), N_MAP, replace=False)
    map_smiles = [pilot[i] for i in map_idx]
    names, X = rdkit_descriptors(map_smiles)
    ok = np.all(np.isfinite(X), axis=0) & (np.nanstd(X, axis=0) > 0)
    names_ok, X_ok = [n for n, k in zip(names, ok) if k], X[:, ok]
    cls = cls_states(model, tok, map_smiles)
    variants = {}
    for act in ("tanh", "relu", "gelu"):
        head = Head(sd, act).to(DEV)
        with torch.no_grad():
            pred = head(cls).cpu().numpy() * norm_std + norm_mean
        good = np.std(pred, axis=0) > 0
        R = np.zeros((199, len(names_ok)))
        R[good] = np.abs(spearman_matrix(pred[:, good], X_ok))
        best_j = R.argmax(1)
        variants[act] = dict(median_best_rho=float(np.median(R.max(1))), R=R, best_j=best_j)
        print(f"head activation {act}: median best |rho| {variants[act]['median_best_rho']:.3f}", flush=True)
    act = HEAD_ACT
    R, best_j = variants[act]["R"], variants[act]["best_j"]
    mapping = [dict(output=o, descriptor=names_ok[best_j[o]], abs_rho=float(R[o, best_j[o]]),
                    accepted=bool(R[o, best_j[o]] >= 0.5)) for o in range(199)]
    n_acc = sum(m["accepted"] for m in mapping)
    n_unique = len({m["descriptor"] for m in mapping if m["accepted"]})
    print(f"head activation (released code): {act}; accepted mappings {n_acc}/199, distinct descriptors {n_unique}",
          flush=True)
    json.dump(dict(head_activation=act, head_activation_source="released code (RobertaRegressionHead, ReLU)",
                   n_molecules=N_MAP,
                   median_best_rho={a: v["median_best_rho"] for a, v in variants.items()},
                   n_descriptors_considered=len(names_ok), accepted=n_acc, distinct_accepted=n_unique,
                   mapping=mapping), open(f"{OUT}/m1_mapping.json", "w"), indent=1)
    head = Head(sd, act).to(DEV)

    # ---------------- step 2: concepts with a mapped fr_ output ----------------
    cons = C.concepts()
    fr_out = {}
    for m in mapping:
        if m["accepted"] and m["descriptor"].startswith("fr_"):
            if m["descriptor"] not in fr_out or m["abs_rho"] > mapping[fr_out[m["descriptor"]]]["abs_rho"]:
                fr_out[m["descriptor"]] = m["output"]
    Lp = sp.load_npz(f"{C.CDIR}/labels_pilot.npz").tocsc()
    mol_idx = np.load(f"{C.CDIR}/atoms_pilot.npz")["mol_idx"]
    rd4 = np.load(f"{C.LB}/results_v4/lifts/chemberta/readable_concept_ids.npy")
    targets = []
    for c in cons:
        if c["partition"] != "readable" or not c["name"].startswith("frag:fr_"):
            continue
        # RDKit's descriptor names replace '=' by '_' (library frag:fr_C=O -> RDKit fr_C_O); added after the
        # first run silently dropped fr_C=O (Table S1)
        fr = c["name"][5:].replace("=", "_")
        if fr not in fr_out or c["id"] not in rd4:
            continue
        mols = np.unique(mol_idx[Lp[:, c["id"]].nonzero()[0]])
        if len(mols) >= 50:
            targets.append(dict(concept=c["name"], cid=c["id"], output=fr_out[fr], pos_mols=mols))
    print(f"{len(targets)} concepts with a mapped fr_ output and >= 50 pilot molecules", flush=True)

    # ---------------- evaluation molecules and token means ----------------
    sets = {}
    for t in targets:
        r = np.random.default_rng(0)
        pos = r.choice(t["pos_mols"], min(N_POS, len(t["pos_mols"])), replace=False)
        neg_pool = np.setdiff1d(np.arange(len(pilot)), t["pos_mols"])
        neg = r.choice(neg_pool, N_NEG, replace=False)
        sets[t["concept"]] = (pos, neg)
    pool = np.unique(np.concatenate([np.concatenate(v) for v in sets.values()]))
    pool_smiles = [pilot[i] for i in pool]
    where = {m: i for i, m in enumerate(pool)}
    means = token_means(model, tok, pool_smiles)
    with torch.no_grad():
        base = head(cls_states(model, tok, pool_smiles)).cpu().numpy()

    # ---------------- step 3-4: units and ablations ----------------
    rows = []
    rr = np.random.default_rng(1)
    for t in targets:
        j = int(np.nonzero(rd4 == t["cid"])[0][0])
        pos, neg = sets[t["concept"]]
        ip, ineg = [where[i] for i in pos], [where[i] for i in neg]
        sub = [pilot[i] for i in np.concatenate([pos, neg])]
        b_sub = base[ip + ineg]
        npos = len(ip)
        for layer in LAYERS:
            pre = np.load(f"{C.LB}/results_v4/lifts/chemberta/{layer}/chemberta_mtr/pre_native.npy")
            u, pole = np.unravel_index(np.argmax(pre[:, :, j]), pre[:, :, j].shape)

            def delta(unit, mode):
                hk = ablation_hook(model, layer, int(unit), float(means[layer][int(unit)]), mode)
                with torch.no_grad():
                    a = head(cls_states(model, tok, sub, hooks=[hk])).cpu().numpy()
                return a - b_sub
            o = t["output"]
            cand = [x for x in range(464) if x != u]
            rand_units = rr.choice(cand, N_RAND, replace=False)
            for mode in MODES:
                d = delta(u, mode)
                mp, mn = d[:npos].mean(0), d[npos:].mean(0)
                target = float(mp[o])
                rank = int(1 + np.sum(np.abs(mp) > abs(target))) if abs(target) > 0 else None
                selectivity = float(abs(mp[o]) / max(abs(mn[o]), 1e-12))
                rand_eff = np.array([abs(delta(x, mode)[:npos, o].mean()) for x in rand_units])
                row = dict(concept=t["concept"], output=o, layer=layer, mode=mode, unit=int(u),
                           pole="positive" if pole == 0 else "negative", lift_H3=float(pre[u, pole, j]),
                           n_pos=npos, target_effect=target, max_abs_change_any_output=float(np.abs(d).max()),
                           specificity_rank=rank, selectivity=selectivity,
                           random_share_ge=float(np.mean(rand_eff >= abs(target))),
                           random_median_abs_effect=float(np.median(rand_eff)))
                rows.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
                      f"{time.time() - t0:.0f}s", flush=True)
    summ = {}
    for mode in MODES:
        summ[mode] = {}
        for layer in LAYERS:
            rs = [r for r in rows if r["layer"] == layer and r["mode"] == mode]
            ranks = [r["specificity_rank"] for r in rs if r["specificity_rank"] is not None]
            summ[mode][layer] = dict(
                n_concepts=len(rs), n_with_nonzero_effect=len(ranks),
                median_specificity_rank=float(np.median(ranks)) if ranks else None,
                share_top5=float(np.mean([r["specificity_rank"] is not None and r["specificity_rank"] <= 5 for r in rs])),
                share_beats_all_random=float(np.mean([r["random_share_ge"] == 0 and abs(r["target_effect"]) > 0
                                                      for r in rs])),
                median_abs_target_effect=float(np.median([abs(r["target_effect"]) for r in rs])),
                median_random_abs_effect=float(np.median([r["random_median_abs_effect"] for r in rs])),
                max_abs_change_any_output=float(max(r["max_abs_change_any_output"] for r in rs)))
    json.dump(dict(plan="lineB/POSTHOC_M1_M4_PLAN.md + lineB/POSTHOC_M1_ADDENDUM.md", head_activation=act,
                   modes=list(MODES), token_mean_pool_molecules=len(pool), summary=summ, rows=rows),
              open(f"{OUT}/m1_ablation.json", "w"), indent=1)
    with open(f"{OUT}/m1_ablation.csv", "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
