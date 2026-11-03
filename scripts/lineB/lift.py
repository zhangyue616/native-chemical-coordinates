"""Line B protocol v1, O4: exact average precision (sklearn definition, tie-aware) on GPU and lift arrays."""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import numpy as np
import torch


def ap_all_concepts(scores, L, pos):
    """scores: (N,) float tensor; L: (N, C) uint8 tensor (same device); pos: (C,) float tensor of positives.

    Returns (C,) average precision with sklearn's definition: thresholds at distinct score values,
    AP = sum_k (R_k - R_{k-1}) * P_k.
    """
    order = torch.argsort(scores, descending=True)
    s = scores[order]
    tp = torch.cumsum(L[order].to(torch.int32), dim=0)
    ends = torch.nonzero(s[1:] != s[:-1]).squeeze(1)
    ends = torch.cat([ends, torch.tensor([s.numel() - 1], device=s.device)])
    tp_e = tp[ends].to(torch.float64)
    n_e = (ends + 1).to(torch.float64).unsqueeze(1)
    tp_prev = torch.cat([torch.zeros(1, tp_e.shape[1], dtype=torch.float64, device=s.device), tp_e[:-1]])
    ap = ((tp_e - tp_prev) * (tp_e / n_e)).sum(0) / pos.to(torch.float64)
    return ap


class Positives:
    """Sparse (concept, atom) positive pairs, prepared once per label matrix."""

    def __init__(self, L):
        idx = torch.nonzero(L)  # (P, 2): atom, concept
        self.atom = idx[:, 0].contiguous()
        self.concept = idx[:, 1].contiguous()
        self.N, self.C = L.shape
        self.pos = torch.bincount(self.concept, minlength=self.C).to(torch.float64)


def ap_all_concepts_sparse(scores, P):
    """Same definition as ap_all_concepts, computed from the positives only.

    AP_c = (1/P_c) * sum over positives i of c of TP_c(g(i)) / n(g(i)), where g(i) is the tie group of i,
    n(g) the number of atoms ranked at or above the end of group g, TP_c(g) the positives of c up to that end.
    """
    N = P.N
    order = torch.argsort(scores, descending=True)
    s = scores[order]
    change = s[1:] != s[:-1]
    gid_sorted = torch.cat([torch.zeros(1, dtype=torch.int64, device=s.device), torch.cumsum(change, 0)])
    ends = torch.cat([torch.nonzero(change).squeeze(1), torch.tensor([N - 1], device=s.device)])
    n_end = (ends + 1).to(torch.float64)
    gid = torch.empty(N, dtype=torch.int64, device=s.device)
    gid[order] = gid_sorted
    g = gid[P.atom]
    key = P.concept.to(torch.int64) * (N + 1) + g
    key_sorted, perm = torch.sort(key)
    tp = torch.searchsorted(key_sorted, key_sorted, right=True)
    seg_start = torch.searchsorted(key_sorted, torch.arange(P.C, device=s.device, dtype=torch.int64) * (N + 1))
    c_sorted = P.concept[perm]
    tp = (tp - seg_start[c_sorted]).to(torch.float64)
    contrib = tp / n_end[g[perm]]
    ap = torch.zeros(P.C, dtype=torch.float64, device=s.device).index_add_(0, c_sorted, contrib)
    return ap / P.pos


def ap_batch_sparse(S, P):
    """Batched ap_all_concepts_sparse. S: (N, B) scores for B units. Returns (B, C) AP."""
    N, B = S.shape
    dev = S.device
    order = torch.argsort(S, dim=0, descending=True)
    s = torch.gather(S, 0, order)
    is_end = torch.ones_like(s, dtype=torch.bool)
    is_end[:-1] = s[1:] != s[:-1]
    gid_sorted = torch.cat([torch.zeros(1, B, dtype=torch.int64, device=dev),
                            torch.cumsum(is_end[:-1], 0)])
    idx = torch.arange(N, device=dev).unsqueeze(1).expand(N, B)
    end_pos = torch.where(is_end, idx, torch.full_like(idx, N))
    end_pos = torch.flip(torch.cummin(torch.flip(end_pos, [0]), 0).values, [0])
    gid = torch.empty_like(gid_sorted).scatter_(0, order, gid_sorted)
    n_atom = torch.empty(N, B, dtype=torch.float64, device=dev).scatter_(0, order, (end_pos + 1).to(torch.float64))
    g = gid[P.atom]                        # (Npos, B)
    key = (P.concept.to(torch.int64).unsqueeze(1) * (N + 1) + g).t().contiguous()   # (B, Npos)
    key_sorted, perm = torch.sort(key, dim=1)
    tp = torch.searchsorted(key_sorted, key_sorted, right=True)
    seg_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                           torch.cumsum(P.pos.to(torch.int64), 0)[:-1]])
    c_sorted = P.concept.to(torch.int64)[perm]           # (B, Npos)
    tp = (tp - seg_start[c_sorted]).to(torch.float64)
    n_sorted = torch.gather(n_atom[P.atom].t(), 1, perm)
    ap = torch.zeros(B, P.C, dtype=torch.float64, device=dev).scatter_add_(1, c_sorted, tp / n_sorted)
    return ap / P.pos.unsqueeze(0)


def lift_array_batched(Z, L_or_P, prevalence, device="cuda", batch=32, act="relu"):
    """Same output as lift_array, computed `batch` units at a time. Pole 0 = act(Z) (relu or gelu), pole 1 = -Z."""
    Zt = Z if isinstance(Z, torch.Tensor) else torch.from_numpy(Z)
    Zt = Zt.to(device=device, dtype=torch.float32)
    P = L_or_P if isinstance(L_or_P, Positives) else Positives(L_or_P)
    prev = torch.as_tensor(prevalence, dtype=torch.float64, device=device)
    D = Zt.shape[1]
    out = torch.empty(D, 2, P.C, dtype=torch.float64, device=device)
    f = {"relu": torch.relu, "gelu": torch.nn.functional.gelu, "identity": (lambda x: x)}[act]
    for b0 in range(0, D, batch):
        zb = Zt[:, b0:b0 + batch]
        out[b0:b0 + zb.shape[1], 0] = ap_batch_sparse(f(zb), P) / prev
        out[b0:b0 + zb.shape[1], 1] = ap_batch_sparse(-zb, P) / prev
    return out.cpu().numpy()


def lift_array(Z, L, prevalence, device="cuda", progress=None):
    """Z: (N, D) pre-activations (numpy or torch, fp32). Returns lift (D, 2, C): pole 0 = ReLU(Z), pole 1 = -Z."""
    Zt = Z if isinstance(Z, torch.Tensor) else torch.from_numpy(Z)
    Zt = Zt.to(device=device, dtype=torch.float32)
    P = Positives(L)
    prev = torch.as_tensor(prevalence, dtype=torch.float64, device=device)
    D = Zt.shape[1]
    out = torch.empty(D, 2, L.shape[1], dtype=torch.float64, device=device)
    for j in range(D):
        z = Zt[:, j]
        out[j, 0] = ap_all_concepts_sparse(torch.relu(z), P) / prev
        out[j, 1] = ap_all_concepts_sparse(-z, P) / prev
        if progress and (j + 1) % progress == 0:
            print(f"    unit {j + 1}/{D}", flush=True)
    return out.cpu().numpy()


def self_test(n_cases=20, seed=0):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    worst = 0.0
    for k in range(n_cases):
        n = int(rng.integers(50, 3000))
        c = int(rng.integers(1, 6))
        raw = rng.normal(size=n)
        if k % 2 == 0:
            raw = np.round(raw, 1)  # heavy ties
        if k % 3 == 0:
            raw = np.maximum(raw, 0)  # ReLU-like tie block at zero
        lab = (rng.random((n, c)) < rng.uniform(0.01, 0.4, size=c)).astype(np.uint8)
        lab[0] = 1
        got = ap_all_concepts(torch.tensor(raw), torch.tensor(lab), torch.tensor(lab.sum(0), dtype=torch.float64))
        got2 = ap_all_concepts_sparse(torch.tensor(raw), Positives(torch.tensor(lab)))
        for ci in range(c):
            worst = max(worst, abs(float(got2[ci]) - average_precision_score(lab[:, ci], raw)))
            ref = average_precision_score(lab[:, ci], raw)
            worst = max(worst, abs(float(got[ci]) - ref))
    return worst
