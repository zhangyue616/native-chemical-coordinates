# R13 portable copy: model root is explicit/environment-relative. Hu/MolCLR MIT notices are in third_party_licenses/.
"""Line B protocol v1, O3: model loading, featurization ports, twins and atom-level activation extraction.

Hu 2020 GIN and MolCLR GIN are ported from the original MIT-licensed code
(snap-stanford/pretrain-gnns chem/model.py + loader.py; yuyangw/MolCLR models/ginet_molclr.py +
dataset/dataset.py) to PyG 2.8. CheMeleon is loaded through chemprop 2.3.1.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from pathlib import Path
ROOT = os.environ.get("NEURONAUDIT_MODEL_ROOT", str(Path(__file__).resolve().parents[2]))
WEIGHTS = {
    "chemeleon": f"{ROOT}/models/chemeleon/chemeleon_mp.pt",
    "hu_masking": f"{ROOT}/models/pretrain_gnns_gin/masking.pth",
    "hu_contextpred": f"{ROOT}/models/pretrain_gnns_gin/contextpred.pth",
    "hu_edgepred": f"{ROOT}/models/pretrain_gnns_gin/edgepred.pth",
    "hu_infomax": f"{ROOT}/models/pretrain_gnns_gin/infomax.pth",
    "molclr_gin": f"{ROOT}/models/molclr/pretrained_gin_model.pth",
    "chemberta_mlm": f"{ROOT}/models/chemberta_77m_mlm",
    "chemberta_mtr": f"{ROOT}/models/chemberta_77m_mtr",
    "molformer": f"{ROOT}/models/molformer_xl_both_10pct",
}
CHEMBERTA_LAYER = 2  # last of 3 encoder layers
MOLFORMER_LAYER = None  # protocol v6: J1 by each network's maximum over layers of the same-layer rank statistic,
                        # J2 at that layer; no fixed main layer
MOLFORMER_MAX_TOKENS = 202  # max_position_embeddings of the released config
# atom tokens of MoLFormer's regex tokenizer: bracket atoms and the organic-subset symbols (aromatic in lower case)
_MF_ATOM = re.compile(r"^(\[[^\]]+\]|Br|Cl|B|C|N|O|S|P|F|I|b|c|n|o|s|p)$")
_TOK = {}


def chemberta_tokenizer():
    """The tokenizer shipped with ChemBERTa-2 (lossy byte-level BPE, as used in pretraining; protocol v3 §2)."""
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained(WEIGHTS["chemberta_mlm"])
    return _TOK["t"]


def chemberta_atom_positions(smiles):
    """Token positions of atom tokens (alphabetic, non-special) in order; None if count != RDKit atom count."""
    tok = chemberta_tokenizer()
    ids = tok(smiles)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    special = set(tok.all_special_tokens)
    pos = [i for i, t in enumerate(toks) if t not in special and t.isalpha()]
    m = Chem.MolFromSmiles(smiles)
    if m is None or len(pos) != m.GetNumAtoms() or len(ids) > 512:
        return None
    return pos


def molformer_tokenizer():
    """The regex (WordLevel) tokenizer shipped with MoLFormer-XL: one token per atom, Cl/Br/bracket atoms intact."""
    if "mf" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["mf"] = AutoTokenizer.from_pretrained(WEIGHTS["molformer"])
    return _TOK["mf"]


def molformer_atom_positions(smiles):
    """Token positions of atom tokens in order (protocol v6); None if the count differs from the RDKit atom count,
    a token is <unk>, or the sequence exceeds the model's maximum length."""
    tok = molformer_tokenizer()
    ids = tok(smiles)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    special = set(tok.all_special_tokens)
    pos = [i for i, t in enumerate(toks) if t not in special and _MF_ATOM.match(t)]
    m = Chem.MolFromSmiles(smiles)
    if m is None or len(pos) != m.GetNumAtoms() or len(ids) > MOLFORMER_MAX_TOKENS or tok.unk_token_id in ids:
        return None
    return pos


TWIN_SEEDS = (1000, 1001, 1002)

# ---------------- featurization port (Hu and MolCLR share the same index scheme) ----------------
_CHIRAL = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
           Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
_BOND = {Chem.rdchem.BondType.SINGLE: 0, Chem.rdchem.BondType.DOUBLE: 1,
         Chem.rdchem.BondType.TRIPLE: 2, Chem.rdchem.BondType.AROMATIC: 3}
_DIR = {Chem.rdchem.BondDir.NONE: 0, Chem.rdchem.BondDir.ENDUPRIGHT: 1, Chem.rdchem.BondDir.ENDDOWNRIGHT: 2}


def gin_featurize(mol):
    """Port of mol_to_graph_data_obj_simple / MolCLR __getitem__ featurization (no masking)."""
    x = torch.tensor([[a.GetAtomicNum() - 1, _CHIRAL.index(a.GetChiralTag())] for a in mol.GetAtoms()],
                     dtype=torch.long)
    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = [_BOND[b.GetBondType()], _DIR[b.GetBondDir()]]
        src += [i, j]
        dst += [j, i]
        attr += [f, f]
    if attr:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    return x, edge_index, edge_attr


# ---------------- GIN port ----------------
class GINConvPort(MessagePassing):
    def __init__(self, emb_dim, num_bond_type, num_bond_direction=3):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        e = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        aggr = self.propagate(edge_index, x=x, edge_attr=e)
        return self.mlp(aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GINPort(nn.Module):
    """Hu: num_atom_type 120, num_bond_type 6, no heads. MolCLR: 119, 5, feat_lin + out_lin."""

    def __init__(self, num_atom_type, num_bond_type, num_layer=5, emb_dim=300, molclr_heads=False, feat_dim=512):
        super().__init__()
        self.num_layer = num_layer
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(3, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList([GINConvPort(emb_dim, num_bond_type) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])
        if molclr_heads:
            self.feat_lin = nn.Linear(emb_dim, feat_dim)
            self.out_lin = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
                                         nn.Linear(feat_dim, feat_dim // 2))

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer < self.num_layer - 1:
                h = F.relu(h)
        return h  # node representation, JK last, dropout 0 in eval


def _gin_kind(name):
    return dict(num_atom_type=119, num_bond_type=5, molclr_heads=True) if name.startswith("molclr") \
        else dict(num_atom_type=120, num_bond_type=6)


def load_model(name, twin_seed=None, device="cuda", molformer_init="pretraining", molformer_orf_seed=None):
    """Return (model, family). twin_seed=None loads pretrained weights strictly; else random init.

    molformer_init (twins only): "pretraining" = the initialization the released weights were trained from
    (PyTorch module defaults: Embedding N(0, 1) for every row, Linear Kaiming-uniform weights and uniform biases,
    LayerNorm 1/0; protocol v6 §3, evidence in lineB/results_v6/checkpoint_facts.json); "hf" = the _init_weights of the
    Hugging Face port (normal, std 0.02, zero biases), used only for the descriptive sensitivity set.
    molformer_orf_seed (pretrained only): redraw the linear attention's random-feature matrices under this seed
    (descriptive set of protocol v6; the checkpoint stores one draw).
    """
    if name.startswith("chemberta"):
        from transformers import RobertaModel, RobertaConfig
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        if twin_seed is not None:
            torch.manual_seed(twin_seed)
            m = RobertaModel(RobertaConfig.from_pretrained(WEIGHTS["chemberta_mlm"]), add_pooling_layer=False)
        else:
            m, info = RobertaModel.from_pretrained(WEIGHTS[name], add_pooling_layer=False, output_loading_info=True)
            missing = [k for k in info["missing_keys"] if k.startswith(("embeddings", "encoder"))]
            assert not missing, f"missing encoder weights: {missing[:5]}"
        return m.to(device).eval(), "chemberta"
    if name == "molformer":
        # MolformerModel from the code distributed with the checkpoint (trust_remote_code); deterministic_eval fixes
        # the random-feature matrices of the linear attention, which the checkpoint stores. Twins draw fresh
        # random-feature matrices and all weights under the twin seed.
        from transformers import AutoConfig, AutoModel
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        cfg = AutoConfig.from_pretrained(WEIGHTS[name], trust_remote_code=True, deterministic_eval=True)
        if twin_seed is not None:
            torch.manual_seed(twin_seed)
            m = AutoModel.from_config(cfg, trust_remote_code=True, add_pooling_layer=False)
            if molformer_init == "pretraining":
                for mod in m.modules():
                    if isinstance(mod, (nn.Linear, nn.LayerNorm)):
                        mod.reset_parameters()
                    elif isinstance(mod, nn.Embedding):
                        nn.init.normal_(mod.weight, 0.0, 1.0)   # the checkpoint's <pad> row is not zero either
            else:
                assert molformer_init == "hf", molformer_init
        else:
            m, info = AutoModel.from_pretrained(WEIGHTS[name], config=cfg, trust_remote_code=True,
                                                add_pooling_layer=False, output_loading_info=True)
            assert not info["missing_keys"], f"missing weights: {sorted(info['missing_keys'])[:5]}"
            assert all(k.startswith("lm_head") for k in info["unexpected_keys"]), info["unexpected_keys"]
        if molformer_orf_seed is not None:
            assert twin_seed is None
            torch.manual_seed(molformer_orf_seed)
            for layer in m.encoder.layer:
                layer.attention.self.feature_map.orthogonal_random_weights()
        assert m.encoder.layer[0].attention.self.feature_map.deterministic
        # the distributed attention code converts the mask with (attention_mask == 0): correct for the eager float mask
        assert getattr(m.config, "_attn_implementation", "eager") == "eager", m.config._attn_implementation
        return m.to(device).eval(), "molformer"
    if name == "chemeleon":
        from chemprop.nn import BondMessagePassing
        ck = torch.load(WEIGHTS[name], map_location="cpu", weights_only=True)
        if twin_seed is not None:
            torch.manual_seed(twin_seed)
        m = BondMessagePassing(**ck["hyper_parameters"])
        if twin_seed is None:
            m.load_state_dict(ck["state_dict"], strict=True)
        return m.to(device).eval(), "chemprop"
    if twin_seed is not None:
        torch.manual_seed(twin_seed)
    m = GINPort(**_gin_kind(name))
    if twin_seed is None:
        sd = torch.load(WEIGHTS[name], map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=True)
    return m.to(device).eval(), "gin"


# ---------------- extraction ----------------
@torch.no_grad()
def extract(model, family, smiles, layer=None, batch_size=512, device="cuda", want_pooled=False, on_gpu=False):
    """Atom-level pre-activations of the audited units, in atom order (molecule order, RDKit index).

    chemprop: W_o output (pre-ReLU), d_h units. gin: gnns[layer].mlp[0] output, 2*emb units.
    Returns (pre float32 [atoms, units], pooled float32 [mols, d] or None, n_atoms per molecule).
    """
    if family == "molformer":
        li = MOLFORMER_LAYER if layer is None else layer
        assert li is not None, "MOLFORMER_LAYER not set"
        Z = extract_multi(model, family, smiles, [f"L{li}"], batch_size=min(batch_size, 256), device=device)[f"L{li}"]
        counts = np.array([Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles])
        return (Z if on_gpu else Z.cpu().numpy()), None, counts
    if layer is None:
        layer = 4
    store = {}
    if family == "chemberta":
        return _extract_chemberta(model, smiles, batch_size=batch_size, device=device, on_gpu=on_gpu)
    if family == "chemprop":
        h = model.W_o.register_forward_hook(lambda mod, inp, out: store.__setitem__("pre", out.detach()))
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        from chemprop.data import BatchMolGraph
        feat = SimpleMoleculeMolGraphFeaturizer()
    else:
        h = model.gnns[layer].mlp[0].register_forward_hook(
            lambda mod, inp, out: store.__setitem__("pre", out.detach()))
    pres, pools, counts = [], [], []
    try:
        for s0 in range(0, len(smiles), batch_size):
            mols = [Chem.MolFromSmiles(s) for s in smiles[s0:s0 + batch_size]]
            n = [m.GetNumAtoms() for m in mols]
            if family == "chemprop":
                bmg = BatchMolGraph([feat(m) for m in mols])
                bmg.to(device)
                H = model(bmg)
                batch_vec = bmg.batch
            else:
                datas = []
                for m in mols:
                    x, ei, ea = gin_featurize(m)
                    datas.append(Data(x=x, edge_index=ei, edge_attr=ea))
                b = Batch.from_data_list(datas).to(device)
                H = model(b.x, b.edge_index, b.edge_attr)
                batch_vec = b.batch
            pre = store["pre"]
            assert pre.shape[0] == sum(n), (pre.shape, sum(n))
            pres.append(pre.float() if on_gpu else pre.float().cpu())
            counts.extend(n)
            if want_pooled:
                nb = len(mols)
                summ = torch.zeros(nb, H.shape[1], device=device).index_add_(0, batch_vec, H)
                cnt = torch.bincount(batch_vec, minlength=nb).clamp(min=1).unsqueeze(1)
                pools.append((summ / cnt).float().cpu())
    finally:
        h.remove()
    pre = torch.cat(pres) if on_gpu else torch.cat(pres).numpy()
    del pres
    pooled = torch.cat(pools).numpy() if want_pooled else None
    return pre, pooled, np.array(counts)


@torch.no_grad()
def extract_multi(model, family, smiles, layers, batch_size=256, device="cuda"):
    """Several atom-level unit sets from one forward pass (protocol v4 §1.1). Returns {layer_name: GPU tensor}.

    chemberta: layers like "L0", "L1", "L2" -> encoder.layer[i].intermediate.dense output (pre-GELU).
    molformer: layers "L0" .. "L11" -> encoder.layer[i].intermediate.dense output (pre-GELU) at atom tokens.
    chemprop (CheMeleon): "main" -> W_o output (pre-ReLU); "mp3" -> bond hidden state after the 3rd message update,
    summed onto target atoms exactly as chemprop aggregates the final step.
    """
    out = {l: [] for l in layers}
    if family in ("chemberta", "molformer"):
        tok = chemberta_tokenizer() if family == "chemberta" else molformer_tokenizer()
        pos_fn = chemberta_atom_positions if family == "chemberta" else molformer_atom_positions
        store, hooks = {}, []
        for l in layers:
            i = int(l[1:])
            hooks.append(model.encoder.layer[i].intermediate.dense.register_forward_hook(
                lambda mod, inp, o, l=l: store.__setitem__(l, o.detach())))
        try:
            for s0 in range(0, len(smiles), batch_size):
                batch = smiles[s0:s0 + batch_size]
                positions = [pos_fn(s) for s in batch]
                assert all(p is not None for p in positions)
                enc = tok(batch, padding=True, return_tensors="pt").to(device)
                model(**enc)
                for l in layers:
                    for b, pos in enumerate(positions):
                        out[l].append(store[l][b, pos].float())
        finally:
            for h in hooks:
                h.remove()
    elif family == "chemprop":
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        from chemprop.data import BatchMolGraph
        feat = SimpleMoleculeMolGraphFeaturizer()
        store = {"updates": []}
        orig_update = model.update

        def update_rec(M_t, H_0):
            H = orig_update(M_t, H_0)
            store["updates"].append(H.detach())
            return H
        model.update = update_rec
        h = model.W_o.register_forward_hook(lambda mod, inp, o: store.__setitem__("main", o.detach()))
        try:
            for s0 in range(0, len(smiles), batch_size):
                mols = [Chem.MolFromSmiles(s) for s in smiles[s0:s0 + batch_size]]
                bmg = BatchMolGraph([feat(m) for m in mols])
                bmg.to(device)
                store["updates"] = []
                model(bmg)
                n_atoms = bmg.V.shape[0]
                if "main" in layers:
                    out["main"].append(store["main"].float())
                if "mp3" in layers:
                    H3 = store["updates"][2]
                    idx = bmg.edge_index[1].unsqueeze(1).repeat(1, H3.shape[1])
                    M3 = torch.zeros(n_atoms, H3.shape[1], dtype=H3.dtype, device=H3.device).scatter_reduce_(
                        0, idx, H3, reduce="sum", include_self=False)
                    out["mp3"].append(M3.float())
        finally:
            h.remove()
            del model.update
    else:
        raise ValueError(family)
    return {l: torch.cat(out.pop(l)) for l in list(out)}   # concatenate one layer at a time (peak memory)


@torch.no_grad()
def _extract_chemberta(model, smiles, batch_size=256, device="cuda", on_gpu=False):
    """Atom-level pre-activations of the last layer's FFN intermediate units (pre-GELU)."""
    tok = chemberta_tokenizer()
    store = {}
    h = model.encoder.layer[CHEMBERTA_LAYER].intermediate.dense.register_forward_hook(
        lambda mod, inp, out: store.__setitem__("pre", out.detach()))
    pres, counts = [], []
    try:
        for s0 in range(0, len(smiles), batch_size):
            batch = smiles[s0:s0 + batch_size]
            positions = [chemberta_atom_positions(s) for s in batch]
            assert all(p is not None for p in positions), "molecule without valid atom alignment in batch"
            enc = tok(batch, padding=True, return_tensors="pt").to(device)
            model(**enc)
            pre = store["pre"]  # (B, T, 464)
            for b, pos in enumerate(positions):
                pres.append(pre[b, pos].float() if on_gpu else pre[b, pos].float().cpu())
                counts.append(len(pos))
    finally:
        h.remove()
    out = torch.cat(pres) if on_gpu else torch.cat(pres).numpy()
    return out, None, np.array(counts)
