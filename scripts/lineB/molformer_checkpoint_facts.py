"""Facts about the released MoLFormer checkpoint that bear on the twin initialization (protocol v6 §3), CPU only.

usage: python -I molformer_checkpoint_facts.py <out_json>
Compares the stored weights with the two twin initializations as built by models.load_model: the pretraining
initialization (Embedding N(0, 1) on every row; Linear Kaiming-uniform weights and uniform biases; LayerNorm 1/0) and
the Hugging Face port's _init_weights (normal, std 0.02, zero biases). The embedding statistics identify the
initialization; the trained Linear weights cannot distinguish the two schemes.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, json, math
import torch
from safetensors import safe_open

from pathlib import Path
import os
P = str(Path(os.environ.get("NEURONAUDIT_MODEL_ROOT", Path(__file__).resolve().parents[2]))/"models/molformer_xl_both_10pct/model.safetensors")
out = {}
with safe_open(P, "pt") as f:
    emb = f.get_tensor("molformer.embeddings.word_embeddings.weight")
    norms = emb.norm(dim=1)
    out["embedding"] = dict(shape=list(emb.shape), std=float(emb.std()), mean=float(emb.mean()),
                            row_norm_min=float(norms.min()), row_norm_max=float(norms.max()),
                            row_norm_median=float(norms.median()), pad_row_norm=float(norms[2]), unk_row_norm=float(norms[2361]),
                            expected_row_norm_if_N01=math.sqrt(emb.shape[1]), expected_std_if_hf_init=0.02)
    lin = {}
    for name in ("molformer.encoder.layer.0.intermediate.dense", "molformer.encoder.layer.5.intermediate.dense",
                 "molformer.encoder.layer.11.intermediate.dense", "molformer.encoder.layer.0.attention.self.query",
                 "molformer.encoder.layer.11.output.dense"):
        w, b = f.get_tensor(name + ".weight"), f.get_tensor(name + ".bias")
        fan_in = w.shape[1]
        lin[name.replace("molformer.encoder.layer.", "L")] = dict(
            weight_std=float(w.std()), bias_std=float(b.std()), bias_absmax=float(b.abs().max()), bias_mean=float(b.mean()),
            kaiming_uniform_weight_std=1 / math.sqrt(3 * fan_in), default_bias_std=1 / math.sqrt(3 * fan_in))
    out["linear"] = lin
    ln = f.get_tensor("molformer.encoder.layer.0.attention.output.LayerNorm.weight")
    out["layernorm_L0_attention_weight"] = dict(mean=float(ln.mean()), std=float(ln.std()))
# the twins as actually built (CPU): the pretraining initialization used for the main set, and the Hugging Face
# port's _init_weights used for the descriptive sensitivity set
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402


def describe(m):
    emb = m.embeddings.word_embeddings.weight
    lin = m.encoder.layer[0].intermediate.dense
    return dict(embedding_std=float(emb.std()), embedding_row_norm_min=float(emb.norm(dim=1).min()),
                embedding_row_norm_max=float(emb.norm(dim=1).max()), pad_row_norm=float(emb[2].norm()),
                linear_weight_std=float(lin.weight.std()), linear_bias_absmax=float(lin.bias.abs().max()),
                layernorm_weight_mean=float(m.encoder.layer[0].output.LayerNorm.weight.mean()),
                feature_map_shape=list(m.encoder.layer[0].attention.self.feature_map.weight.shape))


with torch.no_grad():
    out["twin_pretraining_init_seed4000"] = describe(models.load_model("molformer", twin_seed=4000, device="cpu")[0])
    out["twin_hf_init_seed6000"] = describe(models.load_model("molformer", twin_seed=6000, device="cpu", molformer_init="hf")[0])
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
