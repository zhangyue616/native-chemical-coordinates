"""Does the released MoLFormer checkpoint know stereo tokens? (protocol v6 §1; CPU; textbook molecules, not H4)

usage: python -I molformer_stereo_check.py <out_json>
The model card says isomeric information was removed before pretraining. We mask one stereo token at a time and
record the probability the masked-language-model head puts on stereo tokens ([C@H], [C@@H], /, \\).
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, json
import os
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

P = str(Path(os.environ.get("NEURONAUDIT_MODEL_ROOT", Path(__file__).resolve().parents[2])) / "models/molformer_xl_both_10pct")
MOLS = {"L-alanine": "C[C@H](N)C(=O)O", "D-glucose (open chain)": "OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)C=O",
        "(R)-1-phenylethanol": "C[C@@H](O)c1ccccc1", "(E)-stilbene": "C(=C/c1ccccc1)\\c1ccccc1",
        "(Z)-2-butene": "C/C=C\\C", "ibuprofen (S)": "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"}
tok = AutoTokenizer.from_pretrained(P)
cfg = AutoConfig.from_pretrained(P, trust_remote_code=True, deterministic_eval=True)
m = AutoModelForMaskedLM.from_pretrained(P, config=cfg, trust_remote_code=True).eval()
STEREO = {"[C@H]", "[C@@H]", "[C@]", "[C@@]", "/", "\\"}
stereo_ids = [i for t, i in tok.get_vocab().items() if t in STEREO]
res = {}
with torch.no_grad():
    for name, smi in MOLS.items():
        ids = tok(smi, return_tensors="pt")["input_ids"]
        toks = tok.convert_ids_to_tokens(ids[0])
        rows = []
        for pos, t in enumerate(toks):
            if t in STEREO:
                x = ids.clone()
                x[0, pos] = tok.mask_token_id
                logits = m(input_ids=x).logits[0, pos]
                p = torch.softmax(logits, -1)
                top = torch.topk(p, 3)
                rows.append(dict(position=pos, masked=t, p_stereo=float(p[stereo_ids].sum()),
                                 p_same=float(p[ids[0, pos]]),
                                 top3=[(tok.convert_ids_to_tokens(int(i)), round(float(v), 3)) for v, i in zip(top.values, top.indices)]))
        res[name] = dict(smiles=smi, masked_positions=rows)
        print(name, [(r["masked"], round(r["p_stereo"], 3), round(r["p_same"], 3)) for r in rows], flush=True)
summary = [r["p_stereo"] for v in res.values() for r in v["masked_positions"]]
out = dict(note="CPU; textbook molecules; probability mass on stereo tokens at masked stereo positions",
           n_positions=len(summary), p_stereo_min=min(summary), p_stereo_median=sorted(summary)[len(summary) // 2],
           molecules=res)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("positions", len(summary), "min p_stereo", round(min(summary), 3), "median", round(out["p_stereo_median"], 3))
