"""Which of the 173 readable concepts coincide with a MoLFormer token identity? (protocol v6 §1; pilot set, labels only)

usage: python -I token_visible_concepts.py <pilot_tsv> <concept_dir> <out_json> [threshold=0.8]
Rule: for each readable concept, the share of its positive atoms (frozen pilot labels) whose MoLFormer atom token is
one that ChemBERTa-2's tokenizer does not emit as a single token: a bracket atom ([nH], [N+], [O-], [C@@H], ...), Cl
or Br (ChemBERTa-2 reads Cl as C and Br as B and drops brackets). A concept is token-visible if the share is at least
the threshold. No model is run.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import os, sys, csv, json

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402


def main():
    tsv, cdir, out_json = sys.argv[1:4]
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    tok = models.molformer_tokenizer()
    with open(tsv, encoding="utf-8") as fh:
        smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
    flags = []
    for s in smiles:
        pos = models.molformer_atom_positions(s)
        assert pos is not None
        toks = tok.convert_ids_to_tokens(tok(s)["input_ids"])
        flags.extend(t.startswith("[") or t in ("Cl", "Br") for t in (toks[p] for p in pos))
    flag = np.array(flags, dtype=bool)
    concepts = json.load(open(os.path.join(cdir, "concepts_v1.json"), encoding="utf-8"))
    Lsp = sp.load_npz(os.path.join(cdir, "labels_pilot.npz")).tocsc()
    assert Lsp.shape[0] == len(flag), (Lsp.shape, len(flag))
    rows = []
    for c in concepts:
        if c["partition"] != "readable":
            continue
        col = Lsp[:, c["id"]].toarray().ravel().astype(bool)
        share = float(flag[col].mean()) if col.any() else 0.0
        rows.append(dict(id=c["id"], name=c["name"], n_positive=int(col.sum()), share_on_tokens_absent_in_chemberta=share,
                         token_visible=bool(share >= thr)))
    ids = [r["id"] for r in rows if r["token_visible"]]
    out = dict(rule="share of positive atoms whose MoLFormer token is a bracket atom, Cl or Br (tokens ChemBERTa-2 does not "
                    "emit as single tokens); token-visible if share >= threshold", threshold=thr, molecules=len(smiles),
               atoms=int(len(flag)), share_of_all_atoms=float(flag.mean()), n_token_visible=len(ids), ids=ids, concepts=rows)
    json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("token-visible concepts:", [(r["id"], r["name"], round(r["share_on_tokens_absent_in_chemberta"], 2))
                                      for r in rows if r["token_visible"]])
    print("next highest shares:", sorted([(round(r["share_on_tokens_absent_in_chemberta"], 2), r["id"], r["name"])
                                          for r in rows if not r["token_visible"]], reverse=True)[:6])


if __name__ == "__main__":
    main()
