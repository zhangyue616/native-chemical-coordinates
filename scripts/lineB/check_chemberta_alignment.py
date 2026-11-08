"""Per-atom check of the ChemBERTa-2 token-to-atom alignment (lineB/POSTHOC_M1_ADDENDUM.md §3).

usage: python -I check_chemberta_alignment.py <out_json> <name>=<tsv> [<name>=<tsv> ...]
For every molecule, the k-th alphabetic token (models.chemberta_atom_positions) is compared with RDKit atom k of the
same SMILES. Expected token: first letter of the element symbol, lower-cased for aromatic atoms (Cl -> C, Br -> B,
[nH] -> n, [C@@H] -> C).
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import os, sys, csv, json, collections

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402


def main():
    out_json = sys.argv[1]
    tok = models.chemberta_tokenizer()
    report = {}
    for arg in sys.argv[2:]:
        name, tsv = arg.split("=", 1)
        with open(tsv, encoding="utf-8") as fh:
            smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
        n_mol = n_atom = n_bad = n_unaligned = 0
        bad_examples, by_symbol = [], collections.Counter()
        for s in smiles:
            pos = models.chemberta_atom_positions(s)
            if pos is None:
                n_unaligned += 1
                continue
            toks = tok.convert_ids_to_tokens(tok(s)["input_ids"])
            mol = Chem.MolFromSmiles(s)
            n_mol += 1
            for k, p in enumerate(pos):
                a = mol.GetAtomWithIdx(k)
                exp = a.GetSymbol()[0]
                if a.GetIsAromatic():
                    exp = exp.lower()
                n_atom += 1
                by_symbol[a.GetSymbol() + ("(ar)" if a.GetIsAromatic() else "")] += 1
                if toks[p] != exp:
                    n_bad += 1
                    if len(bad_examples) < 10:
                        bad_examples.append(dict(smiles=s, atom=k, symbol=a.GetSymbol(), token=toks[p]))
        report[name] = dict(molecules=len(smiles), aligned_molecules=n_mol, unaligned_molecules=n_unaligned,
                            atoms_checked=n_atom, mismatches=n_bad, mismatch_examples=bad_examples,
                            atoms_by_symbol=dict(by_symbol.most_common()))
        print(name, {k: v for k, v in report[name].items() if k not in ("mismatch_examples", "atoms_by_symbol")},
              flush=True)
    json.dump(report, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
