"""Per-atom check of the MoLFormer token-to-atom alignment (protocol v6 §1).

usage: python -I check_molformer_alignment.py <out_json> <name>=<tsv> [<name>=<tsv> ...]
For every molecule, the k-th atom token (models.molformer_atom_positions) is compared with RDKit atom k of the same
SMILES. Expected token: the element symbol, lower-cased for aromatic atoms; for bracket atoms the element inside the
bracket ([nH] -> n, [C@@H] -> C, [N+] -> N). Molecules without a valid alignment (count mismatch, <unk>, too long)
are counted separately.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import os, sys, csv, json, re, collections

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402

ELEM = re.compile(r"^\[\d*([A-Za-z][a-z]?)")


def token_element(t):
    return ELEM.match(t).group(1) if t.startswith("[") else t


def main():
    out_json = sys.argv[1]
    tok = models.molformer_tokenizer()
    report = {}
    for arg in sys.argv[2:]:
        name, tsv = arg.split("=", 1)
        with open(tsv, encoding="utf-8") as fh:
            smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
        n_mol = n_atom = n_bad = n_case = n_unaligned = 0
        max_tokens = 0
        bad_examples, case_examples, by_symbol = [], [], collections.Counter()
        for s in smiles:
            pos = models.molformer_atom_positions(s)
            if pos is None:
                n_unaligned += 1
                continue
            ids = tok(s)["input_ids"]
            max_tokens = max(max_tokens, len(ids))
            toks = tok.convert_ids_to_tokens(ids)
            mol = Chem.MolFromSmiles(s)
            n_mol += 1
            for k, p in enumerate(pos):
                a = mol.GetAtomWithIdx(k)
                exp = a.GetSymbol().lower() if a.GetIsAromatic() else a.GetSymbol()
                got = token_element(toks[p])
                n_atom += 1
                by_symbol[a.GetSymbol() + ("(ar)" if a.GetIsAromatic() else "")] += 1
                if got != exp:
                    if got.lower() == exp.lower():
                        n_case += 1
                        if len(case_examples) < 10:
                            case_examples.append(dict(smiles=s, atom=k, symbol=a.GetSymbol(), token=toks[p]))
                    else:
                        n_bad += 1
                        if len(bad_examples) < 10:
                            bad_examples.append(dict(smiles=s, atom=k, symbol=a.GetSymbol(), token=toks[p]))
        report[name] = dict(molecules=len(smiles), aligned_molecules=n_mol, unaligned_molecules=n_unaligned,
                            max_tokens=max_tokens, atoms_checked=n_atom, element_mismatches=n_bad,
                            case_only_mismatches=n_case, element_mismatch_examples=bad_examples,
                            case_only_examples=case_examples, atoms_by_symbol=dict(by_symbol.most_common()))
        print(name, {k: v for k, v in report[name].items() if "examples" not in k and k != "atoms_by_symbol"},
              flush=True)
    json.dump(report, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
