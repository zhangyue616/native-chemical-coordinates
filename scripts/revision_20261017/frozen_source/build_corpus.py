"""Line B protocol v1, O1: hash-sample a pure-chemistry corpus from PubChem CID-SMILES.

usage: python -I build_corpus.py <CID-SMILES.gz> <out_dir>
Writes corpus_50k.tsv, pilot_20k.tsv, val_10k.tsv, reserve_20k.tsv and corpus_build.json.
"""
import sys, os, gzip, hashlib, json, collections

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SALT = b"lineB-v1:"
KEEP_BELOW = 2000          # of 1_000_000
ALLOWED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
CHIRAL_OK = {Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW}
BOND_OK = {Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
           Chem.rdchem.BondType.AROMATIC}
DIR_OK = {Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT}
PEPTIDE = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)")
NUCLEOSIDE = Chem.MolFromSmarts("[n][CX4;r5]1[OX2;r5][CX4;r5][CX4;r5][CX4;r5]1")
N_TOTAL, N_PILOT, N_VAL = 50000, 20000, 10000


def hval(cid):
    return int(hashlib.sha256(SALT + cid.encode()).hexdigest()[:12], 16) % 1_000_000


def standardize(smi):
    """Return (canonical_smiles, None) or (None, reason)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "parse_fail"
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    best, best_n = None, -1
    for f in frags:
        n = f.GetNumHeavyAtoms()
        if n > best_n:
            best, best_n = f, n
    m = Chem.RemoveHs(best)
    if any(a.GetAtomicNum() not in ALLOWED_Z for a in m.GetAtoms()):
        return None, "element"
    if not 5 <= m.GetNumHeavyAtoms() <= 50 or any(a.GetAtomicNum() == 1 for a in m.GetAtoms()):
        return None, "heavy_atoms"
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in m.GetAtoms()):
        return None, "radical_or_isotope"
    if any(a.GetChiralTag() not in CHIRAL_OK for a in m.GetAtoms()):
        return None, "chiral_tag"
    if any(b.GetBondType() not in BOND_OK or b.GetBondDir() not in DIR_OK for b in m.GetBonds()):
        return None, "bond_type_or_dir"
    if m.HasSubstructMatch(PEPTIDE) or m.HasSubstructMatch(NUCLEOSIDE):
        return None, "bio_structure"
    can = Chem.MolToSmiles(m)
    # the canonical SMILES must round-trip to an equally valid molecule
    m2 = Chem.MolFromSmiles(can)
    if m2 is None or m2.GetNumAtoms() != m.GetNumAtoms():
        return None, "roundtrip"
    if any(a.GetChiralTag() not in CHIRAL_OK for a in m2.GetAtoms()) or \
            any(b.GetBondType() not in BOND_OK or b.GetBondDir() not in DIR_OK for b in m2.GetBonds()):
        return None, "roundtrip"
    return can, None


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    n_lines, cand = 0, []
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            tab = line.find("\t")
            if tab < 0:
                continue
            cid = line[:tab]
            h = hval(cid)
            if h < KEEP_BELOW:
                cand.append((h, int(cid), line[tab + 1:].strip()))
            if n_lines % 20_000_000 == 0:
                print(f"read {n_lines} lines, candidates {len(cand)}", flush=True)
    print(f"total lines {n_lines}, candidates {len(cand)}", flush=True)
    cand.sort(key=lambda t: (t[0], t[1]))
    reasons = collections.Counter()
    seen, kept = set(), []
    for h, cid, smi in cand:
        can, why = standardize(smi)
        if why:
            reasons[why] += 1
            continue
        if can in seen:
            reasons["duplicate_canonical"] += 1
            continue
        seen.add(can)
        kept.append((h, cid, can))
    print(f"kept after filters {len(kept)}; removed {dict(reasons)}", flush=True)
    if len(kept) < N_TOTAL:
        sys.exit(f"STOP: only {len(kept)} molecules pass filters, need {N_TOTAL}")
    kept = kept[:N_TOTAL]
    parts = {"corpus_50k.tsv": kept, "pilot_20k.tsv": kept[:N_PILOT],
             "val_10k.tsv": kept[N_PILOT:N_PILOT + N_VAL], "reserve_20k.tsv": kept[N_PILOT + N_VAL:]}
    hashes = {}
    for name, rows in parts.items():
        p = os.path.join(out, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("rank\tcid\thash\tsmiles\n")
            start = 0 if name in ("corpus_50k.tsv", "pilot_20k.tsv") else (N_PILOT if name == "val_10k.tsv" else N_PILOT + N_VAL)
            for i, (h, cid, can) in enumerate(rows):
                fo.write(f"{start + i}\t{cid}\t{h}\t{can}\n")
        hashes[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    info = dict(protocol="lineB PROTOCOL_v1 O1", source=os.path.basename(src), lines=n_lines,
                candidates=len(cand), removed=dict(reasons), kept_before_truncation=len(seen),
                n_total=N_TOTAL, n_pilot=N_PILOT, n_val=N_VAL, files_sha256=hashes)
    json.dump(info, open(os.path.join(out, "corpus_build.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
