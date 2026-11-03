"""Protocol v5 §1: draw held-out set H4 from PubChem with a new salt, excluding every used CID and structure.

usage: python -I build_corpus_v5.py <CID-SMILES.gz> <corpus_50k.tsv> <out_dir>
Writes h4_candidates.tsv (all molecules that pass, in hash order), H4_used.tsv (first 10,000 with a valid ChemBERTa-2
alignment) and prepare_H4.json.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, os, gzip, csv, json, hashlib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import standardize  # noqa: E402  (identical filters to protocol v1)
import models  # noqa: E402

SALT = b"lineB-v5:"
KEEP_BELOW = 150          # of 1_000_000
N_H4 = 10000


def hval(cid):
    return int(hashlib.sha256(SALT + cid.encode()).hexdigest()[:12], 16) % 1_000_000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    src, used_tsv, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    with open(used_tsv, encoding="utf-8") as fh:
        used = list(csv.DictReader(fh, delimiter="\t"))
    used_cids = {int(r["cid"]) for r in used}
    used_smiles = {r["smiles"] for r in used}
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
    cand.sort(key=lambda t: (t[0], t[1]))
    print(f"lines {n_lines}, candidates {len(cand)}", flush=True)
    reasons = collections.Counter()
    seen, kept = set(), []
    for h, cid, smi in cand:
        if cid in used_cids:
            reasons["used_cid"] += 1
            continue
        can, why = standardize(smi)
        if why:
            reasons[why] += 1
            continue
        if can in used_smiles:
            reasons["used_structure"] += 1
            continue
        if can in seen:
            reasons["duplicate_canonical"] += 1
            continue
        seen.add(can)
        kept.append((h, cid, can))
    with open(os.path.join(out, "h4_candidates.tsv"), "w", encoding="utf-8", newline="\n") as fo:
        fo.write("rank\tcid\thash\tsmiles\n")
        for i, (h, cid, can) in enumerate(kept):
            fo.write(f"{i}\t{cid}\t{h}\t{can}\n")
    final, dropped = [], []
    for row in kept:
        if models.chemberta_atom_positions(row[2]) is None:
            dropped.append(row[2])
            continue
        final.append(row)
        if len(final) == N_H4:
            break
    if len(final) < N_H4:
        sys.exit(f"STOP: only {len(final)} molecules available for H4")
    p = os.path.join(out, "H4_used.tsv")
    with open(p, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("rank\tcid\thash\tsmiles\n")
        for i, (h, cid, can) in enumerate(final):
            fo.write(f"{i}\t{cid}\t{h}\t{can}\n")
    assert not ({c for _, c, _ in final} & used_cids) and not ({s for _, _, s in final} & used_smiles)
    info = dict(protocol="lineB PROTOCOL_v5 §1", source=os.path.basename(src), source_sha256=sha(src),
                salt=SALT.decode(), keep_below=KEEP_BELOW, lines=n_lines, candidates=len(cand),
                removed=dict(reasons), passed_filters=len(kept), dropped_alignment_before_10000=len(dropped),
                dropped_alignment_examples=dropped[:10], n_H4=len(final), H4_sha256=sha(p))
    json.dump(info, open(os.path.join(out, "prepare_H4.json"), "w"), indent=1)
    print(json.dumps(info, indent=1), flush=True)


if __name__ == "__main__":
    main()
