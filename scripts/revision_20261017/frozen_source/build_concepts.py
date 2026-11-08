"""Line B protocol v1, O2: build concept library v1 and atom-level labels on a molecule set.

usage:
  python -I build_concepts.py build <pilot_tsv> <out_dir>
      -> concepts_v1.json (frozen list), labels_pilot.npz, atoms_pilot.npz, build_concepts.json
Labels are heavy-atom level; atom order = molecule order in the tsv, then RDKit atom index
of Chem.MolFromSmiles(smiles) (the same parse every model uses).
"""
import sys, os, json, csv, hashlib, collections, itertools

import numpy as np
import scipy.sparse as sp
from rdkit import Chem, RDConfig, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "IFG"))
from ifg import identify_functional_groups  # noqa: E402

PREV_LO, PREV_HI = 0.0005, 0.50
JACCARD = 0.95
MAXM = 100000


def read_smiles(tsv):
    with open(tsv, encoding="utf-8") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        return [row["smiles"] for row in r]


# ---------------- concept definitions ----------------
def smarts_concepts():
    out = []
    # 1. topology (SMARTS part)
    for name, sm in [("topo:pyridine_type_N", "[n;X2;H0]"), ("topo:pyrrole_type_N", "[n;H1,X3]"),
                     ("topo:aromatic_O", "[o]"), ("topo:aromatic_S", "[s]")]:
        out.append(dict(name=name, source="topology", kind="smarts", definition=sm))
    # 2. RDKit functional group hierarchy
    path = os.path.join(RDConfig.RDDataDir, "Functional_Group_Hierarchy.txt")
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        out.append(dict(name="fg:" + parts[0], source="fg_hierarchy", kind="smarts", definition=parts[1]))
    # 3. fragment descriptors
    path = os.path.join(RDConfig.RDDataDir, "FragmentDescriptors.csv")
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        out.append(dict(name="frag:" + parts[0], source="fragment_descriptors", kind="smarts", definition=parts[2]))
    return out


TOPO_FUNCS = ["topo:ring3", "topo:ring4", "topo:ring5", "topo:ring6", "topo:ring7", "topo:ring8",
              "topo:ring_ge9", "topo:fused_ring_atom", "topo:spiro_atom", "topo:stereocenter"]
ELEMS = {5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
INPUT_FUNCS = ([f"input:element_{s}" for s in ELEMS.values()] +
               ["input:aromatic", "input:charge_pos", "input:charge_neg"] +
               [f"input:totalH_{k}" for k in range(4)] +
               ["input:hyb_SP", "input:hyb_SP2", "input:hyb_SP3"] +
               [f"input:degree_{k}" for k in range(1, 5)] + ["input:in_ring"])


def topo_labels(m):
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    res = {k: set() for k in TOPO_FUNCS}
    for a in m.GetAtoms():
        i = a.GetIdx()
        for k in range(3, 9):
            if a.IsInRingSize(k):
                res[f"topo:ring{k}"].add(i)
        if any(i in r and len(r) >= 9 for r in rings):
            res["topo:ring_ge9"].add(i)
        mine = [r for r in rings if i in r]
        if len(mine) >= 2:
            res["topo:fused_ring_atom"].add(i)
            if any(r1 & r2 == {i} for r1, r2 in itertools.combinations(mine, 2)):
                res["topo:spiro_atom"].add(i)
    for i, _ in Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False):
        res["topo:stereocenter"].add(i)
    return res


def input_labels(m):
    res = {k: set() for k in INPUT_FUNCS}
    H = Chem.rdchem.HybridizationType
    for a in m.GetAtoms():
        i, z = a.GetIdx(), a.GetAtomicNum()
        if z in ELEMS:
            res[f"input:element_{ELEMS[z]}"].add(i)
        if a.GetIsAromatic():
            res["input:aromatic"].add(i)
        if a.GetFormalCharge() > 0:
            res["input:charge_pos"].add(i)
        if a.GetFormalCharge() < 0:
            res["input:charge_neg"].add(i)
        h = a.GetTotalNumHs()
        if h <= 3:
            res[f"input:totalH_{h}"].add(i)
        hy = a.GetHybridization()
        for nm, t in (("SP", H.SP), ("SP2", H.SP2), ("SP3", H.SP3)):
            if hy == t:
                res[f"input:hyb_{nm}"].add(i)
        d = a.GetDegree()
        if 1 <= d <= 4:
            res[f"input:degree_{d}"].add(i)
        if a.IsInRing():
            res["input:in_ring"].add(i)
    return res


def compute_labels(smiles, smarts_defs):
    """Return atoms info, dict name -> list of global atom indices, ertl type names."""
    pats = []
    bad = []
    for c in smarts_defs:
        p = Chem.MolFromSmarts(c["definition"])
        if p is None:
            bad.append(c["name"])
        pats.append(p)
    cols = collections.defaultdict(list)
    mol_idx, atom_idx, mol_heavy = [], [], []
    offset = 0
    for mi, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        n = m.GetNumAtoms()
        mol_idx.extend([mi] * n)
        atom_idx.extend(range(n))
        mol_heavy.extend([n] * n)
        for c, p in zip(smarts_defs, pats):
            if p is None:
                continue
            hit = set()
            for match in m.GetSubstructMatches(p, uniquify=True, maxMatches=MAXM):
                hit.update(match)
            if hit:
                cols[c["name"]].extend(offset + i for i in sorted(hit))
        for d in (topo_labels(m), input_labels(m)):
            for k, v in d.items():
                if v:
                    cols[k].extend(offset + i for i in sorted(v))
        for g in identify_functional_groups(m):
            cols["ertl:" + g.type].extend(offset + i for i in sorted(set(g.atomIds)))
        offset += n
        if (mi + 1) % 5000 == 0:
            print(f"  labelled {mi + 1} molecules, {offset} atoms", flush=True)
    atoms = dict(mol_idx=np.array(mol_idx, np.int32), atom_idx=np.array(atom_idx, np.int16),
                 mol_heavy=np.array(mol_heavy, np.int16))
    return atoms, cols, bad, offset


def label_with_frozen(tsv, concepts_json, out_dir, tag):
    """Apply the frozen concept list (no re-selection) to another molecule set."""
    os.makedirs(out_dir, exist_ok=True)
    smiles = read_smiles(tsv)
    concepts = json.load(open(concepts_json, encoding="utf-8"))
    pats = {c["id"]: Chem.MolFromSmarts(c["definition"]) for c in concepts if c["kind"] == "smarts"}
    ertl_ids = {c["definition"]: c["id"] for c in concepts if c["kind"] == "ertl_type"}
    func_ids = {c["definition"]: c["id"] for c in concepts if c["kind"] == "function"}
    rows, cols = [], []
    mol_idx, atom_idx, mol_heavy = [], [], []
    offset = 0
    for mi, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        n = m.GetNumAtoms()
        mol_idx.extend([mi] * n)
        atom_idx.extend(range(n))
        mol_heavy.extend([n] * n)
        hits = collections.defaultdict(set)
        for cid, p in pats.items():
            for match in m.GetSubstructMatches(p, uniquify=True, maxMatches=MAXM):
                hits[cid].update(match)
        for d in (topo_labels(m), input_labels(m)):
            for k, v in d.items():
                if k in func_ids and v:
                    hits[func_ids[k]].update(v)
        for g in identify_functional_groups(m):
            if g.type in ertl_ids:
                hits[ertl_ids[g.type]].update(g.atomIds)
        for cid, v in hits.items():
            for i in v:
                rows.append(offset + i)
                cols.append(cid)
        offset += n
        if (mi + 1) % 5000 == 0:
            print(f"  labelled {mi + 1} molecules, {offset} atoms", flush=True)
    L = sp.csc_matrix((np.ones(len(rows), bool), (np.array(rows, np.int64), np.array(cols, np.int64))),
                      shape=(offset, len(concepts)))
    L.data[:] = True
    sp.save_npz(os.path.join(out_dir, f"labels_{tag}.npz"), L)
    np.savez(os.path.join(out_dir, f"atoms_{tag}.npz"), mol_idx=np.array(mol_idx, np.int32),
             atom_idx=np.array(atom_idx, np.int16), mol_heavy=np.array(mol_heavy, np.int16))
    pos = np.asarray(L.sum(0)).ravel()
    info = dict(protocol="lineB PROTOCOL_v2 §2", molecules=len(smiles), atoms=offset,
                concepts=len(concepts), concepts_json_sha256=hashlib.sha256(open(concepts_json, "rb").read()).hexdigest(),
                readable_with_zero_positives=[c["name"] for c in concepts if c["partition"] == "readable" and pos[c["id"]] == 0])
    json.dump(info, open(os.path.join(out_dir, f"label_{tag}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))


def main():
    mode = sys.argv[1]
    if mode == "label":
        label_with_frozen(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        return
    tsv, out = sys.argv[2], sys.argv[3]
    assert mode == "build"
    os.makedirs(out, exist_ok=True)
    smiles = read_smiles(tsv)
    sdefs = smarts_concepts()
    atoms, cols, bad, n_atoms = compute_labels(smiles, sdefs)
    print(f"molecules {len(smiles)} atoms {n_atoms}; unparseable SMARTS {bad}", flush=True)

    defs = {}
    for c in sdefs:
        defs[c["name"]] = c
    for k in TOPO_FUNCS:
        defs[k] = dict(name=k, source="topology", kind="function", definition=k)
    for k in INPUT_FUNCS:
        defs[k] = dict(name=k, source="input", kind="function", definition=k)
    for k in cols:
        if k.startswith("ertl:"):
            defs[k] = dict(name=k, source="ertl", kind="ertl_type", definition=k[5:])

    def col_vec(k):
        v = np.zeros(n_atoms, bool)
        v[np.array(sorted(set(cols[k])), dtype=np.int64)] = True
        return v

    priority = {"topology": 0, "fg_hierarchy": 1, "fragment_descriptors": 2, "ertl": 3}
    order_in_source = {}
    for i, c in enumerate(sdefs):
        order_in_source[c["name"]] = i
    for i, k in enumerate(TOPO_FUNCS):
        order_in_source[k] = -100 + i
    ertl_names = sorted(k for k in cols if k.startswith("ertl:"))
    for i, k in enumerate(ertl_names):
        order_in_source[k] = 10000 + i

    candidates = [k for k in defs if defs[k]["source"] != "input" and k in cols]
    candidates.sort(key=lambda k: (priority[defs[k]["source"]], order_in_source[k]))
    prev = {k: len(set(cols[k])) / n_atoms for k in cols}
    stats = collections.Counter()
    stats["defined_noninput"] = sum(1 for k in defs if defs[k]["source"] != "input")
    stats["noninput_with_any_hit"] = len(candidates)
    in_range = [k for k in candidates if PREV_LO <= prev[k] <= PREV_HI]
    stats["noninput_in_prevalence_range"] = len(in_range)

    input_keys = [k for k in INPUT_FUNCS if k in cols]
    inp_vecs = {k: col_vec(k) for k in input_keys}
    kept, kept_vecs, moved_to_input, dup_of = [], [], {}, {}
    for k in in_range:
        v = col_vec(k)
        nv = v.sum()
        hit = None
        for ik, iv in inp_vecs.items():
            inter = np.count_nonzero(v & iv)
            if inter / (nv + iv.sum() - inter) >= JACCARD:
                hit = ik
                break
        if hit:
            moved_to_input[k] = hit
            continue
        dup = None
        for kk, kv in zip(kept, kept_vecs):
            inter = np.count_nonzero(v & kv)
            if inter / (nv + kv.sum() - inter) >= JACCARD:
                dup = kk
                break
        if dup:
            dup_of[k] = dup
            continue
        kept.append(k)
        kept_vecs.append(v)
    stats["moved_to_input_partition"] = len(moved_to_input)
    stats["removed_as_duplicate"] = len(dup_of)
    stats["kept_readable"] = len(kept)

    final = []
    for k in kept:
        d = dict(defs[k]); d.update(partition="readable", prevalence=prev[k], n_pos=int(len(set(cols[k]))))
        final.append(d)
    for k in input_keys:
        d = dict(defs[k]); d.update(partition="input", prevalence=prev[k], n_pos=int(len(set(cols[k]))))
        final.append(d)
    for k, ik in moved_to_input.items():
        d = dict(defs[k]); d.update(partition="input_equivalent", equivalent_to=ik, prevalence=prev[k],
                                    n_pos=int(len(set(cols[k]))))
        final.append(d)
    for i, d in enumerate(final):
        d["id"] = i

    rows, cs = [], []
    for d in final:
        idx = np.array(sorted(set(cols[d["name"]])), dtype=np.int64)
        rows.append(idx)
        cs.append(np.full(len(idx), d["id"], np.int64))
    L = sp.csc_matrix((np.ones(sum(len(r) for r in rows), bool), (np.concatenate(rows), np.concatenate(cs))),
                      shape=(n_atoms, len(final)))
    sp.save_npz(os.path.join(out, "labels_pilot.npz"), L)
    np.savez(os.path.join(out, "atoms_pilot.npz"), **atoms)
    cpath = os.path.join(out, "concepts_v1.json")
    json.dump(final, open(cpath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    info = dict(protocol="lineB PROTOCOL_v1 O2", molecules=len(smiles), atoms=n_atoms, unparseable_smarts=bad,
                stats=dict(stats), dup_of=dup_of, moved_to_input=moved_to_input,
                concepts_sha256=hashlib.sha256(open(cpath, "rb").read()).hexdigest())
    json.dump(info, open(os.path.join(out, "build_concepts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
