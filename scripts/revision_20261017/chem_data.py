"""New chemistry corpus and D-only frozen labels; no model imports or calls.

Standardization is copied, without scientific rule changes, from the received
v1 build_corpus.py:29-61. Topology functions and SMARTS match-union semantics are
copied from received build_concepts.py:79-104,157-203. The historical scripts are
provenance only, not imported. No Ertl/input labels or historical prevalence are
used. Actual source decompression requires the root-owned data-rule lock.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from semantic_mapper import SemanticMapper

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/revision_20261017/chemistry"
ENGINEERING = ROOT / "results/revision_20261017/data_engineering"
READY_PATH = ENGINEERING / "DATA_BUILDER_READY.json"
LOCK_PATH = ROOT / "docs/revision_20261017/CHEM_DATA_RULE_LOCK.json"
CONCEPT_SOURCE = ROOT / "scripts/revision_20261017/frozen_source/concepts_v1.json"
STANDARDIZE_SOURCE = ROOT / "scripts/revision_20261017/frozen_source/build_corpus.py"
LABEL_SOURCE = ROOT / "scripts/revision_20261017/frozen_source/build_concepts.py"
OLD_EXCLUSION = ROOT / "intake/old50k_exclusion/old_50k_exclusion.tsv"
SALT = b"native-axis-transfer-20261017-v1:"
THRESHOLD, MODULUS = 2000, 1000000
N_D, N_T = 5000, 10000
SOURCE_BYTES = 1486837335
SOURCE_SHA256 = "36b683a59272b196ed7ece140903a249f7df475a06c312c963a5c761f1037f1a"
ALLOWED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
CHIRAL_OK = {Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW}
BOND_OK = {Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
           Chem.rdchem.BondType.AROMATIC}
DIR_OK = {Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT}
PEPTIDE_SMARTS = "[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)"
NUCLEOSIDE_SMARTS = "[n][CX4;r5]1[OX2;r5][CX4;r5][CX4;r5][CX4;r5]1"
PEPTIDE, NUCLEOSIDE = Chem.MolFromSmarts(PEPTIDE_SMARTS), Chem.MolFromSmarts(NUCLEOSIDE_SMARTS)
TOPO_FUNCS = ["topo:ring3", "topo:ring4", "topo:ring5", "topo:ring6", "topo:ring7", "topo:ring8",
              "topo:ring_ge9", "topo:fused_ring_atom", "topo:spiro_atom", "topo:stereocenter"]
MAX_MATCHES = 100000
TSV_FIELDS = ["rank", "cid", "cid_sha256", "canonical_smiles_sha256", "source_line", "smiles"]


class DataError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def smiles_identity(smiles):
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def cid_digest(cid_text):
    return hashlib.sha256(SALT + cid_text.encode("ascii")).digest()


def candidate_bucket(digest):
    return int.from_bytes(digest[:6], "big") % MODULUS


def candidate_sort_key(candidate):
    return candidate[0], candidate[1]


def split_for_rank(rank, n_d=N_D, n_t=N_T):
    if not 0 <= rank < n_d + n_t:
        raise DataError("rank is outside fixed D/T allocation")
    return "D" if rank < n_d else "T"


def standardize(smi):
    """Exact received v1 standardization; return canonical SMILES or reason."""
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
    m2 = Chem.MolFromSmiles(can)
    if m2 is None or m2.GetNumAtoms() != m.GetNumAtoms():
        return None, "roundtrip"
    if any(a.GetChiralTag() not in CHIRAL_OK for a in m2.GetAtoms()) or \
            any(b.GetBondType() not in BOND_OK or b.GetBondDir() not in DIR_OK for b in m2.GetBonds()):
        return None, "roundtrip"
    return can, None


def load_exclusions(path=OLD_EXCLUSION, expected_count=50000):
    cids, identities = set(), set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["cid", "canonical_smiles_sha256"]:
            raise DataError("wrong old-exclusion schema")
        for row in reader:
            cid, identity = row["cid"], row["canonical_smiles_sha256"]
            if not cid.isascii() or not cid.isdecimal() or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
                raise DataError("malformed old-exclusion identity")
            if int(cid) in cids or identity in identities:
                raise DataError("duplicate old-exclusion identity")
            cids.add(int(cid))
            identities.add(identity)
    if expected_count is not None and (len(cids) != expected_count or len(identities) != expected_count):
        raise DataError("wrong old-exclusion count")
    return cids, identities


def load_definitions(path=CONCEPT_SOURCE):
    all_defs = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = ("id", "name", "source", "kind", "definition", "partition")
    concepts = sorted(({k: c[k] for k in fields} for c in all_defs if c["partition"] == "readable"
        and c["source"] in {"topology", "fg_hierarchy", "fragment_descriptors"}), key=lambda c: c["id"])
    counts = collections.Counter(c["source"] for c in concepts)
    if len(concepts) != 91 or counts != {"topology": 14, "fg_hierarchy": 10, "fragment_descriptors": 67}:
        raise DataError("frozen 91-definition source drift")
    if len({c["id"] for c in concepts}) != 91:
        raise DataError("duplicate concept IDs")
    for c in concepts:
        if c["kind"] == "smarts":
            if Chem.MolFromSmarts(c["definition"]) is None:
                raise DataError("unparseable frozen SMARTS: " + c["name"])
        elif c["kind"] != "function" or c["definition"] not in TOPO_FUNCS:
            raise DataError("unsupported frozen concept function")
    return concepts


def topology_labels(m):
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
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


class FrozenLabeler:
    def __init__(self, concepts):
        self.concepts = concepts
        self.patterns = [Chem.MolFromSmarts(c["definition"]) if c["kind"] == "smarts" else None for c in concepts]

    def molecule(self, canonical_smiles):
        m = Chem.MolFromSmiles(canonical_smiles)
        if m is None:
            raise DataError("saved D canonical SMILES cannot be parsed")
        labels = np.zeros((m.GetNumAtoms(), len(self.concepts)), dtype=np.bool_)
        occurrences = np.full(len(self.concepts), -1, dtype=np.int32)
        topology = topology_labels(m)
        for column, (c, pattern) in enumerate(zip(self.concepts, self.patterns)):
            if pattern is not None:
                matches = m.GetSubstructMatches(pattern, uniquify=True, maxMatches=MAX_MATCHES)
                occurrences[column] = len(matches)
                hit = set(itertools.chain.from_iterable(matches))
            else:
                hit = topology[c["definition"]]
            if hit:
                labels[np.fromiter(sorted(hit), dtype=np.int64), column] = True
        return labels, occurrences


def coverage_and_panel(concepts, positive_atom_counts):
    n_mol = positive_atom_counts.shape[0]
    pos_mol = np.count_nonzero(positive_atom_counts, axis=0)
    pos_atoms = positive_atom_counts.sum(axis=0, dtype=np.int64)
    rows, qualified = [], []
    for column, c in enumerate(concepts):
        positive, negative, atoms = int(pos_mol[column]), int(n_mol - pos_mol[column]), int(pos_atoms[column])
        eligible = positive >= 100 and negative >= 100 and atoms >= 200
        rows.append({"id": c["id"], "name": c["name"], "source": c["source"],
                     "positive_molecules_D": positive, "all_negative_molecules_D": negative,
                     "positive_atoms_D": atoms, "in_C": eligible})
        if eligible:
            qualified.append(c)
    queues = [collections.deque(c["id"] for c in qualified if c["source"] == source)
              for source in ("topology", "fg_hierarchy", "fragment_descriptors")]
    panel = []
    while len(panel) < 12 and any(queues):
        for queue in queues:
            if queue and len(panel) < 12:
                panel.append(queue.popleft())
    mechanisms = [row["id"] for row in rows if row["source"] == "fragment_descriptors" and row["in_C"]
                  and row["positive_molecules_D"] >= 200 and row["all_negative_molecules_D"] >= 200][:3]
    coverage = {"split": "D", "n_molecules": n_mol, "concepts": rows,
                "C_ids": [c["id"] for c in qualified], "coverage_gate": "PASS" if len(qualified) >= 12 else "FAIL_FEWER_THAN_12",
                "thresholds": {"positive_molecules": 100, "all_negative_molecules": 100, "positive_atoms": 200}}
    selection = {"C_ids": coverage["C_ids"], "panel_ids": panel,
                 "panel_rule": "round-robin topology,fg_hierarchy,fragment_descriptors; original ID ascending within each source; max12",
                 "mechanism_candidate_ids": mechanisms,
                 "mechanism_rule": "first3 fragment IDs in C with D positive and negative molecules each>=200; no replacement after downstream gates",
                 "model_outputs_or_associations_consulted": False}
    return coverage, selection


def label_development(records, concepts, output_dir):
    if any(r["split"] != "D" for r in records):
        raise DataError("D-only label function received a non-D row")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeler = FrozenLabeler(concepts)
    labels, occurrence_rows, positive_counts = [], [], []
    molecule_indices, atom_indices, molecule_heavy, atom_cids, offsets = [], [], [], [], [0]
    for mi, record in enumerate(records):
        label, occurrence = labeler.molecule(record["smiles"])
        n = len(label)
        labels.append(label)
        occurrence_rows.append(occurrence)
        positive_counts.append(label.sum(axis=0, dtype=np.int32))
        molecule_indices.extend([mi] * n)
        atom_indices.extend(range(n))
        molecule_heavy.extend([n] * n)
        atom_cids.extend([record["cid"]] * n)
        offsets.append(offsets[-1] + n)
    all_labels = np.concatenate(labels, axis=0)
    positive_counts = np.asarray(positive_counts, dtype=np.int32)
    concept_ids = np.array([c["id"] for c in concepts], dtype=np.int32)
    np.savez_compressed(output_dir / "labels_D.npz", labels=all_labels, concept_ids=concept_ids)
    np.savez_compressed(output_dir / "atoms_D.npz", mol_idx=np.asarray(molecule_indices, np.int32),
        atom_idx=np.asarray(atom_indices, np.int16), mol_heavy=np.asarray(molecule_heavy, np.int16),
        cid=np.asarray(atom_cids, np.int64), mol_offsets=np.asarray(offsets, np.int64))
    np.savez_compressed(output_dir / "occurrence_counts_D.npz", counts=np.asarray(occurrence_rows, np.int32),
        positive_atom_counts=positive_counts, concept_ids=concept_ids,
        is_smarts=np.array([c["kind"] == "smarts" for c in concepts], dtype=np.bool_))
    coverage, panel = coverage_and_panel(concepts, positive_counts)
    dump_json(output_dir / "coverage.json", coverage)
    dump_json(output_dir / "panel.json", panel)
    return {"molecules_D": len(records), "atoms_D": len(all_labels), "n_frozen_definitions": len(concepts),
            "C_size": len(coverage["C_ids"]), "coverage_gate": coverage["coverage_gate"],
            "panel_ids": panel["panel_ids"], "mechanism_candidate_ids": panel["mechanism_candidate_ids"],
            "SMARTS_occurrences": "unique complete substructure matches; function columns=-1, never atom counts",
            "atom_label_order": "D.tsv order, then RDKit atom index parsed from exact saved canonical SMILES"}


def token_fingerprint(input_ids):
    return hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode("ascii")).hexdigest()


def select_eligible(candidates, old_cids, old_identities, mapper, target, stats):
    """Candidates MUST already be sorted by full digest bytes,numeric CID."""
    seen, previous = set(), None
    for digest, cid, raw_smiles, source_line in candidates:
        key = digest, cid
        if previous is not None and key < previous:
            raise DataError("candidate stream is not in frozen full-digest order")
        previous = key
        if stats["accepted"] == target:
            break
        stats["processed_prefix"] += 1
        if cid in old_cids:
            stats["excluded"]["old_cid"] += 1
            continue
        canonical, why = standardize(raw_smiles)
        if why:
            stats["excluded"]["standardize:" + why] += 1
            continue
        identity = smiles_identity(canonical)
        if identity in old_identities:
            stats["excluded"]["old_canonical_identity"] += 1
            continue
        if identity in seen:
            stats["excluded"]["duplicate_new_canonical"] += 1
            continue
        seen.add(identity)
        audit = mapper.audit_molecule(canonical, molecule_id=str(cid), require_canonical=True, max_tokens=512)
        if not audit["accepted"]:
            stats["excluded"]["mapper_gate"] += 1
            stats["mapper_reasons"].update(audit["exclusion_reasons"])
            continue
        record = {"rank": stats["accepted"], "cid": cid, "cid_sha256": digest.hex(),
                  "canonical_smiles_sha256": identity, "source_line": source_line, "smiles": canonical}
        stats["accepted"] += 1
        yield record, audit
        if stats["accepted"] == target:
            return


def compact_anchor_record(record, audit):
    fields = ("graph_atom_id", "source_atom_span", "source_element_span", "source_atom_spelling",
              "source_element_spelling", "anchor_token_index", "anchor_token_id", "anchor_token_spelling",
              "anchor_raw_offset", "anchor_offset_verified", "identity_lossy", "lost_attributes")
    return {"rank": record["rank"], "cid": record["cid"], "canonical_smiles_sha256": record["canonical_smiles_sha256"],
            "gate": audit["gate"], "full_offset_status": audit["full_offset_status"],
            "full_offset_errors": audit.get("full_offset_errors", []),
            "input_ids": audit["input_ids"], "attention_mask": audit["attention_mask"],
            "input_ids_sha256": token_fingerprint(audit["input_ids"]),
            "offset_unit": "original ASCII zero-based half-open characters=bytes",
            "identity_lossy_atoms": sum(a["identity_lossy"] for a in audit["atoms"]),
            "atoms": [{k: a[k] for k in fields} for a in audit["atoms"]]}


def collision_counts(by_split):
    def within(counter):
        values = [v for v in counter.values() if v > 1]
        return {"groups": len(values), "molecules": sum(values), "pairs": sum(n * (n - 1) // 2 for n in values)}
    d, t = by_split["D"], by_split["T"]
    shared = set(d) & set(t)
    return {"D": within(d), "T": within(t), "cross_D_T": {"fingerprint_groups": len(shared),
            "D_molecules": sum(d[k] for k in shared), "T_molecules": sum(t[k] for k in shared),
            "pairs": sum(d[k] * t[k] for k in shared)},
            "hash_contract": "SHA256 ASCII json.dumps(input_ids,separators=(',',':')); no padding",
            "collision_changes_eligibility": False, "T_chemical_or_collision_examples_disclosed": False}


def rules_summary():
    return {"salt_ascii": SALT.decode("ascii"), "threshold": THRESHOLD, "modulus": MODULUS,
            "candidate_rule": "int(full_sha256_hex[:12],16)%1000000<2000",
            "sort": "full SHA256 digest bytes,numeric CID ascending; not modulo bucket",
            "allocation": {"D": N_D, "T": N_T, "D_ranks": [0, N_D - 1], "T_ranks": [N_D, N_D + N_T - 1]},
            "canonical_identity": "SHA256 exact canonical SMILES UTF-8 with no newline or normalization",
            "standardization": "received v1 exact algorithm; largest fragment first-tie, RemoveHs, original filters, default canonical isomeric MolToSmiles",
            "peptide_smarts": PEPTIDE_SMARTS, "nucleoside_smarts": NUCLEOSIDE_SMARTS,
            "mapper": "audit_molecule require_canonical=True,max_tokens=512,unique verified element anchor,identity loss permitted",
            "definitions": {"total": 91, "topology": 14, "fg_hierarchy": 10, "fragment_descriptors": 67,
                            "historical_counts_copied": False, "ertl_or_input_equivalent_generated": False},
            "D_counts": "all-match atom union labels plus separate SMARTS unique match occurrences; topology occurrence=-1",
            "T": "exact input qualification and minimal anchors only; no labels,counts,activations,headoutputs or scientific summaries"}


def input_fingerprints():
    paths = [Path(__file__), Path(__file__).with_name("test_chem_data.py"),
        ROOT / "docs/revision_20261017/CHEMISTRY_DATA_PROTOCOL.md", ROOT / "docs/revision_20261017/CHEMISTRY_SOURCE_RULE.md",
        Path(__file__).with_name("semantic_mapper.py"), ROOT / "assets/chemberta_mtr_tokenizer/provenance.json",
        ROOT / "assets/chemberta_mtr_tokenizer/tokenizer.json", CONCEPT_SOURCE, STANDARDIZE_SOURCE, LABEL_SOURCE,
        OLD_EXCLUSION, OLD_EXCLUSION.with_name("receipt.json"), BASE / "source/SOURCE_RECEIPT.json",
        BASE / "corpus/concepts_91.json",
        ENGINEERING / "ENGINEERING_CHECKS.json"]
    return {p.relative_to(ROOT).as_posix(): {"size_bytes": p.stat().st_size, "sha256": file_sha(p)} for p in paths}


def source_stat_and_receipt():
    source = BASE / "source/CID-SMILES.gz"
    receipt = json.loads((BASE / "source/SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt["status"] != "PASS_SOURCE_IDENTITY" or receipt["sha256"] != SOURCE_SHA256 or receipt["received_bytes"] != SOURCE_BYTES:
        raise DataError("source receipt identity failed")
    stat = source.stat()
    if stat.st_size != SOURCE_BYTES:
        raise DataError("source stat size differs from accepted identity")
    return {"project_relative_file": source.relative_to(ROOT).as_posix(), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256_from_accepted_receipt": SOURCE_SHA256,
            "source_rehashed": False, "source_decompressed": False}


def prepare_ready():
    if LOCK_PATH.exists():
        raise DataError("root data-rule lock already exists; do not replace its READY binding")
    if rdBase.rdkitVersion != "2026.03.6":
        raise DataError("RDKit version drift")
    checks = json.loads((ENGINEERING / "ENGINEERING_CHECKS.json").read_text(encoding="utf-8"))
    if checks["status"] != "PASS":
        raise DataError("synthetic data-builder engineering checks not PASS")
    concepts = load_definitions()
    old_cids, old_identities = load_exclusions()
    dump_json(BASE / "corpus/concepts_91.json", concepts)
    ready = {"status": "DATA_BUILDER_READY", "prepared_at_utc": utc_now(), "rules": rules_summary(),
             "runtime": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
             "inputs": input_fingerprints(), "source": source_stat_and_receipt(),
             "old_exclusion": {"unique_cids": len(old_cids), "unique_canonical_identities": len(old_identities)},
             "outputs": ["development/D.tsv", "sealed/T.tsv", "development/anchors_D.jsonl.gz", "sealed/anchors_T.jsonl.gz",
                         "development/labels_D.npz", "development/atoms_D.npz", "development/occurrence_counts_D.npz",
                         "development/coverage.json", "development/panel.json"],
             "permission": "WAIT_FOR_ROOT_CHEM_DATA_RULE_LOCK; no actual gzip decompression or sampling performed",
             "T_label_calls": 0, "T_scientific_reads": 0, "model_loads": 0, "model_forwards": 0}
    dump_json(READY_PATH, ready)
    return {"status": ready["status"], "ready_file": READY_PATH.relative_to(ROOT).as_posix(),
            "ready_sha256": file_sha(READY_PATH), "source_decompressed": False}


def require_data_lock(lock_path=LOCK_PATH, ready_path=READY_PATH):
    if not Path(lock_path).is_file() or not Path(ready_path).is_file():
        raise DataError("STOP: root CHEM_DATA_RULE_LOCK and DATA_BUILDER_READY are required before gzip access")
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("data_builder_ready_sha256") != file_sha(ready_path):
        raise DataError("STOP: root data-rule lock does not bind this READY file")
    return json.loads(Path(ready_path).read_text(encoding="utf-8"))


def scan_candidates(source, *, progress_every=5000000):
    """The sole real gzip scan; never called by prepare or synthetic tests."""
    candidates, n_lines, malformed = [], 0, 0
    started = time.monotonic()
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            n_lines += 1
            cid_text, separator, raw_smiles = line.partition("\t")
            if not separator or not cid_text.isascii() or not cid_text.isdecimal():
                malformed += 1
                continue
            digest = cid_digest(cid_text)
            if candidate_bucket(digest) < THRESHOLD:
                candidates.append((digest, int(cid_text), raw_smiles.strip(), n_lines))
            if progress_every and n_lines % progress_every == 0:
                print(json.dumps({"stage": "source_scan", "lines": n_lines, "candidates": len(candidates),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), file=sys.stderr, flush=True)
    candidates.sort(key=candidate_sort_key)
    # Source is a CID table. Conflicting duplicate CID rows are a source error,
    # not an opportunity to select a different molecule under the same identity.
    if len({c[1] for c in candidates}) != len(candidates):
        raise DataError("duplicate CID in fixed source candidate set")
    return candidates, {"source_lines": n_lines, "all_source_candidates": len(candidates),
                        "malformed_source_lines": malformed, "source_decompression_passes": 1}


def run_build():
    ready = require_data_lock()
    if ready["inputs"] != input_fingerprints() or ready["source"] != source_stat_and_receipt() or ready["rules"] != rules_summary():
        raise DataError("STOP: implementation/input/source/rule drift since root lock")
    if rdBase.rdkitVersion != ready["runtime"]["rdkit"] or np.__version__ != ready["runtime"]["numpy"]:
        raise DataError("STOP: runtime drift")
    d_dir, t_dir, corpus_dir = BASE / "development", BASE / "sealed", BASE / "corpus"
    for path in (d_dir / "D.tsv", t_dir / "T.tsv", d_dir / "labels_D.npz", corpus_dir / "BUILD_RECEIPT.json"):
        if path.exists():
            raise DataError("STOP: refusing to overwrite existing actual data products")
    for directory in (d_dir, t_dir, corpus_dir):
        directory.mkdir(parents=True, exist_ok=True)
    candidate_path, scan_path = t_dir / "candidates_sorted.tsv.gz", corpus_dir / "SOURCE_SCAN.json"
    if candidate_path.exists() or scan_path.exists():
        if not candidate_path.is_file() or not scan_path.is_file():
            raise DataError("incomplete candidate checkpoint; retain evidence and request bounded repair")
        scan_info = json.loads(scan_path.read_text(encoding="utf-8"))
        if scan_info["ready_sha256"] != file_sha(READY_PATH) or scan_info["candidate_file_sha256"] != file_sha(candidate_path):
            raise DataError("candidate checkpoint identity drift")
        def saved_candidates():
            with gzip.open(candidate_path, "rt", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    yield bytes.fromhex(row["digest"]), int(row["cid"]), row["raw_smiles"], int(row["source_line"])
        candidates = saved_candidates()
    else:
        candidates, scan_info = scan_candidates(BASE / "source/CID-SMILES.gz")
        with gzip.open(candidate_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["digest", "cid", "source_line", "raw_smiles"])
            writer.writerows((d.hex(), cid, line, smiles) for d, cid, smiles, line in candidates)
        scan_info.update(ready_sha256=file_sha(READY_PATH), candidate_file_sha256=file_sha(candidate_path), status="PASS_COMPLETE_SOURCE_SCAN")
        dump_json(scan_path, scan_info)
    old_cids, old_identities = load_exclusions()
    mapper = SemanticMapper()
    stats = {"processed_prefix": 0, "accepted": 0, "excluded": collections.Counter(), "mapper_reasons": collections.Counter()}
    development, fingerprints = [], {"D": collections.Counter(), "T": collections.Counter()}
    with (d_dir / "D.tsv").open("w", encoding="utf-8", newline="") as df, \
            (t_dir / "T.tsv").open("w", encoding="utf-8", newline="") as tf, \
            gzip.open(d_dir / "anchors_D.jsonl.gz", "wt", encoding="utf-8") as da, \
            gzip.open(t_dir / "anchors_T.jsonl.gz", "wt", encoding="utf-8") as ta:
        writers = {"D": csv.DictWriter(df, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n"),
                   "T": csv.DictWriter(tf, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n")}
        for writer in writers.values():
            writer.writeheader()
        for record, audit in select_eligible(candidates, old_cids, old_identities, mapper, N_D + N_T, stats):
            split = split_for_rank(record["rank"])
            writers[split].writerow(record)
            compact = compact_anchor_record(record, audit)
            (da if split == "D" else ta).write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
            fingerprints[split][compact["input_ids_sha256"]] += 1
            if split == "D":
                development.append({**record, "split": "D"})
            if stats["accepted"] % 1000 == 0:
                print(json.dumps({"stage": "qualification", "processed_prefix": stats["processed_prefix"], "accepted": stats["accepted"]}),
                      file=sys.stderr, flush=True)
    if stats["accepted"] != N_D + N_T:
        dump_json(corpus_dir / "BUILD_FAILURE.json", {"status": "FAIL_FIXED_CANDIDATES_INSUFFICIENT", **scan_info, **stats,
            "salt_or_threshold_adjusted": False, "T_labels_generated": False})
        raise DataError("fixed candidate set insufficient; stop without salt/threshold/source rescue")
    dump_json(corpus_dir / "token_collision_counts.json", collision_counts(fingerprints))
    concepts = load_definitions()
    label_info = label_development(development, concepts, d_dir)
    outputs = [d_dir / "D.tsv", t_dir / "T.tsv", d_dir / "anchors_D.jsonl.gz", t_dir / "anchors_T.jsonl.gz",
        d_dir / "labels_D.npz", d_dir / "atoms_D.npz", d_dir / "occurrence_counts_D.npz",
        d_dir / "coverage.json", d_dir / "panel.json", corpus_dir / "concepts_91.json", corpus_dir / "token_collision_counts.json"]
    receipt = {"status": "PASS_DATA_BUILD" if label_info["coverage_gate"] == "PASS" else "FAIL_D_CONCEPT_COVERAGE",
        "completed_at_utc": utc_now(), "data_rule_lock_sha256": file_sha(LOCK_PATH), "ready_sha256": file_sha(READY_PATH),
        "source_scan": scan_info, "qualification_processed_prefix": stats,
        "n_D": N_D, "n_T": N_T, "development": label_info,
        "T_boundary": {"input_records_constructed": N_T, "label_calls": 0, "concept_count_calls": 0,
                       "scientific_reads": 0, "activation_calls": 0, "head_output_calls": 0},
        "execution_ledger": {"model_loads": 0, "model_forwards": 0, "fits": 0, "bootstrap_or_inference": 0},
        "outputs": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in outputs},
        "scope_limitations": ["hash-by-CID molecule split, not scaffold split", "old50k exclusion does not prove absence from MTR pretraining",
                              "prefix exclusion counts do not estimate full-source filter prevalence", "token collisions do not change locked eligibility"]}
    dump_json(corpus_dir / "BUILD_RECEIPT.json", receipt)
    return {"status": receipt["status"], "source_lines": scan_info["source_lines"],
            "all_source_candidates": scan_info["all_source_candidates"], "processed_prefix": stats["processed_prefix"],
            "excluded": stats["excluded"], "accepted": stats["accepted"], "n_D": N_D, "n_T": N_T,
            "development": label_info, "T_scientific_reads": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "build"])
    args = parser.parse_args()
    try:
        with rdBase.BlockLogs():
            result = prepare_ready() if args.action == "prepare" else run_build()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (DataError, OSError) as exc:
        print(json.dumps({"status": "STOP", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
