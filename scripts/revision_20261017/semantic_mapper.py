"""Offset-grounded ChemBERTa MTR mapping; no model or scientific data access.

The exact map-free input is the ONLY string passed to the tokenizer. Temporary
source maps are used solely to cross-check RDKit graph attribution. A token's
element anchor and its complete bracket-atom context are separate relations;
neither relation selects a scientific observation population.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rdkit import Chem
from tokenizers import Tokenizer

REPOSITORY = "DeepChem/ChemBERTa-77M-MTR"
REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = ROOT / "assets/chemberta_mtr_tokenizer"
BRACKET = re.compile(
    r"(?P<isotope>\d*)(?P<element>se|as|[A-Z][a-z]?|[bcnops])"
    r"(?P<chiral>@@|@)?(?P<hydrogen>H\d*)?"
    r"(?P<charge>\+(?:\d+|\+*)|-(?:\d+|-*))?"
)
ORGANIC = re.compile(r"Cl|Br|[BCNOPSFIbcnops]")


class MappingError(ValueError):
    """The input or attribution contract is unsupported, rather than repaired."""


def _charge(text: str | None) -> int:
    if not text:
        return 0
    sign = 1 if text[0] == "+" else -1
    return sign * (int(text[1:]) if text[1:].isdigit() else len(text))


def _atom_properties(atom):
    return {
        "atomic_number": atom.GetAtomicNum(),
        "element": atom.GetSymbol(),
        "isotope": atom.GetIsotope(),
        "formal_charge": atom.GetFormalCharge(),
        "aromatic": atom.GetIsAromatic(),
        "chiral_tag": str(atom.GetChiralTag()),
        "cip": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
        "explicit_hydrogens": atom.GetNumExplicitHs(),
        "implicit_hydrogens": atom.GetNumImplicitHs(),
        "radical_electrons": atom.GetNumRadicalElectrons(),
    }


def lex_smiles(smiles: str):
    """Read atom spans and independently construct source adjacency.

    Supported: ASCII organic subset, ordinary bracket elements, isotope, @/@@,
    H counts, charges, branches, . and one/two-digit ring labels. Extended
    chirality, CXSMILES, atom-map inputs, wildcards and query atoms are rejected.
    """
    if not smiles or not smiles.isascii() or any(c.isspace() for c in smiles):
        raise MappingError("require nonempty ASCII SMILES without whitespace")
    atoms, syntax, edges, stack, rings = [], [], [], [], {}
    current = None
    pending = None
    cursor = 0

    def connect(a, b, bond):
        if a is None or a == b:
            raise MappingError("invalid source edge")
        key = sorted((a, b))
        if any(e["source_atom_ids"] == key for e in edges):
            raise MappingError("duplicate source edge")
        edges.append({"source_atom_ids": key, "raw_bond": bond or "implicit"})

    while cursor < len(smiles):
        start = cursor
        ch = smiles[cursor]
        if ch == "[":
            end = smiles.find("]", cursor + 1)
            if end < 0:
                raise MappingError("unclosed bracket atom")
            text = smiles[cursor + 1:end]
            if ":" in text:
                raise MappingError("model input must be atom-map-free")
            match = BRACKET.fullmatch(text)
            if match is None:
                raise MappingError("unsupported bracket-atom grammar")
            cursor = end + 1
            element_start = start + 1 + match.start("element")
            parts = {}
            for name in ("isotope", "element", "chiral", "hydrogen", "charge"):
                if match.group(name):
                    parts[name] = [start + 1 + match.start(name), start + 1 + match.end(name)]
            element = match.group("element")
            isotope = int(match.group("isotope") or 0)
            charge = _charge(match.group("charge"))
            bracketed = True
        else:
            match = ORGANIC.match(smiles, cursor)
            if match:
                cursor = match.end()
                element, element_start = match.group(), start
                isotope, charge, bracketed = 0, 0, False
                parts = {"element": [start, cursor]}
            else:
                if ch in "-=#:/\\~":
                    if current is None or pending is not None or ch == "~":
                        raise MappingError("unsupported or misplaced bond")
                    pending = ch
                    cursor += 1
                    kind = "bond"
                elif ch == "(":
                    if current is None:
                        raise MappingError("branch without atom")
                    stack.append(current)
                    cursor += 1
                    kind = "branch_open"
                elif ch == ")":
                    if not stack or pending:
                        raise MappingError("unbalanced source branch")
                    current = stack.pop()
                    cursor += 1
                    kind = "branch_close"
                elif ch == ".":
                    if current is None or pending:
                        raise MappingError("misplaced dot")
                    current = None
                    cursor += 1
                    kind = "component_separator"
                elif ch.isdigit() or ch == "%":
                    if current is None:
                        raise MappingError("ring label without atom")
                    if ch == "%":
                        label = smiles[cursor + 1:cursor + 3]
                        if len(label) != 2 or not label.isdigit():
                            raise MappingError("only %nn ring labels are supported")
                        cursor += 3
                    else:
                        label = ch
                        cursor += 1
                    if label in rings:
                        other, opening_bond = rings.pop(label)
                        if pending and opening_bond and pending != opening_bond:
                            raise MappingError("inconsistent ring bond")
                        connect(other, current, pending or opening_bond)
                    else:
                        rings[label] = (current, pending)
                    pending = None
                    kind = "ring_label"
                else:
                    raise MappingError(f"unsupported character at offset {cursor}")
                syntax.append({"kind": kind, "span": [start, cursor], "text": smiles[start:cursor]})
                continue
        symbol = element.capitalize()
        atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
        if not atomic_number:
            raise MappingError("unknown element")
        source_id = len(atoms) + 1
        atom = {
            "source_atom_id": source_id,
            "span": [start, cursor],
            "text": smiles[start:cursor],
            "element_span": [element_start, element_start + len(element)],
            "parts": parts,
            "lexical_properties": {"atomic_number": atomic_number, "isotope": isotope,
                                   "formal_charge": charge, "aromatic": element.islower()},
            "bracketed": bracketed,
        }
        atoms.append(atom)
        if current is not None:
            connect(current, source_id, pending)
        current, pending = source_id, None
    if stack or rings or pending or current is None:
        raise MappingError("unclosed source branch/ring/bond/component")
    return atoms, syntax, edges


def _mol(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise MappingError("RDKit rejected SMILES")
    return mol


def graph_attribution(smiles, atoms, source_edges):
    """Bind source maps to RDKit nodes, then check properties AND graph edges.

    Bracketizing a bare organic atom changes auxiliary implicit-H/radical
    semantics, so those auxiliary attributes are NEVER used as source truth.
    Original graph attributes are retained in full. Original and auxiliary
    node-index correspondence is explicitly verified by chemical signatures,
    chirality and complete adjacency; source adjacency is a third check.
    """
    original = _mol(smiles)
    fragments, cursor = [], 0
    for atom in atoms:
        start, end = atom["span"]
        fragments.append(smiles[cursor:start])
        text = atom["text"]
        fragments.append((text[:-1] if atom["bracketed"] else "[" + text)
                         + ":" + str(atom["source_atom_id"]) + "]")
        cursor = end
    fragments.append(smiles[cursor:])
    auxiliary = _mol("".join(fragments))
    if original.GetNumAtoms() != len(atoms) or auxiliary.GetNumAtoms() != len(atoms):
        raise MappingError("source/RDKit node set mismatch")
    source_to_graph = {}
    compare = ("atomic_number", "isotope", "formal_charge", "aromatic", "chiral_tag")
    for atom in auxiliary.GetAtoms():
        source_id, graph_id = atom.GetAtomMapNum(), atom.GetIdx()
        if source_id in source_to_graph or not 1 <= source_id <= len(atoms):
            raise MappingError("auxiliary source identity is not unique")
        source_to_graph[source_id] = graph_id
        observed = _atom_properties(original.GetAtomWithIdx(graph_id))
        mapped = _atom_properties(atom)
        if any(observed[k] != mapped[k] for k in compare):
            raise MappingError("original/auxiliary atom signature mismatch")
        lexical = atoms[source_id - 1]["lexical_properties"]
        if any(observed[k] != lexical[k] for k in ("atomic_number", "isotope", "formal_charge")):
            raise MappingError("source element/isotope/charge does not match RDKit node")
        if lexical["aromatic"] and not observed["aromatic"]:
            raise MappingError("source aromatic atom lost during parsing")
        atoms[source_id - 1]["graph_atom_id"] = graph_id
        atoms[source_id - 1]["graph_properties"] = observed
    def edges(mol):
        return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                (b.GetBondTypeAsDouble(), str(b.GetStereo()), str(b.GetBondDir()))
                for b in mol.GetBonds()}
    original_edges = edges(original)
    if original_edges != edges(auxiliary):
        raise MappingError("original/auxiliary graph edge or stereo mismatch")
    reconstructed = {}
    bond_values = {"-": 1., "/": 1., "\\": 1., "=": 2., "#": 3., ":": 1.5}
    for edge in source_edges:
        pair = tuple(sorted(source_to_graph[i] for i in edge["source_atom_ids"]))
        if pair not in original_edges:
            raise MappingError("source adjacency disagrees with RDKit")
        actual = original_edges[pair][0]
        raw = edge["raw_bond"]
        expected = bond_values.get(raw)
        # Sanitized Kekule input has aromatic RDKit bonds. Preserve raw notation.
        if expected is not None and actual != expected and not (actual == 1.5 and expected in (1., 2.)):
            raise MappingError("source bond order disagrees with sanitized RDKit")
        reconstructed[pair] = actual
    if set(reconstructed) != set(original_edges):
        raise MappingError("source/RDKit adjacency edge sets differ")
    return {
        "identity_method": "source atom maps in auxiliary parse + original graph signatures + independent source adjacency",
        "source_maps_tokenized": False,
        "nodes": [{"graph_atom_id": a.GetIdx(), **_atom_properties(a)} for a in original.GetAtoms()],
        "edges": [{"graph_atom_ids": list(pair), "bond_order": v[0], "stereo": v[1], "direction": v[2]}
                  for pair, v in sorted(original_edges.items())],
        "automorphism_policy": "graph_atom_id is the original exact-string RDKit index; distinct source occurrences remain distinct, not canonical cross-SMILES identities",
    }


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


class SemanticMapper:
    def __init__(self, asset_dir=DEFAULT_ASSET_DIR):
        asset_dir = Path(asset_dir)
        provenance = json.loads((asset_dir / "provenance.json").read_text(encoding="utf-8"))
        if (provenance["repository"], provenance["revision"]) != (REPOSITORY, REVISION):
            raise MappingError("wrong tokenizer identity")
        path = asset_dir / "tokenizer.json"
        expected = next(f["sha256"] for f in provenance["files"] if f["relative_file"] == "tokenizer.json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise MappingError("tokenizer.json differs from provenance")
        specification = json.loads(path.read_text(encoding="utf-8"))
        if specification["normalizer"] is not None or specification["pre_tokenizer"]["type"] != "ByteLevel":
            raise MappingError("offset fidelity contract not established for these tokenizer components")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.special_ids = {x["id"] for x in specification["added_tokens"] if x["special"]}
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.identity = {"repository": REPOSITORY, "revision": REVISION, "tokenizer_sha256": expected}

    def map(self, smiles, *, pad_to=None, max_tokens=512):
        atoms, syntax, source_edges = lex_smiles(smiles)
        graph = graph_attribution(smiles, atoms, source_edges)
        encoding = self.tokenizer.encode(smiles, add_special_tokens=True)
        raw = {"ids": list(encoding.ids), "tokens": list(encoding.tokens),
               "offsets": [list(x) for x in encoding.offsets],
               "attention_mask": list(encoding.attention_mask),
               "special_tokens_mask": list(encoding.special_tokens_mask)}
        if pad_to is not None:
            if pad_to < len(raw["ids"]):
                raise MappingError("padding length below actual token count; truncation is forbidden")
            for _ in range(pad_to - len(raw["ids"])):
                for key, value in (("ids", self.pad_id), ("tokens", "[PAD]"), ("offsets", [0, 0]),
                                   ("attention_mask", 0), ("special_tokens_mask", 1)):
                    raw[key].append(value)
        return self.attribute(smiles, atoms, syntax, graph, raw, max_tokens=max_tokens)

    def audit_molecule(self, smiles, *, molecule_id=None, require_canonical=True, max_tokens=512):
        """Return an all-atoms-or-exclude corpus gate; never change model input.

        The caller freezes source sanitation/scientific split roles separately.
        Identity loss alone does not fail this positional element-anchor gate.
        Full-offset defects remain explicit, even when independent anchors pass.
        """
        base = {"schema": "element-anchor-audit/1", "molecule_id": molecule_id,
                "tokenizer": self.identity, "exact_model_input": smiles,
                "offset_unit": "zero-based half-open original ASCII character offsets; bytes are identical",
                "graph_atom_id_convention": "RDKit atom index parsed from exact_model_input with removeHs=False",
                "contract": "one_original_graph_atom_one_verified_element_anchor_no_aggregation",
                "canonical_input_required": require_canonical,
                "source_sanitation_or_scientific_eligibility_evaluated": False}
        try:
            report = self.map(smiles, max_tokens=max_tokens)
            canonical = Chem.MolToSmiles(_mol(smiles), canonical=True, isomericSmiles=True)
        except (MappingError, RuntimeError) as exc:
            return {**base, "gate": "EXCLUDE", "accepted": False,
                    "exclusion_reasons": ["unsupported_or_invalid_source:" + str(exc)],
                    "element_anchor_position_status": "BLOCKED", "full_offset_status": "NOT_EVALUATED",
                    "atoms": [], "accepted_atom_count": 0}
        reasons = list(report["anchor_errors"])
        if require_canonical and smiles != canonical:
            reasons.append("exact_input_is_not_canonical_SMILES")
        if report["element_anchor_position_status"] != "position-aligned" and not reasons:
            reasons.append("element_anchor_bijection_not_established")
        atom_rows = []
        for atom in sorted(report["atoms"], key=lambda a: a["graph_atom_id"]):
            anchor = report["tokens"][atom["anchor_token_indices"][0]] if len(atom["anchor_token_indices"]) == 1 else None
            atom_rows.append({
                "graph_atom_id": atom["graph_atom_id"], "source_atom_id": atom["source_atom_id"],
                "source_atom_span": atom["span"], "source_element_span": atom["element_span"],
                "source_atom_spelling": atom["text"],
                "source_element_spelling": smiles[slice(*atom["element_span"])],
                "graph_properties": atom["graph_properties"],
                "anchor_token_index": anchor["token_index"] if anchor else None,
                "anchor_token_id": anchor["id"] if anchor else None,
                "anchor_token_spelling": anchor["token"] if anchor else None,
                "anchor_raw_offset": anchor["offset"] if anchor else None,
                "source_text_at_anchor_offset": smiles[slice(*anchor["offset"])] if anchor else None,
                "anchor_offset_verified": bool(anchor and anchor["offset_source_text_valid"]),
                "identity_lossy": atom["fidelity"] == "identity-lossy",
                "lost_attributes": atom["lost_attributes"],
                "complete_atom_span_relation": atom["span_relation"],
                "complete_atom_span_token_indices": atom["span_token_indices"],
            })
        accepted = not reasons
        return {**base, "gate": "PASS_POSITIONAL_ELEMENT_ANCHOR" if accepted else "EXCLUDE",
                "accepted": accepted, "exclusion_reasons": sorted(set(reasons)),
                "canonical_check": {"exact_input_is_canonical": smiles == canonical,
                                    "canonical_representation_for_check_only": canonical,
                                    "model_input_changed": False},
                "element_anchor_position_status": report["element_anchor_position_status"],
                "full_offset_status": report["position_status"], "full_offset_errors": report["errors"],
                "full_offset_invalid_tokens": [t for t in report["tokens"] if t["kind"] == "invalid_source_offset"],
                "input_fidelity": report["input_fidelity"], "lost_syntax": report["lost_syntax"],
                "identity_lossy": any(a["identity_lossy"] for a in atom_rows),
                "graph_validation": report["graph"]["identity_method"],
                "input_ids": report["input_ids"], "attention_mask": report["attention_mask"],
                "special_tokens_mask": report["special_tokens_mask"],
                "non_anchor_tokens": [{"token_index": t["token_index"], "token": t["token"],
                    "raw_offset": t["offset"], "kind": t["kind"], "graph_atom_id": None,
                    "atom_context_graph_atom_ids": t["atom_context_graph_atom_ids"]}
                    for t in report["tokens"] if t["graph_atom_id"] is None],
                "atoms": atom_rows, "graph_atom_count": len(atom_rows),
                "accepted_atom_count": len(atom_rows) if accepted else 0}

    def attribute(self, smiles, atoms, syntax, graph, raw, *, max_tokens=512):
        """Validate a real encoding, also exposed for adversarial engineering fixtures."""
        errors, tokens, covered, seen = [], [], set(), set()
        previous_end = 0
        lengths = {len(raw[k]) for k in ("ids", "tokens", "offsets", "attention_mask", "special_tokens_mask")}
        if len(lengths) != 1:
            raise MappingError("encoding fields have inconsistent lengths")
        if len(raw["ids"]) > max_tokens:
            errors.append("over_token_window_no_truncation")
        for i, (token_id, token, offset, attention, special_mask) in enumerate(zip(
                raw["ids"], raw["tokens"], raw["offsets"], raw["attention_mask"], raw["special_tokens_mask"])):
            entry = {"token_index": i, "id": token_id, "token": token, "offset": offset,
                     "attention_mask": attention, "special_tokens_mask": special_mask,
                     "graph_atom_id": None, "atom_context_graph_atom_ids": [], "source_atom_ids": [],
                     "offset_source_text_valid": None}
            if token_id == self.unk_id:
                errors.append(f"unknown_token:{i}")
            if token_id in self.special_ids or special_mask or not attention:
                entry["kind"] = "pad" if token_id == self.pad_id else "special"
                if offset != [0, 0] or (not attention and token_id != self.pad_id):
                    errors.append(f"invalid_special_or_pad_offset:{i}")
                tokens.append(entry)
                continue
            start, end = offset
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(smiles)):
                errors.append(f"invalid_offset:{i}")
                entry["kind"] = "invalid"
                tokens.append(entry)
                continue
            positions = set(range(start, end))
            if start < previous_end:
                errors.append(f"nonmonotonic_offset:{i}")
            previous_end = end
            if positions & covered or tuple(offset) in seen:
                errors.append(f"duplicate_or_overlapping_offset:{i}")
            seen.add(tuple(offset))
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
            contexts = [a for a in atoms if _overlaps(offset, a["span"])]
            anchors = [a for a in atoms if _overlaps(offset, a["element_span"])]
            if len(contexts) > 1 or len(anchors) > 1:
                errors.append(f"token_crosses_multiple_atoms:{i}")
            entry["offset_source_text_valid"] = decoded == smiles[start:end] and self.tokenizer.id_to_token(token_id) == token
            if not entry["offset_source_text_valid"]:
                errors.append(f"token_source_text_mismatch:{i}")
                entry["kind"] = "invalid_source_offset"
                entry["raw_offset_atom_context_graph_atom_ids"] = [a["graph_atom_id"] for a in contexts]
                entry["raw_offset_anchor_graph_atom_ids"] = [a["graph_atom_id"] for a in anchors]
                entry["source_text_at_offset"] = smiles[start:end]
                entry["decoded_token_text"] = decoded
                tokens.append(entry)
                continue
            covered.update(positions)
            entry["source_atom_ids"] = [a["source_atom_id"] for a in contexts]
            entry["atom_context_graph_atom_ids"] = [a["graph_atom_id"] for a in contexts]
            if len(contexts) > 1 or len(anchors) > 1:
                entry["kind"] = "many_atoms"
            elif anchors:
                entry["kind"] = "element_anchor"
                entry["graph_atom_id"] = anchors[0]["graph_atom_id"]
                if not all(anchors[0]["span"][0] <= p < anchors[0]["span"][1] for p in positions):
                    errors.append(f"token_mixes_atom_and_external_syntax:{i}")
            else:
                entry["kind"] = "atom_attribute_context" if contexts else "syntax"
            tokens.append(entry)
        for atom in atoms:
            graph_id = atom["graph_atom_id"]
            atom["span_token_indices"] = [t["token_index"] for t in tokens if graph_id in t["atom_context_graph_atom_ids"]]
            atom["anchor_token_indices"] = [t["token_index"] for t in tokens if t["graph_atom_id"] == graph_id]
            if not atom["anchor_token_indices"]:
                errors.append(f"atom_without_element_anchor:{atom['source_atom_id']}")
            atom["uncovered_offsets"] = [i for i in range(*atom["span"]) if i not in covered]
            lost = []
            for part, attribute in (("element", "atomic_number"), ("isotope", "isotope"),
                                    ("charge", "formal_charge"), ("hydrogen", "explicit_hydrogens"),
                                    ("chiral", "stereochemistry")):
                if part in atom["parts"] and any(i not in covered for i in range(*atom["parts"][part])):
                    lost.append(attribute)
            if atom["bracketed"] and (atom["span"][0] not in covered or atom["span"][1] - 1 not in covered):
                lost.append("bracket_boundaries")
            atom["lost_attributes"] = lost
            atom["fidelity"] = "identity-lossy" if atom["uncovered_offsets"] else "source-text-preserved"
            atom["span_relation"] = ("uncovered" if not atom["span_token_indices"] else
                                     "one_to_one" if len(atom["span_token_indices"]) == 1 else "one_to_many")
        identity_lossy = any(a["uncovered_offsets"] for a in atoms)
        lost_syntax = [{**item, "uncovered_offsets": [p for p in range(*item["span"]) if p not in covered]}
                       for item in syntax if any(p not in covered for p in range(*item["span"]))]
        anchor_unique = all(len(a["anchor_token_indices"]) == 1 for a in atoms)
        span_unique = all(len(a["span_token_indices"]) == 1 for a in atoms)
        # A documented punctuation-offset defect does not invalidate separately
        # verified element anchors. It DOES invalidate the full offset contract.
        anchor_errors = [error for error in errors if not (
            error.startswith("token_source_text_mismatch:")
            and not tokens[int(error.rsplit(":", 1)[1])].get("raw_offset_anchor_graph_atom_ids")
            and not tokens[int(error.rsplit(":", 1)[1])]["token"].isalpha())]
        anchor_position_ok = not anchor_errors and anchor_unique
        result = {
            "schema": "semantic-atom-token-map/1",
            "tokenizer": self.identity,
            "exact_model_input": smiles,
            "input_is_map_free": True,
            "offset_unit": "zero-based half-open original ASCII character offsets; ASCII byte offsets are identical",
            "graph": graph, "atoms": atoms, "syntax": syntax, "tokens": tokens,
            "input_ids": raw["ids"], "attention_mask": raw["attention_mask"],
            "special_tokens_mask": raw["special_tokens_mask"],
            "uncovered_source_offsets": [i for i in range(len(smiles)) if i not in covered],
            "lost_syntax": lost_syntax,
            "input_fidelity": "lossy" if len(covered) < len(smiles) else "source-text-preserved",
            "errors": sorted(set(errors)),
            "anchor_errors": sorted(set(anchor_errors)),
            "position_status": "BLOCKED" if errors else "position-aligned",
            "element_anchor_position_status": "position-aligned" if anchor_position_ok else "BLOCKED",
            "identity_fidelity": "identity-lossy" if identity_lossy else "source-text-preserved",
            "status": "BLOCKED" if errors else "position-aligned identity-lossy" if identity_lossy else "position-aligned source-text-preserved",
            "candidate_contracts": {
                "unique_element_anchor": {"eligible": anchor_position_ok,
                    "meaning": "one positionally anchored element token per graph atom; attribute-only bracket tokens stay null, with atom context; identity loss remains a separate measured property"},
                "complete_atom_span_single_token": {"eligible": not errors and span_unique,
                    "meaning": "one token overlaps each complete source atom span, including bracket attributes; no aggregation"},
                "complete_atom_span_single_token_identity_preserving": {"eligible": not errors and span_unique and not identity_lossy,
                    "meaning": "the preceding optional strict contract plus complete atom source-text coverage"},
                "scientific_population_or_aggregation_selected": False,
            },
        }
        return result


def find_collisions(reports):
    """Record actual token-ID collisions among supplied SYNTHETIC examples only."""
    by_ids = {}
    for report in reports:
        key = tuple(t["id"] for t in report["tokens"] if t["attention_mask"])
        by_ids.setdefault(key, []).append(report["exact_model_input"])
    return [{"input_ids": list(ids), "distinct_exact_inputs": sorted(set(inputs))}
            for ids, inputs in by_ids.items() if len(set(inputs)) > 1]
