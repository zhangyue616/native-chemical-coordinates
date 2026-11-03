# A1: Post hoc protocol for H4/D/T lineage and the fixed D-unseen scaffold slice

> English reader translation of [`A_LINEAGE_PROTOCOL.md`](A_LINEAGE_PROTOCOL.md), added for this public release. It is not a study record: the Chinese file is the record of this protocol and governs if the two differ.

Record date: 2026-10-25. Status: **POST_HOC_PROTOCOL_RECORDED_BEFORE_NEW_OUTPUTS**. This is independent post hoc supplementary evidence; it is not retroactively treated as preregistration and does not overwrite the old protocols, data, results or ledgers. The execution entry point is `scripts/strengthening_20261025/a_lineage.py`, and all outputs are saved separately in `results/strengthening_20261025/lineage/`.

## Fixed objects and read-only inputs

| Set | Exact path | Expected row count | Original row identifier |
|---|---|---:|---|
| H4 | `<legacy-source-root>/data/derived/lineB/corpus/H4_used.tsv` | 10000 | Original rank; a zero-based file row index is also retained |
| D | `./data/revision_20261017/chemistry/development/D.tsv` | 5000 | Same as above |
| T | `./data/revision_20261017/chemistry/sealed/T.tsv` | 10000 | Original rank is 5000–14999; within-T index is 0–9999 |

Only the molecule identity/provenance fields of the TSVs above, this protocol and the implementation may be read; no concept labels, scores, AP, model activations or model outputs may be read. The SHA256 of the actual input bytes is recorded with this read. The adjacent `8th_paper` root is read-only throughout. The large PubChem files must not be read, and there must be no network access, no installation, no training and no Git operations.

The existing standardization sources are checked read-only: `scripts/lineB/build_corpus.py:standardize` and `build_corpus_v5.py` in the adjacent root, and `scripts/revision_20261017/chem_data.py:standardize` and `docs/revision_20261017/CHEMISTRY_DATA_PROTOCOL.md` in this root. H4 uses the original v1 standardization, and the D/T saving rules also explicitly reuse that v1 algorithm: the original build stages all apply default parse/sanitize, selection of the largest fragment by heavy-atom count (ties keep the first), RemoveHs, the original element/size/radical/isotope/chirality/bond/biological-structure filters, and read-back of the canonical stereo SMILES. H4 and D/T differ in sampling salt, ordering, exclusion of old objects and mapper gates. This audit does **not redo** these build steps, and it does not interpret their being the same as overall independence.

## Common identity definitions for this run

RDKit **2026.03.6**, as available on site, is locked. Each saved SMILES undergoes only `Chem.MolFromSmiles(saved_smiles)` (default standard sanitize), then `Chem.RemoveHs(mol)`. For this same molecular graph:

1. CID identity: the integer canonical text of the input decimal CID; structures must not be looked up or repaired from the CID.
2. Canonical stereo identity: `Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)`.
3. Destereoized identity: copy the above mol, `Chem.RemoveStereochemistry(copy)`, then `MolToSmiles(canonical=True, isomericSmiles=False)`.
4. Non-empty Bemis–Murcko scaffold identity: call `MurckoScaffold.GetScaffoldForMol` on the same mol, remove stereochemistry, then output canonically. The element/bond and ring-connection definitions of RDKit Murcko are retained; the scaffold must not be converted to a generic scaffold.

This run must not strip salts, neutralize, perform tautomer standardization, reselect the largest fragment or apply new molecular filters. Input structures are retained both before and after common-identity canonicalization. Parsing, RemoveHs, identity or scaffold errors each keep their row and record the stage/exception reason; failed rows must not be silently deleted. Empty scaffolds with no rings are listed separately; **an empty string must never be treated as a shared scaffold**. Missing/failed identities do not enter the corresponding identity set.

For each of the four identity levels, report separately: per set, the valid row count, unique count, excess duplicate row count, the numbers of duplicate groups and of rows participating in duplicates, and identity failure count; and the unique identity counts of H4–D, H4–T, D–T and the triple intersection. The row-level table retains each identity and whether that identity is present in H4/D/T, whether it crosses sets and whether it is in the triple intersection. The intersection table retains the actual shared identities and the row count per set, to avoid treating unique identity counts as numbers of molecule pairs.

## Purely structural frozen slice

Sole rule: take **all rows in T whose scaffold succeeded and is non-empty, and whose scaffold is not in D's set of non-empty successful scaffolds**. The original zero-based within-T index, original rank, CID and originally saved SMILES are retained. Any H4 identity/scaffold overlap is only reported; these rows are **not additionally removed**.

Save the Boolean `T_D_UNSEEN_SCAFFOLD_MASK.npy` of length exactly 10000 in the original T row order, the corresponding mask TSV of all T rows, and the TSV of selected rows. `FREEZE.json` fixes the above rule, the exact input paths/byte SHA256, the mask SHA256, the total T row count/selected count/empty scaffold count/failure count, UTC and a declaration that scores/labels were not read. This slice is a post hoc structural slice of the existing T, **not a new independent scaffold split confirmation**. Subsequent statistics must not reselect samples, remove H4 overlaps or change the handling of empty scaffolds on the basis of results.

## Budget, failures and engineering checks for this stage

Scientific calls: **0 model forward passes, 0 fits, 0 statistical resamplings**. Single process, with the numerical thread cap set to 1 (not exceeding the approved 4). An independent CPU ledger accumulates by `time.process_time()`, including import, input, computation and output; the hard cap is 900 CPU seconds, with a stop triggered at 895 seconds and 5 seconds reserved for writing partial output/the ledger. The budget is checked row by row; progress is saved every 500 rows. A failure/budget stop retains the completed rows and the reason; the rules must not be changed and there must be no automatic retry that overwrites.

The single appropriate engineering check is merged into the end of this run: verify the expected input row counts/fixed rank ranges, and that all output rows are consistent with the original input CID/SMILES/rank order; that the valid/failed/empty-scaffold breakdown is complete; that unique counts and intersection set bounds are reasonable; that mask type/length and every bit are consistent with the fixed structural rule; and that the NPY read back from disk is consistent with the in-memory mask. The input SHA and mask SHA are computed only once; there must be no full repeat hashing, no re-parsing of all molecules and no rescanning of PubChem. Success of the engineering check proves only the integrity of this construction and does not constitute author adoption/submission clearance.

The actual run command, Python/RDKit/NumPy versions, thread settings, start/end UTC, CPU/wall clock, status and output manifest are written to `CPU_LEDGER.json`; complete counts are written to `SUMMARY.json`.
