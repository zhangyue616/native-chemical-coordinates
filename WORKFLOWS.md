# Frozen protein replay and later molecular validation

This directory contains the scientific code and frozen results for the protein extension, the ChemBERTa D/T confirmation, scaffold and standardized rotation checks, pairwise dose matching, and two-coordinate chemical replacement. It excludes manuscript drafts and review correspondence. The development-only sequence-rule exercise is recorded in `reproduction/protein/HISTORICAL_SEQUENCE_RULES.json`.

Protocols under `docs/` are the Chinese study records, with administrative coordination wording removed or reworded and machine paths normalized; `reproduction/SOURCE_DISTRIBUTION_MAP.json` records the source and distribution hash of each file. Each has an English reader translation alongside it (`*.en.md`), added for this release; the translations carry no lock or distribution hash, and the Chinese files govern.

## Executable protein replay

From the repository root, with Python 3.11 and `requirements-protein.txt`:

```sh
python reproduction/protein/unpack_inputs.py --output-root replay-inputs
python reproduction/protein/replay_protein.py --project-root . --input-root replay-inputs --output-dir replay-output --chemistry-python /path/to/python
```

Use new empty input and output directories. The `--chemistry-python` flag only reads installed-distribution metadata from the specified interpreter without installing packages or accessing molecular scientific data, so a new user can point it to the same interpreter. Running the replay requires neither upstream source acquisition nor an ESM-2 checkpoint.

The archive supplies all 116 inputs: historical raw records needed to rebuild the exclusion graph, both 28-page UniProt snapshots, the frozen 1,304 pairs/2,608 cases, saved T activations, six fitted D kernel programs, and T predictions. The four source modules used by the graph/selector/KRR replay retain their original bytes, with sanitization limited to runtime/model-location metadata in one ready receipt. The replay writes `REPLAY_RECEIPT.json` without running model inference, fitting, bootstrap, or statistical interval/p-value calculations.

UniProt inputs derive from UniProt release 2026_03 (2 September 2026), attributed to the UniProt Consortium and provided under [CC BY 4.0](https://rest.uniprot.org/help/license). This package redistributes selected source records, sequence/annotation projections, component joins, case selections, and derived readout data; these processing steps are changes to the source material. The analysis uses existing PDB-covered secondary-structure annotations without generating new experimental structures or DSSP assignments.

## Molecular source and frozen results

| Question | Scientific implementation | Saved evidence |
|---|---|---|
| D/T corpus and fixed-coordinate confirmation | `scripts/revision_20261017/chem_data.py`, `chem_run.py`, `chem_T_data.py`, `chem_T_run.py`, `chem_statistics.py` | `data/revision_20261017/chemistry/`, `results/revision_20261017/chemistry_D/`, `chemistry_T/` |
| Finite-input and historical full-clamp controls | `chem_input_reference.py`, `chem_D_secondary.py`, `chem_T_secondary.py`, `chem_T_intervention.py` | Corresponding `secondary/` result directories |
| Scaffold slice and mapping lineage | `scripts/strengthening_20261025/a_lineage.py`, `a_h4_mapping.py`, `a_scaffold_subset.py` | `results/strengthening_20261025/lineage/`, `mapping/`, `scaffold_subset/` |
| Fit-set standardized rotations | `scripts/strengthening_20261025/b_standardized_rotation.py` | `results/strengthening_20261025/standardized_rotation/` |
| Pairwise dose matching | `scripts/strengthening_20261025/c_paired_dose.py`, `c_geometry_recovery.py` | `results/strengthening_20261025/paired_dose/` |
| Continuous activation prediction and replacement | `scripts/chemical_replacement_20261027/data_fit.py`, `runner.py`, `summarize.py` | `results/chemical_replacement_20261027/fit/`, `summary/`, saved forward arrays |

The repository retains the oracle memory-resume source (implementing the adopted bounded allocation repair) and the dose geometry-recovery source (implementing the adopted four-molecule numerical restoration) because these repaired executions, rather than their preceding engineering stops, generated the reported results.

`tests/fixtures/runner_engineering/` holds synthetic files from engineering checks of the runner lock guards; they are not study results.

To extract original chemical replacement output arrays into a new directory:

```sh
python reproduction/unpack_chemical_outputs.py --output-root replacement-output-copy
```

This command performs extraction only, recovering the stored 10,000-by-199 fresh baselines, 2-by-5-by-10,000-by-199 condition outputs, realized doses, and permutation metadata without model inference or recalculation of estimates/intervals.

## Acquisition and end-to-end limits

While the protein replay above provides the runnable, bounded reproduction entry, the historical molecular runners are supplied for method inspection and future reconstruction, **not as a validated one-command molecular rerun**.

1. `reproduction/OMITTED_INPUTS.json` enumerates 129 omitted source, cache, result, audit and scratch files by exact relative path, recording the omitted D/T activation caches, sizes, original receipt hashes, original timestamps, and producing extraction stage (leaving any missing receipt hash explicitly null rather than using a newly invented or recomputed identity). Full molecular activation caches total about 3.60 GB and are omitted from this Git distribution alongside other large intermediate reference predictions, per-rotation arrays, and bootstrap arrays that require reconstruction. By contrast, frozen compact D/T atom labels, ordered SMILES, mapping anchors, concept dictionaries, fits, selections, and result summaries are embedded, together with the bootstrap replicates behind the reported D/T transfer, scaffold-slice, lookup and mechanism intervals (`T_FIXED_REPLICATES.npz`, `scaffold_subset/SLICE_FIXED_REPLICATES.npz`, `secondary/input/PANEL_BRIER_BOOTSTRAP.npy`, `secondary/mechanism/`). The validation receipt's count of 148 predates these inventory changes.
2. Pin model sources with `python reproduction/fetch_models.py --model all`. This explicit opt-in script downloads weights from official pinned upstream URLs and verifies recorded sizes/hashes without performing inference (ESM-2: `facebook/esm2_t6_8M_UR50D` at `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; ChemBERTa: `DeepChem/ChemBERTa-77M-MTR` at `66b895cab8adebea0cb59a8effa66b2020f204ca`). Model weights are not redistributed here; this command also obtains the eight ChemBERTa model/configuration and tokenizer files rather than redistributing them, because their redistribution license was not verified and source metadata does not grant a new license. The ChemBERTa first-party wrapper does not require downloading the linked third-party `roberta_regression_official.py`.
3. Corpus reconstruction from the entire PubChem feed requires the September 2026 `CID-SMILES.gz` snapshot (1,486,837,335 bytes; SHA256 in the omitted-input manifest). Because its original FTP URL is mutable and `chem_source.py` rejects a changed size/date/hash, this package does not promise upstream retention of those historical bytes; however, selected ordered D/T inputs are already included.
4. Historical protocol/lock hashes are retained as original study evidence. Because sanitized metadata and the three corpus-source path relocations can change distribution hashes, `reproduction/SOURCE_DISTRIBUTION_MAP.json` distinguishes both identities, and **original locks are not rebound to distribution files.** Fetching weights/caches alone is insufficient because historical guarded commands can reject cleaned source or receipt identities. A fresh independent molecular reconstruction must initialize its own workspace, engineering receipts, source bindings, resource limits, and locally approved locks through the documented prepare/ready stages. That reconstruction is neither performed nor validated in this release, and original locks must not be silently bypassed or described as passing against changed copies.
5. `a_lineage.py` and `a_h4_mapping.py` read `NATIVE_UNIT_LEGACY_ROOT` (default repository root) for the included `data/derived/lineB/corpus/H4_used.tsv`, `scripts/lineB/models.py`, `scripts/lineB/check_chemberta_alignment.py`, `models/download_log.csv`, and saved alignment receipt. The H4 mapping audit uses no weight arrays and expects tokenizers at `${NEURONAUDIT_MODEL_ROOT}/models/chemberta_77m_mlm/` and `${NEURONAUDIT_MODEL_ROOT}/models/chemberta_77m_mtr/` (default `external/models/...`), where the legacy acquisition entry obtains each of the six tokenizer files and `config.json` from `acquisition/EXTERNAL_ASSETS.json` (e.g. `python scripts/lineB/companion.py fetch-asset --id models/chemberta_77m_mlm/tokenizer.json --asset-root external`). Three small original corpus/concept source files are retained under `scripts/revision_20261017/frozen_source/` independently of manuscript storage. Several molecular resource monitors use Windows APIs, and Linux/macOS full runs have not been validated.

The published H4 alignment receipt is stored at `lineB/results_posthoc/alignment_check.json` with its required five-column model download ledger at `models/download_log.csv`, and the strict mapper engineering receipt is at `results/revision_20261017/mapper/ENGINEERING_CHECKS.json`.

## Actual release validation

`reproduction/protein/VALIDATION.json` records one complete same-machine replay of the cleaned distribution (a clean external installation was not tested): 45/45 checks passed in 20.71875 cumulative process CPU seconds, with a maximum saved-score discrepancy of 4.6629367034256575e-15 (tolerance 1e-10) and zero network attempts, model loads/forwards, fits, bootstraps, or p/CI calls. The 45 Python source files covered by that validation check compiled. Archive contents and publication candidates were checked for personal machine paths, every actual publication file is below 100 MiB, and local validation inputs, bytecode, and full machine-path receipts in `.validation/` are excluded.

Replaying these already exposed data does not renew historical result-blindness: the scaffold slice is nested, the standardized rotation check is conditional, and replacement analysis is post hoc. Accordingly, no public preregistration, external clean installation, cross-machine replication, new scientific result, or unconditional end-to-end molecular reproducibility is claimed.
