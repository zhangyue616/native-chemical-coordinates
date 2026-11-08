# Protein replay and molecular analyses

Internal names are defined in the [README Glossary](README.md#glossary).

This guide covers the protein extension, the ChemBERTa development/confirmation (D/T) analyses, the scaffold-unseen and standardized-rotation checks, the paired-dose analysis, and chemical activation replacement. The development-only sequence-rule analysis is recorded in `reproduction/protein/HISTORICAL_SEQUENCE_RULES.json`.

The protocols in `docs/` are the Chinese study records. Each has an English translation (`*.en.md`) prepared for the public release; the Chinese files are authoritative, and the translations are not hash-locked.

## Executable protein replay

From the repository root, with Python 3.11 and `requirements-protein.txt`:

```sh
python reproduction/protein/unpack_inputs.py --output-root replay-inputs
python reproduction/protein/replay_protein.py --project-root . --input-root replay-inputs --output-dir replay-output --chemistry-python /path/to/python
```

Use new, empty input and output directories. `--chemistry-python` only reads installed-package metadata from the given interpreter, so the current interpreter can be used. The replay needs no downloads and no ESM-2 checkpoint.

The archive holds all 116 inputs: the raw records needed to rebuild the exclusion graph, both 28-page UniProt snapshots, the 1,304 pairs (2,608 cases), saved confirmation-set (T) activations, the six kernel programs fitted on the development set (D), and the T predictions. The replay rebuilds the graph and case selection and reruns the kernel readout with the four original source modules, then writes `REPLAY_RECEIPT.json`. It does not run the model, refit, or recompute intervals or p values.

UniProt inputs derive from UniProt release 2026_03 (2 September 2026), attributed to the UniProt Consortium and provided under [CC BY 4.0](https://rest.uniprot.org/help/license). The repository includes selected UniProt records and data derived from them (sequence and annotation projections, component joins, case selections, and readouts); these are changes to the source material. The analysis uses existing PDB-covered secondary-structure annotations; no new structures or DSSP assignments were generated.

## Molecular code and saved results

| Analysis | Code | Saved results |
|---|---|---|
| D/T corpus and fixed-coordinate confirmation | `scripts/revision_20261017/chem_data.py`, `chem_run.py`, `chem_T_data.py`, `chem_T_run.py`, `chem_statistics.py` | `data/revision_20261017/chemistry/`, `results/revision_20261017/chemistry_D/`, `chemistry_T/` |
| Finite-input and full-clamp controls | `chem_input_reference.py`, `chem_D_secondary.py`, `chem_T_secondary.py`, `chem_T_intervention.py` | Corresponding `secondary/` result directories |
| Scaffold-unseen subset, lineage, and anchor mapping | `scripts/strengthening_20261025/a_lineage.py`, `a_h4_mapping.py`, `a_scaffold_subset.py` | `results/strengthening_20261025/lineage/`, `mapping/`, `scaffold_subset/` |
| Standardized rotation | `scripts/strengthening_20261025/b_standardized_rotation.py` | `results/strengthening_20261025/standardized_rotation/` |
| Paired dose | `scripts/strengthening_20261025/c_paired_dose.py`, `c_geometry_recovery.py` | `results/strengthening_20261025/paired_dose/` |
| Activation reconstruction and replacement | `scripts/chemical_replacement_20261027/data_fit.py`, `runner.py`, `summarize.py` | `results/chemical_replacement_20261027/fit/`, `summary/`, saved forward arrays |

`scripts/revision_20261017/chem_T_oracle_memory_resume.py` completed the descriptive oracle re-selection on the confirmation set after a memory-related interruption, and `c_geometry_recovery.py` recomputed the baseline and 18 dose conditions for one molecule using its original four-molecule batch; the reported results come from these completed runs.

To extract the stored chemical replacement output arrays into a new directory:

```sh
python reproduction/unpack_chemical_outputs.py --output-root replacement-output-copy
```

This only extracts the stored arrays: baseline outputs (10,000 × 199), condition outputs (2 × 5 × 10,000 × 199), realized doses, and permutation metadata.

## Inputs and acquisition

The protein replay uses only the included inputs. A molecular rerun needs the following.

1. `reproduction/OMITTED_INPUTS.json` lists the 129 files that are not included (about 3.60 GB of molecular activation caches, plus intermediate reference predictions, per-rotation arrays, and bootstrap arrays), with their paths, sizes, original hashes, timestamps, and the extraction stage that produced them. Included are the D/T atom labels, ordered SMILES, mapping anchors, concept dictionaries, fitted models, selections, result summaries, and the bootstrap replicates behind the reported intervals (`T_FIXED_REPLICATES.npz`, `scaffold_subset/SLICE_FIXED_REPLICATES.npz`, `secondary/input/PANEL_BRIER_BOOTSTRAP.npy`, `secondary/mechanism/`).
2. Download model weights with `python reproduction/fetch_models.py --model all`. The script fetches pinned revisions (ESM-2: `facebook/esm2_t6_8M_UR50D` at `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; ChemBERTa: `DeepChem/ChemBERTa-77M-MTR` at `66b895cab8adebea0cb59a8effa66b2020f204ca`), including the eight ChemBERTa model, configuration, and tokenizer files, and checks their sizes and hashes.
3. Rebuilding the corpus from PubChem requires the September 2026 `CID-SMILES.gz` snapshot (1,486,837,335 bytes; SHA-256 in `reproduction/OMITTED_INPUTS.json`). PubChem replaces this file over time, and `chem_source.py` rejects a file whose size, date, or hash differs; the ordered D/T inputs used in the study are included.
4. Hashes inside the protocol and lock records refer to the original study files. For the files edited for public release (machine paths and administrative notes removed, role names normalized, three corpus-source paths moved), `reproduction/SOURCE_DISTRIBUTION_MAP.json` gives both the original and the distributed SHA-256. A new molecular run creates its own workspace and lock records through each runner's prepare and ready steps.
5. `a_lineage.py` and `a_h4_mapping.py` read `NATIVE_UNIT_LEGACY_ROOT` (default: the repository root) for `data/derived/lineB/corpus/H4_used.tsv`, `scripts/lineB/models.py`, `scripts/lineB/check_chemberta_alignment.py`, `models/download_log.csv`, and the saved alignment check. The H4 mapping comparison loads no weights and expects tokenizers at `${NEURONAUDIT_MODEL_ROOT}/models/chemberta_77m_mlm/` and `${NEURONAUDIT_MODEL_ROOT}/models/chemberta_77m_mtr/` (default `external/models/...`). Fetch the six tokenizer files and `config.json` for each model listed in `acquisition/EXTERNAL_ASSETS.json`, for example `python scripts/lineB/companion.py fetch-asset --id models/chemberta_77m_mlm/tokenizer.json --asset-root external`. Three small corpus and concept source files are in `scripts/revision_20261017/frozen_source/`. Some molecular runners monitor resources through Windows APIs; full runs on Linux or macOS have not been tested.

The H4 tokenizer-alignment check is saved in `lineB/results_posthoc/alignment_check.json`, with the model download log in `models/download_log.csv`; the atom-mapping checks are in `results/revision_20261017/mapper/ENGINEERING_CHECKS.json`.

## Validation

`reproduction/protein/VALIDATION.json` records a complete replay of the protein analysis from the released files: all 45 checks passed, with a maximum score difference of 4.66 × 10⁻¹⁵, and no network access, model calls, refitting, or recomputed statistics.
