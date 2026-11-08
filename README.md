# Native Chemical Coordinates

Code and data accompanying **Native Chemical Coordinates in Pretrained Molecular Models: Transfer, Local-Graph Reconstruction, and Output Preservation**, by Yue Zhang, Dong-Jun Yu, and Qi Zhang.

![Overview of the study](figures/figure_1.png)

## Paper-to-repository map

| Paper section | What it contains | Repository paths |
| --- | --- | --- |
| Sections 2.2, 3.1; Table 2; Figure 2A–B; SI S-A–S-C, S-F | Cross-model comparison | `scripts/lineB/`, `data/derived/lineB/`, `frozen_results/lineB/`, `lineB/results_posthoc/`; [comparison guide](CHEMISTRY_AUDIT.md) |
| Sections 2.3, 3.2–3.3; Figures 2C–D, 3A; SI S-D | Fixed-coordinate transfer and fitted local references | `scripts/revision_20261017/chem_*.py`, `data/revision_20261017/chemistry/`, `results/revision_20261017/chemistry_D/`, `results/revision_20261017/chemistry_T/` |
| Sections 2.4, 3.4 (clamping); Figure 4D; SI S-E | Five-control clamping | `scripts/revision_20261017/chem_D_secondary.py`, `scripts/revision_20261017/chem_T_intervention.py`, `results/revision_20261017/chemistry_D/secondary/MECHANISM_D_PLAN.json`, `results/revision_20261017/chemistry_T/secondary/mechanism/` |
| Sections 2.5, 3.4; Figure 4A–C; SI S-J | Activation reconstruction and replacement | `scripts/chemical_replacement_20261027/`, `results/chemical_replacement_20261027/`, `reproduction/chemical_replacement_outputs.tar.gz` |
| Sections 2.6, 3.5; Figure 3B–D; SI S-G | Protein extension | `scripts/revision_20261017/protein_*.py`, `results/revision_20261017/protein_D/`, `results/revision_20261017/protein_T/`, `reproduction/protein/` |
| SI S-I | Post hoc sensitivity analyses: lineage and anchors, scaffold-unseen subset, standardized rotation, paired dose | `scripts/strengthening_20261025/`, `results/strengthening_20261025/` |
| All figures | Final artwork, source data, and rendering materials | `figures/`, `figure_source/`, `source_data/`, `figure_manifest.json`; [figure guide](FIGURES.md) |

## Quick checks

From the repository root, with Python 3.11 or later:

```sh
python scripts/lineB/companion.py smoke
```

The smoke check uses only the Python standard library and checks the saved results for completeness. For the protein replay, use Python 3.11 and `requirements-protein.txt`:

```sh
python reproduction/protein/unpack_inputs.py --output-root replay-inputs
python reproduction/protein/replay_protein.py --project-root . --input-root replay-inputs --output-dir replay-output --chemistry-python /path/to/python
```

Use new empty input and output directories. Set `/path/to/python` to an existing Python interpreter; this argument supplies installed-package metadata. See [WORKFLOWS.md](WORKFLOWS.md#executable-protein-replay) for details.

| Requirements file | Purpose |
| --- | --- |
| [requirements-protein.txt](requirements-protein.txt) | Saved protein-data and kernel-readout replay |
| [requirements-chemistry.txt](requirements-chemistry.txt) | Molecular development/confirmation analyses |
| [requirements-legacy.txt](requirements-legacy.txt) | Cross-model comparison environment recorded in `models/ENV_neuroaudit_freeze.txt` |
| [requirements-figures.txt](requirements-figures.txt) | Rendering figures from saved results |

## Data sources and redistribution

Molecular records come from PubChem; protein records come from UniProt. Selected records and derived results are included. Pretrained weights, the full PubChem snapshot, and large activation caches are obtained separately. For ChemBERTa tokenizer and configuration files, the repository provides source information. See [RIGHTS.md](RIGHTS.md), [the external-asset inventory](acquisition/EXTERNAL_ASSETS.json), and [the omitted-input inventory](reproduction/OMITTED_INPUTS.json).

## Provenance records

JSON records with LOCK, READY, HANDOFF, GO, RECEIPT, or AMENDMENT in their names are timestamped process records linked by SHA-256 hashes; they record when each protocol, dataset, and output was frozen, and their hashes refer to the original study files. For public release, 41 files, including some of these records and the Chinese protocols, were edited to remove machine paths and administrative notes and to normalize role names; `reproduction/SOURCE_DISTRIBUTION_MAP.json` gives the original and distributed SHA-256 of each file. Under `docs/`, `*.md` files are the Chinese protocol records and `*.en.md` files are their English translations.

## Glossary

| Internal name | Meaning in the paper |
| --- | --- |
| lineB | Cross-model comparison pipeline |
| H1 | Early concept-evaluation cohort from the original 50,000-molecule pool |
| H2 | Initial cross-model cohort (the first reserve subset, 10,000 molecules) |
| H3 | Subset-evaluation cohort from the same pool, evaluated against the cached H2 initialization references |
| H4 | Primary cohort (10,000 molecules), drawn after excluding the original pool |
| v5 | Eight-checkpoint primary-cohort comparison protocol |
| v6 | Separate MoLFormer evaluation protocol on the primary cohort, including selection across 12 layers |
| D / T | Development set / confirmation set |
| M1 | Pilot readout-location check in Table S-E3 |
| M4 | Tensor-shuffling comparison in Table S-F1 |
| J1 / J2 | Coordinate readability / basis preference |
| twin | Matched randomly initialized network |
| v3 | Initial-cohort (H2) comparison protocol |
| gate | Prespecified decision rule |
| audit | Earlier internal name for the cross-model comparison |
| root / supervisor / research operator | Supervising AI sessions described in the Methods section of the paper |
| revision_20261017 | Fixed-coordinate molecular analyses and protein extension |
| strengthening_20261025 | Lineage/anchor, scaffold-unseen, standardized-rotation, and paired-dose analyses |
| chemical_replacement_20261027 | Chemical activation reconstruction and frozen-network replacement analyses |

The v5/v6 definitions follow [the comparison guide](CHEMISTRY_AUDIT.md) and [the design record](protocols/lineB_design.json).

## Reproducibility scope

Verified on the released files: the offline smoke check, the figure-input check, and the saved-data protein replay, which reproduces the saved protein scores to within 4.66 × 10⁻¹⁵. A full molecular rerun from a clean installation has not been tested.

## AI use

Analysis code was written and run with AI coding agents under the authors' direction; see the Methods section of the paper.

## License

First-party code is licensed under [MIT](LICENSE). First-party data, figures, and documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party attribution and terms are listed in [RIGHTS.md](RIGHTS.md).

## How to cite

Zhang, Y.; Yu, D.-J.; Zhang, Q. *Native Chemical Coordinates in Pretrained Molecular Models: Transfer, Local-Graph Reconstruction, and Output Preservation*. Manuscript.

Zhang, Y.; Yu, D.-J.; Zhang, Q. *Native Chemical Coordinates: Research Companion*, version 1.0.0. [GitHub repository](https://github.com/zhangyue616/native-chemical-coordinates), [v1.0.0](https://github.com/zhangyue616/native-chemical-coordinates/releases/tag/v1.0.0).
