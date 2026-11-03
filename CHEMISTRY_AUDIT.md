# Completed cross-model chemical audit: protocols v5 and v6

This package provides the implementations and frozen numerical inputs behind the eight-model H4 audit and the twelve-layer MoLFormer extension, offering retrospective review and reproduction access rather than a new result-blind confirmation. Included materials cover the original audit/lift/test formulas, portable model-location adapters, a 200-entry atom-level concept dictionary (173 readable concepts), H4 CID/standardized-SMILES rows and labels, the old 50,000 CID/SMILES exclusion set, complete compact best-lift vectors, and frozen published gate summaries.

Run from the repository root with Python 3.11 or later:

```text
python scripts/lineB/companion.py --help
python scripts/lineB/companion.py smoke
```

Using only the standard library, the smoke check verifies scientific object identities, 8-model/173-concept sizes, the saved primary result fields, H4's 10,000 unique CIDs, complete 199-twin/9-rotation coverage, all 12 MoLFormer layers, its 39-HF-initialization/9-feature-redraw reference sets, and a small fixed sample of saved numeric vectors, without calling a model or recomputing a p value, bootstrap interval, AP, or statistical decision.

`frozen_results/lineB/{v5,v6}/S_vectors.zip` stores exact NPY bytes for the S vectors and readable-concept IDs only, excluding activation matrices and full unit-by-pole lift matrices, while `VECTOR_INDEX.json` records the complete member set. To inspect or replay from these outputs, unpack them to a new directory:

```text
python scripts/lineB/companion.py unpack-results v5 --output work/lineb_v5
python scripts/lineB/companion.py unpack-results v6 --output work/lineb_v6
```

Separate from the offline smoke check, `scripts/lineB/gates_v5.py` and `gates_v6.py` serve as the scientific recomputation entry points for the numerical gates. While the v5 runner accepts explicit output paths, the v6 original confirmation guards require the original pre-exposure freeze and exact original source bytes, and this portable copy does not invent a new historical freeze. For a retrospective gate replay, `unpack-results v6` stages the saved `molformer/run_record.json` and `fingerprints.json` so the runner can be invoked in the original `--dry` mode with an explicit `--token_visible` argument, retaining the mathematical procedure while labeling the execution as a dry replay rather than a new confirmation. Both gate scripts include bootstrap routines, and neither gate was run during package assembly.

```text
python scripts/lineB/gates_v5.py work/lineb_v5 work/v5_replay.json
python scripts/lineB/gates_v6.py work/lineb_v6 data/derived/lineB/concepts work/v6_replay.json --dry --token_visible frozen_results/lineB/v6/token_visible_concepts.json
```

Because the saved-vector gate replay commands import the legacy audit and model modules, they require the numerical and model dependency stack below even when no checkpoint is loaded and no model forward is performed; only the `companion.py smoke` command is standard-library-only.

Full activation extraction additionally needs NumPy, SciPy, PyTorch/CUDA, RDKit, Transformers, PyTorch Geometric, and Chemprop, with an original observed environment of Python 3.11.17, torch 2.11.0+cu128, torch_geometric 2.8.1, transformers 5.20.0, chemprop 2.3.1, and RDKit 2026.03.6. `requirements-legacy.txt` pins the complete frozen list in `models/ENV_neuroaudit_freeze.txt`. It records an execution snapshot rather than a verified clean-machine lockfile, and no package installation is performed here. The scripts remain GPU extraction methods, as adding a CPU substitute would change this scope.

`models.py` reads an explicit `NEURONAUDIT_MODEL_ROOT` (a directory containing `models/`) or defaults to the repository root. The Hu/MolCLR GIN and feature ports are included with both original MIT licenses in `third_party_licenses/`, incorporating the documented PyG 2.8 port, the activation-hook wrapper, and the R13 portable filesystem path updates; those MIT grants apply to the derived third-party portions; first-party terms are in `RIGHTS.md`, and weights are not covered.

Pretrained weights and the external MoLFormer implementation are not redistributed; instead, `acquisition/EXTERNAL_ASSETS.json` records the actual original publisher URLs, byte counts, source snapshot dates, and exact SHA256 values. To fetch one asset explicitly after considering its publisher terms:

```text
python scripts/lineB/companion.py fetch-asset --id models/molformer_xl_both_10pct/config.json --asset-root external
```

Set `NEURONAUDIT_MODEL_ROOT` to that `external` directory before running scientific extraction (no weights are needed for smoke or frozen-vector gate replay). Downloads are never automatic, changed bytes fail closed, and existing files are not overwritten. In the preserved local model snapshot, ChemBERTa's checkpoint license is unconfirmed—a training-code MIT license does not settle checkpoint rights—while MoLFormer's locally preserved model card states Apache-2.0 and its third-party terms must be obtained with its implementation.

PubChem acquisition uses the official rolling CID-SMILES endpoint recorded in the acquisition JSON, where the 2026-09-27 original file had 124,717,116 rows and SHA256 `36b683a59272b196ed7ece140903a249f7df475a06c312c963a5c761f1037f1a`. Because the rolling URL may no longer serve those exact bytes, the necessary H4 and old-exclusion CID/SMILES records are included here without redistributing the full database. `build_corpus.py`, `build_corpus_v5.py`, and `build_concepts.py` expose the original sampling/label definitions, and PubChem third-party content must not be treated as blanket CC0.

The post hoc output screen on the pilot cohort (Table S-E3) is supplied as `scripts/lineB/posthoc_m1_mtr_ablation.py` with saved results in `lineB/results_posthoc/m1_ablation.json`, `m1_ablation.csv` and `m1_mapping.json`; its pilot inputs are in `data/derived/lineB/`, it needs the fetched checkpoints, and it was not rerun for this release.

The post hoc tensor-shuffling comparison on cohort H2 (Table S-F1) is supplied as `scripts/lineB/posthoc_m4_shuffled.py` with its helper `scripts/lineB/figures/common.py`. `lineB/results_posthoc/m4_shuffled.json` is the saved summary, `lineB/results_posthoc/m4_S_vectors.zip` holds the best-lift vectors of the 39 shuffled networks per checkpoint, and `lineB/figures/cache/v3_*.npz` holds the matching H2 pretrained and initialization vectors, so the folds and p values in the table can be recomputed without model inference using `exact_test` in `gates_v2.py` and `effect` in `common.py`. The H2 corpus, labels and model weights are not included, and the runner was not rerun for this release.

`SOURCE_PROVENANCE.json` identifies copied source layouts, hashes, and adaptations, which are path updates, a dependency-free CLI help guard and JSON whitespace normalization. As already recorded in the historical v3 source receipt, exact v3 `models.py`/`lift.py` bytes were not recovered, so this package does not claim that those older activations can be reproduced exactly. Likewise, cross-machine clean installation, new activation extraction, and full statistical recomputation have not been validated by this package's offline inspection.
