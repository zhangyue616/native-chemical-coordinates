# Cross-model comparison (eight checkpoints and MoLFormer)

Internal names are defined in the [README Glossary](README.md#glossary).

This guide covers the eight-checkpoint comparison on the primary cohort (H4) and the separate 12-layer MoLFormer evaluation. The repository includes the comparison, lift, and test code; model loaders; the 200-entry atom-level concept dictionary (173 readable concepts); H4 CIDs, standardized SMILES, and labels; the 50,000-molecule exclusion set; the best-lift (S) vectors; and the saved decision-rule summaries.

From the repository root, with Python 3.11 or later:

```text
python scripts/lineB/companion.py --help
python scripts/lineB/companion.py smoke
```

The smoke check uses only the standard library. It confirms that the saved results cover 8 checkpoints, 173 readable concepts, 10,000 unique H4 CIDs, 199 matched randomly initialized networks and 9 rotations, all 12 MoLFormer layers, and the MoLFormer reference sets (39 initializations and 9 feature redraws), and it checks a fixed sample of saved vectors. It does not run models or recompute statistics.

`frozen_results/lineB/{v5,v6}/S_vectors.zip` holds the S vectors and readable-concept IDs (activation matrices and full lift matrices are not included); `VECTOR_INDEX.json` lists the contents. Unpack them to a new directory:

```text
python scripts/lineB/companion.py unpack-results v5 --output work/lineb_v5
python scripts/lineB/companion.py unpack-results v6 --output work/lineb_v6
```

`scripts/lineB/gates_v5.py` and `gates_v6.py` recompute the decision rules, including their bootstrap intervals, from these vectors. In its default mode, `gates_v6.py` requires the original locked inputs, so a replay runs it with `--dry`, the saved `molformer/run_record.json` and `fingerprints.json` that `unpack-results v6` stages, and an explicit `--token_visible` list:

```text
python scripts/lineB/gates_v5.py work/lineb_v5 work/v5_replay.json
python scripts/lineB/gates_v6.py work/lineb_v6 data/derived/lineB/concepts work/v6_replay.json --dry --token_visible frozen_results/lineB/v6/token_visible_concepts.json
```

These replays import the model modules, so they need the dependencies below, although no checkpoint is loaded.

Activation extraction runs on GPUs and needs NumPy, SciPy, PyTorch with CUDA, RDKit, Transformers, PyTorch Geometric, and Chemprop. The study used Python 3.11.17, torch 2.11.0+cu128, torch_geometric 2.8.1, transformers 5.20.0, chemprop 2.3.1, and RDKit 2026.03.6; `requirements-legacy.txt` pins the full environment recorded in `models/ENV_neuroaudit_freeze.txt`.

`models.py` loads checkpoints from `NEURONAUDIT_MODEL_ROOT` (a directory containing `models/`; default: the repository root). The Hu et al. and MolCLR GIN models and their feature code are adapted from the original MIT-licensed releases (licenses in `third_party_licenses/`), with a port to PyTorch Geometric 2.8 and an activation hook.

Pretrained weights and the MoLFormer implementation are not included; `acquisition/EXTERNAL_ASSETS.json` lists each file's publisher URL, size, snapshot date, and SHA-256. To fetch one file:

```text
python scripts/lineB/companion.py fetch-asset --id models/molformer_xl_both_10pct/config.json --asset-root external
```

Then set `NEURONAUDIT_MODEL_ROOT` to `external` before running extraction; the smoke check and the decision-rule replays need no weights. The script downloads only the requested file, rejects a file whose SHA-256 differs from the record, and does not overwrite existing files. The MoLFormer model card states Apache-2.0; no license is confirmed for the ChemBERTa checkpoints (see [RIGHTS.md](RIGHTS.md)).

The molecules come from PubChem's CID-SMILES file (URL in `acquisition/EXTERNAL_ASSETS.json`); the 2026-09-27 file had 124,717,116 rows and SHA-256 `36b683a59272b196ed7ece140903a249f7df475a06c312c963a5c761f1037f1a`. Because PubChem updates this file, the H4 and exclusion-set records are included. `build_corpus.py`, `build_corpus_v5.py`, and `build_concepts.py` contain the sampling and labeling code.

The post hoc output screen on the pilot cohort (M1; Table S-E3) is `scripts/lineB/posthoc_m1_mtr_ablation.py`. Its saved results are `lineB/results_posthoc/m1_ablation.json`, `m1_ablation.csv`, and `m1_mapping.json`, its inputs are in `data/derived/lineB/`, and it needs the fetched checkpoints.

The post hoc tensor-shuffling comparison on cohort H2 (M4; Table S-F1) is `scripts/lineB/posthoc_m4_shuffled.py`, with the helper `scripts/lineB/figures/common.py`. `lineB/results_posthoc/m4_shuffled.json` is the saved summary, `lineB/results_posthoc/m4_S_vectors.zip` holds the best-lift vectors of the 39 shuffled networks per checkpoint, and `lineB/figures/cache/v3_*.npz` holds the matching H2 pretrained and initialization vectors. The folds and p values in Table S-F1 can therefore be recomputed without model inference, using `exact_test` in `gates_v2.py` and `effect` in `common.py`. The H2 corpus, labels, and model weights are not included.

`SOURCE_PROVENANCE.json` lists the hash of each copied source file and the changes made for distribution: path updates, a `--help` option that runs without dependencies, and JSON whitespace. The exact `models.py` and `lift.py` used for the H2 analysis (v3) were not recovered.
