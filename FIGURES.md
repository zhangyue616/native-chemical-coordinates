# Figures and source data

Internal names are defined in the [README Glossary](README.md#glossary).

`figures/` contains the 15 PNG figures exactly as they appear in the manuscript and Supporting Information: Figures 1–4, S-B1, S-C1–S-C6, S-E1, S-I1, S-J1, and the TOC graphic. `figure_manifest.json` lists the image, source data, and renderer for each figure; the captions are in the manuscript and Supporting Information.

| Figure | Source data | Renderer and inputs |
|---|---|---|
| 1 | `source_data/figure_1_plot_data.json` | `figure_source/main/plot_figure_1.py`; saved molecular geometry and token evidence |
| 2 | `source_data/figure_2_plot_data.json` | `figure_source/main/plot_figure_2.py`; comparison and transfer inputs and three chemical geometries |
| 3 | `source_data/figure_3_plot_data.json` | `figure_source/main/plot_figure_3.py`; 72 display records, molecular lookup values, protein endpoints, and saved geometry |
| 4 | `source_data/figure_4_plot_data.json` | `figure_source/main/plot_figure_4.py`; activation, output, and CI values, two molecular geometries, and atom activation arrays |
| S-B1 | `source_data/figure_S_B1_plot_data.json` | Not included |
| S-C1 | `source_data/figure_S_C1_plot_data.json`, `source_data/figure_S_C1_tokens.csv` | Not included; the source data contain the tokenizer examples and all 593 vocabulary-row norms |
| S-C2 | `source_data/figure_S_C2_plot_data.json` | Six ChemBERTa-77M-MLM image tiles in `figure_source/assets/historical_examples/` |
| S-C3 | `source_data/figure_S_C3_plot_data.json` | Six ChemBERTa-77M-MTR image tiles in the same directory |
| S-C4 | `source_data/figure_S_C4_plot_data.json` | Six CheMeleon image tiles in the same directory |
| S-C5 | `source_data/figure_S_C5_plot_data.json`, `source_data/figure_S_C5_gates.json`, `source_data/figure_S_C5_matrices/*.npz` | `figure_source/historical/plot_S_C2_to_C5.py`; five saved best-lift matrices |
| S-C6 | `source_data/figure_S_C6_plot_data.json` | Not included |
| S-E1 | `source_data/figure_S_E1_plot_data.json` | Not included |
| S-I1 | `source_data/figure_S_I1_plot_data.json` | Not included |
| S-J1 | `source_data/figure_S_J1_plot_data.json` | Not included |
| TOC | `source_data/toc_graphic_plot_data.json` | `figure_source/main/plot_toc_graphic.py`; query geometry in `figure_source/assets/toc/` |

`source_data/data_S1_best_units.csv` gives the unit-level table for cohorts H2 and H3. Some source-data files keep figure numbers from earlier drafts; the table above gives the current labels. Machine-specific paths and workflow metadata were removed from the source-data files; their numerical values are unchanged.

## Check the inputs

```bash
python -B figure_source/main/check_frozen_inputs.py
```

The check imports the four main-figure renderers and loads all of their inputs without drawing. It passes with Python 3.7.0 and Matplotlib 2.2.3.

## Render from saved results

```bash
python -B figure_source/main/plot_figure_1.py --version local
python -B figure_source/main/plot_figure_2.py --version local
python -B figure_source/main/plot_figure_3.py --version local
python -B figure_source/main/plot_figure_4.py --version local
python -B figure_source/main/plot_toc_graphic.py --version local
```

Rendering needs only Matplotlib; no model weights, fitting, or inference are involved. Output goes to `rendered/figure_<n>/local/` or `rendered/toc_graphic/local/`; use a new version name for each run. The renderers plot the saved values in `source_data/` but use labels and a layout from an earlier draft (for example, "held-out" for the confirmation set), so their output looks different from the published artwork. Without the Arial font, text placement can shift.

`figure_source/historical/plot_S_C2_to_C5.py` (also needs NumPy ≥ 1.17 and Pillow) assembles the S-C2–S-C4 image tiles and draws S-C5 from the saved matrices in an earlier style. Its S-C5 branch reproduces the 16 saved effect estimates to within 1.1 × 10⁻¹⁶.
