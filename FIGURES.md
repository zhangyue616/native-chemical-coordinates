# Figures and frozen source data

This package contains the 15 PNG figures used in the manuscript and Supporting Information—Figures 1–4, S-B1, S-C1–S-C6, S-E1, S-I1, S-J1, and the TOC graphic—each supplied as an unchanged copy of its selected source image. In particular, `figures/figure_1.png` is the original selected PNG for direct embedding in the repository README rather than a redrawn copy.

`figure_manifest.json` maps each figure to its image, source-data file, selected source version, and available renderer, while the manuscript and Supporting Information contain the authoritative captions without revisions to their scientific wording here.

| Figure | Selected artwork | Frozen source data | Rendering material |
|---|---|---|---|
| 1 | R9 v01 | `source_data/figure_1_plot_data.json` and geometry | `figure_source/main/plot_figure_1.py`; saved molecular geometry and offline token evidence |
| 2 | R9 v01 | `source_data/figure_2_plot_data.json` and geometry | `figure_source/main/plot_figure_2.py`; frozen audit/transfer inputs and three chemical geometries |
| 3 | R8 frozen v01 | `source_data/figure_3_plot_data.json` and geometry | `figure_source/main/plot_figure_3.py`; 72 frozen display records, molecular lookup values, protein endpoints, and saved geometry |
| 4 | R9 v02 | `source_data/figure_4_plot_data.json` and geometry | `figure_source/main/plot_figure_4.py`; saved activation/output/CI values, two molecular geometries, and stored atom activation arrays |
| S-B1 | R7 | `source_data/figure_S_B1_plot_data.json` | Frozen numerical source; renderer not included |
| S-C1 | R7 | `source_data/figure_S_C1_plot_data.json`, `figure_S_C1_tokens.csv` | Saved tokenizer examples and all 593 vocabulary-row norm records; renderer not included |
| S-C2 | R7 | `source_data/figure_S_C2_plot_data.json` | Original six MLM raster tiles in `figure_source/assets/historical_examples/` |
| S-C3 | R7 | `source_data/figure_S_C3_plot_data.json` | Original six MTR raster tiles in the same directory |
| S-C4 | R7 | `source_data/figure_S_C4_plot_data.json` | Original six CheMeleon raster tiles in the same directory |
| S-C5 | R7 | `source_data/figure_S_C5_plot_data.json`, `figure_S_C5_gates.json`, `figure_S_C5_matrices/*.npz` | Five frozen best-lift caches; historical renderer is included |
| S-C6 | R7 | `source_data/figure_S_C6_plot_data.json` | Frozen numerical source; renderer not included |
| S-E1 | R7 | `source_data/figure_S_E1_plot_data.json` | Frozen numerical source; renderer not included |
| S-I1 | R7 | `source_data/figure_S_I1_plot_data.json` | Frozen numerical source; renderer not included |
| S-J1 | R7 artwork, inherited R4 Figure 5 | `source_data/figure_S_J1_plot_data.json` | Frozen numerical source; renderer not included |
| TOC | R7 | `source_data/toc_graphic_plot_data.json` | Frozen numerical source; renderer not included |

Inherited JSON files retain their original historical figure-number fields, which map to current manuscript labels through the table above and `figure_manifest.json`. `source_data/data_S1_best_units.csv` supplies the retained H2/H3 unit-level source table rather than substituting for per-atom activation arrays.

## Read the frozen inputs

The main renderers use Python and Matplotlib and resolve every input relative to this package, independent of the working directory. The following check imports all four renderers and reads their selected JSON/geometry inputs without creating a figure, loading a model, or estimating a scientific result:

```bash
python -B figure_source/main/check_frozen_inputs.py
```

Passing under Python 3.7.0 and Matplotlib 2.2.3, this check verifies module import and input portability; the adapted renderers were not run to generate new artwork during packaging, and the selected PNGs remain the presentation reference.

## Render from saved results

Re-rendering the four main figures requires a Python environment with Matplotlib, without new model weights, chemistry preparation, fitting, bootstrapping, or inference. Because the original Arial font is not redistributed here, font substitution or different Matplotlib versions can change text placement.

```bash
python -B figure_source/main/plot_figure_1.py --version local
python -B figure_source/main/plot_figure_2.py --version local
python -B figure_source/main/plot_figure_3.py --version local
python -B figure_source/main/plot_figure_4.py --version local
```

Outputs are written to `rendered/figure_<n>/local/` (use a new version name for another rendering run). These scripts produce presentation renderings of frozen results—where geometry alignment and marker-overlap measurements are display operations—rather than a new scientific analysis.

The historical `figure_source/historical/plot_S_C2_to_C5.py` script (additionally requiring NumPy ≥1.17 and Pillow) reconstructs the earlier S-C2–S-C4 tile assemblies and S-C5 from saved caches; these historical assemblies predate the final R7 styling and are not claimed to reproduce the R7 PNGs pixel for pixel. Its S-C5 path (`fig_s3`) applies the historical effect/null formulas to the stored matrices and was not executed during initial packaging. Following restoration of the original fixed competitor count `K = 39` in `figure_source/historical/historical_helpers.py` (with `plot_S_C2_to_C5.py` unchanged), a targeted local validation of `fig_s3` verified all 16 J1/J2 effect estimates across the eight model/layer rows against `source_data/figure_S_C5_gates.json` within the `1e-9` tolerance (maximum absolute discrepancy `1.1102230246251565e-16`), confirmed finite 39-twin null clouds for all 16 series, and rendered the two-panel S-C5 figure without model inference, fitting, bootstrapping, or p-value recalculation (validation details in `figure_manifest.json`). That targeted check exercised only S-C5, not the S-C2–S-C4 tile branch (`fig_s2`), and leaves the 15 supplied original PNGs, the frozen source data and matrices, and the four main-figure rendering chains unchanged.

## Scope and limits

The four main rendering chains include their selected input values and minimal saved chemical geometry, retaining frozen plot-data values, CI families, cohort boundaries, coordinate/pole identities, and scientific qualifications. Among the supplementary figures, the packaged S-C2–S-C4 summaries contain the displayed unit/concept statistics and original raster tiles without the full per-atom response arrays or a machine-readable list of all depicted molecules; S-C5 includes the exact saved raw best-lift matrices and gate summaries rather than a newly generated table of null-cloud display points; and the remaining supplementary figures provide frozen numerical source files and unchanged final PNGs without their final renderers.

Public copies omit machine-specific paths and internal workflow/communication metadata, and retained numerical JSON fields were verified against the frozen inputs without changing their values. Remaining relative upstream paths inside data files serve as provenance labels that may refer to material outside this figure package, with draft-directory names normalized for this distribution, whereas executable rendering inputs use the package-local paths listed above. Overall, this package supports inspection and presentation from frozen results without claiming a complete training or inference rerun.
