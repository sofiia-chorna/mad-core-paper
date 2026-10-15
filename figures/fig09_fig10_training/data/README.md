# Inputs of Figs. 9 and 10 and of the SI learning curves

All files come from `reports/populations/` of the `scratch_scaling_v3` experiment of the
development repository (see `provenance/extracted_files.tsv`), written by
`workflows/training_from_scratch/scripts/population_figures.py` on 2026-10-08 from the
predictions of the six selected checkpoints on the frozen balanced 4,000-structure test
(2,000 random MAD-COSMO and 2,000 held-out MAD-CORE structures).

| file | content |
|---|---|
| `learning_curves_source_balanced_4000.tsv` | the plotted values of Fig. 9: MAE, RMSE, p8 and SUP of energy (meV/atom) and force components (eV/A) for the MAD-CORE and random trainings at 5k, 10k and 20k structures |
| `learning_curve_ylimits.tsv` | the y range of each panel, frozen from the data range of both evaluated populations so the panels are comparable |
| `metrics_by_part_balanced_4000.tsv` | the same metrics on the two 2,000-structure parts separately (SI learning curves) |
| `metrics_balanced_4000.tsv`, `metrics_compact_balanced_4000.tsv` | pooled metrics with the selected epoch and checkpoint hash of every model |
| `sketchmap_panels_balanced_4000.npz` | Fig. 10: 64 x 64 bins over the sketch-map plane with, per model, the smoothed mean energy and force error, the structure count, the support mask, the library density and the bin edges |
| `sketchmap_bins_balanced_4000.tsv` | the supported bins of that file as a table |
| `sketchmap_panels_union_7847_round3.npz` | the support mask (`keep`) of the union of the two full frozen tests (7,847 structures), which fixes the crop window of the maps. The file the figure script originally read (`union_7847/tables/sketchmap_panels_union_7847.npz`) was deleted untracked on 2026-10-10; this copy is `reports/maps/panels_scaling_selected.npz` of the same experiment, whose `keep` and `count` arrays the population script asserted equal to the deleted file when it wrote it (assertion in `population_figures.py`). With it the force map reproduces the submitted PDF pixel for pixel |
| `MAPS.json` | colour limits of the maps (1st and 99th percentiles of the round-1 panels) and bin counts |
| `POPULATIONS.json` | how the balanced population was drawn (seed 20261008) and its disjointness checks against the training sets |
| `CAPTIONS.md` | the captions written with the report |
