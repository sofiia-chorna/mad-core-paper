# Verification log (audit stage, 2026-10-15/16)

All commands were read-only on the two source repositories. Outputs went to the session
scratchpad; the comparison images and plot logs kept here were copied from it.

## Environment used

- `uenv pytorch/v2.9.1:v2 --view=default`, login node of Alps (Clariden), Python 3.12.
- `/users/schorna/venv-phace`: matplotlib 3.10.9, numpy 2.4.4, pyarrow 24.0.0, torch 2.9.1,
  ase 3.28.0, skmatter 0.1.dev448 (editable), metatrain 0.2.0+phace (editable), scikit-learn 1.8.0.
- `/users/schorna/venv-metatrain-fix`: matplotlib 3.11.1, metatrain 2026.4, metatomic-torch 0.1.17, ase 3.29.0.
- Ghostscript (`gs`) and ImageMagick 7 (`magick compare -metric AE`, 100 dpi rasters).

## 1. Scripted paper figures (papers repo `figures/`, commit 28cfe43)

Command (plot-only, no `--prepare`, output directory in the scratchpad):

    uenv run pytorch/v2.9.1:v2 --view=default -- bash figures/check_figures.sh <scratch>/figcheck

Result, differing pixels at 100 dpi against the committed PDF:

    fig1                          0
    gfre_lfre_heatmap_union       0
    gfre_lfre_heatmap_random_test 0
    mad-kset-1col                 0
    mad-kset-counts               0
    coreset224-kset               0
    coreset224-elements           0
    voronoi_groups24_d8_p20d8     0
    train_curves_4000             no plot.py
    train_force_map_4000          no plot.py
    si_train_curves_400           no plot.py

`git status` of the papers repo after the run: unchanged (same six untracked files as before).
Per-figure stdout is in `*.plot.log` next to this file.

## 2. Training learning curves (dev repo, `scratch_scaling_v3/scripts/paper_figure.py` at 97bec6f)

A copy of the script with `S`, `PAPER` and `LIMITS` redirected (source tree read in place,
output to scratch) was run as `paper_figure.py balanced_4000 curves separate_parts`.

| python | output | reference | differing pixels |
|---|---|---|---|
| venv-phace (mpl 3.10.9) | train_curves_balanced_4000.pdf | dev `reports/populations/paper/train_curves_balanced_4000.pdf` (md5 510faae2) | 2808 of 650x374, text antialiasing only (`train_curves_mpl3.10.9_vs_report_diff.png`) |
| venv-metatrain-fix (mpl 3.11.1) | same | same | **0** |
| venv-phace | si_train_curves_balanced_parts.pdf | dev `si_train_curves_balanced_parts.pdf` (md5 c14c8cb8) | 2169 of 700x384, text antialiasing only |
| venv-metatrain-fix | same | same | **0** |

The reference PDFs carry producer "Matplotlib v3.11.1", which explains the antialiasing
differences with 3.10.9.

The copy committed in the papers repo, `figures/train_curves_4000.pdf` (md5 7acb72ec,
committed 2026-10-10 02:24 in 5f1c2ae), rasterises to 650x381 and matches neither build:

| script revision | result |
|---|---|
| a19d3d4 (2026-10-09) | fails: reads `union_7847/tables/sketchmap_panels_union_7847.npz`, which was untracked and deleted in c06aa1a (2026-10-10) |
| e517b32 (2026-10-10 17:28), union_7847 TSV tables restored from `c06aa1a^` | 650x383, 33,812 differing pixels against the paper copy, 30,611 against the current dev build |
| 97bec6f (current) | 650x374, 0 differing pixels against the current dev build |

Conclusion: the papers-repo copy is an intermediate 2026-10-10 build of a script revision
that depended on a deleted file. Its data source (`learning_curves_source_balanced_4000.tsv`,
2026-10-08 16:06) predates all three builds and is unchanged, so the curves are the same
data; only layout differs.

## 3. Water RDF figure (dev repo, `md/water_delta_18k/scripts/plot_rdf_runs_20_50ps.py`)

Copy with `D` (input root) and `FIGURES` (output) redirected, reading the tracked
`analysis/curves/*.tsv`.

| python | differing pixels against tracked `analysis/figures/rdf_oo_oh_hh_runs_20_50ps.pdf` (md5 d7a73b16, producer Matplotlib v3.11.2) |
|---|---|
| venv-phace (mpl 3.10.9) | 8930, and the raster is 1 px wider (686 vs 685), text antialiasing only (`water_rdf_mpl3.10.9_vs_tracked_diff.png`) |
| venv-metatrain-fix (mpl 3.11.1) | **0** |

## 4. Numerical claims recomputed from committed figure data (papers repo `figures/coreset224-kset/data/`)

From `group_counts.tsv`, `category_counts.tsv`, `coverage_radius.tsv` at k = 2^24:

    groups contributing 341, >10,000 structures: 26, <100 structures: 235
    mad-next rows 340,555 of 16,777,216
    OMat24 0.563, Alexandria 0.188, JARVIS_QE_TB 0.035, MP-ALOE 0.034, JARVIS_OQMD 0.023, nomad 0.023, OMol25 0.022
    OMat24 + Alexandria 0.752
    3D bulk 0.945, molecules 0.030, clusters 0.015, surfaces/2D 0.011, molecular crystals 1,954, solvated 247, atom 88
    first prefix containing each category: bulk 2, atom 2, clusters 32, molecules 128, surfaces 128, solvated 16,384, molecular crystals 32,768
    eps(2^24) = 0.21790655, eps(2^23) = 0.30455601
    ANI/QM/SPICE/OMol25/gdb13 prefixes 0.0326, OC20/22/25 0.0098

All agree with the text (341, 26, 235, 340,555, 94%, three quarters, about 3.5% molecular,
about 1% catalysis, 0.218, 0.304, solvated and molecular crystals within the first 2^15).

## 5. Element-coverage claims recomputed from the store parquets (level 2, 40 s each)

    coreset_224_light.parquet (fps_order, elements): all 102 elements covered by pick 2158
    mad_union_full.parquet   (fps_order, elements): all 102 elements covered by pick 480

Both match the text (2,158 and 480). No script for these numbers existed in either repo;
the one-off code is recorded in `04_gap_list.md` for inclusion as a claim script.

## 6. Values read directly from source files (no computation)

- SI SPACE validation metrics: `aa002/mad-core/models/training/run_2026-06-26_03-09-50/train.log`,
  epoch 92: validation energy MAE 29.070 meV/atom, validation non-conservative force MAE 85.407 meV/A. Text: 29.1 and 85.4.
- SPACE learning rate 0.003 (`options_restart.yaml`), parameters 9,698,688 (`train.log`: "The model has 9.7M parameters").
- Scalers: `store:release/ecumetric_scaler.npz` (mean (512,), scale 125.4608154) equals `store:mad_union/scaler_madunion_phaceatomic_meanstd.pkl` exactly (max |mean difference| 0.0, scale difference 0.0); `store:models/scaler_mad15_phaceatomic_meanstd.pkl` (used for the per-family 10 M FPS) has scale 134.5823 and mean differing by up to 12.68.
- Split sizes from the same run's `indices/`: training 594,769, validation 41,338, test 41,347 (text identical).
- Ecumetric on the 973 r2SCAN structures: `random_test/README.md` and `r2scan_eval.json`: E MAE 16.4, F MAE 109 meV/A, 608 GDB-13 structures. Text identical.
- 99.17%: `claims/out/collection_within_coverage.json` 99.16597755 (certified radius 0.2179152).
- 99.93% within eps(2^23) = 0.304: `claims/out/outside_coverage_maxima.json`, radius_level_23 0.30455962, percent 99.93351485.
- Directed Hausdorff 0.4 for passed structures: `outside_coverage_maxima.json`, maximum_passed distance 0.39998862.
- 117 cells, 115 million, 26-atom GDB-13 molecule: `claims/out/largest_voronoi_cell.json` (117, 114,996,700, natoms 26, C10H13N3).
- 67,975 non-finite rows: `claims/out/nonfinite_descriptors.json`; 1,694,502,355 - 67,975 = 1,694,434,380 (the MAD-COSMO count in the text).
- Table training-set sizes: `scratch_scaling_v3/manifests/SETS.json` atoms_ordinary madcore 32,973 / 63,653 / 120,468, random 186,211 / 369,718 / 730,646. Text identical.
- Table finetune: `finetune_aligned_20k/reports/comparison/metrics.tsv` rows scratch and finetune_aligned, population balanced_4000. Text identical after rounding.
- Table delta-learning: `delta_lda_pbe_18k/reports/main/RESULTS.md` and `direct_r2scan_18k/reports/main/RESULTS.md`. Text identical.
- Table accrete-levels: `aa002/mad-core/release/mad-core/levels.json` reproduces |S|, from-anchor and cov for the seven rows; the margin and t(s) columns are not in any file (see gap list).
- Water snapshot force MAEs 11 / 27 / 31 meV/A: `water_delta_18k/diagnostics/dft_snapshots/REPORT.md` section 4 gives H/O pooled values 10/14 (PET-MAD vs r2SCAN), 25/31 (approx. LDA vs LDA); the per-component pooled values quoted in the text are in `REPORT_SNAPSHOTS.md` (not re-derived here).

## 7. External identifiers checked on 2026-10-16

- https://huggingface.co/lab-cosmo/ecumetric: reachable without login, license bsd-3-clause.
- https://doi.org/10.24435/materialscloud:rs-tt and :58-wq: HTTP 404 (reserved or unpublished drafts).
- https://doi.org/10.5281/zenodo.23266878 (the `madcore_code` DOI in `biblio/others.bib`): HTTP 404 (reserved, not published).
- https://archive.materialscloud.org/record/2025.145 (PET-MAD reference record): CC BY 4.0, 268.9 MiB in data/figures/inputs/models/notebooks tarballs plus README and requirements.txt.

## 0. Untracked and gitignored files that matter (sha256 prefix, bytes)

Papers repo, untracked (`git status`):

    c5a45037316001a8 150420845 claims/out/nomad_provenance/entry_metadata_cache.tsv.gz
    9afa5f49670f94c0  24448876 claims/out/nomad_provenance/madcore/nomad_records_madcore.tsv.gz
    34af8d9c0bcac08e 355394714 claims/out/nomad_provenance/nomad_records.tsv.gz
    c718c07ef1989526    255663 claims/out/nomad_provenance/upload_metadata_cache.json.gz
    2abfbbdaff3b9b12 369493862 claims/out/outside_coverage_rows.npz
    707e276cad134018   2304514 figures/mad1.5_gfre_lfre_compare/random_test/results_random_test/superseded/20261014-004427/selection_random_seed0.npz

Dev repo, gitignored but required by the release:

    af3d2aff11c1f5fb      9520 mad-core-valid/experiments/scratch_scaling_v3/manifests/SETS.json
    8bde2ccf4786e719     84046 mad-core-valid/md/water_delta_18k/analysis/production_runs.json
    8897e123c3d26f30 134217856 matcloud-upload/xyz-2pow24-v2/voronoi_population_record.npy

Full SHA-256 of every proposed file: `../inclusion_list.tsv`.
