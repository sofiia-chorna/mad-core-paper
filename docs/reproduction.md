# Reproduction guide

One row per figure, table and stated number of "Hierarchical universal open structure
dataset for atomistic modeling" (text at commit 28cfe43 of the papers repository, see
`provenance/SOURCES.md`). Every status below was established by running the command from
this repository on 2026-10-16; the commands, environments and comparisons are in
`docs/verification.md`.

**Levels.** L1: regenerate the figure or table from the numerical inputs in this repository.
L2: recompute those inputs from saved predictions, features, logs or trajectories (large
inputs in the supporting-results archive or on the project store). L3: rerun the selection,
training, electronic-structure or simulation workflow (`workflows/`, documented, not rerun).

**Statuses.** `verified`: executed for this release and compared with the submitted file or
value; `located`: inputs and code identified, not executed here; `partial`; `missing`;
`conceptual`: an illustration without computation.

**Environment.** `environment/plot-requirements.txt` (Python 3.12, matplotlib 3.11.1) for
every L1 command; pixel-identical output for Figs. 1, 3 to 8 and SI Fig. 1 needs matplotlib
3.10.9 instead, the version of those PDFs (3.11.1 changes text antialiasing only). All L1
commands run on a laptop in under a minute each except Fig. 8 (reads a 17 MB npz).
`scripts/check_figures.sh [python] [out_dir]` runs every figure command and reports the
differing pixels against `reference_figures/submitted/` at 100 dpi (needs Ghostscript and
ImageMagick 7). Outputs go to `build/`, never to `reference_figures/`.

Paths marked `store:` are on the project store of the original runs (not public; the
layout is `workflows/data_records/FORMAT.md`). Where a public record file replaces them,
the row says so.

## Main-text figures

| figure | command (from the repository root) | inputs | expected check | upstream (L2 / L3) | level, status |
|---|---|---|---|---|---|
| Fig. 1, nested k-set example | `python figures/fig01_nested_kset/plot.py` | none (synthetic points, seed 4) | 0 differing pixels vs `reference_figures/submitted/fig1.pdf` | none | L1 verified |
| Fig. 2, accretion scheme | none | `reference_figures/submitted/accrete.pdf` | n/a | hand-drawn, no source (`figures/fig02_accrete/README.md`) | conceptual |
| Fig. 3, GFRE/LFRE on the MAD-NEXT test split | `python figures/fig03_gfre_lfre/plot.py` | `figures/fig03_gfre_lfre/data/union_{gfre,lfre}_seed0.json` | 0 differing pixels vs `gfre_lfre_heatmap_union.pdf`; the printed matrices match the JSON | L2: `workflows/gfre_lfre/union/gfre_lfre_union.py` on the 677,454-structure teacher memmaps (`store:models/training/atomic_combined`, 90 GB) or on `descriptors_seed0.npz` of the results archive; `prepare_data.py` copies the JSONs from `workflows/gfre_lfre/union/results_union/`. L3: teacher featurisation `workflows/gfre_lfre/teacher_features/` | L1 verified; L2 located (results archive); L3 located |
| Fig. 4, MAD-NEXT subsets per level | `python figures/fig04_madnext_subsets/plot.py` | `data/subset_counts.tsv` | 0 differing pixels vs `mad-kset-1col.pdf` | L2: `prepare_data.py --store-root STORE` reads `store:mad_union/mad_union_full.parquet` (fps_order, subset). L3: `workflows/fps/runs/mad_union_meanstd` | L1 verified; L2 located; L3 located |
| Fig. 5 (a, b), MAD-NEXT element families and periodic table | `python figures/fig05_madnext_elements/plot.py` | `data/element_family_counts.tsv`, `data/element_structure_counts.tsv` | 0 differing pixels vs `mad-kset-counts.pdf` | L2: `prepare_data.py --store-root STORE` on the same parquet (elements) | L1 verified; L2 located |
| Fig. 6 (a, b, c, inset), MAD-CORE decomposition | `python figures/fig06_madcore_decomposition/plot.py` | `data/{group_counts,element_family_counts,category_counts,coverage_radius}.tsv` | 0 differing pixels vs `coreset224-kset.pdf`; `claims/out/element_coverage.json` and the text numbers (341 groups, 26 above 10,000, 235 below 100, 340,555 MAD-NEXT, 94% bulk, 0.752 OMat24+Alexandria, eps 0.218 / 0.304) recomputed from these tables | L2: `prepare_data.py --store-root STORE` reads `store:accrete/.../coreset224/out/coreset_224_light.parquet`, `picks_category_map_224.json`, `store:release/mad-core/dataset_group.npy`. L3: `workflows/accretion/` (64 GPUs, 23 h) | L1 verified; L2 located; L3 located |
| Fig. 7, MAD-CORE periodic table | `python figures/fig07_madcore_elements/plot.py` | `data/element_structure_counts.tsv` | 0 differing pixels vs `coreset224-elements.pdf` | as Fig. 6 | L1 verified; L2 located |
| Fig. 8, sketch-map of MAD-CORE | `python figures/fig08_sketchmap/plot.py` | `data/smap_points.npz` (17 MB) | 0 differing pixels vs `voronoi_groups24_d8_p20d8.pdf` | L2: `prepare_data.py --store-root STORE` projects `store:release/mad-core/ecumetric_features.bin` (34 GB, or the record feature files) with `data/models/ecumetric_sketchmap_projector_p20d8_rec_silu.pt`, weights by `data/voronoi_population_record_level24.npy` (results archive), seed 0, gamma 0.5, about 2 min. L3: `workflows/sketchmap/` (fit 2 nodes 79 min; needs the uncommitted fork, P3) | L1 verified; L2 located; L3 partial |
| Fig. 9, learning curves | `python figures/fig09_fig10_training/plot_training.py curves` | `data/learning_curves_source_balanced_4000.tsv`, `data/learning_curve_ylimits.tsv` | 0 differing pixels vs `train_curves_balanced_4000.pdf` (matplotlib 3.11.1) | L2: `workflows/training_from_scratch/scripts/population_figures.py` from the prediction caches (results archive `training_from_scratch/cache_stats`) and `scripts/evaluate_final.py` from the six checkpoints (models archive). L3: `workflows/training_from_scratch/` (8 trainings, 187,200 updates each) and the PBE labels (`workflows/dft_labels/`, MAD-CORE record) | L1 verified; L2 located (archives); L3 located |
| Fig. 10, force-error maps | `python figures/fig09_fig10_training/plot_training.py force_map` | `data/sketchmap_panels_balanced_4000.npz`, `data/sketchmap_panels_union_7847_round3.npz` (crop window, see `data/README.md`), `data/MAPS.json` | 0 differing pixels vs `train_force_map_balanced_4000.pdf` | L2: `workflows/training_from_scratch/scripts/shared/make_maps.py` and `population_figures.py` (binning of the per-structure errors on the sketch-map coordinates; needs `coords_eval.npz` and `library_xy.npy` of the development tree, not included). L3: as Fig. 9 | L1 verified with a recovered crop-window file; L2 partial (two cache files not included); L3 located |

## Main-text table

| table | check | status |
|---|---|---|
| Table I, overview of the data records | `python tables/check_tables.py --record-dir DIR` counts the structures of the three `dft_*.extxyz.gz` files of the MAD-CORE record: r2SCAN 21,854 as printed; **LDA 79,220 and PBE 77,362 in the record files, against 79,241 and 77,383 in the text** (open point P2). The other entries are descriptive; the record manifests are in `workflows/data_records/record_manifests/` | partial |

## Supporting-information figures

| figure | command | inputs | expected check | upstream | level, status |
|---|---|---|---|---|---|
| SI Fig. 1, GFRE/LFRE on the random MAD-COSMO selection | `python figures/siS1_gfre_lfre_random/plot.py` | `data/random_{gfre,lfre}_seed0.json`, `data/summary_seed0.json` | 0 differing pixels vs `gfre_lfre_heatmap_random_test.pdf` | L2: `workflows/gfre_lfre/random_test/gfre_lfre_random_test.py` on the per-model features (results archive, 1.6 GB; fingerprinted, bit-identical recompute documented in its README). L3: the five featurisation scripts and environments | L1 verified; L2 located (results archive); L3 located |
| SI Fig. 2, learning curves by test part | `python figures/fig09_fig10_training/plot_training.py separate_parts` | `data/metrics_by_part_balanced_4000.tsv` | 0 differing pixels vs `si_train_curves_balanced_parts.pdf` (assumed submitted file, see `reference_figures/README.md`) | as Fig. 9 | L1 verified (reference file assumed) |
| SI Fig. 3, water RDFs | `python figures/siS3_water_rdf/plot.py` | `data/rdf_{oo,oh,hh}_20_50ps.tsv` | 0 differing pixels vs `si_rdf_water_20_50ps.pdf` (matplotlib 3.11.1) | L2: `workflows/water_md/scripts/analyse.py` on the 15 i-PI trajectories (1.1 GB, not included; `production_curves.npy` of the results archive holds the analysed curves). L3: i-PI runs (`workflows/water_md/runs/`, 10 GPU runs of 50 ps) with PET-MAD-1.6-M and the two delta models (models archive) | L1 verified; L2 partial; L3 located |

## Supporting-information tables

`python tables/check_tables.py` compares every printed value with its source file in `tables/data/`.

| table | source file | status |
|---|---|---|
| accrete levels (|S|, from MAD-NEXT, cov, margin, t) | `tables/data/levels.json` (release `levels.json`): size, anchor count and coverage of the seven rows agree. **The margin and t(s) columns have no surviving source**: the accretion run log was deleted and `levels.json` itself cites the printed table for the float32 coverage. They are reported as submitted and cannot be reproduced | partial (3 of 5 columns verified) |
| training-set sizes | `tables/data/training_sets_SETS.json` | verified (6 values) |
| fine-tuning errors | `tables/data/finetune_metrics.tsv` (rows `scratch`, `finetune_aligned`, population `balanced_4000`) | verified (16 values) |
| delta-learning errors | `tables/data/delta_lda_pbe_18k_metrics.json`, `direct_r2scan_18k_metrics.json` | verified (48 values) |

## Numbers stated in the text

| statement | source | command / evidence | status |
|---|---|---|---|
| MAD-COSMO has 1,694,434,380 structures | the release holds 1,694,502,355 rows of which 67,975 have non-finite descriptors and were dropped from the record (`claims/out/nonfinite_descriptors.json`) | `python claims/nonfinite_descriptors.py --release-root STORE/release` (4 min, rerun identical) | verified |
| 99.17% of the library within eps(2^24) = 0.218 | `claims/out/collection_within_coverage.json` (99.166%, certified radius 0.21792) | `python claims/collection_within_coverage.py --release-root STORE/release [workers]` (Slurm, 1.7e9 rows) | located, output consistent |
| 99.93% within eps(2^23) = 0.304, directed Hausdorff 0.4 for structures passing the filters | `claims/out/outside_coverage_maxima.json` (0.30456, 99.934%, 0.39999) | `claims/outside_coverage_scan.py` then `outside_coverage_maxima.py` (Slurm; needs the certificate population arrays, see P5) | located, output consistent |
| half of the records in 117 cells; largest cell 115 million around a 26-atom GDB-13 molecule | `claims/out/largest_voronoi_cell.json` | `python claims/largest_voronoi_cell.py --release-root STORE/release` (6 s, rerun identical) | verified |
| all 102 elements within the first 2,158 MAD-CORE picks and the first 480 MAD-NEXT picks | `claims/out/element_coverage.json` | `python claims/element_coverage.py --store-root STORE` (40 s each; written for this release) | verified |
| 341 of 455 groups contribute, 26 above 10,000, 235 below 100, 340,555 MAD-NEXT structures, 94% bulk, three quarters OMat24 and Alexandria, about 3.5% molecular, about 1% catalysis, solvated and molecular crystals within 2^15 | `figures/fig06_madcore_decomposition/data/*.tsv` | recomputed from the tables (`docs/verification.md` section 4) | verified |
| ZBL thresholds 200 eV/Å, 50 eV/atom, 100 eV/atom; fewer than 0.02% excluded | `workflows/fps/submit_fps_family.sh`, `workflows/accretion/anchor/FILTERS.md`; 215,308 of 1.69e9 rows (`claims/out/madcosmo_populations.json`) | read | verified from files |
| 289,934 NOMAD volume-scan entries in 21 uploads excluded | `workflows/accretion/cut_2pow24/excluded_nomad_volume_scans_*_ids.txt` | the release counts 262,478 `nomad_excluded` rows, entries versus rows not reconciled (P6) | located, unreconciled |
| 454 dataset groups; 403 ColabFit, 232 Materials Cloud, OQMD and 7 Materials Cloud databases via OPTIMADE, 6,150 NOMAD uploads; GDB-13 933,762,319 structures | `claims/out/madcosmo_populations.json` (454 / 455 folders), `workflows/data_records/license_audit/license_audit.tsv` (655 rows by channel), `workflows/collection/` | counts per channel not recomputed | located |
| SPACE: 9.7 M parameters, cutoff 7.5 Å, 2 layers, batch 32, lr 3e-3; 677,454 structures, 17.9 M atoms; split 594,769 / 41,338 / 41,347; validation MAE 29.1 meV/atom, 85.4 meV/Å | `workflows/ecumetric_training/` (`options_restart.yaml`, `build_stats.json`, `run_2026-06-26_03-09-50/train.log` epoch 92: 29.070, 85.407; `indices/`) | read | verified from files |
| ecumetric on the 973 r2SCAN structures: 16.4 meV/atom, 109 meV/Å, 608 GDB-13 molecules | `workflows/gfre_lfre/random_test/results_random_test/r2scan_eval.json` | `workflows/gfre_lfre/random_test/eval_r2scan.py` (GPU, record labels) | located, output consistent |
| GFRE/LFRE protocol (31,300 finite of 41,347; 15,000 stratified; 4,000 / 11,000; ridge 1e-5 to 0.9, 2-fold; random test 5,905 = 4,330 + 1,575) | `workflows/gfre_lfre/union/gfre_lfre_union.py`, `random_test/results_random_test/summary_seed0.json` | read | verified from files |
| MAD-NEXT 216,803 + 600,300 + 38,909 = 856,012, seeded at the hydrogen atom | `workflows/fps/README.md` (mad), `runs/mad_union_meanstd/` | counts read; the seed row is an argument of the FPS run | located |
| accretion: 183 M pool (184,180,658 with the anchor), 10 M cap per family, about 10% retained from MAD-NEXT up to the anchor size (86,910 of 856,012), 23 h on 64 GPUs | `tables/data/levels.json`, `workflows/accretion/anchor/FILTERS.md` (183,324,646 unique), run log deleted | read | located; timing unverifiable |
| sketch-map parameters (sigma 1.24, A 4, B 2, a 2, b 2, PCA init, 400 iterations, weight power 0.25, MLP 512-512-256-128-8 SiLU, R^2 0.9982) | `workflows/sketchmap/README_original.md`, `data/models/ecumetric_sketchmap_projector_p20d8_rec_silu.json` | read | verified from files |
| DFT selections 25k / 25k / 6k with 9 rare-element replacements; converged 22,276 / 24,281 / 5,847; < 300 atoms, forces <= 100 eV/Å | `workflows/dft_labels/lda_pbe/README.md`, `sets/*/selection.tsv`, `scripts/ensure_elements.py` | read | located |
| training protocol (PET-OMat-S hyperparameters, MSE, Adam, batch 64, lr 3e-4, warm-up 0.01, cosine, seed 42, about 187,200 updates; validation 250 + 1,000; test 2,000 + 2,000) | `workflows/training_from_scratch/runs/PROVENANCE.json`, `runs/*/options.yaml`, `figures/fig09_fig10_training/data/POPULATIONS.json` | read | verified from files |
| fine-tuning 90 / 191 vs 177 / 583 meV | `tables/data/finetune_metrics.tsv` | `tables/check_tables.py` | verified |
| delta learning: 20,881 aligned, 18,000 / 1,000 / 1,881; 143 / 382 to 29 / 62; random 62 / 266 to 20 / 41 | `workflows/delta_learning/delta_lda_pbe_18k/manifests/SETS.json`, `tables/data/*_metrics.json` | `tables/check_tables.py` | verified |
| water MD protocol and snapshot force MAEs 11 / 27 / 31 meV/Å | `workflows/water_md/README_original.md`, `verification/*.json`, `diagnostics/dft_snapshots/REPORT_SNAPSHOTS.md` | read | located |

## Open points

- **P1, undocumented prefilter.** The collection was built from the structures that pass a
  Pyykkö covalent-radius contact filter at scale 0.35, plus those rejected at 0.35 but
  passing at 0.25 (`workflows/collection/download_and_prefilter/`); structures rejected at
  0.25 were not featurised and are not in MAD-COSMO. The manuscript and the record READMEs
  describe only the ZBL filter. Whether the omitted structures would all have failed the ZBL
  thresholds cannot be checked, because they were never featurised. The authors have to
  decide whether the text needs a sentence on this prefilter.
- **P2, DFT label counts of Table I.** The record files hold 79,220 LDA and 77,362 PBE
  labelled structures; the text prints 79,241 and 77,383 (21 more each). The source of the
  printed numbers was not found in either repository.
- **P3, sketch-map fit code.** The torch sketch-map implementation that fitted the map of
  Fig. 8 exists as uncommitted changes in the author's scikit-matter fork; it is not in this
  release. The fitted embedding, the projector and the projection are.
- **P4, PET-MAD-1.6-M baseline.** The water runs used an export of PET-MAD-M v1.6.0 with a
  local attention-mask fix (`workflows/water_md/baseline_model/`); the two exports agree to
  float32 rounding on single structures, but the public PET-MAD 1.6 release was not checked
  for the same fix.
- **P5, Slurm-scale claims.** `collection_within_coverage.py`, `outside_coverage_scan.py` and
  `madcosmo_populations.py` scan all 1.7e9 release rows (hours, hundreds of GB of memory)
  and read certificate and population arrays that are not in this release (38 GB and 2.9 GB
  on the development tree). Their committed outputs are included and consistent with the text;
  they were not rerun.
- **P6, NOMAD exclusion count.** 289,934 excluded entries versus 262,478 `nomad_excluded`
  rows in the release; the difference (duplicates, other filters, dropped non-finite rows) was
  not reconciled.
- **P7, accrete table.** Two of the five columns of the SI accrete table are from a deleted
  log (see above).
- **P8, data-collection code provenance.** `workflows/collection/` comes from two
  unversioned trees; only file hashes at copy time are recorded.
