
# voronoi/scripts

## Production assignment pipeline (run in this order)

| step | script                                                             | what it does                                                                                                                                                                                                                                                                                           |
| ---- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | `build_manifest.py`                                              | lists all 8,096 input shards from both store roots + anchor →`out/manifest_all.tsv` (1,695,275,276 rows total)                                                                                                                                                                                      |
| 2    | `build_center_tree.py` + `submit_tree.sh`                      | extracts the 2^24 coreset224 features, scales them, writes the fps-ordered center matrix`out/tree/center_feats_scaled.f32` (34 GB) + `center_configuration_ids.npy`                                                                                                                                |
| 3    | `assign_shards.py` + `submit_assign.sh` / `submit_assist.sh` | the sweep worker: for every structure, exact nearest center at all 25 levels (running min over the fps-ordered centers, snapshot at each 2^k) →`out/assign/<family>/*.parquet`. Resumable (skip-if-exists); `submit_assist.sh` runs the same worker with `--reverse` to help from the other end |
| 4    | `global_integrity_check.py`                                      | wall-to-wall verification of all 8,096 output files (row counts, c↔a consistency, monotone distances, self-assignment)                                                                                                                                                                                |

`voronoi_net.py` — the core math, imported by steps 2–3: `brute_all_levels` (the
production chunked-matmul running-min kernel) and `load_scaler`; also the legacy
navigating-net `descend` (kept for the record — measured 18× slower, see `tex/`).

## analysis/ — results of the assignment

- `aggregate_voronoi.py` — one pass over all assignment files → group populations per
  level, distance histograms, level-16 dataset composition (`out/analysis/*.npz`)
- `compute_pca_spectrum.py` — uniform + voronoi-weighted PCA spectrum of the first 2^20
  points (`out/analysis/pca_spectrum_2p20.npz`)
- `plot_pca_residual.py` — residual-variance figure from that spectrum
- `make_final_plots.py` — distance_vs_coreset_size, structures_per_representative,
  concentration_vs_coreset_size (`plots/`)

## experiments/ — side checks, not part of the pipeline

- `blocked_experiment.py` — block-pruned exact search speed test (negative result:
  bit-identical but only 1.02–1.07× faster)
- `trajdist_224_parallel.py` — the Jonathan check: pairwise distances between
  same-material Alexandria trajectory picks in coreset224 (min = 0.2179 = eps exactly)
