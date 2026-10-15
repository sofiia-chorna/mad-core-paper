# Round 3, pooled-test learning curves and maps in two versions (2026-10-08)

Two evaluation populations, each used for both the overall learning curves and the sketch maps, for the round 3
models (MAD-CORE and RANDOM training, each training set plus the 102 isolated-atom references, seed 42,
validation-selected checkpoints). The figures show 5k, 10k and 20k; the 2k models are left out of the figures but
kept in the metric tables. Nothing was retrained or reselected; all errors come from the stored test predictions
already used for the round 3 tables. The separate-test figures in `../figures/` (SI) are unchanged. Captions are in
`CAPTIONS.md`.

## 1. Populations

| version | part | structures | atoms | force components | share of structures | share of atoms and force components |
|---|---|---|---|---|---|---|
| A, full union (`union_7847`) | RANDOM test, all | 5,847 | 214,002 | 642,006 | 74.5 % | 94.7 % |
| | MAD-CORE test, all | 2,000 | 11,957 | 35,871 | 25.5 % | 5.3 % |
| | total | 7,847 | 225,959 | 677,877 | | |
| B, equal structure counts (`balanced_4000`) | RANDOM test, sampled | 2,000 | 73,209 | 219,627 | 50.0 % | 86.0 % |
| | MAD-CORE test, all | 2,000 | 11,957 | 35,871 | 50.0 % | 14.0 % |
| | total | 4,000 | 85,166 | 255,498 | | |

- The B sample is `numpy.random.default_rng(20261008).choice(5847, 2000, replace=False)` over the frozen RANDOM test
  rows in rank order, indices sorted. It was drawn by id only, before any prediction was read
  (`scripts/build_populations.py` reads manifests only).
- Frozen manifests: `manifests/union_7847.tsv` (sha256 `61eceb5c…393b`) and `manifests/balanced_4000.tsv` (sha256
  `a9b0a135…f2c1`), listed in `manifests/CHECKSUMS.sha256`. A rerun rebuilds the same bytes and checks them (status
  `verified` in job 5002869). The source test manifests are checked against the checksums recorded when the split
  was built.
- Disjointness, checked by configuration id, frozen id, geometry hash and invariant fingerprint, all 0 in both
  versions:
  - RANDOM part against MAD-CORE part;
  - each population against the union of the eight round 3 training sets;
  - each population against both validation sets.
  Configuration ids and geometry hashes are also unique inside each population (`POPULATIONS.json`).
- B is balanced by structure count only. The RANDOM part averages 36.6 atoms per structure and the MAD-CORE part 6.0,
  so in B the RANDOM part still holds 86.0 % of the force components.

## 2. Metrics

- **Energy:** |ΔE| / N per structure, pooled over the structures of the population.
- **Forces:** absolute Cartesian component errors, pooled over all atoms and components of the population. This is
  the metatrain definition, as in the round 3 tables.
- **p8 and SUP:** (mean e^8)^(1/8) and max, with the same pooling.

Everything is computed from per-structure sums (count, Σ|e|, Σe², Σe^8, max |e|) of the stored predictions, so each
population is normalised by its own structure or component count. It is not an average of the two tests' metrics.
Recomputed for each full test, these sums reproduce all 128 values of `selected_checkpoints/metrics_mtt.tsv` exactly
at the printed precision (`scripts/component_stats.py`, cache in `selected_checkpoints/cache_stats/`).

Units: energy in meV/atom, forces in eV/Å.

**A, full union, 7,847 structures**

| training set | size | E MAE / RMSE | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|
| MAD-CORE | 2k | 518.4 / 828.4 | 1.085 / 1.991 | 3236.6 | 10.72 | 9369.1 | 46.04 |
| MAD-CORE | 5k | 317.0 / 476.9 | 0.953 / 1.521 | 1510.8 | 6.95 | 4034.6 | 28.02 |
| MAD-CORE | 10k | 197.4 / 285.9 | 0.635 / 0.996 | 1280.9 | 4.95 | 3914.9 | 22.71 |
| MAD-CORE | 20k | 138.6 / 211.6 | 0.597 / 1.010 | 992.4 | 4.78 | 3007.2 | 22.17 |
| RANDOM | 2k | 1377.1 / 3443.8 | 0.491 / 3.032 | 14152.6 | 28.09 | 42051.3 | 87.59 |
| RANDOM | 5k | 1080.6 / 2839.9 | 0.320 / 2.387 | 11458.9 | 52.00 | 33223.4 | 253.25 |
| RANDOM | 10k | 580.0 / 1662.6 | 0.275 / 2.190 | 11324.3 | 34.29 | 34386.9 | 172.15 |
| RANDOM | 20k | 419.9 / 1264.1 | 0.224 / 1.869 | 9668.3 | 30.62 | 29402.7 | 142.22 |

**B, equal structure counts, 4,000 structures**

| training set | size | E MAE / RMSE | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|
| MAD-CORE | 2k | 644.5 / 988.2 | 1.110 / 2.093 | 3518.6 | 12.09 | 9369.1 | 46.04 |
| MAD-CORE | 5k | 395.0 / 575.3 | 0.939 / 1.545 | 1641.2 | 7.67 | 4034.6 | 28.02 |
| MAD-CORE | 10k | 232.0 / 337.5 | 0.628 / 1.024 | 1393.3 | 5.57 | 3914.9 | 22.71 |
| MAD-CORE | 20k | 176.7 / 262.8 | 0.583 / 1.008 | 1079.6 | 5.25 | 3007.2 | 22.17 |
| RANDOM | 2k | 2615.7 / 4818.7 | 0.961 / 4.908 | 15396.3 | 31.73 | 42051.3 | 87.59 |
| RANDOM | 5k | 2085.6 / 3976.7 | 0.667 / 3.875 | 12465.9 | 58.75 | 33223.4 | 253.25 |
| RANDOM | 10k | 1105.1 / 2328.0 | 0.578 / 3.560 | 12319.5 | 38.74 | 34386.9 | 172.15 |
| RANDOM | 20k | 809.8 / 1770.2 | 0.479 / 3.038 | 10517.9 | 34.60 | 29402.7 | 142.22 |

Full-precision tables, with the selected epoch and the `model_best.pt` sha256 of every model, are in
`<version>/tables/metrics_<version>.tsv`. Side-by-side ratios are in `comparison.tsv`.

## 3. How the population choice changes the overall curves

- **Which training set is lower never changes.** This holds for every quantity, metric and size. MAD-CORE training
  is lower in all energy metrics and in force RMSE, p8 and SUP. RANDOM training is lower in force MAE.
- **SUP is identical in both versions.** Every model's largest energy and force error lies in the MAD-CORE test set,
  which both versions contain in full.
- **B moves every pooled metric towards its MAD-CORE-test value.** In B the MAD-CORE test holds 50 % of the
  structures (against 25.5 % in A) and 14.0 % of the force components (against 5.3 %). Both arms have larger errors
  on the MAD-CORE test than on the RANDOM test in every metric, with one exception: MAD-CORE training's force MAE at
  5k, 10k and 20k. The excess is much larger for RANDOM training, so the gap between the arms changes size:

  | metric, plotted sizes 5k to 20k | RANDOM / MAD-CORE in A | RANDOM / MAD-CORE in B |
  |---|---|---|
  | energy MAE | 2.9 to 3.4 | 4.6 to 5.3 |
  | energy RMSE | 5.8 to 6.0 | 6.7 to 6.9 |
  | force RMSE | 1.6 to 2.2 | 2.5 to 3.5 |
  | force MAE | 0.34 to 0.43 | 0.71 to 0.92 |

  For force MAE, RANDOM's advantage shrinks: its force MAE roughly doubles from A to B (×2.1), while MAD-CORE's falls
  by 1.1 to 2.4 %.

  p8 rises 8.6 to 8.8 % in energy and 9.9 to 13.0 % in force for both arms, and the p8 ratios between the arms are
  almost the same in A and B. This is mostly normalisation. The e^8 sums are dominated by structures that both versions contain, so p8 scales close to
  (N_A / N_B)^(1/8): 1.088 for energy (structures) and 1.130 for forces (components).
- **Curve shapes are the same in both versions, with one exception.** MAD-CORE force RMSE from 10k to 20k is
  0.996 → 1.010 eV/Å in A and 1.024 → 1.008 eV/Å in B, flat within 2 % in both. Every other plotted curve decreases
  from 5k to 20k in both versions, except RANDOM energy SUP (10k above 5k in both).
- **Sampling of the RANDOM part in B** (`metrics_by_part_*.tsv`). On the 2,000 sampled structures, MAE and RMSE are
  within −2 to +9 % of the full RANDOM test. p8 and SUP vary more (SUP 0.59 to 1.00 of the full-test value), as
  expected for tails in a one-third subsample. All four RANDOM-training force p8 values on this part are 1.143 times
  the full-test value. That factor is (5847 / 2000)^(1/8) to within rounding: the e^8 sum is dominated by a structure
  that is in the sample, so only the normalisation changes.

This section describes the effect of the population; it does not recommend either version.

## 4. Maps

- **Columns:** 5k, 10k and 20k, as in the delivered round 3 maps; 2k is left out.
- **Unchanged inputs:** projection coordinates (`scratch_scaling/reports/maps/coords_eval.npz`), bin boundaries
  (`grid.json`) and colour limits are those of the earlier maps. The colour limits are the 1st and 99th percentile of
  the round 1 bin values, as in the delivered round 3 maps.
- **Recomputed for each population:** bin counts, smoothed error sums, the support mask and the per-bin composition.
  Support:
  - A: 559 bins, holding 7,624 of 7,847 structures;
  - B: 544 bins, holding 3,768 of 4,000 structures.
- **Shared between the versions:** the crop window (taken from A) and the count colour scale.
- **Check:** A reproduces the delivered round 3 map panels (5k, 10k, 20k) bin for bin, asserted in the script.
- **Aggregation, unchanged:** one value per structure, |ΔE| / N or the mean absolute force-component error over its
  3N components; bin value = smoothed sum / smoothed count. Bins weigh structures equally, while the global force
  metrics pool components.
- **Colour bar:** a plain bar, as for the counts. Bins outside the fixed limits are drawn in the end colours. Only
  RANDOM-training panels have such bins; no MAD-CORE-training bin is outside. Counts per figure are in `MAPS.json`.

## 5. 5k + 5k: not feasible from existing labels

A MAD-CORE test of 5,000 needs 3,000 more eligible labelled MAD-CORE structures outside every round 3 training,
validation and test set. They do not exist.

| MAD-CORE structures in the labelled pool (`prep_20261005/reports/pool_20261005.tsv`) | count |
|---|---|
| total | 25,000 |
| eligible under the rule used by every round (`eligible_pair`) | 22,276 |
| used: training (`madcore_20k`, which contains the 2k, 5k and 10k sets) | 20,000 |
| used: `val_madcore_250` | 250 |
| used: MAD-CORE test | 2,000 |
| **eligible and unused** | **26** |
| under PBE-only eligibility | 89 |
| any PBE label, ignoring the screen (2,175 of them fail the 100 eV/Å force cut) | 2,262 |

The unused counts were checked against all round 3 sets by configuration id, frozen id, geometry hash and invariant
fingerprint.

The RANDOM side would be possible (5,847 ≥ 5,000). A 5k + 5k version would need either new DFT labels for MAD-CORE
structures beyond the labelled 25,000, or moving structures from MAD-CORE training into the test, which changes the
splits and needs retraining. Nothing was added or changed.

## 6. Files

All paths are under `mad-core-valid/experiments/scratch_scaling_v3/`.

- **Scripts** (`scripts/`), run by `submit_populations_daint.sh` (debug, lp133, job 5002869, 40 s, COMPLETED):
  - `build_populations.py`: selection and checks;
  - `component_stats.py`: per-structure sums, checked against the tables;
  - `population_figures.py`: metrics, figures and tables.
- **`reports/populations/`:**
  - `REPORT.md`, `CAPTIONS.md`, `METRICS.md`, `comparison.tsv`, `POPULATIONS.json`, `MAPS.json`, `manifests/`;
  - `union_7847/figures/` and `balanced_4000/figures/`: `learning_curves_<version>`, `sketchmap_energy_<version>` and
    `sketchmap_force_<version>`, each as `.png` and `.pdf`;
  - `<version>/tables/`:
    - `metrics_<version>.tsv` and `metrics_compact_<version>.{tsv,md}`;
    - `metrics_by_part_<version>.tsv`;
    - `learning_curves_source_<version>.tsv` (the plotted values);
    - `sketchmap_bins_<version>.tsv` (the plotted bin values of the supported bins);
    - `sketchmap_panels_<version>.npz`.

## 7. Paper figures (`paper/`, `scripts/paper_figure.py`)

Drawn from the tables and panels above, with no new evaluation or smoothing. Each version has its own files
(`<version>` = `balanced_4000` or `union_7847`).

- **Main text, 6.5 in wide:**
  - `train_curves_<version>`: pooled learning curves, energy and force, MAE / RMSE / p8 / SUP, 5k to 20k;
  - `train_force_map_<version>`: force maps, rows RANDOM and MAD-CORE, columns 5k / 10k / 20k, fourth column library
    density above evaluated structures, one shared force colour bar in eV/Å.
- **SI, 6.5 in wide:**
  - `si_train_energy_map_<version>`: energy maps, same layout and test structures;
  - `si_train_curves_separate_tests`: each test set separately, both tests in every panel, 5k to 20k.
- **Alternative:** `mad-core_train_<version>`, the earlier single-page combined figure (7 in wide).
- **Readability** (`readability_<version>.tsv`): `train_curves_<version>` uses fonts 1.35 times larger, tighter panel gaps
  and tick and axis labels closer to the axes, with one shared x label; its smallest text is 10.1 pt (9.9 pt at 6.38 in). In the other split figures the smallest text is 7.5 pt at 6.5 in, which
  prints at 7.4 pt at 6.38 in (article 11pt, 2.4 cm margins, A4) and 8.1 pt at 7.0 in. Exponents of the 10ⁿ tick
  labels are drawn smaller, as usual. The combined figure prints at 6.8 pt at 6.38 in.
