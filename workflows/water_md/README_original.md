# PET-MAD-1.6-M minus the 18k learned r²SCAN−LDA and r²SCAN−PBE corrections: 32-water NVT at 1.0 g/cm³ (2026-10-12)

Formerly `md/petmad_m_delta_water_shared18k`, renamed 2026-10-13.

**Question.** Repeat the 1.0 g/cm³ water experiment (`md/water_delta_10k_1gcm3/`) with the corrections trained on the shared
18,000-structure split (`experiments/delta_lda_pbe_18k/`), everything else unchanged. The corrected potentials are approximate
LDA and approximate PBE models (PET-MAD-1.6-M minus a learned correction), not DFT trajectories.

**Scope.** Separate experiment. The 1.0 g/cm³ folder and its results are read, never written. No DFT, no training, no model change.

## Potentials

| | file | sha256 | output | weight |
|---|---|---|---|---|
| PET-MAD-1.6-M | `md/lips_delta_10k/models/pet-mad-m-v1.6.0.fixed.pt` (the corrected-attention-mask export; `upstream.pt` is the untouched download) | `7e578c97…` | `energy` | +1 |
| r²SCAN−LDA correction, 18k | `experiments/delta_lda_pbe_18k/runs/main/delta_lda--omat-s--madcore_shared18000--s42/model_best.pt` (validation best, epoch 102) | `3ace01e3…` | `mtt::delta_energy` | −1 |
| r²SCAN−PBE correction, 18k | `experiments/delta_lda_pbe_18k/runs/main/delta_pbe--omat-s--madcore_shared18000--s42/model_best.pt` (validation best, epoch 101) | `aebe2505…` | `mtt::delta_energy` | −1 |

- Both exported models equal the `best_model_state_dict` of their checkpoint to 0.0 (checked 2026-10-12); float64 copies for the
  verification in `verification/float64/` (`scripts/export_float64.py`), baseline float64 copy from the LiPS experiment.
- i-PI combines two force-field clients with weights +1 and −1 on energies and conservative forces, both clients receive the same
  coordinates at every step (checked from the geometry logs by `scripts/pilot_check.py`). Driver `scripts/metatomic_output_driver.py`,
  byte-identical to the LiPS driver (`760ccb14…`). Socket prefix `pk_`.

## Protocol, recovered from the 1.0 g/cm³ inputs and unchanged

- 32 H₂O, periodic 9.855480 Å cube, density 1.000 g/cm³ (`source/water_32_1gcm3.extxyz`, sha256 `3c2c263c…`, byte copy).
- NVT 300 K, SVR thermostat τ = 10 fs, `fixcom`, 0.25 fs timestep, 200,000 steps = 50 ps per run.
- Output every 50 steps (positions, velocities, forces, properties), checkpoint every 2,000 steps, hard-kill-safe `run.sh` resuming
  from the furthest readable checkpoint, refusing a fresh start over existing outputs.
- Five velocity seeds 20261010 + k and thermostat seeds 20261020 + k, k = 0 to 4; `starts/` are byte copies, so positions, velocities
  and seeds are identical across the three potentials and identical to the 1.0 g/cm³ runs (checked: `init.extxyz` and momenta lines).
- Analysis: `scripts/analyse.py`, `plot.py`, `plot_rdf_20_50ps.py`, `plot_rdf_panels_18k.py`, `plot_rdf_runs_20_50ps.py` as in the 1.0 g/cm³ experiment; RDF bins
  0.02 Å to 4.92 Å normalised to the cell density, blocks 5–20, 20–35, 35–50 ps, structural window 20–50 ps (the last 30 ps), kinetic
  temperatures from the saved velocities, thresholds 0.6 Å, 36 eV/Å, 600 K reported as crossings, not failures.
- No discrepancy was found between these inputs and the requested protocol.

## Baseline reuse

The five PET-MAD-1.6-M production runs of the 1.0 g/cm³ experiment use the same model file, start files, seeds and settings, so they
are reused unchanged: `runs/production/baseline_seed{0..4}` and `runs/pilots/*/baseline_seed0` are symlinks into the read-only
1.0 g/cm³ folder. Only the ten corrected runs are new.

## Jobs

- pilots and checks: `runs/submit_pilots.sh` (NVE 0.25 fs, NVE 0.125 fs, NVT in hard-killed segments, seed 0, both corrected
  potentials; `scripts/verify_water.py` on GPU), then `runs/submit_pilot_check.sh` (`verify_water.py` on CPU, `pilot_check.py`).
- production: `runs/submit_debug_production.sh`, Clariden debug chain, 3 nodes x 30 min segments (10 runs, one per GPU), self-queued
  with afterany until every run is complete, then `runs/submit_analysis.sh`.
- Status and results are appended below once the runs exist.

## Status (2026-10-13 02:49): complete

- Pilots and checks: jobs 3648964 and 3649020, `verification/water_checks.json` (17 checks) and `pilot_checks.json` (16 checks), all pass:
  hashes, cell and density, identical starts, float32 against float64 combined forces within 3.5e-5 eV/Å, finite differences within
  3.7e-5 eV/Å, both clients fed identical coordinates, i-PI energies and forces equal independent evaluation and subtraction within
  7e-5, NVE drift at most 0.006 meV/atom/ps with the dt² fluctuation scaling, NVT pilots at 295 to 296 K, restarts continuous.
- Production: chain 3649026, 3649028, 3649176, 3649339, 3649541 (30 min segments, timed out as designed) and 3649666 (finished),
  3649805 found all runs complete and queued the analysis 3649860. 10 corrected runs plus the 5 reused baseline runs, all 15 complete
  to step 200,000 with 4,001 paired frames, no confirmed numerical failure, no threshold crossing, no integration warning; conserved
  drift at most 0.0011 meV/atom/ps, largest force 7.0 eV/Å, shortest distance 0.874 Å.
- Temperatures: every run 299.4 to 301.1 K over 5 to 50 ps. Approximate LDA: H and O temperatures equal within noise from 5 ps.
  Approximate PBE: O stays hotter than H, T_H − T_O = −12.6 ± 4.1 K (5–20 ps), −8.4 ± 7.4 (20–35), −3.4 ± 4.8 (35–50), the same slow
  equipartition seen with the 10k correction at this density.
- Results: `analysis/REPORT.md`, tables `analysis/production_*.tsv`, figures `analysis/figures/` (three-panel RDF, mean of five replicas
  `rdf_oo_oh_hh_20_50ps.png`, the same with the five runs and O–H from zero for the paper
  `rdf_oo_oh_hh_runs_20_50ps.{png,pdf}`, two-panel O–O `rdf_oo_lda_pbe_panels_20_50ps.png`, standard figures), hydrogen-transfer diagnostic
  `diagnostics/proton_transfer/`.

### O–O structure, 20–50 ps, mean over five runs ± one standard error (10k-correction values from the 1.0 g/cm³ README in brackets)

| | PET-MAD-1.6-M (reused) | approx. LDA, 18k | approx. PBE, 18k |
|---|---|---|---|
| first peak r, g | 2.75 Å, 3.14 ± 0.06 | 2.57 Å, 4.81 ± 0.14 [2.59 Å, 4.94] | 2.73 Å, 3.48 ± 0.14 [2.74 Å, 3.06] |
| first minimum r, g | 3.34 Å, 0.68 ± 0.03 | 3.09 Å, 0.19 ± 0.03 [3.07 Å, 0.24] | 3.28 Å, 0.39 ± 0.06 [3.31 Å, 0.56] |
| second peak r, g | 4.43 Å, 1.29 ± 0.02 | 4.21 Å, 1.61 ± 0.04 [4.33 Å, 1.52] | 4.37 Å, 1.51 ± 0.05 [4.41 Å, 1.41] |
| O neighbours within 3.3 Å | 4.33 ± 0.01 | 4.15 ± 0.03 [4.27] | 4.08 ± 0.01 [4.06] |
| nearest O–H | 0.979 Å | 1.023 Å [1.020] | 0.991 Å [0.990] |
| virial pressure, 5–50 ps | −0.3 ± 0.2 kbar | −7.6 ± 1.0 kbar [−10.7] | +7.0 ± 0.4 kbar [+5.8] |
| frames with a coordination change after 5 ps | 0 | 2173, 166, 236, 331, 584 (seeds 0 to 4) [all five runs] | 0 |

- Seed-to-seed RDF spread (O–O, integrated |Δg| over r ≥ 1.3 Å): 0.17 baseline, 0.33 approximate LDA, 0.29 approximate PBE; the last
  two production blocks differ by less than the seed spread for every condition (ratios 0.58 to 0.89).
- Reading, limited to what the trajectories show: the approximate LDA potential is again far more structured than PET-MAD-1.6-M with a
  compressed first shell, a longer O–H bond and coordination changes in every run; the approximate PBE potential now has a first peak
  above the baseline (3.48 ± 0.14 against 3.14 ± 0.06) with a deeper first minimum and a higher second peak, where the 10k correction
  had a first peak equal to the baseline. These are machine-learned approximations of LDA and PBE, not DFT trajectories, and their
  accuracy against the functionals is not verified here.

## Reassessment 2026-10-13 (diagnostics/)

Collaborator comments on the LDA water led to three read-only diagnostics plus one bounded DFT check, see
`diagnostics/dft_snapshots/REPORT.md` for the summary: `diagnostics/proton_transfer/` (events), `diagnostics/molecular_integrity/`
(O-H from zero, bond and angle distributions, H-H decomposition, movies, OVITO trajectories), `diagnostics/dft_snapshots/`
(literature table, existing labels, nine snapshots at LDA, PBE and r2SCAN; jobs 3651137 and 3651188). Outcome: the approximate
potentials reproduce their functionals on these water geometries (forces within 25 to 33 meV/A, relative energies within
2 meV/atom, proton forces within 0.05 eV/A), LDA DFT ranks the shared-proton and ionic event frames below the intact one where
r2SCAN ranks them above (whole-cell energies within the 0.45 eV fluctuation scale), and 97 % of molecule-frames
in the approximate LDA runs are intact water. The wording stays approximate LDA and approximate PBE; no retraining indicated.

## Diagnostics 2026-10-13

- `diagnostics/dft_snapshots/`: bounded single-point check (nine snapshots x LDA, PBE, r2SCAN, jobs 3651137 and 3651188), the
  literature comparison, the event classification and the labelled-water force check; summary in its `REPORT.md`. Each
  combined potential reproduces its functional to about 30 meV/A and 1 to 2 meV/atom on these geometries; LDA DFT supports
  the shared-proton energetics.
- `diagnostics/molecular_integrity/`: O-H and H-H curves from zero, bond and angle distributions, H-H decomposition,
  defect time series and two 1 ps movies with molecules made whole; `REPORT.md` there.
