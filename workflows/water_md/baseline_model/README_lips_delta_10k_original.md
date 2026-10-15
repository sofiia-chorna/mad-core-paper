# PET-MAD-M v1.6.0 minus learned r²SCAN−LDA and r²SCAN−PBE corrections: adapted LiPS stability test (2026-10-09)

Formerly `md/petmad_m_minus_lda_delta`, renamed 2026-10-13.

**Question.** Can a correction learned on 10k MAD-CORE structures adapt the pretrained r²SCAN potential PET-MAD-M to LDA
and still give stable MD?

**What this experiment adds.** A new combined potential:

    E = E_PET-MAD-M − E_delta,    F = F_PET-MAD-M − F_delta

Earlier delta results used actual LDA calculations plus the correction to predict r²SCAN, so they say nothing about
this combination.

Three conditions, with identical starts, thermostat and seeds:

| condition | run folders | potential |
|---|---|---|
| A | `baseline_start*` | PET-MAD-M alone |
| B | `combined_start*` | PET-MAD-M minus the r²SCAN−LDA correction |
| C | `combined_pbe_start*` | PET-MAD-M minus the r²SCAN−PBE correction, added 2026-10-09 23:20 |

**Status.**
- Calculator verification, static evaluation and the NVE and NVT pilots are done.
- 50 ps production is running.
- No DFT was run, and no model was trained or changed.

## Models

| | baseline | correction |
|---|---|---|
| model | PET-MAD-M v1.6.0 | r²SCAN−LDA delta, `experiments/delta_lda_pbe_10k/runs/main_clariden/delta_lda--omat-s--madcore_10k--s42` |
| source | `lab-cosmo/upet`, `models/pet-mad-m-v1.6.0.ckpt`, revision `06774bd774562c6f801024f380c5af3026c66b60` | selected checkpoint, epoch 85 of 150, by validation mae_prod on 198 MAD-CORE structures (from its own experiment, unchanged) |
| file used | `models/pet-mad-m-v1.6.0.fixed.pt`, sha256 `7e578c97…` (exported from the checkpoint, sha256 `044418af…` = Hugging Face LFS id) | `model_best.pt`, sha256 `3324b33e…`; checkpoint `837d0316…`; the same files as the stored delta predictions |
| output used | `energy` (an LLPR uncertainty wrapper around PET); its `non_conservative_*` heads are not used | `mtt::delta_energy` |
| target | r²SCAN energy (MAD-1.6 conventions) | A_r²SCAN − A_LDA, atomization energies, each functional against its own 102 isolated-atom references |
| forces | conservative, −dE/dx by autograd | conservative, −dE/dx by autograd |
| units and types | float32, eV, Å, elements 1 to 102 | float32, eV, Å, elements 1 to 102 |

The r²SCAN−PBE correction:
- **Run:** `experiments/delta_lda_pbe_10k/runs/main_clariden/delta_pbe--omat-s--madcore_10k--s42`.
- **Selected checkpoint:** epoch 65 of 150, by validation mae_prod.
- **Files:** `model_best.pt` sha256 `3b496349…` and checkpoint `dd358427…`, the same files as its stored predictions.
- **Output and target:** output `mtt::delta_energy`; target A_r²SCAN − A_PBE with each functional's own references.
- **Units and forces:** float32, eV, Å, conservative forces.
- **Use:** it enters condition C with weight −1, with its original sign.

No dispersion correction is added.

The combined model is i-PI with two socket force fields: weight +1 for the baseline and −1 for the correction, which
enters with its original sign. i-PI applies the weight to energy, forces and virial (`engine/forces.py`).

## Environment

- **Runtime:** `${HOME}/venv-ipi-petmad`.
  - Packages: ipi 3.3.0, metatomic-torch 0.1.17, metatensor-torch 0.10.3, vesin 0.6.1, ase 3.29.0.
  - torch 2.9.1 from uenv `pytorch/v2.9.1:v2`.
  - Package lists: `verification/environment_*.txt`.
- **Driver:** `scripts/metatomic_output_driver.py`. It subclasses i-PI's own `MetatomicDriver` and changes only:
  - the energy output it requests (`output=`);
  - non-conservative heads, ensembles and uncertainties are forced off;
  - it logs hashes of the geometry and cell it receives.
- **Exports:** made with `${HOME}/venv-metatrain-fix`, metatrain 2026.4 with the PET attention-mask fix, the code
  that trained the correction. PET-MAD-M was also exported with plain metatrain 2026.4. One structure at a time, the
  two exports agree to 3.1e-5 eV/atom and 1.5e-4 eV/Å, the size of float32 rounding.
- **GPU neighbour lists:** `VESIN_CUDA_MAX_PAIRS_PER_POINT=16384`. Small dense test cells exceed vesin's default
  capacity; LiPS needs about 130 pairs per atom.

## Energy reference (training data only, nothing fitted on test data)

PET-MAD-M's `energy` is an r²SCAN atomization energy, relative to the MAD-1.6 isolated atoms, not a total energy:
- **MAD-1.6 training structures:** on 35 of them, among 500 sampled delta-training structures (seed 20261009), the
  energy minus the r²SCAN atomization label is −0.9 ± 4.5 meV/atom.
- **Total-energy reading:** off by about 10⁵ eV/atom, so it is ruled out.

Consequences:
- E_PET-MAD-M − E_delta compares directly with the LDA atomization energy, with LDA isolated-atom references.
- No per-element offsets are needed or fitted.

On the 465 sampled MAD-CORE r²SCAN training structures (`botr`), the same difference is −156 ± 478 meV/atom (median
−142). That is PET-MAD-M's own error on these strongly distorted structures, not a convention offset.

## Calculator verification (`scripts/verify_calculators.py`, `verification/calculator_checks.json`)

- **Identity:** checkpoint and export hashes are as above. The correction is the exact model behind the stored
  `delta_lda` test predictions.
- **Float32 consistency:** the patched and plain PET exports, and the LLPR wrapper against the bare PET, agree within
  float32 precision.
- **Gradients:** combined forces equal central differences of the combined energy.
  - Float64 copies of the same weights, start 0, 12 components: 8.4e-6 eV/Å at h = 0.001 Å, growing as h²
    (8.4e-4 at 0.01 Å).
  - Float32 against float64 combined forces on the 5 starts: at most 1.7e-5 eV/Å.
- **Correction model:** my one-structure evaluator reproduces the stored delta predictions to 3.4e-6 eV/atom and
  2.1e-5 eV/Å.

## Static evaluation (`scripts/static_eval.py`, `static/metrics.tsv`)

Test sets of the delta experiment against our FHI-aims LDA labels.
- Energy: |ΔE|/N per structure, meV/atom.
- Forces: Cartesian components, meV/Å.

| subset | model vs reference | E MAE / RMSE / p8 / max | mean E error | F MAE / RMSE / p8 / max |
|---|---|---|---|---|
| random (973) | PET-MAD-M vs LDA | 56.5 / 119.4 / 385 / 855 | +39.9 | 202 / 271 / 609 / 2213 |
| random (973) | PET-MAD-M − correction vs LDA | 11.8 / 21.3 / 71 / 137 | −0.8 | 55 / 91 / 590 / 2167 |
| random (973) | PET-MAD-M vs r²SCAN (context) | 5.7 / 13.7 / 83 / 194 | −3.0 | 42 / 83 / 600 / 2176 |
| MAD-CORE (1,631) | PET-MAD-M vs LDA | 939.7 / 2001.5 / 25554 / 64419 | +915.9 | 620 / 6487 / 168854 / 523647 |
| MAD-CORE (1,631) | PET-MAD-M − correction vs LDA | 327.0 / 1670.0 / 24918 / 62816 | −49.4 | 567 / 6477 / 168753 / 523289 |
| MAD-CORE (1,631) | PET-MAD-M vs r²SCAN (context) | 318.4 / 1666.1 / 24874 / 62704 | −50.7 | 566 / 6477 / 168726 / 523205 |
| MAD-CORE outside MAD-1.6 (1,451) | PET-MAD-M − correction vs LDA | 353.9 / 1768.9 / 25285 / 62816 | −54.0 | 590 / 6677 / 170046 / 523289 |
| MAD-CORE in MAD-1.6 train (170) | PET-MAD-M − correction vs LDA | 111.3 / 225.2 / 1168 / 2219 | −16.7 | 207 / 383 / 1517 / 3433 |
| random (973) | PET-MAD-M vs PBE | 440.0 / 451.4 / 571 / 1015 | −439.9 | 124 / 177 / 653 / 2231 |
| random (973) | PET-MAD-M − PBE correction vs PBE | 15.5 / 19.9 / 59 / 124 | +10.6 | 49 / 86 / 597 / 2185 |
| MAD-CORE (1,631) | PET-MAD-M − PBE correction vs PBE | 325.3 / 1669.6 / 24908 / 62790 | −48.6 | 565 / 6476 / 168742 / 523236 |

The PBE rows confirm the sign and energy convention of the PBE correction empirically.

`static/metrics.tsv` has every combination.

**Overlap with PET-MAD-M's training data:**
- The model card at the pinned revision is empty, so its training set is not documented there.
- Using MAD-1.6 membership by geometry hash as a proxy:
  - random test: 0 of 973;
  - MAD-CORE test: 170 in MAD-1.6 train, 5 in val, 5 in test.
- Neither set is claimed to be independent of the baseline; the MAD-CORE part outside MAD-1.6 is reported separately.

**Reading the table:**
- On the random set the correction removes most of PET-MAD-M's distance to LDA.
- On MAD-CORE the error is dominated by PET-MAD-M's own error against r²SCAN on extreme structures, with the largest
  force errors in the hundreds of eV/Å.
- Model selection was fixed beforehand; nothing here was used to choose a checkpoint.

## MD protocol: an adapted LiPS stability test, not a reproduction of "Forces are not enough"

**Kept from the benchmark and the previous LiPS runs (`md/md_augmentation/PROTOCOL.md`):**
- System: Li27P12S44, 83 atoms, fixed triclinic cell of 1676.139 Å³.
- Starts: the same five, with the same positions, cell and saved Maxwell–Boltzmann velocities (`lips_start{0..4}.data`).
- Masses: ASE, Li 6.94, P 30.973762, S 32.06.
- Dynamics: 0.25 fs timestep, 200,000 steps (50 ps), output every 50 steps (12.5 fs).

**Departures, recorded:**
- **Thermostat:** global SVR (Bussi) in i-PI with τ = 10 fs for both conditions. This is a predefined protocol choice
  following the Atomistic Cookbook thermostat example, not a conversion of the benchmark's Nosé–Hoover thermostat.
  i-PI 3.3.0, also the newest release, has no Nosé–Hoover.
- **Temperature and centre of mass:**
  - i-PI target 520 K with `fixcom = True` in both conditions, so 3N−3 degrees of freedom: a physical 520 K.
  - The previous LAMMPS runs set `compute_modify thermo_temp extra/dof 0` and `compute_modify nvt_temp extra/dof 0` in
    all 25 inputs, with no custom temperature computes and no `fix_modify temp`. That overrides the 3N−3 default of
    LAMMPS, so their 520 K was counted over 3N.
  - Their centre-of-mass momentum was removed once (`velocity all zero linear`) and stayed at about 1e-5 amu·Å/ps.
  - So they sampled 520 × 249/246 = 526.3 K over the real degrees of freedom. The new runs therefore have about 1.2 %
    less kinetic energy than the old ones; matching the old kinetic energy would mean a target of 526.3 K. The target
    was deliberately not changed.
- **Engine:** i-PI (velocity Verlet with an SVR step) instead of LAMMPS.
- **Thermostat seeds:** `20261210 + start`, the same in both conditions for a given start.
- **Initialisation:**
  - Positions, cell, labels and masses come from `init.extxyz` (`<file mode='ase'>`).
  - Momenta come from the saved velocities, in atomic units, using i-PI's own unit factors.
  - i-PI 3.3.0 `mode='manual'` positions left every atom at the origin in the first pilot, so they are not used.

## Pilots

**NVE** (start 0, 2 ps, `verification/pilot_checks_baseline_combined.json`, first written as `pilot_nve_checks.json`):

| run | drift (meV/atom/ps) | RMS fluctuation (meV/atom) | max deviation (meV/atom) | speed (steps/s, 1 run per GPU) |
|---|---|---|---|---|
| PET-MAD-M, 0.25 fs | −0.0001 | 0.0006 | 0.0048 | 26.8 |
| PET-MAD-M − correction, 0.25 fs | −0.0006 | 0.0017 | 0.0058 | 22.3 |
| PET-MAD-M, 0.125 fs | −0.0000 | 0.0003 | 0.0009 | 24.2 |
| PET-MAD-M − correction, 0.125 fs | +0.0003 | 0.0018 | 0.0060 | 22.7 |

There is no timestep dependence of concern.

**Implementation checks in the same run:**
- **Identical inputs:** both clients received identical positions and cells in all 20 logged calls.
- **Units:** received geometry equals the start file to 1.4e-8 relative. That is the ratio of i-PI's Å/bohr factor
  (1.8897261) to ASE's (1.8897261258).
- **Weighted combination:** i-PI's weighted energies and forces equal independent evaluation and subtraction on 20
  saved frames, to 4.5e-4 eV (5e-6 eV/atom, GPU against CPU in float32) and 3.5e-5 eV/Å.

**NVT** (start 0, 1 ps, SVR τ = 10 fs):
- Both conditions average 518 ± 49 K after 0.25 ps.
- The SVR conserved quantity shows no drift.

**Restart test, run as 9 to 10 segments cut at 60 s:**
- The stitched outputs are continuous: steps 0 to 4000, 81 frames, no gaps.
- The first version failed, and was fixed:
  - i-PI's timed soft exit closes the sockets before the step ends and leaves an empty or missing `RESTART`, plus stale
    sockets in `/tmp`.
  - `run.sh` now resumes from whichever of `RESTART` and the periodic `sim.chk` holds the furthest step, and removes
    this run's stale sockets.
  - Production does not use a timed soft exit (`total_time 0`).

**Throughput:** two runs sharing one GPU each run at about half speed, so production uses one run per GPU.

## Analysis rules (`scripts/analyse.py`, `scripts/lips_observables.py`, `scripts/check_runs.py`)

- **No truncation by RDF.** No trajectory is truncated or classified as unstable because its RDF differs from the
  benchmark reference.
- **Usable trajectory:** ends only at the first frame that is non-finite, has two atoms closer than 1.2 Å, or has an
  atomic force above 100 eV/Å.
- **Analysis window:** from 5 ps to the end of the usable trajectory.
- **The benchmark's VASP-PBE AIMD RDF:**
  - it is reported separately as a diagnostic (sliding-window first exceedance and the integrated difference);
  - for condition C it is a relevant reference;
  - our PBE labels are FHI-aims all-electron (tight basis, ZORA) while the benchmark is VASP PAW, so a C-to-reference
    difference includes that code difference;
  - for conditions A and B it is context only, and not an LDA reference.
- **Frame pairing:**
  - positions and forces are read separately from all restart segments;
  - each step keeps its latest occurrence;
  - frames are paired by step only;
  - missing steps, unexpected steps and steps present in only one file are reported, never paired silently.
- **Copied definitions:** `lips_observables.py` copies the RDF, MSD and failure definitions of the previous LiPS
  analysis, whose module paths predate the 2026-10-03 layout move.
- **Completion:** `check_runs.py <phase> [runs]` reads it from the trajectory files: last property step, last paired
  frame step, frame counts against the expected 4,001, restart and checkpoint steps, and every i-PI exit code.
  `submit_production.sh` and `submit_production_pbe.sh` report completion only when it passes. Arrays submitted before
  this change (3629130 and 3629131) keep their stored script and are checked with `check_runs.py` separately.

## Production

- **Jobs, normal partition, cancelled before they started (2026-10-10 02:16):**
  - `runs/submit_production.sh`: arrays 3629130 and 3629131;
  - `runs/submit_production_pbe.sh`: arrays 3629182 and 3629183.

  They stayed pending with `QOSGrpNodeLimit` because the ab010 group node limit was full.
- **Jobs, debug partition, used instead:** `runs/submit_debug_production.sh`.
  - Limits: 90 node-minutes per job, one running and two submitted jobs per user.
  - Each job is a 22-minute segment on 4 nodes, one run per GPU (`runs/run_node.sh`), covering every incomplete run.
  - Each segment queues its successor and logs to `logs/debug_chain.log`.
  - The chain ends when `check_runs.py` finds all 15 runs complete, when a segment makes no progress, or after 14
    segments.
  - Runs resume from the furthest readable checkpoint; an unreadable restart file is set aside.
  - First segment 3629548 started 2026-10-10 02:18.
- **PBE verification and pilots:** debug job 3629177 (`runs/submit_pilot_pbe.sh`); results in
  `verification/calculator_checks.json` and `verification/pilot_checks_combined_pbe.json`.
- **Cost:** 200,000 steps take about 2.5 h for each corrected potential (2.8 h at the 19.6 steps/s of the PBE pilot)
  and 2.1 h for PET-MAD-M alone.
- **Analysis:** `scripts/analyse.py production`, discarding the first 5 ps. It reports per trajectory:
  - completed time, failures, temperature, conserved-quantity drift, maximum atomic force and minimum distance;
  - all-atom and per-element-pair RDFs, the P–S bond distribution and PS4 fraction, and Li MSD.

## Results (2026-10-10)

All 15 production runs are complete and verified, with no failures. The three-condition comparison is in
`analysis/COMPARISON.md`, with tables in `analysis/COMPARISON_TABLES.md` and figures in `analysis/*.png`.

## References for LiPS

There is no LDA-labelled LiPS frame. The two benchmark-collection frames in our LDA labels (`random_25k`) are 9- and
21-atom molecules from the other systems of that collection.

Frames for a possible later LDA evaluation, rule fixed before production results:
- the 20 frozen benchmark frames of `md/md_augmentation/diagnostic/tables/proposed_lips_recalculation.tsv`;
- for each combined-potential production run, the saved frames at 25 ps and 50 ps.

**The reference AIMD is a separate simulation from the benchmark's ML MD protocol.**
- Per the NequIP paper (Batzner et al., Nat. Commun. 13, 2453, 2022, Methods), the LiPS data come from an earlier study.
  The structure was equilibrated and then run in NVT at 520 K for 50 ps with a 2.0 fs time step, giving 25,001
  frames.
- The reference AIMD used a Nosé–Hoover thermostat, which the NequIP paper does not state; its degree-of-freedom
  convention was not verified here. Our runs use SVR, so the thermostats differ.
- The single Nosé–Hoover thermostat with ttime 20 and the 0.25 fs step belong to the ML MD protocol (NequIP's custom
  Nosé–Hoover, tuned to match the AIMD temperature fluctuations, and the benchmark code's `lips.yml`), not to the
  reference.
- The RDF at 520 K is a canonical-ensemble property, so the thermostat matters little for it if both runs sample
  correctly. Li MSD and diffusion depend more on thermostat coupling.

**What the results can and cannot show:**
- The benchmark's AIMD reference RDF is VASP-PBE. It is context only and is not an LDA reference, so a difference
  from it is not an LDA prediction error.
- Stability alone does not establish accurate LDA dynamics.

## Layout

- `models/`: the downloaded checkpoint and the two exports.
- `scripts/`:
  - `models.py`: independent evaluator and LiPS starts.
  - `metatomic_output_driver.py`: i-PI driver.
  - `verify_calculators.py`.
  - `static_eval.py` (`atomization` convention).
  - `make_ipi_inputs.py`: all inputs and run scripts.
  - `pilot_check.py`.
  - `analyse.py`.
- `runs/production/<condition>_start<k>/`: one folder per production run, with `input.xml`, `run.sh`, `run.json`, i-PI
  outputs and driver logs.
- `runs/pilots/{nve,nve_half_dt,nvt}/<condition>_start0/`: the short pilot test runs, the same files (moved there from
  `runs/pilot_*` on 2026-10-10).
- `runs/submit_*.sh`: one job script per step.
- `verification/`, `static/`, `analysis/`, `logs/`.
