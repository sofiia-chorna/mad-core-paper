# delta_lda_pbe_18k

Formerly `experiments/delta_lda_pbe_shared`, renamed 2026-10-13.

Two fresh correction models, r2SCAN-LDA and r2SCAN-PBE, trained on one shared random split of the verified paired MAD-CORE pool.
This protocol supersedes the frozen splits of `delta_lda` and `delta_lda_18774` and the stratified split of `delta_lda_18k_dev_superseded`
(prepared, never trained). All previous runs and manifests are untouched.

## Data

- Pool: 18,774 old training + 232 eligible of the old MAD-CORE validation 250 + 1,875 old MAD-CORE test = 20,881 structures, every one
  re resolved with both pairings under the existing rules (r2SCAN label from BotR round 1, MAD-1.6 v4 or BotR-0; identical geometry and
  pbc across the three functionals; finite; max atomic force norm <= 100 eV/A for all three; not fragmented). LDA deltas and r2SCAN
  sources identical to the previous manifests (gap 0.0 eV).
- Split (`scripts/build_shared_split.py`): pool sorted by configuration_id, one numpy RandomState(42) permutation, positions 0 to
  17,999 train, 18,000 to 18,999 validation, 19,000 to 20,880 MAD-CORE test; each split written in stable index order. No stratification.
- Random test: the unchanged 973 r2SCAN random test structures with both pairings, separate.
- Zero overlap between all four splits by configuration id, frozen id, geometry hash and lattice aware fingerprint; no internal
  duplicates. Counts 18,000 / 1,000 / 1,881 / 973. Composition in `manifests/SPLITS.md`, everything in `manifests/SETS.json`,
  checksums in `manifests/CHECKSUMS.sha256`, frames in `manifests/xyz/` (r2SCAN, LDA and PBE labels and both deltas in every frame).
- Flagged: the 10 structures whose r2SCAN label comes from the MAD-1.6 validation or test files (previously kept out of training by
  the training rule) all fell into the training split under the permutation. The Gd and Xe single atoms stay eligible (both in training).
  No reference atom calculation is a training structure.

## Training

`scripts/make_runs.py` writes `runs/<stage>/delta_{lda,pbe}--omat-s--madcore_shared18000--s42/options.yaml` from the established
delta template. The two options files differ only in the target keys (delta_energy_lda / delta_forces_lda vs delta_energy_pbe /
delta_forces_pbe) and wandb names; both are identical to the 18,774 run's options up to names and paths: PET-OMat-S checkpoint
`models/pet-omat-s-v1.0.0.ckpt`, full fine tune, batch 64, 150 epochs, lr 1e-4, warmup 0.01, seed 42, checkpoint every 25 epochs,
best_model_metric mae_prod (product of the validation MAE entries, energy per atom times force components, lowest wins, every epoch,
applied separately to each model on the same 1,000 validation structures). Each run fits its own composition baseline and scale on its
delta targets over the 18,000 training structures, prunes the pretrained energy head and starts a fresh delta head with fresh optimizer
state; nothing is initialised from an existing delta model.

## Jobs

- smoke: `runs/smoke/submit.sh`, Clariden debug, both corrections for 2 epochs on 256 structures, evaluation and report on the smoke files.
- main: `runs/main/submit.sh`, Clariden debug chain because the normal queue held over 1,000 pending jobs: both corrections train in
  the same allocation, one GPU each, 1 h 30 links chained with afterany (the debug QoS allows two queued jobs per user, so later
  links are submitted by a watcher), every link resumes each run from the newest checkpoint of its own run directory, the link that
  finds both trainings finished evaluates (`scripts/evaluate.py`, batch 16) and reports (`scripts/report.py` -> `reports/main/RESULTS.md`,
  `metrics.json`: MAE, RMSE, p8 = (mean |e|^8)^(1/8), SUP = max |e|, signed energy bias, whole test plus nonperiodic and periodic views,
  LDA and PBE uncorrected rows next to the corrected ones). Ids in `runs/*/JOBID*`.

## Outcome 2026-10-12

Smoke 3648033 (training fine, evaluation step fixed and verified on the login node, 37 checks). Chain: link 1 3648046 TIMEOUT after
epochs 0 to 105 (LDA) and 0 to 99 (PBE), link 2 3648047 resumed from the epoch 100 (LDA) and epoch 75 (PBE) checkpoints, finished
training, evaluated and reported (REPORT PASSED, 39 checks), link 3 3648367 found the marker and left. Best epochs 102 (LDA) and 101
(PBE), 42,150 optimizer updates each in the final model. Restart audit `reports/main/RESTART_AUDIT.md` passes for both runs (one
near tie for the LDA run between epochs 94 and 102 is resolved by the checkpoint metric, 9e-6 relative apart, below the log rounding).
Results `reports/main/RESULTS.md`; checkpoints `runs/main/<run>/model_best.pt` (validation selected) and `model.ckpt` (best and last).
