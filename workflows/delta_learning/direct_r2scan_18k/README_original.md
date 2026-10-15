# direct_r2scan_18k

Formerly `experiments/direct_r2scan_shared18k`, renamed 2026-10-13.

Matched direct r2SCAN fine tune completing the comparison with the two 18k corrections of `delta_lda_pbe_18k`.

## Data

The exact `delta_lda_pbe_18k` manifests, read in place and not copied: `manifests/xyz/train.xyz` (18,000), `val.xyz` (1,000),
`test_madcore.xyz` (1,881), `test_random.xyz` (973), smoke subsets `smoke_*.xyz`. Targets are the r2SCAN atomization energies
(`atomization_energy`, the established r2SCAN isolated atom references subtracted) and forces (`forces`) stored in those frames.
No reference atom is a training structure.

## Training

`scripts/make_runs.py` writes `runs/<stage>/direct_r2scan--omat-s--madcore_shared18000--s42/options.yaml` from the established direct
template (`delta_pbe_10k_vs_mad16/config/options_direct.yaml`) with one change: the target is named `mtt::r2scan_energy` (quantity energy)
instead of `energy`. With the pretrained target name metatrain would keep the pretrained energy head; with a new name it prunes that head
and initialises a fresh one, the same head replacement the delta runs get through `mtt::delta_energy`. The 10k direct run of
`delta_lda` kept the pretrained head, so this is a recorded difference from that earlier run, not from the two 18k corrections.
Everything else equals the delta runs: PET-OMat-S `models/pet-omat-s-v1.0.0.ckpt`, full fine tune, batch 64, 150 epochs, lr 1e-4,
warmup 0.01, seed 42, checkpoint every 25 epochs, mae_prod selection on the 1,000 validation structures (product of the validation
energy per atom MAE and force component MAE, lowest epoch). Composition baseline and scale are fitted by metatrain on the r2SCAN
targets of the 18,000 training structures (checked against an independent least squares fit in the report).

## Jobs

`runs/main/submit.sh`, Clariden normal partition, one allocation of 4 h: smoke stage (2 epochs on 256 structures, evaluation and
report), then the 150 epoch run, `scripts/evaluate.py` (standalone predictions on the two shared tests, batch 16, no DFT added),
`scripts/report.py` (`reports/main/RESULTS.md`, `metrics.json`: the direct model next to the two corrections read from the shared
experiment, same metrics, views and definitions) and `scripts/audit_restart.py`. The normal partition was chosen because the debug
QoS allows two queued jobs per user and the water MD chain holds them. Job id in `runs/main/JOBID`.

## Status 2026-10-13

Job 3649160 ran 01:45 to 03:54 (2 h 09 min) in one allocation without a restart: SMOKE DONE, MAIN DONE (150 epochs, 281 steps per
epoch, 42,150 updates), evaluation, REPORT PASSED (16 checks). Slurm records the job as FAILED only because the last step,
`scripts/audit_restart.py`, crashed: it was copied from the shared delta experiment and still parsed the `mtt::delta_energy` log lines
and buffers, so it saw no epochs. The target name was changed to `mtt::r2scan_energy` and the audit rerun on the login node, AUDIT
PASSED (`reports/main/RESTART_AUDIT.md`, `restart_audit.json`): lineage 0 to 149 without gaps, optimizer and scheduler steps equal
(epoch + 1) x 281 in every checkpoint, normalization identical across checkpoints, best epoch 113 is the validation mae_prod minimum.

Best checkpoint epoch 113 (validation E MAE 153.2 meV/atom, F MAE 371.6 meV/A), exported to
`runs/main/direct_r2scan--omat-s--madcore_shared18000--s42/model_best.pt`. Standalone test results (`reports/main/RESULTS.md`),
E MAE in meV/atom and F MAE in meV/A, direct / LDA + delta / PBE + delta: MAD-CORE 1,881 143.2 (bias -6.9) / 28.5 / 24.0 and
382 / 62 / 51; random 973 61.6 (bias +46.8, molecules +49.9) / 20.3 / 6.2 and 266 / 41 / 24. The 10k direct model (different test,
pretrained head kept) had 164.6 / 387 on its MAD-CORE test and 94.4 / 182 on the random test.

Paper: the direct row is in `paper_demo/manuscript/tables/delta_shared.tex` (column DFT at inference, none for the direct model) and
`delta_corrections.tex`.
