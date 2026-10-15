# finetune_aligned_20k: PET-OMat-S fine-tune on the scratch model's data (2026-10-09)

One predefined run, `runs/omat-s--madcore_20k_aligned--s42`. Its training and validation data are aligned with the 20k
MAD-CORE PBE model trained from scratch (round 3). It is evaluated on the frozen balanced 4,000-structure test set next
to that scratch model and the previous fine-tune (`voronoi_weighted/pbe_20k/runs/pbe/corrected_20k/omat-s--madcore_20k--s42`).
No settings were tuned, and the checkpoint is chosen by validation only.

## What is aligned and what still differs

| | scratch 20k (round 3) | previous fine-tune | this run |
|---|---|---|---|
| training structures | 20,000 MAD-CORE + 102 isolated atoms | 20,000 MAD-CORE (19,756 shared) | the scratch file, 20,000 + 102 |
| validation | 1,000 random + 250 MAD-CORE | 1,000 random | the scratch file, 1,000 + 250 |
| initialisation | random | PET-OMat-S v1.0.0, all parameters | PET-OMat-S v1.0.0, all parameters |
| energy head, composition baseline, scale | fitted on the training set | inherited unchanged | inherited unchanged |
| learning rate, epochs, updates | 3e-4, 596, 187,144 | 1e-4, 150, 46,800 | 1e-4, 150, 47,100 (314 steps per epoch) |
| schedule, clipping, loss, selection | Adam, 1 % warm-up then cosine to 0, clip 1, MSE energy and forces 1:1, best validation mae_prod, seed 42 | same | same |

The training set, isolated-atom references, validation set and selection weighting are now the same as for the
scratch model. What still differs from the scratch model:
- initialisation, together with the inherited head, composition baseline and energy scale;
- the learning rate;
- the number of updates.

A difference between this run and the scratch model is therefore not caused by pretraining alone either.

## Configuration

`runs/omat-s--madcore_20k_aligned--s42/options.yaml` is the previous fine-tune's `options.yaml`, unchanged except for:
- `training_set.systems.read_from`: `scratch_scaling_v3/manifests/xyz/madcore_20k_aug.xyz`;
- `validation_set.systems.read_from`: `scratch_scaling_v2/manifests/xyz/val_combined_1250.xyz`;
- the wandb name, group and tags.

The model architecture comes from the checkpoint. `checkpoint_interval: 2` is kept as before; it does not affect training.

Energy head, baseline and scale, as in the previous fine-tune:
- the target is named `energy`, as in PET-OMat-S, so no new head is created;
- the pretrained energy head, its per-element composition baseline (102 elements) and its energy scale (0.18707635)
  are inherited and not refitted.

The targets are PBE atomization energies: energy minus the campaign's PBE isolated-atom energies. The 102 references
are those isolated atoms, with atomization energy 0 and zero forces.

Checked after launch: the composition buffer and energy scaler buffer in this run's epoch 0 checkpoint are identical
(max difference 0) to PET-OMat-S v1.0.0, and so are those of the previous fine-tune's selected checkpoint. The
"Calculating composition weights" and "Calculating scaler weights" lines in both training logs leave them unchanged.

## Input checks (`scripts/check_inputs.py`, `reports/input_checks.json`; repeated in the job before training)

- **Files:** training and validation files equal the scratch model's by checksum and hold exactly its 20,102 and
  1,250 structures. Identifiers in file order: `manifests/train_ids.tsv` and `manifests/val_ids.tsv`.
- **Disjointness:** training (ordinary and references), random validation, MAD-CORE validation, the full random test
  and the full MAD-CORE test are pairwise disjoint by configuration id, frozen id and geometry hash. The test
  population is `scratch_scaling_v3/reports/populations/manifests/balanced_4000.tsv`.
- **References:**
  - 102 single atoms, elements 1 to 102 once each, atomization energy exactly 0 and zero forces;
  - every element is a pretrained atomic type with a composition weight;
  - the reference energies equal the campaign's PBE isolated-atom energies.
- **Energy convention:** 200 sampled training atomization energies equal the PBE energy minus the same isolated-atom
  energies, to within 6.5e-9 eV.
- **Fit with the inherited baseline:** the starting offset per atom, label minus inherited baseline, has
  - median 7.6 eV/atom and 1 to 99 % range −1.6 to 32.3 eV/atom for ordinary structures;
  - median 4.5 eV/atom and range −3.2 to 14.3 eV/atom for the references.

  101 of 102 references lie inside the ordinary range; the exception is Xe at −3.2. The references therefore ask the
  head for offsets of the same size as the ordinary structures do.
- **Isolated atoms in the tests:** no test structure is a non-periodic single atom.

## Running

```
python scripts/check_inputs.py     # login node, read only
sbatch runs/submit.sh              # Clariden normal, ab010: checks, training, evaluation, comparison
```

Steps of `runs/submit.sh`:
1. `voronoi_weighted/pbe_20k/scripts/launch_runs_fix.sh` trains with the corrected PET package.
2. `scripts/evaluate.py` exports the selected checkpoint and evaluates both full frozen tests, writing hashes to
   `reports/preds/*.provenance.json`.
3. `scripts/compare.py` writes `reports/comparison/`.

Python: `uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python`.

## Outputs (`reports/comparison/`)

- `metrics.tsv`: the three models on the balanced 4,000 structures and on each 2,000-structure part. Energy is
  |ΔE|/N per structure in meV/atom; forces are Cartesian components in meV/Å; MAE, RMSE, p8 and max.
- `table_pooled.tex`, `table_by_subset.tex`.
- `provenance.json`: for each model, checkpoint and exported model hashes, the exported model equal to the selected
  state, prediction hashes, epochs, selected epoch, steps per epoch and updates, and every check.

## Selected checkpoint

Epoch index 83 of 0 to 149, the lowest validation energy MAE per atom times force MAE (64.14 meV/atom times 173.2 meV/Å)
over all 150 epochs, recomputed from `train.csv`. 314 steps per epoch, 47,100 updates. Its exported model carries the
selected state (111/111 tensors).

## Results (balanced 4,000 test set, `reports/comparison/metrics.tsv`)

Energy in meV/atom, forces in meV/Å, MAE / RMSE / p8 / max.

| model | test | energy | forces |
|---|---|---|---|
| from scratch | pooled 4,000 | 176.7 / 262.8 / 1080 / 3007 | 583 / 1008 / 5253 / 22173 |
| fine-tuned, previous | pooled 4,000 | 95.3 / 141.0 / 602 / 1521 | 194 / 374 / 6257 / 27779 |
| fine-tuned, aligned | pooled 4,000 | 90.0 / 143.7 / 586 / 1465 | 191 / 360 / 5717 / 24924 |
| from scratch | random 2,000 | 100.6 / 144.3 / 381 / 788 | 597 / 997 / 4310 / 14884 |
| fine-tuned, previous | random 2,000 | 68.0 / 74.3 / 135 / 334 | 173 / 263 / 1027 / 3799 |
| fine-tuned, aligned | random 2,000 | 47.8 / 53.8 / 109 / 259 | 171 / 251 / 1026 / 3821 |
| from scratch | MAD-CORE 2,000 | 252.7 / 342.5 / 1177 / 3007 | 496 / 1077 / 6554 / 22173 |
| fine-tuned, previous | MAD-CORE 2,000 | 122.5 / 185.1 / 656 / 1521 | 321 / 758 / 7997 / 27779 |
| fine-tuned, aligned | MAD-CORE 2,000 | 132.1 / 195.9 / 639 / 1465 | 314 / 731 / 7307 / 24924 |

Reading the numbers:
- **Alignment changed little.** Aligning the training set, references, validation set and selection weighting moved
  the fine-tune only from 95.3 to 90.0 meV/atom and from 194 to 191 meV/Å pooled.
- **What still separates fine-tune and scratch:** the pretrained initialisation with its inherited head, baseline and
  scale, the learning rate (1e-4 vs 3e-4) and the number of updates (47,100 vs 187,144). The gap is therefore still
  not due to pretraining alone.
- **Energy differences between the two fine-tunes are within selection noise.**
  - Over epochs 100 to 149, validation energy MAE ranges from 79.8 to 114.4 meV/atom in the previous run and from 72.8
    to 125.1 in this run.
  - Validation force MAE moves only from 155.5 to 168.1 and from 166.3 to 180.6 meV/Å.
  - The two runs use different validation sets, so compare these values within a run only.
  - The energy gaps between the two fine-tunes (random 47.8 vs 68.0, MAD-CORE 132.1 vs 122.5 meV/atom) are smaller
    than this epoch-to-epoch swing; forces are the stable comparison.
- **The random-test energy error looks like a near-constant offset.** RMSE is close to MAE for both fine-tunes (53.8
  vs 47.8, 74.3 vs 68.0), consistent with an offset from the inherited baseline. The mean signed error was not computed.
- **Force tails on MAD-CORE:** p8 and max are still higher than from scratch (7307 and 24924 vs 6554 and 22173 meV/Å).

All checks in `reports/comparison/provenance.json` passed:
- the balanced 4,000 manifest matches its frozen checksum;
- for each model, the checkpoint, exported model and prediction hashes match their records, and the exported model
  carries the selected state;
- every prediction is aligned with its test structure by atom count and positions within 1e-4 Å;
- the scratch and previous fine-tune numbers reproduce `scratch_scaling_v3/reports/populations/finetune_vs_scratch/metrics.tsv`.

## Jobs

- **3628079** (Clariden normal, ab010), submitted 2026-10-09 18:24, ran 18:54 to 21:45.
  - The input checks passed and training finished all 150 epochs, with `Training finished!` at 21:42.
  - The export of the selected checkpoint succeeded.
  - `scripts/evaluate.py` then stopped at `torch.load` of `model.ckpt` because `metatomic.torch` was not imported, so
    no test predictions were written. The job is marked FAILED for this reason only.
- **3628887** (Clariden debug, ab010, `runs/submit_eval.sh`), 21:46 to 21:48.
  - It reran only `evaluate.py`, with the missing import added, and `compare.py` on the exported model.
  - Training was not repeated, and the checkpoint choice was not changed.
