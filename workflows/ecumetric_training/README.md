# PhACE-atomic — training reproduction bundle

Everything needed to reproduce the training of `phace_atomic_ep92`
(`../phace_atomic_ep92.pt`). Copied from
`${DEV_ROOT}/phace-atomic` on 2026-07-11.

Model: atom-level PhACE distillation on full MAD (1.5 + next + sol) with 5
per-atom teacher heads (pet, mace, uma, dpa_omat, dpa_omol).
wandb run: `universal_features / phace-atomic-madfull`.

## Contents

```
atomic_combined/              training dataset (memmap) — this is what /tmp/train was
  a.bin x.bin c.bin e.bin f.bin   structures: species, coords, cells, DFT energy, forces
  na.npy ns.npy                   per-structure atom offsets / #structures
  pet.bin mace.bin uma.bin        teacher features (per-atom distillation targets),
  dpa_omat.bin dpa_omol.bin       mean/scale-normalized in place
  val_indices_in_all.txt          validation split (indices into all structures)
  test_indices_in_all.txt         test split; train = everything else
  build_stats.json                order, ns, atom counts, per-teacher scale + NaN counts

options.yaml                  training config actually passed to submit.sh
options_restart.yaml          same config, resolved/expanded by metatrain (defaults filled in)
submit.sh                     multi-node launch (copies atomic_combined -> /tmp/train per node)
submit_debug_chain.sh         debug-partition chained variant
prep_features.py              builds atomic_combined/ from the 3 source xyz + teacher .npy
extxyz_to_memmap.py           extxyz -> memmap helper used by prep_features.py

run_2026-06-26_03-09-50/      record of the run that produced ep92
  indices/{training,validation,test}.txt   resolved split actually used
  train.csv train.log                      loss curves + log (through epoch 92)
```

## Dataset composition (build_stats.json)

| subset   | #structures | energy key         | source xyz |
|----------|-------------|--------------------|------------|
| mad15    | 216,803     | atomization_energy | `${STORE_ROOT}/universal_features/mad-1.5/mad-1.5-r2scan-all.xyz` |
| mad_next | 421,640     | corrected_energy   | `.../phace-atomic/data/mad_next/mad_next.xyz` |
| mad_sol  | 39,011      | corrected_energy   | `.../phace-atomic/data/mad_sol/mad_sol.xyz` |
| **total**| **677,454** (17,960,928 atoms) | | |

Teacher feature dims: pet=512, mace=144, uma=128, dpa_omat=240, dpa_omol=240.
Some teachers are missing for part of the atoms (stored as NaN, filtered in loss):
mace 920k, uma 4.15M, dpa_omol 1.49M NaN atoms; pet and dpa_omat fully dense.

## How the pieces fit

1. `prep_features.py` reads the 3 source xyz (via `extxyz_to_memmap.py`) and the
   per-subset teacher `.npy`/`.npz` files, concatenates them in order
   [mad15, mad_next, mad_sol], normalizes each teacher, and writes `atomic_combined/`.
   For mad15 it reuses the fixed `val/test_indices_in_all.txt` from the mad-1.5
   store; mad_next/mad_sol get a 5%/5% random split (seed 42).
2. `submit.sh` copies `atomic_combined/` to node-local `/tmp/train` and runs
   metatrain with `options.yaml` (which points every `read_from` at `/tmp/train`).
3. Training ran to epoch 92 → `phace_atomic_ep92.pt` (in the parent `models/` dir).

## To re-run

Point `DATA_DIR` at this `atomic_combined/` (skips the prep step) and launch:

```bash
DATA_DIR=${STORE_ROOT}/mad-core/models/training/atomic_combined \
OPTS=${STORE_ROOT}/mad-core/models/training/options.yaml \
sbatch submit.sh
```

To rebuild `atomic_combined/` from scratch instead, run `prep_features.py`
(needs the 3 source xyz above + per-subset teacher features).

Note: reproducing the exact weights also requires the same metatrain / PhACE
code version used for the original run (`architecture.name: experimental.phace`).
