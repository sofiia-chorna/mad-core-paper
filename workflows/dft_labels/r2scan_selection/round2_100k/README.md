# BoTR round 2 — the next 100k

Extension of the BoTR ("Best of the Rest") selection by 100,000 structures for DFT
recalculation, decided 2026-08-20: MAD-2.0 grows only on the BoTR side, so nothing
upstream of it has to be redone. 100k rather than the ~88k minimum so some can be
lost to the LLPR filter afterwards.

## Rule — identical to round 1

Continue the same walk of the madcore FPS ordering where round 1 stopped
(`fps_rank` 122,378) and keep the next 100,000 structures that are

1. **not in the MAD union** (MAD-1.5 + MAD-NEXT + MAD-SOL — this is the same
   exclusion round 1 used, and it is what excludes mad-next), and
2. **under 300 atoms** (`n_atoms < 300`).

## Source — the release, not the old run parquets

Round 1 walked `madcore_10M/out/coreset_10M_full.parquet`, which no longer exists on
scratch. Round 2 walks `release/mad-core/` (frozen + validated), where row i = the
i-th FPS pick and rows [0, 856,012) are the accrete anchor in accrete order — the
same ordering round 1 used.

Correctness anchor: re-running the round-1 rule over the release reproduces the
stored round-1 selection **rank-for-rank** (all 100,000 ranks 1..122,378 identical,
`n_atoms` identical), and the MAD mask via id-join to the `mad-next-coreset` tier
equals the mask via `dataset_group == "mad-next"` everywhere in the walked window.
The script asserts both on every run.

**Id caveat:** the release ids were unified on 2026-08-14, so this round carries the
current `<dataset_group>:<hex32>` configuration_ids while the round-1 xyz carries
stale pre-unification cids. Join the two rounds (or round 1 to the release) by
`fps_rank`, never by cid.

## Result (`out/cscs-madcore_next100k_non-mad_max300.{xyz,parquet}`)

| | |
|---|---|
| structures | 100,000 |
| `fps_rank` range | 122,379 … **238,371** (still inside the 856k accrete anchor block) |
| walked / skipped | 115,993 / 15,993 (15,989 MAD, 4 with ≥300 atoms) |
| n_atoms | 1 … 271, mean 6.7 |
| periodicity | 3D 99,598 · molecules 400 · surfaces 2 |
| top families | Alexandria 31,853 · MP-ALOE 19,090 · JARVIS_QE_TB 17,377 · OMat24 12,656 |

Round 1 for comparison: Alexandria 42,302 · JARVIS_QE_TB 20,234 · MP-ALOE 12,825 ·
OMat24 8,054.

One selected structure (`fps_rank` 233,694, nomad) is a far-from-origin case whose
released coordinates are folded into the cell — physically identical under PBC and
fine for DFT; only byte-level reproduction of its stored features needs the original
unfolded coordinates (`release/mad-core` image flags).

The extxyz frames carry `fps_rank, configuration_id, dataset_id, family, n_atoms,
periodicity` and no prediction fields, matching the round-1 file. Positions/cells
come from the release binaries (float32, ~1e-5 Å — the published values).

## Files

| file | what |
|---|---|
| `01_select_round2.py` | the whole selection incl. the reproduction + integrity asserts |
| `out/` | the xyz + parquet index (gitignored; regenerable in ~2 min on a login node) |

Run on a login node: `uenv run pytorch/v2.9.1:v2 --view=default -- bash -c
"source ~/venv-phace/bin/activate && python 01_select_round2.py"`.
