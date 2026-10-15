# BoTR round 3 — the next 20k

Extension by 20,000 structures (BoTR picks 200,001 … 220,000), requested 2026-10-03 by
the DFT collaborators because some round-1/2 structures did not converge.

## Rule — identical to rounds 1 and 2

Continue the same walk of the madcore FPS ordering over `release/mad-core/` where round 2
stopped (`fps_rank` 238,371) and keep the next 20,000 structures that are not in the MAD
union and have `n_atoms < 300`.

Correctness anchor: the script re-runs the rule from rank 0 and asserts it reproduces the
stored round-1 + round-2 selections (store parquets in `aa002/mad-core/botr/`) rank-for-rank,
and that the MAD mask via id-join equals `dataset_group == "mad-next"`.

## Result (`out/cscs-madcore_next20k_non-mad_max300.{xyz,parquet}`)

| | |
|---|---|
| structures | 20,000 |
| `fps_rank` range | 238,372 … **258,372** (inside the 856k accrete anchor block) |
| walked / skipped | 20,001 / 1 (0 MAD, 1 with ≥300 atoms) |
| n_atoms | 1 … 254, mean 6.9 |
| periodicity | 3D 19,904 · molecules 96 |
| top families | Alexandria 6,229 · MP-ALOE 3,780 · JARVIS_QE_TB 3,132 · OMat24 3,013 |
| folded far-from-origin | none |

No MAD row falls in ranks 238,372 … 258,373 (they cluster in the accrete order: 24,614 MAD
rows sit in 258k … 400k), hence the near-zero skip count.

Same extxyz info fields as round 2 (`fps_rank, configuration_id, dataset_id, family,
n_atoms, periodicity`), current unified ids. Join rounds by `fps_rank`.

## Files

| file | what |
|---|---|
| `01_select_round3.py` | the whole selection incl. the reproduction + integrity asserts |
| `out/` | the xyz + parquet index (gitignored; regenerable in ~2 min on a login node) |

Run on a login node: `uenv run pytorch/v2.9.1:v2 --view=default -- bash -c
"source ~/venv-phace/bin/activate && python 01_select_round3.py"`.
