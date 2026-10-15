# BoTR — "Best of the Rest"

The non-MAD structures picked out of the madcore coreset for DFT recalculation.
MAD-2.0 is composed as `MAD-1.5 + BoTR + MAD-NEXT`; BoTR is the non-MAD part, and it
is the only component that can grow without redoing the workflow upstream of it.

Selection rule (both rounds): walk the madcore FPS ordering from the start and keep
structures that are **not in the MAD union** (MAD-1.5 + MAD-NEXT + MAD-SOL — this is
what excludes mad-next) and have **fewer than 300 atoms**. Round 2 continues the same
walk where round 1 stopped, so the batches are disjoint and contiguous in FPS order.

| round | folder | fps_rank range | picked | walked | skipped |
|---|---|---|---|---|---|
| 1 (2026-07) | `round1_100k/` (was `accrete/runs/madcore_production/bor/`) | 1 … 122,378 | 100,000 | 122,378 | 22,378 |
| 2 (2026-08) | `round2_100k/` | 122,379 … 238,371 | 100,000 | 115,993 | 15,993 |
| 3 (2026-10) | `round3_20k/` | 238,372 … 258,372 | 20,000 | 20,001 | 1 |

**Data lives in the store**, this tree holds the code:
`${STORE_ROOT}/mad-core/botr/round{1,2}_100k/` and `round3_20k/` — the xyz + parquet
index of each batch, round 1's as-sent file and id map, a copy of the scripts, and
per-folder `SHA256SUMS`. The `out/` folders here are gitignored working copies.

Round 1 was originally built from the (since deleted) `coreset_10M_full.parquet`; it
was restored from `release/mad-core` on 2026-08-21 and verified against the file sent
to the collaborators — 99,995 of 100,000 frames identical atom-for-atom, the other 5
the documented far-from-origin folds (see `round1_100k/README.md`). Round 2 is built
from the release directly and reproduces round 1 rank-for-rank as its correctness
check.

**Ids:** round 1's as-sent file predates the 2026-08-14 id unification and carries
`CO_…` ids; everything else carries current `<dataset_group>:<hex32>` ids. Join
across rounds, or to the release, by **`fps_rank`** — never by configuration_id.
