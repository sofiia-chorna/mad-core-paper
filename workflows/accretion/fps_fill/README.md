# madcore_10M — phase 2: FPS fill to 10M

Continues FPS from the 856,012 anchor built in `../madcore_production`.

## Which file to use

**`out/coreset_10M_full.parquet`** (24 GB, 10,000,000 rows, 20 columns). This is the
enriched output and it is on the **production lineage**: its first 200,000 ids are identical
to `coreset224`, and 122,379/122,379 identical over the BoR walk range.

## Two families of file here, and they number differently

There are two kinds of coreset parquet in `out/`, and **they are the same structures with
different `fps_order` numbering**. Joining on `fps_order` across the two gives silently
wrong ranks.

| file | rows | cols | `fps_order` numbering |
|---|---|---|---|
| `coreset_10M_full.parquet` | 10,000,000 | 20 | **production** — matches coreset224 |
| `coreset_10M.parquet` | 10,000,000 | 6 | raw FPS |
| `coreset_25M.parquet` | 25,000,000 | 6 | raw FPS |

Verified: the two raw files agree with each other 200,000/200,000, and the enriched file
agrees with production 200,000/200,000, but raw vs production matches only 1/200,000.

**They are not different selections.** The id *sets* are identical at every prefix tested
(856,012 / 2,000,000 / 10,000,000 — 100% overlap each time). Only the rank numbers differ.
The raw files are stored physically in `fps_order`; the enriched ones are stored in another
row order with `fps_order` as a column. Both are valid `0..N-1` permutations.

Telling them apart: the enriched files have 20 columns including `source_block` and `k_set`;
the raw ones have 6 and neither.

`coreset_25M.parquet` here is **not** the production 25M — that is
`../madcore_25M/out/coreset_25M_full.parquet`.

## Disk

232 GB, of which **204 GB is 1,677 `out/fpsckpt10m_*` files** — FPS resume checkpoints from
a finished run whose output is already on the store. Almost certainly reclaimable, but
nothing has been deleted.

## Other outputs

`recalc_100k/first100k_nonMAD_lt300atoms.{xyz,parquet}` — the BoR selection (100,000
non-MAD structures under 300 atoms). Built by `extract_recalc_100k.py`; a copy of that
script plus the follow-up analysis lives in `../madcore_production/bor/`.
