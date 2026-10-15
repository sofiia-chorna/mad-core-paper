# madcore_25M — phase 3: FPS fill to 25M, the production lineage

**`out/coreset_25M_full.parquet`** (60 GB, 25,000,000 rows) is the deepest FPS run, and
**the published coreset224 is its first 16,777,216 rows**.

Verified: the first 300,000 ids of this file and of
`coreset224/out/coreset_224_full.parquet` are identical (300,000/300,000).

## Why the cut is at 2^24 and not 25M

`coreset224` takes 2^24 = 16,777,216. The 17–25M tail is distorted by OMat24 pool
exhaustion and is analysis-only — do not use it as a coreset.

## Pipeline

| phase | folder | output |
|---|---|---|
| 1. accrete | `../madcore_production` | `out/accrete.parquet` — 856,012 anchor |
| 2. FPS fill | `../madcore_10M` | `out/coreset_10M_full.parquet` — 10,000,000 |
| 3. FPS fill | **this folder** | `out/coreset_25M_full.parquet` — 25,000,000 |
| 4. production cut | store | `coreset224/out/coreset_224_full.parquet` — 16,777,216 |

All three enriched files share one consistent `fps_order` numbering. Note the *raw* FPS
parquets in `../madcore_10M/out/` (`coreset_10M.parquet`, `coreset_25M.parquet`) contain the
same structures under a **different** numbering — see that folder's README before joining on
`fps_order` across folders.

## Contents

`out/` also holds `cid_index_25M.parquet`, the light coreset, and category/family count
files. 62 GB total.
