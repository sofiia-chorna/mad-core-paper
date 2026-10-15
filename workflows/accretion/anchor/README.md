# madcore_production — phase 1: the accrete anchor

**The name is misleading: this folder does not hold the production coreset.** It holds the
**anchor** that seeds it, plus the pool build and the analysis scripts.

| what | where |
|---|---|
| the anchor, 856,012 structures | `out/accrete.parquet` (29 MB) |
| pool construction | `submit_build_pool.sh`, `build_pool.py` |
| the production coreset (2^24) | **not here** — see the pipeline below |

## Pipeline

| phase | folder | output |
|---|---|---|
| 1. accrete | **this folder** | `out/accrete.parquet` — 856,012 anchor |
| 2. FPS fill | `../madcore_10M` | `out/coreset_10M_full.parquet` — 10,000,000 |
| 3. FPS fill | `../madcore_25M` | `out/coreset_25M_full.parquet` — 25,000,000 |
| 4. production cut | store | `coreset224/out/coreset_224_full.parquet` — 16,777,216 = 2^24 |

The cut is at 2^24 rather than 25M because 17–25M is distorted by OMat24 pool exhaustion
and is analysis-only.

Verified: the first 300,000 ids of `coreset_25M_full` and `coreset224` are identical, and
`coreset_10M_full` agrees with both.

## Subfolders

- `bor/` — MOVED to the repo root: `botr/round1_100k/` (the 100k non-MAD selection for DFT
  recalculation + the failed-flashmd check). Round-2 selection lives in `botr/round2_100k/`.
