# madcore_production — pool provenance & filters applied

Pool for the production accrete = **big-family FPS coresets** (`aa002/mad-core/fps/phace_atomic_ep92/<fam>/with_features`)

+ **small families whole** (`lp133/.../accrete_pools/madcore_prod/small/<fam>`). Anchor = MAD union
  `aa002/mad-core/mad_union/anchor.parquet` (856,012), union scaler `…/mad_union/scaler_madunion_phaceatomic_meanstd.pkl`.

## Filters applied (all verified — see pre-flight below)

1. **nsites ≤ 1000** — at the store/featurization level (larger structures excluded).
2. **finite features** — rows with any non-finite `phace_features_bb_mean` dropped (NaN/inf).
3. **ZBL cuts** — `zbl_max_force < 200` & `zbl_energy_per_atom ≤ 50` & `zbl_max_atomic_energy ≤ 100`.
   Applied at FPS time for big families (`fps.parquet.meta.json`: max_force 200 / epa 50 / atomicE 100) and at
   pool-build time for small families (`build_pool.py`). Consistent thresholds across the whole pool.
4. **nomad-Ahmetcik excluded** — 289,934 Ahmetcik entries removed from the `/nomad/` family (build_pool
   `not_ahmetcik_mask`). Verified 0 remain (below).
5. **configuration_id dedup** — keep-first per unique cid (content hash), small + big (`dedup_pool.py`).
6. **anchor overlap removed** — `mad-1.5`, `mad-next`, `Massive_Atomic_Diversity_MAD` deleted from the pool
   (they are the anchor seed, not candidates). `Massive_Atomic_Diversity_bench` intentionally KEPT.

## Pre-flight verification (2026-07-03)

- pool = **183,324,646 unique** (big 99,938,439 + small 83,386,207); anchor 856,012 → accrete input 184,180,658.
- manifest 1,568 shards — **0 missing / 0 empty / 0 unreadable**.
- **nomad-Ahmetcik cids present = 0** (5,057,529 nomad rows scanned).
- all **459** store datasets covered (0 missing); feature dim **512** (mean⊕std); union scaler loads (512-d).
- memory 377 GB → 4 nodes = 23.6 GB/GPU (safe).

## NOT in the pool (by design)

mad-1.5, mad-next, Massive_Atomic_Diversity_MAD (anchor). DFT labels (energy/forces/stress) are NOT in the
store — join selected cids back to `aa002/zbl_filtering/output` after selection to build the labeled set.

## Excluded NOMAD volume-scan entries

`excluded_nomad_volume_scans_entry_ids.txt` — the 289,934 NOMAD entry_ids excluded by
filter 4 above (2016 upload of equation-of-state volume scans of binary compounds; no
dataset DOI). `excluded_nomad_volume_scans_upload_ids.txt` lists the 21 NOMAD upload_ids
covering these entries. Referenced by the paper footnote in the "Atomistic datasets" section.
