# MAD reference data (phace-atomic ep92)

The MAD anchor for accreting the big datasets into MAD-core. Each subset is **one parquet**
holding structures + phace ep92 meanstd features (512-d) + `configuration_id` together.

| subset | dataset.parquet | rows | source structures |
|---|---|---|---|
| MAD-1.5  | `mad15/dataset.parquet`    | 216,803 | `/capstor/store/.../aa002/universal_features/mad-1.5/mad-1.5-r2scan-all.xyz` |
| MAD-NEXT | `mad_next/dataset.parquet` | 600,300 | `../data/mad_next/mad_next.xyz` + `mad_next/src/dimers_trimers/` |
| MAD-SOL  | `mad_sol/dataset.parquet`  |  38,909 | `../data/mad_sol/mad_sol.xyz` |

**MAD-NEXT update (2026-07-01):** dropped `moldimers`+`moltrimers` (61,340), added the new
dimers/trimers (240,000, all splits). Kept `newtrimers`, `madcat`, `flashmd_*`. Old feature
file backed up at `_backup/mad_next_pre_dimtri.parquet`.

## union/  — the MAD coreset (FPS ordering over the 3 datasets)
- `manifest.txt` — the 3 dataset parquets
- `submit_fps.sh` — full FPS (orders all 856,012); `fps.parquet` = result
- `anchor.parquet` — concatenated union features for `run_accrete.py --anchor-features` (regenerated post-FPS)

## Conventions
- `configuration_id` is minted by `../ingest/ids.py` (blake2b of canonical geometry + provenance),
  the same scheme used for every dataset. Verified to reproduce the original MAD-NEXT cids.
- Source xyz for MAD-NEXT/MAD-SOL live under `../data/` because the training pipeline
  (`prep_features.py`) reads them there. `data/` = training; `mad/` = accrete reference.
