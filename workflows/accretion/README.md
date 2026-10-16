The MAD-CORE ordering. anchor builds the candidate pool, 183,324,646 unique structures
after the ZBL cuts, the NOMAD volume scans and the duplicates are removed
(build_pool.py, dedup_pool.py), and accretes it into the MAD-NEXT ordering with
run_accrete.py on 4 nodes (submit_accrete.sh). fps_fill continues with plain FPS to 10 M and
then 25 M structures (submit_fps_continue.sh, submit_fps_25M.sh). cut_2pow24 cuts the first
16,777,216 rows, which are MAD-CORE, and holds the manifests of the pool and the excluded
NOMAD entry ids.

The raw FPS parquets and the enriched ones number fps_order differently, join on
configuration_id. The run logs of the accretion were not kept.
