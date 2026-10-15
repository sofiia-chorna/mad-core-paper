# Within-cell FPS ordering: method and strategy

Goal: a complete farthest-point-sampling (FPS) ordering of the population of every
level-24 Voronoi cell of coreset224 (2^24 = 16,777,216 cells; every collection
structure assigned to its nearest center in the scaled ecumetric metric). The rank
of a structure inside its cell extends the nested coreset hierarchy below the last
coreset level: taking the first m ranks of every cell yields a balanced, maximally
diverse refinement of the full collection.

## Metric and pool

- Metric: squared Euclidean distance on `(features - mean) / scale` with the
  released ecumetric scaler (mean (512,), scalar scale). The scalar scale makes the
  FPS ordering invariant to the scaling; all reported `eps_sq` values are in the
  scaled metric, directly comparable to the assignment distances `d24` and to the
  coreset `epsilon_sq` curve.
- Pool: all clean distinct release rows — collection rows with `filter_code == 0`
  and `is_duplicate == false`, plus the 856,012 mad_coreset rows (MAD ships only as
  a coreset tier, so those rows are not in the collection). Total pool
  1,694,597,952 rows scanned, of which the clean distinct subset is ranked.
- Exclusions: `nan_features` rows cannot be ranked (no features exist).
  Duplicates are excluded because `configuration_id` is content-addressed: a
  duplicate is the same structure, its distance to its twin is featurization
  jitter (~1e-7), and FPS would place all duplicates in a meaningless
  noise-ordered tail. A duplicate's rank is defined by inheritance: join to the
  ranked row with the same `configuration_id`.

## Seeding and epsilon semantics

Each cell's FPS is seeded with its own center (the coreset224 pick): the selected
set is initialized with the center's feature vector, so rank 0 onward continues
the hierarchy "given that the center is already selected".

- If the center's own row is in the pool (the normal case; centers self-assign),
  it is forced to rank 0 and `eps_sq[0]` is its recomputed distance to the center
  feature (0 up to f32 jitter).
- If the center's row is absent (rare filtered edge case), the selected set still
  starts from the center feature and rank 0 is the farthest member from it.
- `eps_sq[k]` = squared distance of the rank-k pick to the previously selected set
  (center included) at the moment of picking, i.e. the standard non-increasing FPS
  radius curve of the cell. `eps_sq[1]` equals the squared cell radius
  (max member `d24`), which is verified against the independently computed
  assignment distances.

## Scale statistics and three-regime strategy

Level-24 populations are extremely skewed: 29% of cells are singletons, the median
is 3, 99% of cells have <= 256 members — but 87 mega-cells with >= 2^21 members
hold 798.5M points, topped by the 115.0M-member cell. A full ordering costs
sum(n_i^2) ~= 2.8e16 pair distances, half of it in the single largest cell. One
implementation cannot serve all three regimes:

1. **Batched small cells (n <= 512).** Millions of independent tiny FPS problems
   are launch-latency-bound, not bandwidth-bound. Cells are sorted by size and
   packed into padded (B, n_max, 512) batches (padding bounded 2x by size-class
   batching); one fully vectorized torch FPS iterates all cells of a batch
   simultaneously, finished cells masked with -inf.
2. **Single-GPU kernel path (512 < n < 2^21).** The custom CUDA kernels are driven
   with a fully device-side loop: two-phase deterministic argmax, center commit,
   active-center marking, masked triangle-bound (dSL) computation, Voronoi-skip
   Hausdorff update. The winning index and epsilon are captured on device each
   pick; there is no host synchronization inside the loop.
3. **Sharded multi-GPU path (n >= 2^21, 87 cells).** `fps_sharded` with
   round-robin-sharded X_sel (`CORESETS_FPS_SHARD_XSEL=1`): both the feature
   matrix and the per-pick dSL bound computation are divided across ranks and the
   small dSL vector is all-gathered. Output is byte-identical to the replicated
   and single-GPU paths. Checkpoint/resume every 4M picks makes the largest cells
   walltime-robust.

Rows are staged for this by one shuffle pass over the release binaries: features
are scaled once, clean rows routed to 128 hash buckets (cell mod 128, one GPU
task per bucket) and to 87 per-cell part directories for the sharded jobs.

## Cost model: why Voronoi-skip, and why sharding

The FPS update step is a bandwidth-bound GEMV (one new center against all rows) —
there is no GEMM data reuse to preserve, unlike the assignment problem where
brute-force GEMM beat pruning. Per step and per row, the dense update reads the
2 KB feature row; the Voronoi-skip update first reads a 16-byte warp-uniform bound
(hausdorff, cell id, dSL of the row's current cell) and skips the row read when
the triangle inequality proves the update cannot improve.

The growing term is the per-pick dSL bound recomputation, O(n_sel * D). Two
mitigations: (a) marking active centers first restricts it to cells that still
contain unselected rows; (b) in the sharded path each rank computes only its own
centers' bounds — the one term that would otherwise not scale with world size is
divided by it.

**Measured (A/B on a real 434k-member cell, full ordering, one GH200):** dense
69.9 s vs Voronoi-skip 76.0 s — for FULL orderings the skip machinery is
break-even to slightly negative (0.92x), because at depth nearly every selected
cell still contains unselected rows, so the masked dSL pass approaches the cost
the skip saves. This mirrors the assignment-time finding that brute force is hard
to beat. The skip path is kept everywhere regardless: the production coreset FPS,
the sharded driver and this pipeline all share it, so orderings are reproducible
across reruns and world sizes within one code path. Note dense and skip orderings
are NOT bit-identical: skipping changes f32 rounding of the update stream, so
near-degenerate ties can resolve differently — both results are exact FPS
orderings up to f32 rounding; a per-cell run is deterministic and reproducible.

**Measured throughput:** batched+kernel bucket pass: 847.0M rows / 16,777,129
cells in 31.2 GPU-hours total (all cells < 2^21). Single-GPU kernel path:
0.29-0.43 ns per point-pair at n ~ 0.5-2M. Sharded path on one node (4 ranks):
n = 2.18M in 1,100 s, 4.67M in 3,701 s, 8.58M in 11,120 s — apparent t ~ n^1.9,
~0.6 ns per pair (at n ~ 2M the single-GPU path is ~3x cheaper in GPU-seconds;
sharding buys wall-time and memory, not efficiency).

**The two-term cost law (and why n^1.9 misleads).** At the largest cells the true
per-pick cost separates cleanly into two terms, fitted live on 8 nodes / 32 ranks:

    cost per pick = a + b * n_sel,   a ~ 2.10 ms, b ~ 0.053 ns   (n = 115.0M)
    cost per pick = a + b * n_sel,   a ~ 1.43 ms, b ~ 0.054 ns   (n =  71.0M)

`a` is exactly the dense row scan: n/world * 2 KB at ~3.5 TB/s HBM predicts
2.10 ms and 1.30 ms — i.e. the Voronoi skip saves almost nothing at depth, the
same conclusion as the 434k A/B. `b * n_sel` is the dSL bound pass, the skip
machinery itself; it is negligible below ~10M picks but reaches 6 ms/pick at
n_sel = 115M, tripling the cost of a pick. Integrated, a full ordering costs
a*N + (b/2)*N^2 — asymptotically QUADRATIC in the cell size on top of the
N * (row-scan) term, not n^1.9. Fitting scaling on cells <= 8.6M (where `a`
dominates) underestimated the two largest cells by ~3x: measured/projected
1,310 node-hours for the 115M cell versus 428 predicted.

Consequence for anyone repeating this: for very large FPS problems the dSL/Voronoi
bound is a net cost, and a dense update (drop the bound and its per-pick
all-gather) is the cheaper exact algorithm. The two largest cells were therefore
switched to a dense sharded driver (`fps_dense_sharded.py`) after ~25% and ~48%
of their picks: same argmax / pack / all-gather / commit structure as
`fps_sharded`, but `fps_update_cuda` (dense) replaces the
dSL-bound + `voronoi_update_cuda` pair, which also removes one collective per
pick and the entire replicated/sharded X_sel array.

**Validation:** run from scratch on an already-completed mega cell (16490638,
2.10M members), the dense driver reproduced the production skip-path ordering
BIT-IDENTICALLY over the first 200,001 picks (100.0000%, no divergence). The two
paths are the same algorithm; only the work skipped differs.

**Measured effect:** dense holds a FLAT pick rate (2,309 -> 2,355 pick/s over
200k picks on 4 GPUs, rising slightly as the argmax scan shortens) where the skip
path decays monotonically with n_sel. Projected cost for the remaining picks of
the two largest cells: 522 node-hours dense versus 1,502 on the bound path.

Because a resume needs the Hausdorff state, and rebuilding it costs
O(N * k * D) (~30-45 min at k = 28M on 8 nodes), the dense driver also
checkpoints the per-rank Hausdorff array alongside the pick list; only the first
resume after the switch pays the rebuild, subsequent chain links load it
directly. Reloading requires the same world_size and shard layout, which is
asserted against the stored array length.

## Determinism and precision

All distance updates run in f32 with the norms identity ||a-b||^2 =
||a||^2 + ||b||^2 - 2ab, matching the production coreset FPS. The custom argmax
kernel breaks ties deterministically toward the smaller index, making single-GPU,
multi-GPU, and resumed runs byte-identical; the batched torch path is
deterministic per cell (first-occurrence argmax). Epsilon values are clamped at 0.

## Resume and the epsilon monotonicity artifact

Epsilon is non-increasing by construction within a continuous run (the picked value
is the max of an array that only shrinks). Across a RESUME boundary it can rise by
a few 1e-7: the Hausdorff state is recomputed from the selected set with chunked
GEMM, whereas the in-loop update uses warp dot products, so the two disagree in the
last fp32 bits and the max over ~1e8 perturbed values can land marginally higher.
Observed on the 115M cell: exactly 2 steps of 114,974,137, at ranks 16,000,000 and
28,000,000 (its two resume points), of size 8.9e-7 and 9.5e-7 on eps^2 ~ 3e-3
(~3e-4 relative, ~8e-6 in distance against a 0.215 cell radius). The size is what
fp32 predicts: a 512-term dot accumulates ~2.7e-6 relative error, amplified by the
cancellation ratio ||x||^2/d^2 ~ 330. It is not driver-specific — the rank-16M
instance occurred on a `fps_sharded` resume. The pick set remains an exact
permutation of the pool in every case.

Consequence: a resumed run is not bit-identical to an uninterrupted one, because
near-tied argmax comparisons can flip after the state rebuild. Both are valid FPS
orderings; exact reproduction requires the same resume points. The final eps of a
full ordering can also be slightly negative (~-8e-7) since the last pick's true
distance is zero and the norms identity cancels.

## Verification gates

Per bucket / per mega cell, recorded in the run summaries:
- every pool row ranked exactly once (rank arrays are permutations);
- `eps_sq` non-increasing from rank 1 (exact FPS invariant, checked with f32 slack;
  isolated violations at resume boundaries are the artifact described above, not a
  defect — check that flagged steps coincide with checkpoint steps);
- `eps_sq[1]` equals the squared max member `d24` from the independent assignment
  pipeline. The tolerance needs an ABSOLUTE term as well as the 5% relative one:
  the production run flagged 1,126 of 16,777,216 cells (0.0067%) on the relative
  test alone, and characterising them showed every single one is a near-degenerate
  cell — radius 0.0100-0.0200 (none above 0.02, against a 0.218 coreset
  separation), median 7 members, worst absolute disagreement 2.3e-5. Both sides
  are f32 squared distances produced by different routes, so ~1e-5 of noise is
  expected and is >5% relative at radius^2 ~ 1e-4. The check now requires
  `rel > 0.05 AND abs > 1e-4`; summaries written by the production run predate
  this and carry the relative-only counts;
- the center row, where present, holds rank 0;
- sharded runs additionally: seed is the center row, selected is a permutation,
  all rows carry the expected cell id.
Globally (`verify_cell_fps.py`): shuffle row accounting matches the release
(1,694,597,952 rows), and per-cell ranked counts match the per-cell clean-row
counts accumulated independently during the shuffle. Separately
(`check_written_output.py`) the WRITTEN parquets are re-read and checked
structurally — rows grouped by cell, every cell's ranks exactly 0..n-1 in row
order, no repeated configuration_id within a cell — which is what catches
output-path bugs that in-memory assertions cannot.

## Final result (2026-08-02)

All 16,777,216 level-24 cells ordered; **1,642,071,166 rows ranked**
(846,992,684 in 16,777,129 bucket cells + 795,078,482 in the 87 mega cells),
exactly matching the independently accumulated clean-distinct pool count. Zero
per-cell count mismatches, zero cells missing, zero mega cells failing a
structural gate, and the written parquets pass the structural re-read.
`duplicates_inherited.parquet` gives 52,040,439 of 52,053,362 duplicate rows the
rank of their content-identical twin; the 12,923 unmatched are cids whose only
non-duplicate occurrence was itself filtered, so no copy ever entered the pool.

The single remaining flag class is `radius_mismatch_cells`: 1,126 cells of
16,777,216 (0.0067%) where `eps_sq[1]` disagrees with the stored `d24`-derived
cell radius by more than 5%. Investigated and closed — an independent float64
recomputation of the max distance to the center agrees with `eps_sq[1]` to
0.2-0.7%, while the stored fp32 `d24` differs by up to 12%. The error is in the
comparison reference (the assignment pipeline's fp32 distances suffer
cancellation at radii ~0.01), not in the cell FPS. The gate is a useful
cross-check but must not be read as a defect at this distance scale.

## Output

`voronoi/out/cell_fps/fps/` — one parquet per bucket and per mega cell, rows
sorted by (cell, cell_rank): `cell` (int32, center fps_order), `cell_rank`
(int32), `eps_sq` (float32, scaled metric), `configuration_id`, `dataset_group`,
`dataset_id`. Accounting sidecar JSON per file.
