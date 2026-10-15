# Nested Voronoi assignment of the full collection to coreset224

Goal: assign EVERY point of the full featurized collection (~1.8B structures, both store roots
+ the MAD-union anchor) to its nearest center among the first 2^k picks of the production
coreset (`store/…/coreset224/out/coreset_224_full.parquet`), for every level k = 0…24.
The result is a 25-level nested Voronoi tessellation of the whole library — per-level cell
populations, cell radii, and a resolution-conditional map of the collection.

Metric: Euclidean distance in the scaled phace-atomic ep92 space — raw
`phace_features_bb_mean` (512-d) transformed with the SAME global scaler used for accrete/FPS
(`mad_union/scaler_madunion_phaceatomic_meanstd.pkl`: x → (x − mean_) / scale_). Centers and
query points are scaled identically; distances are fp32.

---

## 1. Why brute force is wrong and what structure we exploit

Brute force is Σ_k 2^k ≈ 3.4·10^7 distance evaluations per point (≈ 3·10^19 FLOPs over 1.8B
points) — borderline feasible on many GPUs but wasteful and not the point. The coreset is not
an arbitrary center set: it is a **nested FPS/accrete ordering** with the sep ≥ cov guarantee at
every level. That makes {C_0 ⊂ C_1 ⊂ … ⊂ C_24} (C_ℓ = first 2^ℓ picks) a **navigating net /
cover-tree-like hierarchy** in the metric space, and nearest-center search over ALL levels can
be done in one top-down descent with triangle-inequality pruning — O(beam · 25) distances per
point instead of O(2^24). This descent is the scientific object: the same structure that makes
the library nested makes it *searchable*, and any future dataset can be assigned at the same
polylogarithmic cost.

Related prior art: the Voronoi-FPS trick (only cells with d(center, new-center) < 2·cell-radius
can lose members — Imbalzano et al., J. Chem. Phys. 149, 241730 (2018); scikit-matter
`VoronoiFPS`); Elkan/Hamerly k-means bounds (lower bounds from center-center distances);
cover trees / navigating nets (Beygelzimer et al. 2006; Krauthgamer & Lee 2004) whose packing
arguments bound the beam by the doubling constant; sketch-map's landmark Voronoi assignment of
out-of-sample points (Ceriotti, Tribello, Parrinello, JCTC 9, 1521 (2013) — the PDF in
`papers/`). Our contribution: the net is FREE (it is the production coreset itself), the levels
are the deliverable, and the descent yields the exact per-level assignment for all 25 levels in
one pass.

## 2. The center net (Phase A — over the 2^24 centers only)

Centers in coreset224 fps_order (anchor block in accrete-hierarchy order — the canonical
nesting). Level of birth of center i: ℓ(i) = ceil(log2(i+1)) (c_0 is level 0).

Build, level by level ℓ = 0…23, using the descent of §3 restricted to the levels built so far:
- **parent** p(c) of each center c born at level ℓ+1 = its exact nearest neighbor in C_ℓ,
  with distance dp(c) = d(c, p(c)).
- **children CSR** per level: child_{ℓ+1}(j) = {c born at ℓ+1 : p(c) = j}, stored as
  (indptr over j < 2^ℓ, child ids, child dp) — one CSR per level.
- **pcd_ℓ = max_c born at ℓ dp(c)** (max parent distance of the level), and the suffix sums
  **S_ℓ = Σ_{m ≥ ℓ} pcd_m**. With FPS, dp(c) ≤ cov(C_ℓ) ≈ eps_ℓ which decays geometrically, so
  S_ℓ = O(eps_ℓ) — this is what keeps beams small.

Output: `out/tree/` — scaled center features (memmap fp32 (2^24, 512)), per-level CSR arrays,
pcd/S arrays, parent/dp arrays (npz + .pt). Cost: 16.7M descents over ≤ 24 levels ≈ minutes of
GPU; one GH200 (features 34 GB resident).

## 3. The descent (per query point x; exact per-level NN, provably)

State: beam B_ℓ = evaluated centers of C_ℓ with their exact distances; d*_ℓ = min distance so
far; a_ℓ = argmin (per-level assignment, recorded for every ℓ).

- Init: evaluate d(x, c_0); B_0 = {c_0}.
- Step ℓ → ℓ+1:
  1. **Candidates** K = ∪_{c ∈ B_ℓ} child_{ℓ+1}(c).
  2. **Elkan pre-prune** (no feature access): for candidate c with parent p (p ∈ B_ℓ, d(x,p)
     known; dp(c) stored): lower bound lb(c) = |d(x,p) − dp(c)|. Discard c if
     lb(c) ≥ d*_ℓ + S_{ℓ+2} (it can neither win nor matter for deeper levels).
  3. Evaluate surviving candidates exactly; update d*_{ℓ+1}, a_{ℓ+1} (keep old center on ties —
     matches first-argmin over ascending fps_order, since older ids are smaller).
  4. **New beam** B_{ℓ+1} = {c ∈ B_ℓ ∪ evaluated : d(x,c) ≤ τ_{ℓ+1}}, τ_{ℓ+1} = d*_{ℓ+1} + S_{ℓ+2}.

**Correctness** (invariant: B_ℓ ⊇ {c ∈ C_ℓ : d(x,c) ≤ d*_ℓ + S_{ℓ+1}}): any c′ ∈ C_{ℓ+1} with
d(x,c′) ≤ d*_{ℓ+1} + S_{ℓ+2} is either old — then d(x,c′) ≤ d*_ℓ + S_{ℓ+1} so c′ ∈ B_ℓ — or
born at ℓ+1 with parent p, d(x,p) ≤ d(x,c′) + pcd_{ℓ+1} ≤ d*_ℓ + S_{ℓ+1}, so p ∈ B_ℓ and c′ ∈ K.
In particular the exact NN at every level is always found (d*_{ℓ+1} ≤ τ). The suffix-sum slack is
the price of guaranteeing an unbroken ancestor chain down to level 24 (navigating-net argument);
with geometric eps decay S_ℓ ≈ 2–5 × eps_ℓ, and the beam is contained in ball(x, d* + S) whose
packing at separation sep_ℓ ≈ eps_ℓ is bounded by the doubling constant — a small, dimension-
dependent number measured in the pilot (expected O(10–100)).

**Engineering:** beam capped at W = 512 per point (fp32 dist + int32 id); candidates gathered per
batch via the CSR (loop over beam slots, flat torch ops, no per-point Python); full center matrix
(34 GB) resident on GPU so candidate features are direct row gathers. Points whose beam overflows
W are flagged and finished by exact brute force at the affected levels (expected ≪ 1%; measured).

## 4. The sweep (Phase B — 1.8B points)

- Input: `out/manifest_all.tsv` (all shards of both store roots + anchor.parquet; features
  column verified present). Workers process disjoint shard slices (`shards[task::n_tasks]`),
  fully independent (no collectives), resumable (skip shards whose output exists).
- Batch size 131,072 points; per batch: read `configuration_id` + features → GPU → scale →
  descent → write.
- Output (per input shard, mirrored layout): `out/assign/<family>/<subdir>__<stem>.parquet`
  with columns `configuration_id`, `row` (int32, row in source shard), `a00…a24` (int32
  fps_order of the assigned center per level), `d00…d24` (fp32 distance per level). Levels are
  the columns, so any "centers for 2^k" view is a column projection; zstd compresses the highly
  redundant a-columns well. ~0.3–0.5 TB total.
- Per-level center lists (tiny): `out/centers/centers_l{k}.parquet` (fps_order,
  configuration_id) for k = 0…24 — written by Phase A.
- Cost estimate: compute is trivial after pruning (~10^3–10^4 distance evals/point); the sweep
  is **IO-bound**: ~3.7 TB of features decoded. With 64 single-GPU workers (16 nodes) and
  multithreaded arrow decode, wall time is a few hours.

## 5. Optional accelerations (documented, not in v1)

- **Per-family FPS warm start** (`store/fps/phace_atomic_ep92/<family>/`): the 10 big families
  have their own FPS orderings; a member's nearest same-family FPS pick gives an upper bound
  d(x, C_24) before the descent (tightens τ from level 0). Requires the per-point nearest-pick
  distance from those runs (not currently materialized) — skip in v1, mention in paper as the
  "preselection" variant.
- fp16/bf16 candidate evaluation with fp32 re-check of near-ties.
- Level-0..10 in one shot (2^10 = 1024 centers brute force per point is one small matmul),
  descent from level 10 — trades exactness bookkeeping for fewer levels; v1 descends from 0.

## 6. Verification

`verify_sample.py`: 100k random points across random shards; exact brute-force NN (GPU, chunked
over all 2^ℓ centers) at levels {0, 4, 8, 10, 13, 16, 20, 24} compared to the descent output —
ids must match exactly (ties resolved to lowest fps_order in both) and distances to 1e-5 rel.
Runs automatically after the pilot.

## 7. Execution sequence (today)

1. `scripts/build_manifest.py` — running. Confirms the ~1.8B total.
2. `scripts/build_center_tree.py` (job `vor-tree`, 1 GPU, ~1 h): net + per-level center lists.
3. `scripts/verify_sample.py` (job `vor-verify`, afterok tree): pilot correctness + beam-width /
   overflow statistics → sets the final W.
4. `scripts/assign_shards.py` via `submit_assign_array.sh` (array of 64 single-GPU tasks,
   normal partition, 12 h, resumable) — LAUNCH AFTER PILOT REVIEW.

## 8. Deliverables for the paper

- The descent algorithm + proof sketch (§3) — "the nested coreset is its own search index".
- Beam-width distribution & pruning factors vs brute force (measured on 1.8B points).
- Per-level Voronoi statistics: cell populations (density at 25 resolutions), cell radii
  vs eps_ℓ (empirical coverage of the FULL collection, not just the pool — directly tests how
  well the 184M-pool coreset covers the 1.8B universe), per-dataset composition per cell.
- Assignment of any future dataset in O(25 · beam) distances/point.
