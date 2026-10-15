# Stage C design review — direct fit of the 2^20

## 1. Their numbers: all confirmed

Re-derived independently (`src/check_2p20.py`, 24k-point weighted subsample, GPU fp64):

| claim | theirs | measured | verdict |
|---|---|---|---|
| N_eff at wgamma=0.25 | 698,224 / 1,048,576 (66.6%) | **698,224 (66.6%)** | exact |
| distance span (5–95%) | 2.8x | **2.80x** | exact |
| pair-weighted peak | 1.38 | **1.397** | confirmed |
| sigma = 0.9 x peak | 1.24 | **1.257** | confirmed |
| mid-sigmoid at sigma=1.24 | 96.6% | **96.8%** | confirmed |

The sigma-must-come-from-the-weighted-histogram argument is right, and their value is right.

## 2. The verification kills the premise

At sigma=1.24, wgamma=0.25, A=4, B=2 the pair-weight splits:

```
close (F<0.1)   0.0%
far   (F>0.9)   3.2%
mid            96.8%
```

**close = 0.0%.** The sigmoid never once says "this pair is close". For comparison:

| set | close | far | mid | outcome |
|---|---|---|---|---|
| 2^14 control A (sigma=3.0) | 0.1% | 8.5% | 90.2% | map == PCA (vsPCA 0.913) |
| 2^14 W2 (sigma=2.0) — best | ~0% | 27.1% | ~73% | best map we have (1.42x) |
| **2^20 (sigma=1.24)** | **0.0%** | **3.2%** | **96.8%** | **more inert than the broken control** |

Raising sigma does not rescue it — at sigma=2.0 the 2^20 gives close=4.6%, far=0.9%, mid=94.5%.
The cause is structural: the 2^20 distance span is **2.80x** (5th->95th), *narrower* than the
level-14 landmarks' 3.50x, and a soft sigmoid needs ~6x of range to traverse 0->1. More
landmarks => more distance concentration => less filtering. This is the curse of
dimensionality working against us, and no parameter fixes it.

Their own prediction ("the direct fit may come out more PCA-like, not less"; "if Stage C
lands at ~1.67x it has not moved off its init") is therefore **not a risk — it is the
expected outcome, derivable before spending a GPU-hour.**

## 3. Feasibility: possible, but skmatter's `fit` is unusable

Not a patch job. `fit` materialises N x N **three times** before the optimiser is ever called:

| line | allocation | at N=2^20 (fp64) |
|---|---|---|
| 658 | `cdist(X, X)` | 8.8 TB |
| ~692 | `sigmoid_transform(hd_distances)` | 8.8 TB |
| 710 | `np.outer(w, w)` | 8.8 TB |
| 711 | `np.triu(weights, k=1)` | 8.8 TB (temp) |

`torch_patch.py` replaces `_stress_and_grad` only — which runs *after* all four. **It does not
help at 2^20.** Stage C requires a fresh tiled implementation of the whole objective.

## 4. The tiled formulation (what a new implementation would do)

Two facts make it tractable:

- **`total_weight` has a closed form.** `sum_{i<j} w_i w_j = 0.5 * (1 - sum_i w_i^2)` when
  `sum_i w_i = 1`. No N x N needed. (Removes the line-710/711 blocker outright.)
- **F(D) is constant across iterations** — only Y moves. So it is cache-or-recompute, never
  re-derive.

Per gradient eval, for row-block I and column-chunk C:
```
D_IC = cdist(X_I, X_C)        # 512-d   <- the cost
F_IC = sigmoid(D_IC, sigma, A, B)
d_IC = cdist(Y_I, Y_C)        # 8-d     <- cheap
f_IC, f'_IC = sigmoid_fdf(d_IC, sigma, a, b)
stress_I += sum(w_I w_C [(1-m)(F-f)^2 + m(D-d)^2])
grad_I   += -2/tw * sum_C c_IC (Y_I - Y_C),  c = w_I w_C [(1-m)(F-f) f' + m(D-d)] / d
```
Y is replicated (N x 8 fp32 = **34 MB**), so rank 0 can run L-BFGS and broadcast Y each eval;
ranks reduce their grad blocks back. Column chunking is mandatory: one rank's full row-block
at 64 GPUs is 16384 x 1048576 = **68 GB fp32**, so chunk C to ~65536 (4.3 GB).

## 5. Cost

Dominant term is the D recompute: `2 N^2 x 512 = 1.13e15 FLOP/eval`. Low-D side is
`2 N^2 x 8 = 1.8e13` (~60x cheaper). Assume ~60 TFLOP/s realistic fp32 per GH200.

| config | per eval | 3000 evals | notes |
|---|---|---|---|
| 4 GPUs (1 node), recompute | 4.7 s | **~4 h** | simplest; no multi-node |
| 16 GPUs (4 nodes), recompute | 1.2 s | ~1 h | multi-node NCCL |
| 64 GPUs (16 nodes), recompute | 0.29 s | ~15 min | multi-node NCCL |
| 64 GPUs + F(D) cached fp16 | ~0.05 s | **~3 min** | cache = 2.2 TB = 34 GB/GPU; fits 96 GB only at 64 GPUs |

Eval count is the big unknown: skmatter runs MDS-refine (skipped with `init`) + main (1000
steps) + global annealing (5 mixing stages). The 2^14 d=2 fit used ~1300 evals; 2000-6000 is
the plausible band, so scale the table accordingly.

**Queue reality (checked): 807 jobs pending on `normal`, 303 running; `debug` is mostly
down/drained/reserved.** A 16-node request is a bad bet on wall-clock even though its compute
is the cheapest. Short jobs backfill; long multi-node jobs do not.

## 6. Recommendation: ladder up, do not jump to 2^20

The question Stage C asks is *"does the direct fit move off PCA?"*. That question does **not
need 2^20 to answer** — and answering it at 2^20 first is the expensive way round.

Direct fit is cheap at smaller N because cost goes as N^2:

| N | pairs | F(D) fp16 cache | fits on | est. fit |
|---|---|---|---|---|
| 2^16 = 65,536 | 4.3e9 | **8.6 GB** | 1 GPU | minutes |
| 2^17 = 131,072 | 1.7e10 | 34 GB | 1 GPU | minutes |
| 2^18 = 262,144 | 6.9e10 | 138 GB | 2 GPUs | ~10 min |
| 2^20 = 1,048,576 | 1.1e12 | 2.2 TB | 64 GPUs | see above |

So: implement the tiled fit **once**, validate it at 2^14 against the existing runs (D = 1.75x,
and the 09_skmatter_compare fixture at n=1024), then run 2^16 -> 2^17 -> 2^18 **on one node**
and plot `vsPCA` and family locality against N.

- If vsPCA **rises toward 1.0** and family locality converges to PCA's as N grows — which the
  filter statistics predict, since mid-sigmoid goes 90.2% (2^14) -> 96.8% (2^20) — then 2^20
  is answered, for ~1 node-hour instead of 16 nodes, and the finding is *stronger* because it
  is a trend rather than a single point.
- If the map **does** move off PCA at 2^18, that justifies the 64-GPU run, and by then the
  implementation is already validated at four values of N.

Either way the tiled implementation is written and reusable, nothing is wasted, and we never
debug at 2^20 (which their doc also warns against).

## 7. Open items that should be settled first

- **wgamma is still Michele's call.** At 2^20, N_eff = 698,224 (66.6%) — genuinely healthy,
  unlike the landmarks' 3,853/16,384. But the value shapes what the map means and Stage C is
  expensive to redo. Their doc flags this; it is a real blocker for the 64-GPU run, not for
  the ladder.
- **sigma must be re-derived per N.** 2^14 -> 2.00, 2^20 -> 1.26. Each rung of the ladder
  needs its own weighted-histogram peak. `src/check_2p20.py` does this; parameterise it by N.
