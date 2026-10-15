
# Next runs: we violated the a/A = d/D rule at d=8

## The finding

Re-read Appendix A/B of the transferability paper (ref [39]). Three rules, verbatim:

- **sigma**: *"a good value for sigma places the inflection point of the filter in the
  vicinity of the **first prominent feature** in the histogram of pairwise distances"*
- **A, B**: *"A should thus be set to a large number — **D is often a good first guess**.
  We generally set **B << A**, as we want the tail of the function for large R to be
  considerably longer than the small R tail."*
- **a**: *"if we set a and A so that **a/A = d/D**, we can obtain the close match between the
  distributions of transformed distances between points taken from high and low dimensional
  Gaussian distributions"*

So **a = A * d / D**. Sanity-check against the paper's own LJ38 numbers: D=16, A=8, d=2 ->
a = 8*2/16 = **1**, and they used a=1. The rule holds.

Applied to our level-14 landmark set (D = **intrinsic** dimension ~= 14 — the ambient 512 is
unusable, `2**(A/B)` overflows):

|                    | we ran      | rule                   | status                                                                                  |
| ------------------ | ----------- | ---------------------- | --------------------------------------------------------------------------------------- |
| sigma              | 3.0         | 0.9 * peak(3.38) = 3.0 | **correct** — Appendix A; skmatter's `sigma = 0.9*peak_distance` implements it |
| A                  | 4           | ~= D =**14**     | too small                                                                               |
| B                  | 2           | << A                   | ok                                                                                      |
| a at d=2           | 2           | 14*2/14 = 2            | ok                                                                                      |
| **a at d=8** | **2** | **14*8/14 = 8**  | **VIOLATED**                                                                      |

**At d=8 we hard-wired `a_low=2`, which is MAD's *d=2* value.** It was copied without
re-deriving it for d=8. skmatter's own auto-suggest already gets this right —
`_sketchmap_utils.py:359` is `a_low = b_low = float(max(n_components, 1))`, i.e. exactly
`a = A*d/D` with A=D — and we overrode it by passing params explicitly.

This is a plausible reason the d=8 map is not better than it is. Untested.

## Runs — d=2 first (Sofiia wants a visual check), then d=8

Landmarks and weights unchanged: `features_first65536.npy[:2**14]`,
`populations_level14.npy ** 0.25` normalised (`-wgamma 0.25`), `center=False`.

### Stage 1 — d=2, ~13 min each

| run          | sigma | A            | B | a | b | tests                               |
| ------------ | ----- | ------------ | - | - | - | ----------------------------------- |
| **P1** | 3.0   | **14** | 2 | 2 | 2 | A = D (Appendix B); a/A = d/D holds |
| P2           | 3.0   | 4            | 2 | 2 | 2 | control = existing run A            |

At d=2 both already satisfy a/A ~ d/D loosely, so expect little difference — this is mainly
to see what A=14 does to the picture before spending d=8 runs.

### Stage 2 — d=8, the actual target

| run          | sigma | A            | B    | **a** | b    | tests                                                     |
| ------------ | ----- | ------------ | ---- | ----------- | ---- | --------------------------------------------------------- |
| **P3** | 3.0   | **14** | 2    | **8** | 2    | fully paper-compliant                                     |
| **P4** | 3.0   | 4            | 2    | **8** | 2    | isolates the a fix alone                                  |
| P5           | 3.0   | 4            | 2    | 2           | 2    | control = existing run D (family locality**1.75x**) |
| **P6** | auto  | auto         | auto | auto        | auto | `suggest_sigmoid_params(pdist(X), 8)`, pass nothing     |

P6 matters: skmatter implements Appendix A and the a/A rule itself, so it is the honest
baseline. Note it takes **distances**, not X — `suggest_sigmoid_params(pdist(X), n_components)`.
Passing X silently returns sigma ~ 0.02 and a garbage map.

Ambiguity worth resolving with P6 vs P3: skmatter sets `b_low = n_components` (so b=8 at
d=8), whereas both papers used **b = B** (LJ38 and MAD both had b = B = 2). P3 uses b=2, P6
uses b=8. The paper's own argument is that differing b distorts long distances:
*"when the b parameters are set differently, distances greater than sigma will be distorted
because for R/sigma >> 1, [F(R) - f(r)] = 0 when r/sigma ~= (R/sigma)^(B/b)"*.

## Metrics — judge on these, in this order

1. **family locality** (5-NN same-family rate / composition baseline). Run D scores **1.75x**.
   This is the metric that matters.
2. **corr(d_2d, D_512)** — a sanity floor (~0.9), **not** something to maximise: sketch-map is
   *supposed* to distort distances, so maximising it just gets MDS back.
3. **collapsed (NN<1e-3)** — want ~0. Run D had 0.
4. **max_r/median_r** — want < ~8. Run D had 7.15.

## Do not re-tune sigma

sigma = 3.0 is **correct** per Appendix A (0.9 x the 3.38 peak of the level-14 landmark
histogram). The earlier sweep to sigma=4.25 was mine and it was wrong-headed — it also moved
B to 6, so it conflated three changes, and it lost at d=8 (1.57x vs 1.75x). Hold sigma at
3.0 and vary only A and a.

## Note for later: parameters do not carry across levels

They are properties of the landmark **set**:

| set  | histogram peak | sigma = 0.9*peak | % Alexandria |
| ---- | -------------- | ---------------- | ------------ |
| 2^14 | 3.38           | **3.0**    | 60.6%        |
| 2^20 | 1.49           | **1.34**   | 28.2%        |

So if we move to 2^16/2^20 landmarks, sigma must be re-derived from that set's own histogram
(and A from its intrinsic dimension). Also relevant to Sofiia's "too much Alexandria"
complaint: it is 60.6% of the level-14 centers and 28.2% at 2^20, so more landmarks halves
it. Note the Alexandria spread is **not** a map defect — it genuinely has the largest 512-d
median pair distance (4.39 vs ~2.9 for the other families), and the map reproduces each
family's extent faithfully (Alexandria ratio 1.27 vs the global 1.25).
