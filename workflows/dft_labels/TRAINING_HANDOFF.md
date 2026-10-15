# LDA/PBE training handoff — 2026-09-25, updated 2026-10-05

Everything needed to start training. **The DFT campaign is closed (2026-10-05): no
calculation is pending or will be run, and the snapshot below is the final version.**

## 1. Snapshot to use

    ${STORE_ROOT}/mad-core/lda_pbe/final/

(renamed from `product/labelled_20261005` on 2026-10-05; after the reorganisation script the build inputs live in `build/`, the raw outputs in `dft_outputs/`, see `README.md`). **Verified 2026-10-05: `sha256sum -c SHA256SUMS` →
13/13 OK.** Per arm: `<arm>/lda.xyz`, `<arm>/pbe.xyz`, `<arm>/manifest.tsv`, plus
`merge_labels.log` (the record of the build) and `SHA256SUMS`.

All earlier snapshots (`labelled`, `labelled_20260922` … `labelled_20260929b`) were strict
subsets of this one, frame for frame, and were removed; `build/snapshot_history/` keeps
their checksum lists and build logs. The frozen 22k experiment
(`mad-core-valid/experiments/prod_experiment_22k`) was built from `labelled_20260929b`; its
`madcore_22k` and `random_22k` sets rebuild identically from the final snapshot, its
`voronoi_22k` does not (reuse its frozen manifest, never rebuild it).

## 2. Counts per arm

| arm | selected | LDA labels | PBE labels | paired | LDA-only | PBE-only |
|---|---:|---:|---:|---:|---:|---:|
| madcore_25k | 25,000 | 24,719 | 24,514 | 24,447 | 272 | 67 |
| random_25k | 25,000 | 24,516 | 24,295 | 24,283 | 233 | 12 |
| voronoi_25k | 25,000 | 24,119 | 22,714 | 22,697 | 1,422 | 17 |
| random_test_6k | 6,000 | 5,887 | 5,860 | 5,848 | 39 | 12 |
| **total** | **81,000** | **79,241** | **77,383** | **77,275** | **1,966** | **108** |

Training-eligible after the force filter (max per-atom |F| ≤ 100 eV/Å, recomputed in float64
from the xyz forces on 2026-10-05, no non-finite labels, no labelled fragmented structure):

| arm | PBE-eligible (no LDA requirement) | pair-eligible |
|---|---:|---:|
| madcore_25k | 22,339 | 22,276 |
| random_25k | 24,293 | 24,281 |
| voronoi_25k | 22,577 | 22,560 |
| random_test_6k | 5,859 | 5,847 |
| **total** | **75,068** | **74,964** |

Largest matched training budget after the frozen val (1,000 from random) and test (5,847)
exclusions: **22,276** under the pair rule (MAD-CORE binding), 22,339 under the PBE-only rule.
Full derivation: `mad-core-valid/experiments/voronoi_weighted/prep_20261005/reports/PREP_20261005.md`.

## 3. Convention, references, units, filtering

**Energy.** `energy` in every frame is the **electronic free energy, E − TS**.
`free_energy` carries the same number. This is the quantity to train on.

> Do **not** use `build/computed_*/` or the pipeline's `merged/` for training:
> those still carry E − TS/2. Only `final/` (and the removed `labelled_*` snapshots) has the swap
> applied. `merge_labels.py:swap_to_free_energy()` performs it against
> `references/free_energies.tsv` and asserts every label matches to 1e-5 eV.

**Atomic references.** `atomization_energy` = `corrected_energy` =
`energy − Σ_atoms E0(species)`, with the isolated-atom E0 taken per functional
from `build/references/monomers_lda.xyz` and `monomers_pbe.xyz` — **102
elements each**, same settings as the campaign. A structure whose species are not
all covered is rejected by the builder, so every label present is referenced.

**Units.** Å, eV, eV/Å. `manifest.tsv` carries `e_lda`, `e_pbe` (free energies),
`half_ts_lda`, `half_ts_pbe` (the subtracted TS/2 amount), `atomization_e_*`,
`corrected_e_*` and `fmax_*`.

**Filtering.** The established `exp_a` rule, unchanged:

    max per-atom |F_DFT| <= 100 eV/A, applied to every source (train, val, test)

`exp_a/SPEC.md` §3.1: *"Force filter first: drop any frame with max per-atom
|F_DFT| > 100 eV/Å from every source (train, val, test alike)."* Select with
`has_lda`/`has_pbe` and `fmax_* <= 100` from the manifest, or read the
`*_eligible` columns of the accounting tables, which apply exactly this rule.

## 4. Scripts, configurations, split manifests

| what | path |
|---|---|
| training options template | `mad-core-valid/prod_experiment/config/options_template.yaml` |
| delta-experiment options | `mad-core-valid/delta_experiment/config/{options_direct,options_delta}.yaml` |
| frozen training protocol | `mad-core-valid/exp_a/SPEC.md` §6 |
| selection + split manifests | `build/sets/<arm>/{selection.tsv, ttt.extxyz, surfaces.extxyz, fff.extxyz}` |
| frozen→release id map | `build/sets/<arm>/release_ids.tsv` |
| madcore ∩ voronoi overlap | `build/sets/voronoi_25k/madcore_intersection.txt` |
| label builder | `build/scripts/merge_labels.py` (scratch copy `mad-core-valid/dft/lda_pbe/scripts/merge_labels.py`) (+ dated `_staged_*` variants) |

Protocol as frozen in `exp_a/SPEC.md` §6: full fine-tune, lr 1e-4, 150 epochs,
batch 64, seed 42, targets = `atomization_energy` + forces, `stress: false`,
best checkpoint by `mae_prod` on VAL. The options template already encodes
`key: atomization_energy`, `unit: eV`, `length_unit: angstrom`, `stress: false`.

**Splits.** `random_test_6k` is the held-out test arm; the other three are
training pools. Within an arm, `split_file` is `ttt` (bulk periodic), `surfaces`
or `fff` (isolated), and `split_row` is the row in the matching
`sets/<arm>/<split>.extxyz`. `has_lda & has_pbe` gives the common structure set
for identical splits across functionals.

## 5. Paired versus single-functional experiments

These are **two different experiments** and the counts are not interchangeable.

- **Paired** (`pair_eligible`, 74,964): both functionals present and both under
  the force cut. This is the existing experiment — LDA/PBE comparisons, delta
  learning, anything that needs the same structure at both levels. Unchanged.
- **Single-functional** (`lda_eligible` 76,996 / `pbe_eligible` 75,068): each
  functional judged on its own merits. A valid PBE result counts even when its
  LDA partner is missing or failed, and vice versa.

Single-functional adds **+2,032 LDA** and **+104 PBE** structures over the paired
set. For Voronoi specifically there are **17 PBE-only** structures, all 17
eligible; in every one the LDA ran and failed SCF, so PBE was cold-started by
design and succeeded. Report the two experiments separately; do not pool them.

## 6. Provenance limitations

**MAD-CORE raw outputs are purged.** That arm was computed by an earlier,
separate toolchain whose scratch working directory was deleted. Only the
collected XYZ survives. Consequences:

- **49,235 records** over 24,786 structures have a label but **none** of the
  seven mandatory output files — no matrices and no `aims.out`. Recorded in
  `dft_outputs/clariden_handoff/ledger/known_loss_madcore.tsv` +
  `KNOWN_LOSS_madcore.md`. **This does not affect any label**: all 24,447
  MAD-CORE pairs and 22,276 pair-eligible structures are present and usable.
- **767 records** (281 LDA, 486 PBE) have **no label**, classified
  `historical_outcome_unknown_raw_records_purged`. The outcome is unknowable —
  not a convergence failure, not a non-submission, not a lost success. The purge
  removed the evidence, not the label: the surviving XYZ holds exactly 24,719 and
  24,514 frames, precisely what is labelled, and already lacked those 767 when it
  was collected on 2026-09-04.
- The **shared structure** `mad_next:8f0ef3d9dde9d993bba32683b7815a36` — Voronoi
  `periodic_ttt/frame_15442`, computed as `madcore_25k/frame_16973` — inherits
  this. It is a complete eligible pair; only its raw outputs are gone. It is
  visible only by following `label_source`, not the arm it appears in.
- The search is **closed**. The one candidate alternative copy,
  `build/share/lda_pbe_2026-09-09/`, is built by `build_share.py` from
  `labelled/`, so it is a derivative and holds nothing extra. Reopen only if a
  specific unpurged copy is named.

Everything **outside** MAD-CORE is complete: `missing_matrices.tsv` (the open,
retrievable list) is **empty**. random_25k and random_test_6k have the full
output set for every pair.

## 7. Remaining DFT, owners, delivery status

**Superseded 2026-10-05: the campaign is closed.** Voronoi ends at 22,697 paired /
22,560 pair-eligible. The last import was Clariden batch 1 (2 pairs, 1 LDA-only;
`dft_outputs/clariden_handoff/completion_20260929/BATCH1_RESULTS_20261005.md`). Lyra's round 10
(2 frames), the frames it was running, and the claimed PBE of `frame_23509` are dropped.
The rest of this section is the 2026-09-25 state, kept as history.

Voronoi is the only arm with open work. Ledger (25,000 structures):

| category | structures |
|---|---:|
| complete pair in the product | **22,640** |
| terminal SCF failure | 2,000 |
| unfinished, unsubmitted | 234 |
| unfinished, running or queued | 96 |
| deferred huge | 17 |
| unresolved or excluded | 13 |

Ceiling without recovering terminal failures: **23,000**.

| owner | open | status |
|---|---:|---|
| clariden | 93 + 6 huge | LDA converged, PBE unattempted (density lost). Not queued — checked against the live queue; the `owner_job` values are shard names, not Slurm IDs. Candidate pool with measured runtimes in `dft_outputs/clariden_handoff/completion_20260923/candidates_93_evidence.tsv`. |
| lyra | 2 | `frame_05829`, `frame_20224` — pair rerun queued on Lyra, array 130046 tasks 10 and 42. Their LDA is already labelled and its full output set was delivered. |
| unowned | 1 | `frame_10051` — needs ownership confirmation; EPFL has been running block_10 RETRY frames. |
| — | 234 | no terminal status anywhere; not claimed as "never attempted". |

**Deliveries.** Six from EPFL, all verified on the destination copy and imported:
rounds 1–3, the compact retrieval, deltas 4–5, and the supplementary LDA. Details
and checksum results in `dft_outputs/clariden_handoff/deliveries/DELIVERIES.md`.

**Awaiting**: 97 records converged in EPFL's log whose outputs have not shipped —
EPFL sends a frame only once it is terminal at **both** functionals, so these
arrive when their partner finishes. 3 more are importable now but are LDA-only
(PBE terminally failed), so they add no trainable pair.
`dft_outputs/clariden_handoff/ledger/awaiting_import.tsv`.

**Compute spend**: the 2026-09-23 pilot and completion batch used **10.12 of the
24 node-hour cap** and produced 7 complete pairs, 1.45 node-hours each. Nothing
is running.

## 8. Where to look

`dft_outputs/clariden_handoff/INDEX.md` lists every authoritative file.
