# GFRE/LFRE on the independent random MAD-COSMO test selection

Extension of `../gfre_lfre_union.py` (fig:gfre_lfre, MAD-NEXT test split) to the 6,000-structure
random MAD-COSMO draw behind the random test set of the paper (seed 20260814,
`phace-atomic/mad-core-valid/dft/lda_pbe/sets/random_test_6k/{fff,ttt,surfaces}.extxyz`;
5,847 of them are DFT-converged and form the labeled random test set, the labels are not used).

## Scripts

| file | role |
| --- | --- |
| `common.py` | frame loading in the canonical order (fff, ttt, surfaces), the NaN-initialised per-atom feature writer, batched evaluation with per-structure retry, output of `results_random_test/features/<model>_ll.npy` and `<model>_exclusions.json` |
| `featurize_metatomic.py <phace-atomic\|pet-mad>` | metatomic models, output `features` (ecumetric, 256) or `mtt::aux::energy_last_layer_features` (PET-MAD, 512) |
| `featurize_orb.py orb-v3-omat` | `node_features` of orb-v3-conservative-inf-omat (256) |
| `featurize_mace.py mace-omat` | readout features of every layer through the `LLPRModel` of the LLPR fork of MACE (layer 1 scalars 128 plus layer 2 readout MLP 16 = 144) |
| `featurize_uma.py uma-omat` | input of the last linear layer of the UMA energy head, task `omat` (128), the two forward hooks of the `expose_feat` fairchem fork |
| `featurize_dpa.py <dpa-omat\|dpa-omol>` | `DeepPot.eval_fitting_last_layer` (240), DPA3-OMol restricted to its first 83 elements and given charge 0 and spin multiplicity 1 |
| `verify_mad15.py <model>` | recomputes MAD-1.5 frames and compares with the stored teacher features, see below |
| `gfre_lfre_random_test.py` | pooling, filters, GFRE and LFRE on the random selection |
| `submit_featurize.sh <model> ...` | Slurm wrapper that picks the venv per model, `--export=ALL,LIMIT=20` for a smoke test |
| `submit_verify.sh <model> ...` | Slurm wrapper of `verify_mad15.py` (`N`, `START` as environment variables) |
| `submit_gfre_lfre.sh` | Slurm wrapper of `gfre_lfre_random_test.py --with-lfre` |
| `audit_overlap.py` | overlap of the usable and reconstruction-test structures with the SPACE training snapshot and the MAD-NEXT tier, by id and by geometry |
| `audit_features.py` | NaN pattern per model, the DPA3-OMol exclusions, ecumetric descriptors against the release |
| `audit_matched.py` | re-derives and recomputes the size-matched MAD-NEXT control that was dropped from the paper |
| `submit_audit_overlap.sh`, `submit_audit_features.sh`, `submit_audit_matched.sh` | Slurm wrappers of the three audits |
| `eval_r2scan.py` | ecumetric energy and force errors against the 973 r2SCAN labels of this draw, release cross-check, overlap of the 973 with training |
| `submit_eval_r2scan.sh` | Slurm wrapper of `eval_r2scan.py` |

The extraction recipes are those of `sofiia-chorna/merge-datasets:scripts/0_feats/mad15/*`, the
scripts that produced the teacher features of MAD-1.5 and MAD-NEXT. Checkpoints are in
`merge-datasets/teacher_models/`: `pet-mad-s-v1.5.0.pt` (exported with `mtt export` from
`lab-cosmo/upet` `models/pet-mad-s-v1.5.0.ckpt`), `mace-omat-0-medium.model`, `uma-s-1p2.pt`,
`DPA-3.2-5M-OMat24.pth` (OMat24 branch frozen from `DPA-3.2-5M.pt` with
`deepmd.pt.entrypoints.main.freeze`), `DPA3-Omol-Large.pt`; the ORB weights are cached in
`merge-datasets/orb_ckpt_cache`; ecumetric is the release `ecumetric_model.pt`.

## Pipeline

1. `sbatch submit_featurize.sh phace-atomic pet-mad orb-v3-omat mace-omat uma-omat dpa-omat dpa-omol`
   writes `results_random_test/features/` (NaN rows for excluded structures, exclusion reasons
   unsupported element, mixed pbc, runtime error, non finite output, plus `frames_meta.npz`).
2. `sbatch submit_gfre_lfre.sh` pools [mean, std], applies the finite and 50x-median-norm filters,
   `Ridge2FoldCV` and LFRE k = max dim exactly as `../gfre_lfre_union.py`. A random share of the
   usable structures (4,000/15,000 as in the main figure) evaluates the reconstruction, the rest
   fits scaler and ridge (skmatter `test_idx`, no leakage).
   Output `results_random_test/random_{gfre,lfre}_seed0.json`, `selection_random_seed0.npz`,
   `summary_seed0.json` and `fingerprint_seed0.json`. The fingerprint holds sha256 digests of the
   filtered descriptors of every model, of the kept and test indices, of the pooling, GFRE/LFRE,
   ridge and scaler code, plus the settings. A rerun resumes from existing pair results only when
   the fingerprint is identical; otherwise the existing outputs are moved to
   `results_random_test/superseded/<time>/` and everything is recomputed.
3. Figure `paper/figures/gfre_lfre_heatmap_random_test/` (`prepare_data.py` copies the JSONs,
   `plot.py` draws GFRE and LFRE side by side with the helpers and layout of the main figure). A copy of the
   resulting `gfre_lfre_heatmap_random_test.pdf` is kept in this folder next to the scripts.

`logs/` keeps the Slurm logs of the runs behind the current results: `feat-1` to `feat-4` for
the features (they predate the split into one script per environment), `gfre-final` for the
first GFRE and LFRE run, `gfre-lfre-random-test_3657701` for the fingerprinted recompute that
reproduced it bit for bit, `gfre-lfre-random-test_3657810` for the resume check, `verify-mad15`
and the three `audit-*` logs for the checks below.

## Environments (inside `uenv pytorch/v2.9.1:v2`, torch 2.9.1, `merge-datasets/venvs_feats/`)

- `orb/`: orb-models 0.7.0, protobuf pinned to the uenv's 6.33.1.
- `mace/`: `SanggyuChong/mace@LLPR_universal` (provides `LLPRModel`) over the mace-torch deps.
- `uma2/`: upstream fairchem-core 2.19.0 installed `--no-deps`, the first release that loads the
  current uma-s-1p2 checkpoint; `uma/` holds the `expose_feat` fork, which predates that
  checkpoint and is not used.
- `dpa/`: deepmd-kit 3.2.0 wheel (built for torch 2.11); local patch in `deepmd/pt/cxx_op.py`
  makes the MPI and custom-op loading optional (`.orig` kept), DPA3 inference needs neither.
- ecumetric: `phace-atomic/huggingface/ecumetric_work/venv-ecumetric` (metatomic 0.1.17);
  PET-MAD: `${HOME}/venv-metatrain`.

The teacher features of the MAD-NEXT cache were centred and divided by one scalar per model
(`phace-atomic/prep_features.py`, scales in the store `build_stats.json`), which the skmatter
scaler removes, so the raw features computed here are directly comparable.

## Verification of the feature recipes (2026-10-13)

`sbatch submit_verify.sh pet-mad mace-omat uma-omat dpa-omat dpa-omol` recomputes MAD-1.5 frames
150,000 to 150,039 (665 atoms, 3 to 48 atoms per frame) and compares them with the normalized
teacher bins the union figure was built from (store
`mad-core/models/training/atomic_combined/*.bin`, mad15 block; new features divided by the
`build_stats.json` scale, both sides centred over the window):

| model | max abs. difference | correlation |
| --- | --- | --- |
| PET-MAD-1.5 | 3.5e-6 | 1.000000 |
| MACE-OMat | 5.4e-6 | 1.000000 |
| UMA-OMat | 3.7e-6 | 1.000000 |
| DPA3.2-OMat | 7.2e-7 | 1.000000 |
| DPA3-OMol | 1.3e-5 | 1.000000 |

PET-MAD also matches the raw per-atom array `universal_features/mad-1.5/pet/mad-s-v1.5.0_ll.npy`
to 4.6e-5. The other raw arrays in `universal_features/mad-1.5/` (`mace/mace_full.npy`,
`uma/uma-omat.npy`, `dpa/dpa-Omat24_ll.npy`) and the per-structure `*.bin` files in its root do
not match (correlation 0.05 to 0.6): they come from other model runs (mace-mp-0b3, DPA-3.1) and
are not what the paper used. ORB has no stored reference (the old arrays were purged) and uses the
unchanged upstream recipe.

## Results (seed 0)

6,000 structures, 5,905 with finite descriptors in all seven models (95 excluded by the
83-element domain of DPA3-OMol, none by the norm filter). Of the 5,905, 4,275 are non-periodic,
976 fully periodic and 654 surfaces, 58% come from GDB-13, and 5,763 are in the 5,847-structure
labeled random test set. Split 4,330 fit / 1,575 test.

Against the main figure, 31 of 42 GFRE and 27 of 42 LFRE entries are lower (median -0.07 and
-0.03), although the fitting set is smaller. ecumetric has the lowest mean error as source under
both metrics (0.408 and 0.377) and is the best source for 5 of 6 targets (UMA-OMat is again best
reconstructed by DPA3.2-OMat). PET-MAD-1.5 is harder to reconstruct than on MAD-NEXT, from 5 of 6
models under the GFRE and from all 6 under the LFRE.

## Audit (2026-10-14)

Overlap with SPACE training data (`audit_overlap.py`, `results_random_test/overlap_audit.json`).
The random draw (`phace-atomic/mad/madcore_vs_random_20k/random_sampler.py`) is uniform over all
1,694,502,355 release rows, the MAD-NEXT tier included, and the release `is_duplicate` flag only
marks repeated configuration ids, so disjointness was checked directly. Same composition and
sorted interatomic distances (minimum image) within 0.01 A count as the same geometry. The SPACE
training snapshot is the store `atomic_combined` (677,454 structures, split by its
`val_indices_in_all.txt` and `test_indices_in_all.txt`), the MAD-NEXT tier is the release
`mad-next-coreset` (856,012).

| | usable 5,905 | reconstruction test 1,575 |
| --- | --- | --- |
| id in the MAD-NEXT tier | 3 | 0 |
| same geometry in the MAD-NEXT tier | 4 | 0 |
| same geometry in SPACE train / val / test | 3 / 0 / 0 | 0 / 0 / 0 |

The 4 are two MAD-1.5 rows and one new MAD-NEXT dimer drawn from the MAD tier, plus one
`mcloudarchive__7v6p9-92t31` structure identical to a MAD-1.5 training structure. All 4 are in the
fit subset (0.09 percent of it), none is in the reconstruction test or in the r2SCAN 1,000, so no
result needs updating.

Features (`audit_features.py`, `results_random_test/feature_audit.json`). Every frame is fully
finite or fully NaN in every model. The 95 DPA3-OMol exclusions are exactly the frames with an
element beyond its first 83 (Ac, Th, Pa, U, Np, Pu). The pooled ecumetric descriptors equal the
release `ecumetric_features.bin` of the same rows for all 5,905 (max relative difference 7e-7).
Pooling is permutation invariant, so only the frame order has to agree between models, and all
models write into the same frame index.

Provenance. The delivered JSONs had no fingerprint. Recomputed from scratch with the final inputs
(log `gfre-lfre-random-test_3657701`), all 56 GFRE and LFRE values, the selection and the summary
are bitwise identical; the delivered copies are kept in `superseded/20261014-004427/`.

Size-matched control (`audit_matched.py`, `results_random_test/audit_matched/`). Re-derived from
the seed it is the same 5,905 MAD-NEXT structures and 1,575 test indices as the dropped control,
and its recompute reproduces the original values exactly.

| median difference | GFRE | LFRE |
| --- | --- | --- |
| matched control minus main figure (sample size) | +0.022 | +0.045 |
| random minus matched control (structures) | -0.091 | -0.081 |
| random minus main figure | -0.071 | -0.025 |

These are single-seed numerical differences, not significance-tested. In the main figure SPACE
is not the lowest-mean source under GFRE (DPA3.2-OMat is, 0.450 against 0.461); on the random
selection it is (0.408). The LFRE neighbourhood is the larger descriptor dimension in both, which
is up to 24 percent of the 4,330 fit structures here and up to 9 percent of the 11,000 of the main
figure, so LFRE values are compared most fairly with the size-matched control.

## ecumetric energy and forces against r2SCAN (2026-10-14)

`sbatch submit_eval_r2scan.sh` (log `eval-r2scan_3659723`, output
`results_random_test/r2scan_eval.json`) runs the release `ecumetric_model.pt` on the 973
converged r2SCAN labels of this draw (store `mad-core/r2scan_random_test/results/test_1000_r2scan.xyz`,
1,000 drawn from the 5,847 labeled, 27 not converged). The labels are FHI-aims r2SCAN, spin none,
and their `atomization_energy` subtracts the same 102 isolated-atom energies as the MAD-1.5 r2SCAN
record that ecumetric was trained on (identical to every digit), so energies are compared directly,
with no offset. The isolated atoms of MAD-1.5 have atomization energy 0, as in `e.bin`.

Metrics as in the delta-learning table, energy per atom over structures, forces over all cartesian
components, forces are the non-conservative head.

| subset | n | E MAE / RMSE (meV/atom) | F MAE / RMSE (meV/A) | E p8 | F p8 | E SUP | F SUP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 973 | 16.4 / 25.1 | 109 / 200 | 87.2 | 1435 | 187.9 | 5434 |
| molecules | 729 | 14.0 / 19.5 | 105 / 197 | 54.9 | 1398 | 118.9 | 5117 |
| bulk | 136 | 27.7 / 44.9 | 94 / 198 | 111.2 | 1644 | 187.9 | 5434 |
| surfaces | 108 | 17.8 / 23.8 | 133 / 208 | 44.9 | 892 | 75.1 | 2975 |

The mean signed energy error is -8.0 meV/atom. The 608 GDB-13 structures have an energy MAE of
12.6 meV/atom, the other 365 of 22.6. For reference, the epoch-92 validation MAE in the appendix is
29.1 meV/atom and 85.4 meV/A, on a different structure mix.

Checks. The predictions equal the release `ecumetric_energy.bin` and `ecumetric_nc_forces.bin`
of the same rows (same species order, positions within 5e-9 A, energy within 3.6e-6 eV/atom,
forces within 6.1e-5 eV/A). None of the 973 has the same geometry as a structure of the SPACE
training snapshot or of the MAD-NEXT tier (same test as `audit_overlap.py`), which also covers the
11 of them with elements outside DPA3-OMol that the earlier audit did not see.
