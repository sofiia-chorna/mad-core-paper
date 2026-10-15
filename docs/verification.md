# Verification report

Release built on 2026-10-16 from the papers repository at 28cfe4307989f4fe032dd56c1c28dfb273e1f903
and the development repository at 97bec6f7f334f8b07f0d73c6fedd1d9c85e10ce5. The release
commit is recorded in `dist/1.0.0/mad-core-paper-1.0.0-src.commit` next to the archives and
the archive checksums in `dist/1.0.0/SHA256SUMS`.

## Environments used

- `plot`: Python 3.12, matplotlib 3.11.1, numpy 2.4.4, scipy 1.18.0, pandas 3.0.3, pyarrow
  24.0.0, ase 3.29.0, torch 2.9.1 (the `venv-metatrain-fix` environment of the
  development tree, inside the CSCS uenv `pytorch/v2.9.1:v2`, login node of Alps).
- `plot-3.10.9`: the same with matplotlib 3.10.9, numpy 2.4.4, pyarrow 24.0.0
  (`venv-phace`), the environment of the papers-repository figures.
- Ghostscript and ImageMagick 7 for the pixel comparisons (100 dpi rasters, `magick
  compare -metric AE`, absolute number of differing pixels).

## 1. Figures, from this repository

`bash scripts/check_figures.sh <python>`, outputs in `build/check/`, compared with
`reference_figures/submitted/`:

| figure | command | environment | differing pixels |
|---|---|---|---|
| Fig. 1 | `figures/fig01_nested_kset/plot.py` | plot-3.10.9 | 0 |
| Fig. 3 | `figures/fig03_gfre_lfre/plot.py` | plot-3.10.9 | 0 |
| Fig. 4 | `figures/fig04_madnext_subsets/plot.py` | plot-3.10.9 | 0 |
| Fig. 5 | `figures/fig05_madnext_elements/plot.py` | plot-3.10.9 | 0 |
| Fig. 6 | `figures/fig06_madcore_decomposition/plot.py` | plot-3.10.9 | 0 |
| Fig. 7 | `figures/fig07_madcore_elements/plot.py` | plot-3.10.9 | 0 |
| Fig. 8 | `figures/fig08_sketchmap/plot.py` | plot-3.10.9 | 0 |
| Fig. 9 | `figures/fig09_fig10_training/plot_training.py curves` | plot | 0 |
| Fig. 10 | `figures/fig09_fig10_training/plot_training.py force_map` | plot | 0 (crop window from `sketchmap_panels_union_7847_round3.npz`, see `figures/fig09_fig10_training/data/README.md`) |
| SI Fig. 1 | `figures/siS1_gfre_lfre_random/plot.py` | plot-3.10.9 | 0 |
| SI Fig. 2 | `figures/fig09_fig10_training/plot_training.py separate_parts` | plot | 0 against the assumed submitted file |
| SI Fig. 3 | `figures/siS3_water_rdf/plot.py` | plot | 0 |

With the other matplotlib minor version the same figures differ by a few thousand pixels
in text antialiasing only (checked during the audit: 2,808 of 650x374 for Fig. 9, 8,930 of
686x242 for SI Fig. 3 with matplotlib 3.10.9; diff images in `audit/evidence/`).

The rendered figures were inspected: panel counts, curves, units, axis labels and legends
match the submitted files (a zero pixel difference at 100 dpi implies this).

## 2. Tables

`python tables/check_tables.py --record-dir <MAD-CORE record files>` (environment: any
Python 3): training-set sizes 6 of 6, fine-tune 16 of 16, delta learning 48 of 48, accrete
levels 21 of 21 sourced values agree with the printed values after rounding to the printed
digits; the margin and t(s) columns of the accrete table have no source. Record label
counts: r2SCAN 21,854 agrees; LDA 79,220 and PBE 77,362 counted against 79,241 and 77,383
printed (open point P2 of `docs/reproduction.md`).

## 3. Claims rerun against the aa002 release (internal layout)

| script | run time | result |
|---|---|---|
| `claims/largest_voronoi_cell.py --release-root STORE/release` | 6 s | output identical to `claims/out/largest_voronoi_cell.json` (117 cells, 114,996,700 records, C10H13N3) |
| `claims/nonfinite_descriptors.py --release-root STORE/release` | 4 min | output identical to `claims/out/nonfinite_descriptors.json` (67,975 non-finite rows) |
| `claims/element_coverage.py --store-root STORE` | 80 s | 2,158 (MAD-CORE) and 480 (MAD-NEXT), as stated; output `claims/out/element_coverage.json` written for this release |

Not rerun (Slurm scale, open point P5): `collection_within_coverage.py`,
`outside_coverage_scan.py`, `outside_coverage_maxima.py`, `outside_coverage_report.py`,
`madcosmo_populations.py`; their committed outputs carry the values quoted in the text
(99.166%, 0.21792; 99.934%, 0.30456, 0.39999; 454 groups; 215,308 ZBL-rejected rows).

## 4. Numbers recomputed from the figure data

From `figures/fig06_madcore_decomposition/data/` at k = 2^24: 341 contributing groups, 26
with more than 10,000 structures, 235 with fewer than 100, 340,555 MAD-NEXT rows, OMat24
0.563 and Alexandria 0.188 (0.752 together), 3D bulk 0.945, molecular datasets (ANI, QM,
SPICE, OMol25, GDB-13 prefixes) 0.0326, Open Catalyst 0.0098, first prefix with solvated
systems 16,384 and with molecular crystals 32,768, eps(2^24) 0.21791, eps(2^23) 0.30456.
All agree with the text.

## 5. Scalers and model identity

`data/ecumetric_scaler.npz` equals the MAD-NEXT scaler pickle exactly (mean difference 0.0,
scale 125.4608 in both). The released `ecumetric_model.pt` (md5 2038c977...) is the
epoch-92 checkpoint re-exported under the public SPACE architecture name with all 530
weight tensors bit-identical (`workflows/ecumetric_export/space_swap/backup_README.md`).

## 6. Clean extraction

`git archive` of commit d74f02d (the release candidate) was extracted into an empty
directory outside the repository and the source trees (`/tmp/.../scratchpad/clean/mad-core-paper-1.0.0`, 1,057
files, 164 MB), and the checks were run from there with `PYTHONPATH` unset and the working
directory inside the extraction:

| check | environment | result |
|---|---|---|
| `scripts/check_figures.sh`, Figs. 1, 3 to 8, SI Fig. 1 | plot-3.10.9, `python -I` (no user site, no PYTHONPATH) | 0 differing pixels for all eight |
| `scripts/check_figures.sh`, Figs. 9, 10, SI Figs. 2, 3 | plot (matplotlib 3.11.1) | 0 differing pixels for all four |
| `tables/check_tables.py --record-dir <record files>` | system Python 3, `-I` | as in section 2 (all table values agree; the two record counts differ, P2) |
| `claims/largest_voronoi_cell.py --release-root STORE/release` | plot-3.10.9, `-I` | output identical to the committed JSON |
| references to the development trees | `grep` over the extraction outside `provenance/` and `audit/` | only through the documented placeholders (`${DEV_ROOT}`, `${STORE_ROOT}`, ...) in the workflow code and notes; no absolute path of either source repository remains in the figure, table or claim code |
| shared modules resolve inside the extraction | the figure scripts put only their own `plotting/` on `sys.path` | yes |

The eight 3.10.9 figures were checked again from the archive of commit 0392628 with the
same result. The history was then rewritten into a few small commits; the figure, table and claim files
are unchanged, the accretion workflow folder was reduced to the three pipeline steps and the
long documents moved under `docs/`.

One environment caveat was found: the development `venv-metatrain-fix` does not contain
the `packaging` module itself and takes it from the user site-packages, so the 3.11.1 pass
fails with `PYTHONNOUSERSITE=1`. A fresh environment from `environment/plot-requirements.txt`
installs `packaging` as a matplotlib dependency; the pass above used the development
environment with its user site, as the historical builds did.

The archives are built from the release commit by `scripts/build_archives.sh 1.0.0`:
`git archive` for the source (reproducible from the commit) and tar plus gzip for the two
data archives. Rebuilding the data archives from the same staged files gave tarballs that
differ by a few hundred bytes (tar and gzip metadata), so their hashes are those of the
shipped files in `dist/1.0.0/SHA256SUMS`, not a property of the content; the content is
verified file by file against `provenance/archive_files.tsv` with
`sha256sum -c`. The three archives, their sizes and SHA-256 are listed in
`SHA256SUMS` next to them and in the final report of the release.

Per-file hashes of the archive contents: `provenance/archive_files.tsv` (checked with
`sha256sum -c`).

## 7. What remains unverified

- Level 2 and level 3 of every workflow (not rerun; documented with resources).
- The Slurm-scale claims (P5) and the NOMAD exclusion count (P6).
- Which build of SI Fig. 2 was submitted (assumed from date and caption).
- The margin and timing columns of the SI accrete table (P7) and the 21-structure
  difference of the LDA and PBE label counts (P2).
- The Materials Cloud DOIs and the Zenodo DOI return 404 on 2026-10-16 (unpublished).
