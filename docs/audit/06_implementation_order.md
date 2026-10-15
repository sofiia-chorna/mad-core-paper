# Implementation order for stage 2

Ordered by usefulness per unit of work and by how recoverable each item is. Each step names
the verification that closes it. Nothing here launches training, DFT, full featurisation,
full FPS or long MD.

1. **Freeze provenance.** Record the three source commits (paper 28cfe43, phace-atomic
   97bec6f, coresets 14f72a3), the scikit-matter and metatrain clone states, and the hashes of
   the gitignored files the release carries (`inclusion_list.tsv`). Resolve Q1 (submitted
   figure variants) and Q2 (license) first, since they decide what goes into
   `reference_figures/` and the file headers.
2. **Figure folders, L1 verified.** Copy the eight papers-repo figure folders and the shared
   `plotting/` module, replace the two cross-folder imports (G15) and the absolute store paths
   in `prepare_data.py` by a `--release-root` argument (G12). Verification: `check_figures.sh`
   from a fresh extraction gives 0 differing pixels against `reference_figures/` for all eight
   (same matplotlib 3.10.9 or 3.11.1 as recorded; document the version dependence).
3. **Training and water figures, L1.** Extract `paper_figure.py` and the three helper
   functions it uses into `figures/fig09_fig10_training/` (G14), copy the five TSV/JSON inputs
   and `sketchmap_panels_balanced_4000.npz`; copy the water script and its three curve TSVs.
   Verification: 0 differing pixels against the dev builds with matplotlib 3.11.1 (already shown
   in this audit from the source tree). Decide G1 for the force map: attempt the L2 rebuild of the
   union_7847 panels with `make_maps.py` (needs the predictions and the projector, minutes on one
   GPU); if it is not feasible, record the crop window and label the redraw as reconstructed.
4. **Tables.** `tables/check_tables.py` reading `SETS.json`, `metrics.tsv`, the two
   `metrics.json`, `levels.json`; prints the LaTeX rows and asserts equality with the submitted
   values (tolerance: last printed digit). Document G2 for the accrete table.
5. **Claims.** Copy `claims/` with `--release-root` and `--out` (G20); add
   `element_coverage.py` (G27). Verification: rerun `largest_voronoi_cell.py` (6 s) and
   `nonfinite_descriptors.py` (4 min) from the fresh extraction against the aa002 release and diff
   the JSON against the committed outputs; mark the Slurm-scale claims as located/untested unless
   the user wants them resubmitted (hours).
6. **REPRODUCTION.md** with the matrix of this audit as its skeleton, one row per figure,
   table and number, updated from the stage-2 runs. README, CITATION.cff, environment files
   (G16, G24) with the tested versions listed separately from the historical ones.
7. **Workflows (documented, not rerun).** Copy the code and small records listed in
   `02_inclusion_list.md` section 4, strip the private identifiers and accounts (G8, G28),
   replace scratch paths by variables, and write one README per workflow folder giving the
   command, inputs, outputs, resources and the data-record files that replace the internal
   layout. Bounded checks only: `coresets` python FPS on the toy data (G26), a 100-row
   `run_inference.py` smoke test on the ecumetric model if the ML environment is available on a
   GPU node, `fit_dist.py --help`. The scikit-matter fork (G3) and the metatrain patch (G4)
   must be committed by the user or vendored with notices before this step closes.
8. **Clean-extraction test.** `scripts/build_archive.sh` producing the three tarballs and
   `SHA256SUMS` from a tagged commit; extract into an empty directory with the two source repos
   unreadable (chmod or a different user), run steps 2 to 5 there with `PYTHONNOUSERSITE=1`
   and no scratch paths in the environment; write the verification report from that run.
9. **Zenodo metadata draft** from `authors.tex`, `acknowledgements.tex` (grants 205602,
   200020_214879, Ambizione 233444, NCCR Separations 229280, project aa002), the record DOIs
   once they resolve, the HF model URL, the `coresets` repository and the reserved Zenodo DOI
   `10.5281/zenodo.23266878` from `biblio/others.bib` (to be confirmed as the reserved DOI of
   this deposit, not invented here).
10. **Optional archives** (Q3): results and models tarballs only if the user wants L2 offline.

Expected effort: steps 1 to 6 are a day of work and fully verifiable on a login node; step 7
is mostly careful copying and documentation; step 8 half a day; 9 and 10 an hour each.
