# Sources of this release

| source | location at extraction | revision | role |
|---|---|---|---|
| papers repository (`sofiia-chorna/datasets-paper`) | `${DEV_ROOT}/paper` | `28cfe4307989f4fe032dd56c1c28dfb273e1f903` ("submitted paper", 2026-10-15), working tree clean apart from six untracked NOMAD caches and intermediate files that are not part of the release | manuscript and SI text, figure folders, claim scripts, GFRE/LFRE comparisons |
| development repository (`sofiia-chorna/phace-atomic`) | `${DEV_ROOT}/phace-atomic` | `97bec6f7f334f8b07f0d73c6fedd1d9c85e10ce5` ("record matcloud"), working tree clean; some copied files are gitignored there (marked in `extracted_files.csv`) | ecumetric training, featurisation, FPS, accretion, Voronoi, sketch-map, DFT, trainings, MD, data records |
| ingestion tree (`mad-core`, remote `sofiia-chorna/mad-core` empty) | `${DEV_ROOT}/mad-core` | unversioned, snapshot 2026-10-16 | non-ColabFit ingestion, GDB-13 conversion |
| collection scripts | `${STORE_ROOT}/dataset_collection` | unversioned, snapshot 2026-10-16 | downloads, license policy, covalent-radius prefilter |
| `merge-datasets` scripts repository (`sofiia-chorna/merge-datasets`) | `${DEV_ROOT}/_gh_scripts` | `9f58daa` (2026-05-12) | teacher feature extraction on MAD-1.5 |
| project store | `${STORE_ROOT}/mad-core` | files dated in `extracted_files.csv` | release metadata (`levels.json`, scalers, projector), SPACE training record |
| `coresets` package (`sofiia-chorna/coresets`, MIT) | `${DEV_ROOT}/coresets` | `14f72a3` (develop); the July runs used commits up to `67a932d` | not copied; dependency of the FPS and accretion workflows |

`extracted_files.csv`: one row per file of the repository that was copied from a source
(release path, original path, source, revision, sha256 of the original at copy time).
`archive_files.csv`: the same for the supporting-results and model archives, with the hash of the shipped file as well (the `options.yaml` next to each model were scrubbed of site paths).
`modified_files.csv`: the copied files whose content in the release differs from the
original (edited for portability, or scrubbed by `scrub.sh`), with both hashes.

Scripts: `copy_from_sources.sh` copies the listed files from their original paths,
`assemble_archives.sh` the archive contents, `scrub.sh` replaces the site- and user-specific
path prefixes and the private draft identifiers by placeholders (explained in
`workflows/README.md`), `mark_modified.py` writes `modified_files.csv`. The figure, table and
claim scripts that this release runs were edited by hand after extraction; `docs/reproduction.md`
and the file headers say what changed (inputs and outputs as arguments, a shared heatmap
module, the inlined map helpers). No scientific logic was changed; every edited figure script
was verified to reproduce the submitted PDF pixel for pixel.

Files written for this release (not copied): `README.md`, `docs/reproduction.md`,
`docs/verification.md`, `LICENSE`, `LICENSE-DATA`, `CITATION.cff`, `docs/zenodo.json`,
`environment/*`, `plotting/gfre_heatmap.py` (extracted from the union figure script),
`figures/fig09_fig10_training/plot_training.py` (assembled from `paper_figure_original.py`
and `plot_maps_original.py`), `figures/*/README.md`, `data/README.md`,
`reference_figures/README.md`, `tables/check_tables.py`, `claims/element_coverage.py`,
`scripts/*`, `workflows/README.md`, the files of `provenance/`, and the stage-1 audit in
`docs/audit/` (whose statuses are superseded by `docs/reproduction.md`).
