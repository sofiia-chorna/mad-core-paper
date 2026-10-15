# Upstream workflows

The code and small records of every computational step behind the paper, by scientific
purpose. These are level-2 and level-3 workflows in the sense of `docs/reproduction.md`: they
are documented and their inputs, outputs and resources recorded, but they were **not rerun**
for this release. The original analysis code is copied as it was (one file per original,
see `provenance/extracted_files.tsv` for the source path, revision and hash), with two
mechanical changes applied by `provenance/scrub.sh`: site- and user-specific path prefixes
were replaced by the placeholders below, and the private identifiers of the unpublished
Materials Cloud drafts by `${MC_DRAFT_MADCORE}` and `${MC_DRAFT_MADCOSMO}`. Slurm scripts
keep their original accounts and partitions (CSCS Alps, GH200 nodes) as a record of what ran.
Files named `README_original.md` or `*_original.py` are the notes and scripts of the
development tree, unchanged apart from the scrub.

| placeholder | meaning |
|---|---|
| `${DEV_ROOT}` | the development workspace holding the `phace-atomic`, `paper`, `mad-core` and `coresets` trees |
| `${STORE_ROOT}` | the durable project store (`mad-core/` with `release/`, `mad_union/`, `accrete/`, `fps/`, `models/`, `experiments/`, `lda_pbe/`; `dataset_collection/`; `zbl_filtering/`) |
| `${STORE_USER_ROOT}`, `${STORE_LP133}`, `${COLLABORATOR_STORE}`, `${TRANSFER_ROOT}`, `${KUMA_*}`, `${IOPS_SCRATCH_ROOT}`, `${OTHER_WORKSPACE}` | other user or cluster trees of the original runs |
| `${HOME}` | the user home holding the Python virtual environments (`venv-phace`, `venv-metatrain-fix`, `venv-ipi-petmad`) |

The order below is the order of the pipeline. Resource figures are those recorded in the
scripts and notes.

## collection/

How the 1.7 billion structures of MAD-COSMO were gathered. Two parts, both from unversioned
trees (hashes in `provenance/extracted_files.tsv` are the only provenance):

- `download_and_prefilter/`: bulk downloads from ColabFit (Hugging Face mirror,
  `download_colabfit.py`, `download_oc20.py`), NOMAD (`nomad/download_nomad_2.py`, REST API
  with cursor pagination; `collect_dois.py` for the dataset DOIs of the entries), OPTIMADE
  providers (`optimade/bulk_download.py`, `jsonl_to_xyz.py`, `probe_license.py`), the
  license policy (`note_on_licenses.txt`, `license.py`), and the covalent-radius prefilter
  (`pyykko_filter*.py`, `run_filter_all.py`, `run.sh`): a structure is rejected when two
  atoms are closer than `scale` times the sum of their Pyykkö covalent radii. The collection
  was built from the structures that pass at scale 0.35 (`filtered`) plus those rejected at
  0.35 but passing at 0.25 (`filtered_rej_passes_at_025`); structures rejected at 0.25 were
  not featurised (apart from 8 shards) and are not in MAD-COSMO. **This prefilter is not
  described in the manuscript or in the record READMEs**, which describe only the ZBL filter
  applied afterwards; see `docs/reproduction.md`, open point P1.
- `ingest/`: conversion of the non-ColabFit sources to the common parquet schema with
  content-addressed `configuration_id` (`convert.py`, `ids.py`, `records.py`,
  `normalize.py`, one reader per source format in `readers/`), the GDB-13 SMILES to 3D
  conversion (`preprocess/smiles2extxyz.py`: RDKit ETKDGv3 with seed 0xf00d, MMFF94
  optimisation, molecules that fail to embed dropped), the Materials Cloud archive retrieval
  (`runs/mcloud_archive_tarzip/download_extract.sh`), and the per-source run records
  (`runs/<source>/submit.sh`, `manifest.txt`, `NOTES.txt`). The featurisation these runs
  fused with the conversion used an earlier model (`m292`); the features of the collection
  were recomputed with the ecumetric model by `featurize_collection/`. `ids_phace_atomic.py`
  is the id module as used by the later tree (the two differ in helpers only; the hash
  parameters are frozen).

## ecumetric_training/

Training of the SPACE model (ecumetric): `options.yaml` (architecture and loss as submitted
to metatrain: cutoff 7.5 Å, 2 message-passing layers, Huber losses, five per-atom teacher
heads with batch-mean handling of missing teachers), `options_restart.yaml` (the resolved
options, learning rate 3e-3, batch 32), `prep_features.py` and `extxyz_to_memmap.py` (the
training memmaps from the three MAD source files and the teacher features), `submit.sh`
(1 node, 4 GH200, about 15 min per epoch), `build_stats.json` (677,454 structures, 17.9 M
atoms, teacher scales), `run_2026-06-26_03-09-50/` (train.csv and train.log through epoch 92,
the resolved split indices 594,769 / 41,338 / 41,347). The teacher features themselves were
produced by `gfre_lfre/teacher_features/`. The epoch-92 checkpoint became the released
`ecumetric_model.pt` (Hugging Face `lab-cosmo/ecumetric`); `ecumetric_export/` records how.
MAD-NEXT, the training set, is not released (it will be part of MAD-2); the training memmaps
exist only on the project store.

## ecumetric_export/

Packaging of the model for the data records and Hugging Face: `space_swap/` (the 2026-10-08
re-export of the epoch-92 weights under the public `experimental.space` architecture name,
with the checks that every tensor and prediction is unchanged), `export_model.sh`,
`build_explorer.py` (the sketch-map explorer, ecumetric + scaler + projector in one metatomic
model), `check.py`, the model cards, `upload.py`.

## featurize_collection/

Featurising every structure of the collection with the ecumetric model (`run_inference.py`,
`--meanstd` concatenation of mean and standard deviation of the 256-d backbone features,
`--max-nsites 1000`), the work-stealing Slurm harness and manifests (`build_manifest.py`,
`build_index.py`), the ZBL diagnostics carried from the source. `PLAN.md` gives the scale:
6,744 shards, 756 M structures, 1,600 to 2,500 GPU hours on 16 nodes.

## fps/

`fit_scaler_union.py` (the global scaler of the metric, on the 856,012 MAD-NEXT structures),
`run_fps.py` (multi-GPU NCCL farthest point sampling with the `coresets` package),
`runs/mad_union_meanstd/` (the MAD-NEXT ordering seeded at the hydrogen atom, Fig. 4 and 5),
`submit_fps_family.sh` (the per-family pre-selection of at most 10 M structures for the
large datasets, with the ZBL thresholds 200 eV/Å, 50 eV/atom, 100 eV/atom; the earlier
MAD-1.5 scaler, see `data/README.md`).

## accretion/

`run_accrete.py` (the accretion of the 183 M candidate pool into the MAD-NEXT ordering,
`coresets.cuda.multi.sharded.accrete_sharded`, 4 nodes for the anchor phase), `anchor/`
(pool build, deduplication, filters, `FILTERS.md`, the accretion run), `fps_fill/` (the plain
FPS fill to 10 M and 25 M structures), `cut_2pow24/` (the cut at 2^24, the pool and FPS
manifests, the excluded NOMAD volume-scan entry ids). Level 3 only; the run logs with the
timings of the SI table were deleted.

## coverage_certificate/

The exact separation and coverage checks of every level of the ordering on the released
features (stages A to C), `levels.json` of the release (the source of the size, anchor and
coverage columns of the SI accrete table), `epsilon_exact/` (float64 recomputation of
epsilon_sq). The per-folder arrays (38 GB) stay on the development tree; the JSON results are here.

## voronoi/

Assignment of every structure of the collection to its nearest MAD-CORE structure at all 25
levels (`scripts/voronoi_net.py`, `build_center_tree.py`, `assign_shards.py`), the
population checks, and the within-cell FPS (`cell_fps/`, with `METHODS.md` and `FORMAT.md`).

## sketchmap/

The sketch-map of Fig. 8: sigma from the pair-weighted histogram (`src/derive_params_2p20.py`),
the fit on the first 2^20 structures weighted by population^0.25 (`src/fit_dist.py`, 2
nodes, 79 min), the SiLU projector (`src/train_projector.py`), the projection of the 2^24
(`src/apply_projector.py`), the record populations at level 20 (`record_populations/`), the
chemiscope export. The fit needs the torch sketch-map of the author's scikit-matter fork,
which was not committed at the time of this release (open point P3).

## gfre_lfre/

The feature-space reconstruction errors of Fig. 3 (`union/gfre_lfre_union.py` on the
MAD-NEXT test split: 15,000 stratified structures, 4,000 evaluate and 11,000 fit, ridge
2-fold CV, scikit-matter) and of SI Fig. 1 (`random_test/`, the 6,000-structure random
draw, one featurisation script per model and environment, the bit-identical recompute and
the overlap audits described in its README, `eval_r2scan.py` for the ecumetric energy and
force errors on the 973 r2SCAN structures). `teacher_features/` holds the scripts that
extracted the last-layer features of the five teacher models on MAD-1.5 (from the
`merge-datasets` scripts repository). The result JSONs are here; the per-structure
descriptors (1.1 GB) and the random-test features (1.6 GB) are in the supporting-results archive.

## dft_labels/

`DFT_CAMPAIGN.md` (the frozen FHI-aims protocol, version 260622, tight basis, k-grid density
8, Gaussian smearing 0.05 eV), `lda_pbe/` (the four 25,000 / 6,000 selections with their ids,
the selection and label-building scripts, the 102 isolated-atom references),
`r2scan_random_test/` (the 1,000-structure r2SCAN subsample), `r2scan_selection/` (the
rounds that selected MAD-CORE structures for r2SCAN labels), `record_packaging/` (how the
labels were packaged into the MAD-CORE record). The labels are in the record, not here.

## training_from_scratch/

The eight PET-S models of Fig. 9 and 10 (`scripts/build_sets.py`, `make_runs.py`, the
`options.yaml` of every run, `runs/PROVENANCE.json`: 187,200 optimiser updates, seed 42,
`driver.sh`), the frozen test populations and their disjointness checks
(`reports/populations/manifests/`, `POPULATIONS.json`), the evaluation
(`scripts/evaluate_final.py`, `reports/selected_checkpoints/metrics*.tsv`), the figure data
builder (`scripts/population_figures.py`) and the shared map helpers (`scripts/shared/`).
`manifests/SETS.json` and the id lists define the training sets; the labelled xyz files are
rebuilt from the MAD-CORE record labels with `scripts/build_sets.py`. The six evaluated
checkpoints are in the models archive.

## finetune/, delta_learning/

The PET-OMat-S fine-tune of the SI fine-tune table (`finetune/`), and the r2SCAN−LDA and
r2SCAN−PBE corrections plus the direct r2SCAN fine-tune of the SI delta-learning table
(`delta_learning/`): split construction (`build_shared_split.py`, RandomState(42) over the
20,881 aligned structures), options, reports with the checks (composition baseline,
checkpoint selection by validation `mae_prod`, restart audits), the pretrained-model
provenance (`pretrained_models_PROVENANCE.txt`). The checkpoints are in the models archive.

## water_md/

The 32-molecule water simulations of SI Fig. 3 with PET-MAD-1.6-M minus the learned
corrections: i-PI inputs of the 15 production runs, start files, driver, pilots and
verification JSONs, the analysis (`scripts/analyse.py`, `observables.py`) that wrote the
curves of `figures/siS3_water_rdf/data`, and the bounded DFT single-point check of nine
snapshots (`diagnostics/dft_snapshots/`). `baseline_model/` records the attention-mask-fixed
export of PET-MAD-1.6-M used as the baseline (open point P4). The trajectories (1.1 GB, i-PI
output) are not included; `analysis/production_runs.json` and the curves are.

## data_records/

`FORMAT.md` (the internal release layout the claim scripts read), the build and validation
scripts of that layout, the export scripts of the two Materials Cloud records (`matcloud/`),
the license audit of every source dataset (`license_audit/`), and the record manifests,
READMEs, LICENSES and example scripts as uploaded (`record_manifests/`).
