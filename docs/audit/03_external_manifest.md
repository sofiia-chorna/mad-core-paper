# External dependency, data and model manifest

Status checked on 2026-10-16. "local copy" gives the file on scratch or the aa002 store that
was used; nothing here is to be redistributed in the code release unless marked.

## Data records (authoritative, referenced from the release)

| item | identifier | status | local copy / manifest | notes |
|---|---|---|---|---|
| MAD-CORE record (2^24 structures extxyz, 512-d features HDF5, LDA/PBE/r2SCAN labels, isolated atoms, control.in templates, chemiscope 2^20 sketch-map, dataset_metadata.csv, LICENSES.md) | DOI `10.24435/materialscloud:rs-tt` as cited in `biblio/others.bib` (`madcore_matcloud`); Materials Cloud draft id in the dev repo | DOI returns 404 (reserved, record unpublished) | `phace-atomic/matcloud-upload/xyz-2pow24-v2/files/` (37 GB), `SHA256SUMS` and `index.json` (27 files, per-file bytes and sha256, content hashes of the gz members); copy on `store:matcloud/xyz-2pow24-v2` | the draft identifier is a private preview token and must not appear in release docs; the record is the source of `dft_{lda,pbe,r2scan}.extxyz.gz` (63.5 / 62.3 / 7.0 MB), `dft_isolated_atoms_*.xyz`, `dft_control.tar.gz` and `dataset_metadata.csv` (1.36 MB) |
| MAD-COSMO record (1.7e9 structures in Voronoi-cell HDF5, features, dataset_metadata.tsv) | DOI `10.24435/materialscloud:58-wq` (`madcosmo_matcloud`) | DOI returns 404; only intervals [0, 2^20) uploaded, the rest waits for quota (dev README) | `matcloud-upload/mad-cosmo-v1/uploaded/files/` and `to_upload/` (3.75 TiB scratch only); `store:matcloud/mad-cosmo-v1` | same remark on the draft id |
| MAD-NEXT / MAD-2 labels | not released (text: "will be available as part of MAD-2") | preprint arXiv 2603.02089 (`Malosso2026`) | `store:universal_features/mad-1.5/mad-1.5-r2scan-all.xyz`, `phace-atomic/data/mad_next/*.xyz` (gitignored) | MAD-1.5 record 2026.167 v4 cited in `dft/sources/mad-1.6/record_2026.167_v4/` (dev repo) |
| Source datasets (ColabFit, OPTIMADE, NOMAD, Materials Cloud, Alexandria, OMat24, OMol25, OC20/22/25, OC20-NEB, ODAC25, GDB-13) | per-dataset DOIs in `biblio/{colabfit,nomad,materialscloud,optimade,extra_datasets}.bib`; licenses per dataset in the record `dataset_metadata.csv` and `LICENSES.md` | external | `release/build/dataset_metadata.tsv` (dev repo) | GDB-13 is redistributed under permission of its authors (text); the SMILES-to-3D conversion code is not in either audited repo |

## Models

| item | identifier / version | license | size, checksum | local copy | used for |
|---|---|---|---|---|---|
| ecumetric model (`ecumetric_model.pt`, metatomic export of the SPACE checkpoint) | https://huggingface.co/lab-cosmo/ecumetric, tag `v1.0.0`, metatomic-torch 0.1.17 | BSD-3-Clause (model card) | 53,381,407 B, md5 2038c977851de01753df1b62da4199f1 (identical in `store:release/ecumetric_model.pt` and `huggingface/upload/`) | yes | featurisation of every structure, GFRE/LFRE, r2SCAN evaluation |
| ecumetric explorer (`ecumetric_explorer.pt`, ecumetric + scaler + SiLU projector) | same repo, tag `v1.0.0`; `chemiscope.explore(..., featurizer="ecumetric-1.0")` | BSD-3-Clause | 55,117,919 B | `huggingface/upload/` | projection of new structures |
| raw training checkpoint `phace_atomic_ep92.pt` (epoch 92, metatrain `experimental.phace`) | not published | n/a | 47,058,643 B, md5 422f1aff84cebc15f11d1b12b7b95704 | `store:models/phace_atomic_ep92.pt` (scratch copy `pre/` deleted) | historical featurisation of the pool; the released model is the same weights re-exported through the SPACE swap (`release/build/ecumetric_space_swap/`, to be confirmed in stage 2) |
| sketch-map projector `ecumetric_sketchmap.pt` (`p20d8_rec_silu`) | inside the HF explorer; standalone file not archived publicly | as above | 1,721,701 B | `store:release/ecumetric_sketchmap.pt`, `skmatter-run/sketchmap/projector/projector_d8_p20d8_rec_silu.pt` | L1 input of Fig. 8 (question Q3) |
| PET-OMat-S v1.0.0 (`pet-omat-s-v1.0.0.ckpt`) | https://huggingface.co/lab-cosmo/upet (models/) | not recorded locally; check the upet model card | 103,774,114 B, sha256 7fc9c915... (`experiments/models/PROVENANCE.txt`) | `mad-core-valid/experiments/models/` | fine-tune, delta and direct r2SCAN runs |
| PET-MAD-1.6-M (`pet-mad-m-v1.6.0`) | lab-cosmo release of MAD-2 / PET-MAD 1.6 (`Malosso2026`) | not recorded locally | 454 MB; sha256 of the attention-mask-fixed export 7e578c97... in `water_delta_18k/scripts/water.py` | `md/lips_delta_10k/models/{pet-mad-m-v1.6.0.ckpt,.fixed.pt,.upstream.pt}` | water MD baseline; the ".fixed" export carries a local attention-mask correction that must be documented (gap G6) |
| Teacher models for the SPACE heads and GFRE: PET-MAD-S 1.5.0, MACE-OMat-0-medium, UMA-S-1p2 (omat), DPA-3.2-5M-OMat24, DPA3-OMol-Large, ORB-v3-conservative-inf-omat | names in `si.tex` and `random_test/README.md`; checkpoints in `merge-datasets/teacher_models/` and `orb_ckpt_cache/` (outside both repos) | per upstream (MACE MIT, fairchem/UMA custom, DeePMD LGPL, ORB Apache-2.0: to be verified on the upstream pages) | not hashed here | outside the audited repos | L3 of Figs. 3 and SI 1 |
| Learned corrections and scratch models (13 checkpoints, 100 MB each) | none | n/a | listed in `inclusion_list.tsv` (data_archive tier) | dev repo `runs/` (gitignored), `store:experiments/` | L2 of Figs. 9, 10, SI tables; L3 of the water MD |

## Software

| package | version used (evidence) | license | role | redistribution |
|---|---|---|---|---|
| `coresets` (multi-GPU FPS and accrete) | github.com/sofiia-chorna/coresets, local clone at 14f72a3 (develop, 2026-10-12); the July runs used commits up to 4285ea0 / 67a932d (2026-07-10); branch `phace-atomic-prod`; no release tag | MIT (`LICENSE`), `pyproject.toml` version 0.1.0 | `run_fps.py`, `run_accrete.py` (`from coresets.cuda.multi.sharded import accrete_sharded`) | reference by commit; a tagged release or Zenodo archive of `coresets` is advisable (question Q5) |
| `scikit-matter` fork with torch sketch-map | github.com/sofiia-chorna/scikit-matter, local checkout at cdfe0fb on branch `sketchmap-tiled`, with **staged, uncommitted** changes (28 files, incl. `_sketchmap_torch.py`, `test_sketchmap_torch.py`) relative to `origin/sketchmap-tiled`; upstream v0.4.0 has `_sketchmap.py` | BSD-3-Clause | sketch-map fit (`fit_dist.py`), GFRE/LFRE (`Ridge2FoldCV`, `global/local_reconstruction_error`) | the fit code the paper used is not committed anywhere (gap G3) |
| `metatrain` | venv-phace: `0.2.0+phace.parquet.prod` (editable, PhACE-era, used for training ep92 and the parquet featuriser); venv-metatrain-fix: `2026.4` (fine-tune, delta, evaluation); ecumetric export: clone `/users/schorna/src/metatrain` branch `fix/space-feature-device` with uncommitted edits to `experimental/space/model.py` | BSD-3-Clause | SPACE architecture, PET training, exports | the SPACE feature-device fix must land upstream or be recorded as a patch (gap G4) |
| `metatomic-torch` | 0.1.11 (venv-phace), 0.1.17 (export and MD) | BSD-3-Clause | model interface | pip |
| `torch` | 2.9.1 (uenv `pytorch/v2.9.1:v2`, GH200) | BSD-3 | all | uenv image on Alps |
| i-PI | version in `venv-ipi-petmad` (not recorded in the audited files; record in stage 2) | MIT | water MD | pip |
| FHI-aims | version 260622 (SI), `control.in` templates in the MAD-CORE record | proprietary academic license | DFT labels | cannot be redistributed; inputs only |
| matplotlib | 3.10.9 (figure folders, venv-phace); 3.11.1 (training figures, venv-metatrain-fix); 3.11.2 (water figure, `venv-ipi-petmad` or the `venvs_feats` set) | PSF-style | plotting | pip; pixel-identical reproduction needs the matching minor version |
| numpy 2.4.4, pyarrow 24.0.0, ase 3.28.0/3.29.0, scikit-learn 1.8.0, h5py 3.16.0, pandas (version not recorded) | venv-phace dist-info | BSD / MIT | data handling | pip |
| orb-models 0.7.0, mace (SanggyuChong/mace@LLPR_universal), fairchem-core 2.19.0, deepmd-kit 3.2.0 (patched `cxx_op.py`) | `random_test/README.md` | Apache-2.0 / MIT / fairchem custom / LGPL-3.0 | random-test featurisation only | not redistributed; document the pins |
| chemiscope | version not recorded | BSD-3-Clause | explorer | pip |
| Ghostscript, ImageMagick 7 | system | AGPL / Apache-2.0 | pixel checks | system |

## Compute resources recorded in the sources

| workflow | resource (evidence) |
|---|---|
| SPACE training to epoch 92 | 1 node x 4 GH200, about 15 min per epoch (`train.log` timestamps), aa002 |
| featurising the collection | 16 nodes x 4 GPU, 1,600 to 2,500 GPU hours (`PLAN.md`) |
| per-family FPS (10 M cap) | 1 to 4 nodes per family (`submit_fps_family.sh`) |
| accretion anchor | 4 nodes x 4 GPU, 2 h allocation (`submit_accrete.sh`); the SI says 23 h on 64 GPUs for the full run through the 25 M fill |
| Voronoi tessellation and cell FPS | multi-node, see `voronoi/PLAN.md`, `cell_fps/mega_job_plan.txt` |
| sketch-map fit | 2 nodes, 79 min (REPRODUCE.md); projector on 1 GPU |
| scratch trainings | 8 runs, 187,200 updates each, two node chains (`runs/PROVENANCE.json`) |
| fine-tune / delta / direct | 1 GPU, 150 epochs, 2 to 3 h each (READMEs) |
| water MD | 10 GPU runs x 50 ps in 30 min segments, plus 5 reused baseline runs |
| DFT | FHI-aims, 72 MPI ranks per frame, campaign of about 80k frames on Alps and Lyra (`DFT_CAMPAIGN.md`) |

## Credentials and private links found

- Materials Cloud draft identifiers for both records (preview tokens, values not reproduced
  here) occur in 19 tracked text files of the dev repo and none of the papers repo:
  `matcloud-upload/README.md`, `matcloud-upload/mad-cosmo-v1/uploaded/REVIEW.txt`,
  `matcloud-upload/xyz-2pow24-v2/upload_state.json`, `release/build/matcloud/README.md`,
  `release/build/matcloud/mad-cosmo-v1/{RUN.md,STATUS.txt,audit.py,finish.py,reupload.py,test_upload_approval.py,upload.py}`,
  `release/build/matcloud/madcore-r2scan-v1/{refresh_store.sh,run_upload.sh}`,
  `release/build/matcloud/xyz-2pow24-v1/{PLAN.md,refresh_store_v3.sh,run_push_check.sh,run_upload.sh,run_upload_v2.sh,run_upload_v3.sh}`.
  Of these, only `matcloud-upload/README.md`, `release/build/matcloud/README.md`, `RUN.md`,
  `reupload.py`, `PLAN.md`, `run_upload.sh` and `refresh_store.sh` are on the inclusion list and
  must be scrubbed (G28).
- Neither source repository has a `LICENSE` file at its root; the dev repo's CLAUDE.md and
  READMEs carry no license statement either. This is why question Q2 is open.
- Hugging Face upload uses the cached `hf auth login`; no token in any tracked file
  (grep for `hf_`, `AKIA`, `WANDB_API_KEY`, `PRIVATE KEY`, `Bearer` found nothing in tracked text files).
- `wandb` project and run names appear in `options.yaml` files (`universal_features /
  phace-atomic-madfull`); harmless but can be dropped from the release copies.
- `.aws/`, `.claude/`, `.agents/`, `.codex/` directories exist in both repos (empty or
  settings only); exclude.
