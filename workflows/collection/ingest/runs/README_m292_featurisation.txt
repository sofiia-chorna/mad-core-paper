inference/runs — reproducible PhACE m292 featurization runs (colabfit + non-colabfit)
======================================================================================

Convention (mirrors fps/runs/): one dir per run, fully self-contained --
  submit.sh        CSCS launch (sbatch submit.sh)            -> calls runs/_run_one_cscs.sh
  submit_kuma.sh   Kuma launch (some runs ran on Kuma)       -> calls inference/_run_one_kuma.sh
  manifest.txt     the run's input list (+ manifest_kuma.txt where it also ran on Kuma)
  NOTES.txt        what / where / status
  logs/            SLURM logs (submit.sh --output's here; git-ignored like fps/runs)

Worker = runs/_run_one_cscs.sh (CSCS paths PINNED; the shared inference/_run_one.sh flips
CSCS<->Kuma so historical scripts use this pinned copy). All steps idempotent (skip if the
_with_phace output exists). Older runs used the generic inference/submit_array.sh + MANIFEST
env; these dirs are retrofitted from the saved manifests + logs. Per-shard provenance also
lives in each *_with_phace.json sidecar (slurm_job_id, host, struct_per_s, completed_at).

CSCS runs (exact)
  omat_m292            OMat24 filtered (646 shards)            phace-omat-m292   DONE
  omat_m292_rej025     OMat24 rej_passes_at_025 (626)          phace-rej025      DONE
  vesin053_refeat      vesin-0.5.3 remediation recompute (779) phace-array       DONE  (+submit_kuma.sh)
  colabfit_rej025      colabfit rej_passes_at_025 (297)        phace-rej025      DONE
  spice                SPICE_2023 (2)                          phace-spice       DONE
  vqm24                Vector-QM24 DFT_uniques (3)              phace-vqm24       DONE
  vqm24_alex2d         VQM24 + Alexandria_PBE_2D (89)          phace-vqm24       DONE
  oc20_is2re_refeat    OC20-IS2RE 16 corrupt-shard refeat      refeat-oc20-d4    DONE
  oc22_is2re_refeat    OC22-IS2RE 2 corrupt-shard refeat       refeat-oc22       DONE
  colabfit_filtered    ORIGINAL big colabfit featurize (778, relative manifest) phace-array  DONE  [BEST-EFFORT]

Kuma runs (best-effort reconstructions; driver inference/_run_one_kuma.sh, /transfer paths)
  alexandria_pbe3d     Alexandria PBE_3D (2054)                                   DONE
  materials_project    Materials_Project (64)                                     DONE
  sn2                  SN2_JCTC_2019 (1)                                          DONE
  oc_recompute         OC re-featurize (6)                                        DONE

Non-colabfit (new pipeline, inference/nc_ingest/) -- STILL in inference/nc_ingest/runs/
  while their jobs run; MOVE here once complete:
  OMol25_val, OMol25_train_passed, OMol25_train_rej025, Alexandria, nomad
  (fused convert+featurize via inference/nc_ingest/_nc_one.sh, metatomic/vesin 0.5.8, lean cols).
  OC20_S2EF_train_all still pending (needs the 26k-file grouping).

NOT runs (helper / transfer / feature-list manifests, left in inference/manifests/):
  *.relpaths.txt, *_FEATURES.relpaths.txt, *_push.txt, *_feature_dirs.txt, *_feature_files.txt,
  *_kuma_source_dirs.txt, kuma_sources*.txt
