# scratch_scaling_v3: isolated atom reference round

Eight PET-S models from scratch (MAD-CORE vs random, 2k / 5k / 10k / 20k ordinary structures from the round 2
orderings) with the 102 isolated atom references (one non periodic atom per element, zero atomization energy and
zero forces, cut from `dft/lda_pbe/references/monomers_pbe.xyz`) appended to every training set. Validation and
selection as in round 2 (combined 1000 random + 250 MAD-CORE, native mae_prod). Motivated by the isolated atom
failures documented in `../scratch_scaling/reports/audit/final_mae_10k/`.

- `scripts/build_sets.py` writes `manifests/` (SETS.json, CHECKSUMS.sha256, `*_ordinary.tsv`, `*_aug.tsv`, xyz).
- `scripts/make_runs.py` writes `runs/` (options, node scripts with `--nice=2000`, mixed batch smoke, PROVENANCE.json).
- `scripts/check_scratch_init.py`, `check_selection.py`, `check_references.py`: smoke checks.
- `scripts/driver.sh`: detached driver (smoke, production chains, isolated atom tracking, per run init checks,
  evaluation, report, dated 08:00 copy of the status).
- `scripts/iso_track.py` + `isolated_atoms.py`: isolated atom diagnostic over snapshots.
- `scripts/submit_eval.sh`, `evaluate_final.py`, `val_subsets.py`, `maps_common.py`, `make_maps.py`: evaluation and figures.
- `scripts/build_report.py` writes `reports/STATUS.md`.

Everything here is uncommitted and unstaged.
