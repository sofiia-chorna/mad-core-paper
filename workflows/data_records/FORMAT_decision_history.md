# FORMAT.md decision history

Process notes stripped from the released FORMAT.md during the production rewrite
(2026-08-11). The released document states only what IS; the whys and whens live
here.

- **MAD not duplicated in the collection** — decided 2026-07-24: the MAD union
  ships once, as the mad-next-coreset tier.
- **dataset_id readable names** — renamed 2026-08-03 from the opaque ColabFit
  DS-ids (`DS_fblez5twbr07_0`) that were stored previously; 403 ids across 454
  directories. See `release/build/rename_dataset_ids.py` and
  `colabfit_name_overrides.tsv` for the applied DS-id → name mapping. The DS-ids
  are preserved in `dataset_metadata.tsv` (`colabfit_ds_id`).
- **voronoi_population correction** — corrected 2026-08-03 from an earlier version
  that summed to 1,695,275,276, the internal assignment universe, which also
  counted the 677,324 raw mad-1.5/mad-next source frames (216,803 + 460,521); the
  released file sums to 1,694,502,355 = every release row. See
  `release/build/fix_voronoi_population.py`.
- **split mapping proof** — see `release/build/verify_split_mapping.py` for the
  re-hash that pins the mad-1.5-r2scan frames to the MAD-1.5 parquet rows.
- **per-directory id width W** — settled 2026-07-23: W is per directory (max id
  byte length within it). A global W would be 228 bytes (mcloudarchive ids embed
  archive paths) and would bloat the collection id file from ~68 GB to ~386 GB.
  The collection ships as per-group directories only, with plain fixed-width
  string files for dataset identity — no code/vocab indirection.
- **voronoi_d precision** — measurement scripts:
  `release/build/verify_voronoi_recompute.py`,
  `check_voronoi_distance_precision.py`.
- **per-cell FPS cap** — decided 2026-08-11: NO capping; the full within-cell
  ordering ships in `mad-core/cell_fps/` and users cap by epsilon or count.
  The "Later additions (in_cell_fps_rank.npy, cell_members)" sketch in earlier
  FORMAT versions is superseded by that directory (see
  `voronoi/cell_fps/FORMAT.md` and `METHODS.md` for build internals, including
  the checkpoint-resume epsilon artifact).
- **tier renames** — 2026-08-11: `coreset224` → `mad-core`, `mad_coreset` →
  `mad-next-coreset`; dataset_group value `mad-union` → `mad-next` (340,555
  mad-core rows + 249 extxyz chunk headers, see
  `release/build/rename_mad_union_to_mad_next.py`).
- **coordinate fold** — 2026-08-10/11: ~65k structures stored with unwrapped
  periodic coordinates were folded in place, with LAMMPS-style image-flag
  sidecars added and affected extxyz chunks regenerated; mechanism and proofs in
  `release/build/far_from_origin_featurization_bug.md`,
  `fold_release_coordinates.py`, `verify_fold_apply.py`. Manifest
  `format_version: 2` marks the four affected directories.
- **ecumetric model provenance** — `ecumetric_model.pt` is the phace_atomic
  training run at epoch 92 (`outputs/2026-06-26/03-09-50/model_92.ckpt` is the
  matching training checkpoint, shipped as `ecumetric_model.ckpt`).
  `ecumetric_sketchmap.pt` is the SiLU d=8 projector retrain
  (`skmatter-run/sketchmap/projector/projector_d8_p20d8_silu.pt`).
- **configuration_id unification** — 2026-08-13: every native id (442 families,
  1,350,679,490 rows, including gdb13's SMILES-embedding ids and all ColabFit
  `CO_...` ids) replaced by minted `<dataset_group>:<blake2b-16>` ids per
  `release/build/ID_UNIFICATION_SPEC.md`: minted once per distinct structure
  from the first occurrence's release geometry via `inference/ingest/ids.py`,
  so all duplicate/equality relations are preserved one-for-one (verified:
  distinct-id counts unchanged in every family). The old↔new mappings live
  store-side in `mad-core/id_maps/` (never in the release — the gdb13 map
  contains SMILES). gdb13 `source_reference` scrubbed of the `|SMILES` element
  in the same pass (collection/gdb13 + mad-core). MAD tier ids and the 12
  already-minted nc_ingest families untouched.
