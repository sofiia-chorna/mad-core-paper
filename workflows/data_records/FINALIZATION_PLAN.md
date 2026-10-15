# Release finalization plan — fold, validate, freeze

Written 2026-08-10; STATUS 2026-08-11: Step 1 APPLIED (64,959 rows, log
`release/build/logs/fold_release_apply.log`), Step 2 VERIFICATION PASSED (worst
reconstruction 1.5e-05 A, untouched bytes identical everywhere, log
`logs/verify_fold_apply.log`), Q1 DONE (`dataset_metadata.tsv` at release root,
655 entries, totals verified). Step 3 DONE (24 of 256 coreset224 extxyz
chunks regenerated + verified, manifest hashes updated, .prefold backups kept;
mad_coreset extxyz untouched, no displaced rows). Q7 DONE (image-flags section added
to FORMAT.md). IMPORTANT NEW ITEM for the manifest pass: the per-dir MANIFEST.json
sha256 entries for x.bin in the 4 patched dirs are now STALE (x.bin changed) and the
image_flag files need entries — fix together with Q10 (format_version bump).
2026-08-11 later: Q12 DONE (256 extxyz chunks renamed to frames_rSTART-END with
level-alignment documented in manifest + FORMAT.md). Q11 RESOLVED as keep+document
(dataset-id naming rule now in FORMAT.md, DS-id mapping via dataset_metadata.tsv).
Prefold policy agreed: delete extxyz .prefold + move x.bin.prefold to audit/ AFTER
Step 4 validation passes.
Next: Step 4 validation matrix, decisions Q2, Q10, model self-containment,
stale MANIFEST sha256 refresh for the 4 patched dirs.

2026-08-11 afternoon: Q13 DONE — coreset224 renamed to mad-core everywhere (dir,
both MANIFEST tier fields, FORMAT.md 16 refs, audit.md, audit json filename, the
four repo scripts). Q14 DONE — ecumetric_projector_silu.pt/.json (SiLU d8 retrain)
and phace_atomic_ep92.pt copied to release root next to ecumetric_scaler.npz
(model self-containment gap CLOSED — add sha256 entries in the manifest pass).
Q15 decided: NO capping, ship the full ordering. SPACE FACT: /capstor/store/cscs
is 93% full, 1.2 TB free — the raw 3.9 TB cell_fps scratch output CANNOT and need
not move; the release form (in_cell_fps_rank.npy ~6.8 GB int32 + cell_members
offsets+blob ~7-14 GB) fits easily. Build those two artifacts from
voronoi/out/cell_fps/{binary,fps} (ignore *_stale_* dirs) in the clean session.

2026-08-11 evening renames DONE: release root now has ecumetric_model.pt (was
phace_atomic_ep92.pt — PROVENANCE: phace_atomic epoch 92, record this in FORMAT.md),
ecumetric_model.ckpt (copied from outputs/2026-06-26/03-09-50/model_92.ckpt, the
training checkpoint), ecumetric_sketchmap.pt + ecumetric_sketchmap.json (SiLU d8
projector; json KEPT — 451 bytes holding architecture + quality metrics R2/MSE,
recommend keeping, Sofiia may delete). mad_coreset renamed to mad-next-coreset
everywhere (dir, both manifests incl. extxyz file key, extxyz file itself,
FORMAT.md 11 refs, audit.md, audit json name, fold script).
STILL OPEN from this: the dataset_group VALUE "mad-union" inside mad-core and
mad-next-coreset dataset_group.npy files — rename value to "mad-next"? (data
change in 2 tiers, needs Sofiia's yes = the old Q2, now the last mad-union trace).
CELL-FPS INTEGRATION FACTS: voronoi/out/cell_fps/binary/ is release-ready:
center_configuration_id.npy + center_offsets.npy (per mad-core cell), then per
member fps_order_in_cell.npy, epsilon_sq.npy, configuration_id (ragged
bytes+offsets, 51+13 GB), AND direct pointers release_directory.npy +
release_row_in_directory.npy (9.6 GB) + release_manifest_sha256.npy pins.
~90 GB total (skip the ids/ and _id_lengths.npy intermediates). NOTE: the
manifest sha pins will be invalidated by the upcoming MANIFEST refresh — record
old+new pins or rebuild the pin file. Decision: ship ids blob AND pointers
(~90 GB) or pointers only (~30 GB)? Filesystem is 93% full.

New items 2026-08-11 (from Sofiia):
- **Q13 rename coreset224 -> mad-core**: COLLISION FLAG — parent dir is already
  mad-core and sibling tier is mad_coreset (hyphen vs underscore mix). Cheap to do
  (mv + tier fields in manifests + FORMAT.md + audit names, no data rewrite) but
  needs the exact final name and her confirmation of the collision.
- **Q14 ecumetric checkpoint to release root**: candidates
  `skmatter-run/sketchmap/projector/projector_d8_p20d8_silu.pt` (+ .json, SiLU
  retrain — believed production) vs older `projector_d8_p20d8.pt`. Confirm which,
  then copy next to ecumetric_scaler.npz + manifest entry.
- **FORMAT.md production rewrite**: strip internal decision notes ("ask Michele",
  Settled-dates, corrected-from footnotes) into repo-side provenance, fix stale
  counts (257 -> 256 chunk files), her review before landing.
- **Q15 voronoi in-cell orderings**: cell-FPS outputs DONE in
  `voronoi/out/cell_fps/` (3.9 TB, ignore *_stale_* variants). Release integration
  (in_cell_fps_rank.npy + cell_members per FORMAT.md "Later additions") never
  built. Needs the per-cell cap decision (FORMAT.md open question) OR ship the
  uncapped rank column (~6.8 GB) and let users cap. Then build + validate.

2026-08-11 night (clean session): TODO 1 DONE — mad-union -> mad-next applied
(`rename_mad_union_to_mad_next.py --apply`): mad-core/dataset_group.npy 340,555
rows + the same 340,555 frames across 249 extxyz chunk headers + 2 FORMAT.md
mentions. IMPORTANT premise correction: mad-next-coreset NEVER contained
"mad-union" (S8 array, values mad15/mad_next/mad_sol; its extxyz clean, verified)
— the plan's claim of 856,012 mad-union rows there was wrong; no change needed.
TODO 2 cell_fps: 90 GB copied to release/mad-core/cell_fps/ (rsync, excl ids/,
_id_lengths, logs), stale name mad_coreset -> mad-next-coreset patched in
release_directory_names.npy. Validated with new `validate_cell_fps.py` against
voronoi/cell_fps/FORMAT.md invariants (NOT the plan's naive ones: rank -1 =
no-ordering rows [486,653 documented], duplicates repeat twin's rank+eps,
eps rises ~1e-6 allowed at resume boundaries). mad-core epsilon_sq has exactly ONE
resume rise at row 10,000,000 (rel 5.5e-5) — checkpoint boundary of production
FPS, tolerated + recorded. New Step-4 orchestrator `validate_release_final.py`
(parts: coreset/madnext/crosstier/extxyz/scan) + `submit_validate_release.sh`;
`refresh_manifests.py` written (run AFTER validation green; root manifest step
re-runnable after FORMAT.md rewrite lands). FORMAT.md production rewrite DRAFTED
repo-side (awaiting Sofiia's review; decision notes moved to
`release/build/FORMAT_decision_history.md`); release copy NOT yet replaced.

2026-08-11 late evening (fleet + findings): Step-4 scan parallelized WITHIN
directories (range tasks cut at equal-atom boundaries; gdb13 = 62 tasks, 7.5 min
vs >90 single-stream). Sofiia-approved additions DONE: (a) fps_order injected
into ALL extxyz (mad-next file + 256 mad-core chunks, verified sequential,
validator asserts fps_order==row; extxyz part re-PASSED); (b) mad-core cross-tier
pointer files built from cell_fps rank-0 pointers and FULLY verified (16.7M-row
id round-trip + dataset_group==folder rule for every row) — STAGED at
release/build/staging/madcore_pointers/, copy into release/mad-core/ pending.
Validation findings: stray collection/OMol25/.claude (147-byte settings file,
Jul 29) deleted with approval; mad-core row 115363 is a vv6sy no-cell molecule
(documented Q5 exception) inside the coreset — validator now allows far rows
whose dataset_group is vv6sy. REAL CATCH: 5,154 nomad rows (all filter_code 1,
none in any coreset, coords to 1.47e10 A, structure extent itself ~1e10) were
MISSED by the fold: fold_release_coordinates.py globbed only nomad/nomad/*.parquet,
missing the filtered_rej_passes_at_025/ and rejected/ tail-shard subdirs (proven:
all 5,154 lie in release rows [5,396,859..5,408,927), the two tail shards). Fix:
recursive glob in fold + verify scripts; expected nomad flagged rows
63,839 -> 68,993 (FOLDED_DIRS + FORMAT.md draft updated). OVERNIGHT SLURM CHAIN
(afterok): fold dryrun 3057943 -> apply 3058003 -> {verify_fold 3058004,
scan4 3058005, scan5 3058006} -> merge 3057776->3058007. Shards 0-3 PASSED
(shard-3 rerun after .claude removal); light + extxyz + cell_fps + crosstier all
PASSED. Apply is idempotent: existing x.bin.prefold PRESERVED (true original),
sidecars regenerated as the union in row order.

2026-08-12 night COMPLETE: overnight chain ALL GREEN (apply 70,113 rows, verify
worst dev 3.8e-6 A, scans 4+5 PASSED, merge PASSED incl exact voronoi closure).
Pointer files copied into release/mad-core/ + revalidated in place. Manifest
refresh DONE (456 tiers stamped v1.0, format_version 2 on the 4 flag dirs,
pointer + image-flag + fps_order entries, cell_fps MANIFEST, pins rebuilt w/ old
archived, root MANIFEST.json = 7 files + 459 dir manifests). DISCOVERY: the 4
x.bin.prefold were HARD LINKS to live x.bin (same inode — never real snapshots;
prefold byte-identity checks were vacuous). Closed with verify_untouched_rows.py
(debug job): EVERY unflagged row in the 4 patched dirs byte-identical to store
f32, full coverage — then all prefold files deleted (extxyz .prefold + the 4
links, Sofiia approved). audit/RELEASE_VALIDATED_2026-08-11.md written (also
mirrored to repo release/build/audit/).

REMAINING: 1) Sofiia reviews FORMAT.md rewrite (repo diff of release/FORMAT.md;
decision notes in release/build/FORMAT_decision_history.md) — after approval:
cp to release root + `refresh_manifests.py --only root`; 2) Q10 final: DOI still
null placeholder, release_version "1.0" — change if wanted (then rerun refresh);
3) Sofiia git add/commit + one Slack line with audit summary link.

2026-08-13 evening — ID UNIFICATION, second pass (unattended chain running).
The first pass (minting per family) BROKE global duplicate identity: bare
ColabFit ids (`CO_...`) are shared ACROSS families and that sharing is what the
global `is_duplicate` dedup — and therefore the FPS pool — was built on.
12,459,512 structures got two ids. Caught by the cell_fps tie invariant; the
per-family bijection gate could not see it. Full analysis + decision:
`release/build/ID_UNIFICATION_SPEC.md` (AMENDMENT section).
FINAL SCHEME (Sofiia's): hash minted once per global class (at the
`is_duplicate == False` row), each row keeps its OWN family prefix →
`A:9455…` / `B:9455…`, so **is_duplicate ⇔ equal hash part** and every prefix
still equals its own directory. FORMAT.md updated accordingly.
BACKUPS moved off scratch to `aa002/mad-core/id_backups/{old_cids,cellfps,refs}`
(181 GB, outside release/), next to `id_maps/`.
EDGE CASE found by the dry-run (nothing written, it stopped): 14 duplicate
classes have NO unflagged row in the release. Cause: dedup ran over 1,693,741,940
rows but only 1,693,646,343 shipped, so ~95,597 deduped-against rows were never
released; a class whose original is among them keeps only flagged copies.
Resolution: canonical hash = the class's EARLIEST RELEASED row (same first-wins
rule, applied to what shipped). check_id_consistency.py asserts exactly
ORPHAN_CLASSES = 14 such classes rather than failing. Detail job: 3075929.
CHAIN v2 (afterok, staging/idfix_chain.txt): dry 3076097 -> fix 3076098 ->
madcore 3076099 -> {cellfps 3076100, extxyz 3076101} -> manifests 3076102 ->
{validate 3076103 (full matrix + cell_fps), consistency 3076104, restore 3076105}.
NEXT SESSION, FIRST ACTION (agreed, deferred because SLURM spools the batch
script and the queued jobs call release/build/<script>.py by absolute path — a
move while the chain runs breaks them): git mv the id-unification work into
release/build/id_unification/ — ID_UNIFICATION_SPEC.md, ID_BACKUPS_README.md,
build_id_maps.py, fix_shared_hashes.py, apply_new_cids.py, rebuild_cellfps_ids.py,
scrub_gdb13_references.py, add_colabfit_to_references.py, rewrite_extxyz_ids.py,
check_id_consistency.py, check_id_restore.py, run_id_fix_pipeline.sh (update the
$build paths inside it). refresh_manifests.py / validate_release_final.py /
validate_cell_fps.py STAY in release/build/ — general release tooling.

Then: check those verdicts first. `check_id_consistency.py` proves the
new semantics globally (not sampled); `check_id_restore.py` proves the old ids
are recoverable. If green: delete `release/build/staging/` leftovers, write the
audit addendum, and the release is frozen again. If any job failed, the chain
stops there and nothing downstream ran.

2026-08-14 — ID UNIFICATION COMPLETE AND VERIFIED. Three passes: re-mint of
1,350,679,490 native ids (442 families); shared-hash fix re-pointing 12,560,338
rows after the ColabFit cross-family sharing trap; prefix unification of
OC25/ODAC/OMol25 (207,570,620 rows, prefix only). Final scheme:
`<dataset_group>:<32-hex>`, hash = structure, prefix = own directory, so
is_duplicate <=> equal hash part. ALL GATES GREEN: full matrix PASSED (456 dirs,
Voronoi closure, extxyz, cross-tier); consistency EXHAUSTIVE — 1,641,592,998
unflagged rows all distinct, 52,053,328/52,053,345 duplicates matched, 14 known
orphan classes (17 rows, Massive_Atomic_Diversity_bench, originals in the MAD
tier), and 1,641,592,998 + 52,053,345 = 1,693,646,343 exactly; cell_fps PASSED;
restore PASSED; GDB13 SMILES removal proven by scanning all 30,753,452,423 bytes
of the gdb13 reference blob (zero `|` separators) + all 1,720 mad-core gdb13
rows. Audit addendum: audit/RELEASE_VALIDATED_2026-08-14_id_unification.md.
Scripts reorganized into release/build/id_unification/. ColabFit CO_ ids appended
to source_reference (206 families + 12.35M mad-core rows) + manifest refresh.

REMAINING: (1) Sofiia reviews FORMAT.md (git diff) then commits the staged work;
(2) DOI still `null` in every manifest — fill once Materials Cloud mints it, then
rerun `refresh_manifests.py --only root`; (3) send the Materials Cloud upload
request (~6 TB) and the GDB13 usage-policy note via Michele; (4) old exported
artefacts (e.g. madcore_r2scan xyz) carry pre-migration ids — join them by row
pointers or through aa002/mad-core/id_maps/.

Self-contained: a fresh session starts here. Goal: the release at
`${STORE_ROOT}/mad-core/release` fully folded, internally
consistent, validated file-by-file, with manifests and naming settled.

Context in one paragraph: ~65k structures (nomad + 2 mcloudarchive groups) were stored
with unwrapped periodic coordinates up to 10^10 A. Their stored phace predictions and
features were computed on those coordinates and are only reproducible from them
(mechanism + proofs: `release/build/far_from_origin_featurization_bug.md`). Decision
(thread, 2026-08-10): keep all predictions exactly as computed; fold the released
coordinates; store integer image flags (LAMMPS-style) so `original = folded +
flags @ cell` and everything stays reproducible from the release alone; xyz exports
show folded coordinates only. Verified in advance: reconstruction gives bit-identical
neighbour lists (170/170 sampled) and |dE| <= 1e-6 eV, feature dist <= 2.3e-5 (rerun
noise is ~3e-4): `test_lossless_fold.py`, `test_lossless_fold_predictions.py`.

## Step 1 — apply the fold (the ONLY step that modifies existing release bytes)

Tool: `release/build/fold_release_coordinates.py` (dry-run by default; `--apply` to
write). Dry-run output 2026-08-10 (logs/fold_release_dryrun2.log):

| release dir | rows folded | atoms | sidecar |
|---|---|---|---|
| coreset224 | 29 | 469 | ~13 KB |
| collection/nomad | 63,839 | 875,455 | ~25 MB |
| collection/mcloudarchive__2ftdw-8cr18 | 1,088 | 160,640 | ~3.8 MB |
| collection/mcloudarchive__b4kgx-rtd78 | 3 | 144 | ~4 KB |
| mad_coreset + all other dirs | 0 | — | — |

What `--apply` does, exactly, per affected dir:
1. Hard-links `x.bin -> x.bin.prefold` (instant snapshot; rollback = `mv` back).
2. Overwrites ONLY the affected rows' spans inside `x.bin` (offsets from `na.npy`)
   with float32 folded positions computed in float64 from the store. A guard skips
   any row whose store/release atom counts disagree (prints SKIP, never writes).
3. Writes three NEW files: `image_flag_rows.npy` (int64 row indices),
   `image_flags.bin` (int64 per-atom 3-vectors, rows concatenated in listed order),
   `image_flag_cells.bin` (float64 3x3 per listed row — needed because flags ~2e9
   times the float32 cell error would be ~500 A; the float64 cell makes
   reconstruction exact to ~1e-6 A).
No other file is opened for writing. `mad_coreset` is untouched. Molecular structures
with no cell (44 in vv6sy) are untouched — nothing to fold, see Q5.

Displacement handling for partial cells: fold along active (nonzero) lattice axes
only; flags on inactive axes are 0.

## Step 2 — post-apply verification (before anything else happens)

New script `release/build/verify_fold_apply.py` (to write; all read-only):
1. Per affected dir: reconstruct `x[r] + flags @ cell` for every flagged row and
   compare against the store float64 originals: max deviation must be < 1e-4 A.
2. Every NON-flagged row of every patched `x.bin`: byte-identical to `x.bin.prefold`.
3. Every flagged row: max|x| now < cell diagonal * 2 (sane), and == store fold to f32.
4. `image_flag_rows.npy` sorted, unique, in range; flags nonzero per listed row;
   `image_flags.bin` length == sum of listed rows' atoms * 24 bytes.
5. Model-level spot check (GPU, ~20 rows incl. the 29): featurize(reconstruction)
   vs stored prediction path — |dE| <= 1e-5 eV.
Write results to `audit/fold_2026-08-10/` (see Q9).

## Step 3 — regenerate xyz exports

`coreset224/extxyz/` (257 files) was exported from pre-fold `x.bin`. Regenerate with
`release/scripts/export_extxyz.py` (folded coordinates only, per thread decision —
no flags in the xyz). `mad_coreset/extxyz` has no displaced rows: verify one file
byte-identical after re-export, or leave untouched. Validate: parse every regenerated
file, frame count == ns, coordinates == x.bin values, global max|coordinate| < 1e4.

## Step 4 — full validation matrix (every file, every tier)

Run as one new committed script `release/build/validate_release_final.py`, results
into `audit/`. Checks marked (*) are the non-obvious ones.

Common to every dir (collection families + coreset224 + mad_coreset):
- sizes: `x.bin == na[-1]*12`, `a.bin == na[-1]*4`-or-dtype, `c.bin == ns*36`,
  every per-structure .npy has length ns; `na` monotone, `na[0]==0`, `ns.npy == ns`.
- `configuration_id`: family prefix matches dir; no empty ids.
- `pbc.npy` consistent with nonzero rows of `c.bin` (*): pbc true => cell row nonzero.
- MANIFEST.json: `ns` matches, `row_order_shards` counts sum to ns.
- no coordinate anywhere above 1e4 A except documented vv6sy molecules (*) — the
  single check that proves the fold landed everywhere it should.

coreset224 specifics (all files):
- `fps_order.npy` is a permutation of 0..ns-1 (*).
- `epsilon_sq.npy` non-increasing along fps_order (*) — the FPS invariant; any
  violation means ordering corruption.
- `k_set.npy` consistent with doubling levels; the first 2^k rows by fps_order equal
  the k-level set (*) — nested-coreset invariant, check for k = 10..24.
- `voronoi_population.npy`: sums to 1,694,502,355 exactly (documented number);
  min >= 1; the 29 folded rows keep their populations (untouched by fold).
- `ecumetric_features.bin`: ns * d * 4 bytes; recompute from store phace features
  through `ecumetric_scaler.npz` for 1000 random rows + the 29 -> match stored (*).
  For the 29 this doubles as proof the fold did NOT touch features.
- `ecumetric_energy` / `ecumetric_nc_forces`: finite everywhere; nc_forces length
  == na[-1]*12; spot-recompute 100 rows.
- `zbl_*.npy`: spot-recompute on RAW (store) coordinates for 200 rows — stored
  values are as-computed-at-selection by design (see Q3/Q4).
- `source_reference_*`: offsets monotone; every entry parses; resolve 50 random
  entries + all 29 far rows to their upstream file/frame (*).
- `dataset_group` / `dataset_id`: value sets match dataset_metadata.tsv (see Q1);
  per-group counts match FORMAT.md numbers.
- cross-tier (*): all coreset224 ids exist in their collection family (sampled
  10k + all 29); the 340,555 mad_coreset overlap rows match id-for-id (documented).
- Voronoi closure (*): histogram of `voronoi_a.bin` over ALL collection families
  == `voronoi_population.npy` exactly (big streaming job, 1 node; the strongest
  single consistency check in the whole release); each of the 29 centers is
  assigned to itself with d == 0 in its collection row.
- extxyz: as Step 3.

mad_coreset specifics:
- `split.npy` values in expected set, counts vs FORMAT.md; `voronoi_a.bin` values
  < 2^24; `voronoi_d.bin` finite, >= 0; overlap rows' voronoi_a equal the
  coreset224 self-index mapping (*); no displaced rows (assert no image_flag files).

collection specifics (454 families):
- per-family: the common checks; `filter_code`/`is_duplicate` present, released
  count (code==0 & !dup) matches audit numbers; `voronoi_a/d` present and in range.
- the four patched families: image-flag checks from Step 2.

audit dir: refresh `audit.md` + per-tier jsons with the new checks' outputs; add
the fold event with date, script hash, and dry-run/apply logs.

## Step 5 — flagged decisions (answer before or during; none block Step 1 except Q2/Q7 wording)

- **Q1 dataset_metadata.tsv is MISSING from the release** although FORMAT.md line 25
  documents it as the home of every DOI / license / upstream URL (655 entries).
  `release/build/build_dataset_metadata.py` exists. Rebuild and place at release
  root? (Strong yes recommended — without it the release has no citation info.)
- **Q2 "mad-union" vs "mad-next"**: `dataset_group` in mad_coreset uses `mad-union`;
  FORMAT.md explains mad-1.5/mad-next/mad-sol were "assigned internally but are not
  part of the release". You said "use mad-next instead of mad-union" — decide the
  exact renaming scope: (a) only FORMAT.md wording, (b) also the `dataset_group.npy`
  values (data change! needs its own mini-verify), or (c) keep `mad-union` and
  document it. Note 340,555 coreset224 rows carry the same group value — renaming
  must touch both tiers identically.
- **Q3 the 8 released rows whose folded ZBL exceeds the cut** (worst 30,262 vs 200;
  `pool_far_zbl_folded.tsv`): stored zbl columns + filter_code stay as-computed
  (consistent with the predictions-unchanged philosophy) with a note in FORMAT.md,
  or flip their `filter_code` / update zbl columns? (Recommend: keep + document;
  they are collection rows, none in any coreset.)
- **Q4 zbl columns of far rows generally** are raw-coordinate values — same
  keep-and-document question as Q3, wider scope (~65k rows).
- **Q5 the 44 vv6sy no-cell molecules** at up to 3.2e4 A: leave as-is (recommended;
  open boundaries, predictions fine to ~1e-3) or translate to origin with a
  per-structure shift sidecar?
- **Q6 x.bin.prefold backups** after validation passes: keep in place, move under
  `audit/prefold/`, or delete? (Recommend move under audit/.)
- **Q7 naming of the new files**: `image_flag_rows.npy` / `image_flags.bin` /
  `image_flag_cells.bin` (LAMMPS "image flags" convention) — confirm, then FORMAT.md
  gains a section with the reconstruction formula and the caveat: featurizing the
  folded coordinates directly gives correct-but-different values for flagged rows;
  reproducing stored columns goes through the flags.
- **Q8 collection keeps filter-rejected rows** (filter_code != 0), including most of
  the 63,839 folded nomad rows — confirm this is intended and documented (it is how
  the release was built; folding them anyway is recommended and is what Step 1 does).
- **Q9 audit location** for fold + validation outputs: `audit/fold_2026-08-10/` ok?
- **Q11 `colabfit_` prefix on dataset_id**: 91 of ~118 dataset_id values carry the
  prefix (`colabfit_ANI-1`), non-colabfit ones are bare (`Alexandria_pbesol`). No raw
  DS_xxx ids exist anywhere (checked). Options: (a) keep + document — the prefix is
  the provenance marker, zero cost (recommended), or (b) strip it — touches
  dataset_id.npy in all tiers, dataset_metadata.tsv, FORMAT.md, AND the dataset_id
  values embedded in every extxyz frame (full 27 GB re-export + mad_coreset file).
- **Q12 extxyz arrangement**: chunks of 65,536 = 2^16 frames mean the files ALIGN
  with coreset levels: frames_0000 alone IS the 2^16 coreset, files 0-1 = 2^17,
  0-3 = 2^18, first 2^(k-16) files = level k. Options: (a) document this in
  FORMAT.md + extxyz MANIFEST (zero cost, recommended), (b) also rename files to
  embed row ranges (frames_r00000000-00065535.extxyz — pure rename + manifest
  update), (c) group into per-level subdirectories (more churn, little gain).
- **Prefold backups, refined**: extxyz `.prefold` files (24, ~2.8 GB) contain the
  OLD unwrapped coordinates inside the release tree — after Step 4 validation is
  green, DELETE them (regenerable from x.bin any time). The four `x.bin.prefold`
  hard links: move under `audit/prefold/` or delete after validation (store keeps
  the originals regardless) — Q6 decision.
- **Q10 release-level DOI / version stamp**: should FORMAT.md + each MANIFEST.json
  carry a release version (e.g. `format_version: 2` for dirs gaining image flags)
  and the dataset DOI once minted? Decide the version string now so every manifest
  is stamped in one pass.

## Step 6 — freeze

1. All validation green -> write `audit/RELEASE_VALIDATED_<date>.md` summarizing
   every check + counts.
2. Update FORMAT.md: image-flags section, fold event note, Q-decisions as decided.
3. git add (Sofiia commits): `fold_release_coordinates.py`, `verify_fold_apply.py`,
   `validate_release_final.py`, FINALIZATION_PLAN.md, FORMAT.md, the bug writeup,
   and the test_* scripts from the investigation.
4. Slack: one line + link to audit summary.

## THE ORDERED TODO FOR THE CLEAN SESSION (state as of 2026-08-11 end of day)

All decisions are made. Steps 1-3 applied and verified. Execute in this order:

1. **mad-union -> mad-next value rename** (APPROVED): in `mad-core/dataset_group.npy`
   (340,555 rows) and `mad-next-coreset/dataset_group.npy` (856,012 rows), plus the
   remaining `mad_union` mentions in FORMAT.md. Fixed-width S arrays: "mad-next"
   (8 bytes) fits inside the existing width. Verify: value counts before/after
   identical, no other value touched, extxyz dataset_group fields in
   mad-next-coreset.extxyz (it embeds dataset_group per frame? CHECK — if yes the
   single extxyz file needs re-export too).
2. **Cell-FPS integration** (APPROVED: ship EVERYTHING — ids blob AND pointers,
   ~90 GB): copy voronoi/out/cell_fps/binary/ (excluding ids/, _id_lengths.npy,
   logs/) into `release/mad-core/cell_fps/`, then validate: center count == 2^24,
   center_configuration_id matches mad-core/configuration_id row-for-row,
   member counts sum == collection total, fps_order_in_cell is a permutation
   within every sampled cell, epsilon_sq non-increasing within cells,
   release_directory/row pointers resolve to the right configuration_id for
   sampled members, and the level-24 cell membership agrees with collection
   voronoi_a. Then FORMAT.md section for it.
3. **Step 4 validation matrix** (`validate_release_final.py`, see Step 4 above) —
   including voronoi_a/d presence in all 454 families and the population closure.
4. **MANIFEST refresh, one pass**: recompute sha256 for changed files (x.bin in 4
   dirs, both extxyz manifests' parents), add entries for all new files
   (image_flag_*, cell_fps/, root-level model/scaler/sketchmap/metadata files),
   bump format_version, stamp release version (Q10: pick string, add DOI field
   placeholder), rebuild release_manifest_sha256.npy pins inside cell_fps.
5. **FORMAT.md production rewrite** (approved): clean reference doc, decision
   history to repo-side, record ecumetric_model.pt = phace_atomic epoch 92,
   image-flags section stays, cell_fps section added, fix any stale counts.
   Sofiia reviews before it lands.
6. **Prefold cleanup** (after 3 is green): delete extxyz *.prefold (24 files),
   move the four x.bin.prefold into audit/prefold/ (or delete — Sofiia's call
   at that moment). Refresh audit/ summaries with all new checks.
7. git add everything repo-side (Sofiia commits), one Slack line with the audit
   summary link.

Open micro-decisions left (none block the start): ecumetric_sketchmap.json keep
or delete (kept for now, recommended keep), x.bin.prefold final fate in step 6.

## Standing safety rules for the session

- Everything under `${STORE_ROOT}/` is modified ONLY by
  `fold_release_coordinates.py --apply` (Step 1) and the extxyz re-export (Step 3),
  each announced before running. Everything else is read-only.
- Any check that fails: stop, report, decide together — no silent fixes.
- The store parquets are never touched (they are the provenance archive).
