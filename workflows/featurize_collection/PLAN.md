# Re-featurize zbl_filtering/output with phace-atomic (ep92) — execution plan

## Goal
Re-run PhACE inference with the **new phace-atomic checkpoint** on **every** structure in
`${STORE_ROOT}/zbl_filtering/output` (except gdb13), writing a clean,
versioned, permanent parquet store. Storage holds features for everything; ZBL/size selection
is a **downstream FPS concern**, not baked into the store.

## Scope (measured 2026-06-30)
- Source total: **7,724 files, 766,013,730 structures, 2.9 TB** across ALL variants
  (`filtered`, `co_passed`, `filtered_rej_passes_at_025`, `rejected`, collection dirs).
- **Exclude `gdb13/`** (980 files, 10,000,000 structures — treated separately later).
  ⇒ in scope: **~6,744 files, ~756M structures**.
- Per-file rows: max 3.6M, p99 700k, p50 50k, mean 99k.

## Model
- `pre/phace_atomic_ep92.pt` (epoch 92, Jun 26; md5 `422f1aff84cebc15f11d1b12b7b95704`). Latest.

## Filesystem reality (measured 2026-06-30) — there is NO /work
| FS | free space | free inodes | purged? |
|---|---|---|---|
| `/capstor/store/.../swissai/aa002` | **3.5 TB** (78% used) | **~18k** (82% used) | no (persistent, **shared**) |
| `${SCRATCH_ROOT}` | 148 TB | ~0.9M | **YES (purged)** |
| `${HOME}` (home) | large | — | no |

- **Output is ~0.5–1.0 TB / ~6,744 files** (measured ~710 B/row × 756M; feature dim 2×D=512). Fits
  aa002's 3.5 TB; the binding constraint is **inodes** (~18k free) ⇒ **NO per-file sidecars in the
  store** (mandatory — sidecars would double the file count). Adding ~6,744 files leaves aa002 at
  ~89k/100k inodes; monitor since it's a shared project.
- **Purge fear**: the **output is safe** (persistent store) and the **venv is safe**
  (`${HOME}/venv-phace`, home). At risk = the **checkpoint** (scratch-only) and the **repo**
  (scratch-only, **no git remote**) — mitigations below.

## Store location & layout (versioned — best practice)
Root: **`${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92/`** (decided: aa002).
Checkpoint-versioned (`phace_atomic_ep92/`) so a future model never clobbers this one — the exact
ambiguity that forced this re-run (zbl_filtering already carried an *older* model's phace cols).
- **Mirror the source tree 1:1**: `phace_atomic_ep92/<relpath under zbl_filtering/output>`, same name.
- `_model/phace_atomic_ep92.pt` (+ `.md5`) — **copy the checkpoint into the store** (reproducibility +
  purge-safety).
- `README.md` / `MANIFEST.json` — human + machine provenance (source root, model path+md5, keep-cols,
  filter policy, meanstd, NL backend, dtype, frozen input file list + per-file source row counts).
- `index.parquet` — one row per output file: relpath, src_relpath, n_src_rows, n_out_rows,
  n_dropped_nsites>1000, model, host, struct_per_s, completed_at (completeness audit + the recoverable
  record of omitted >1000-atom structures).
- **NO per-file `.json` sidecars in the store** (inode hygiene) — sidecars + SLURM logs go to a
  **scratch** logs dir and are aggregated into `index.parquet`. Store stays ~6,744 files.

## Code / checkpoint safety (scratch is purged, repo has no remote)
- Copy `pre/phace_atomic_ep92.pt` → `<store>/phace_atomic_ep92/_model/` (done as a plan step).
- Recommend: add a git remote and push the repo (or mirror it to store). Not blocking, flagged.

## Columns per output row
Carried from source (**lenient** keep — keep those present in each file):
- IDs/provenance: `configuration_id` (universal) + `structure_hash`, `dataset_id`, `names`
  (colabfit-only; non-colabfit/mcloudarchive files omit the ones they lack)
- Geometry: `atomic_numbers`, `positions`, `cell`, `pbc`, `nsites`
- ZBL: `zbl_max_force`, `zbl_energy_per_atom`, `zbl_max_atomic_energy` (drop `passed`; kept so
  FPS can apply thresholds downstream)
- **No DFT labels** (energy/atomic_forces/cauchy_stress dropped).

Added fresh by ep92 inference (the old `phace_*` cols are NOT carried, so these replace them):
- `phace_energy`, `phace_features_bb_mean` (**mean+std concat, 2×D**), `phace_non_conservative_forces`

## Filtering (storage policy = NONE on ZBL; only the OOM hard limit)
- **No ZBL filter at storage** — store features for all structures; FPS filters at selection time.
- **Skip `nsites > 1000`** (featurizer OOM): only 1,276 structures globally. Enforced as a
  per-chunk row mask (`nsites <= 1000`) built from the source column before constructing systems,
  so they cost no inference. **Omitted from the store and recorded in `index.parquet`**
  (recoverable from source if a large-structure pass is ever built).

## Two small code changes to `inference/run_inference.py`
1. `--max-nsites N` (default off): per-chunk drop of rows with `nsites > N`; also record
   `n_src_rows` and `n_dropped_nsites` in the sidecar for the index.
2. `--keep-cols-missing-ok` flag: keep only keep-cols present in the file (log dropped) instead
   of `SystemExit` — needed for non-colabfit/mcloudarchive files lacking `structure_hash` etc.

(No row-slicing change: with a 24 h walltime even the 3.6M-row file finishes (~6–7 h). Splitting
a single file is a documented fallback only if a unit ever exceeds the walltime.)

## Parallelization (reuse the proven candidate_test work-stealing harness)
One **work-unit = one source parquet file** → one output file. N nodes × 4 GPU ranks, each rank
strides the manifest by `ntasks` (packs all GPUs, self-balances). NOT a SLURM array.
- **`build_manifest.py`** — glob all in-scope files (skip `gdb13/`), read `num_rows`, write
  `feat_manifest.txt` lines `SRC<TAB>OUT<TAB>BATCH` **sorted big-first**; `BATCH`=4 for
  big-molecule sets (ODAC, OMol25, OC20/OC22/OC25, solvated_protein, Chig-AIMD, OrbNet_Denali,
  OMC25), else 32; pre-create output dirs; also emits `MANIFEST.json`.
- **`run_one.sh`** — call `run_inference.py --model …ep92.pt
  --keep-cols configuration_id,structure_hash,dataset_id,names,atomic_numbers,positions,cell,pbc,nsites,zbl_max_force,zbl_energy_per_atom,zbl_max_atomic_energy
  --keep-cols-missing-ok --meanstd --max-nsites 1000 --nl-backend metatomic --batch-size <BATCH>`.
  Idempotent: `run_inference` skips if the output exists.
- **`submit.sh`** — `#SBATCH --account=aa002 --partition=normal --uenv=pytorch/v2.9.1:v2
  --view=default`, `--nodes=16 --ntasks-per-node=4 --gpus-per-task=1 --time=24:00:00`
  (copy candidate_test/submit_featurize.sh work-stealing body). **Re-submit to resume**
  (skip-if-exists drains the remainder); `.tmp`+atomic-rename means no partial file poisons a resume.

## Compute budget
~155 struct/s/GPU light, ~30–80 heavy ⇒ blended ~100–130. 756M ⇒ **~1,600–2,500 GPU-h**
≈ **1–1.6 days at 64 GPUs** (16 nodes), ~half at 32 nodes. Run in resumable 24 h batches.
**Calibrate on the pilot first**, then size node count.

## Execution order
1. Patch `run_inference.py` (`--max-nsites`, `--keep-cols-missing-ok`, sidecar counts); smoke 1 file.
2. `build_manifest.py` → `feat_manifest.txt` + `MANIFEST.json` (print unit count, total rows, any
   file too big for the walltime).
3. **Pilot** (1 node): small + 1 big (3.6M) + 1 mcloudarchive (lenient cols) + 1 ODAC (batch 4).
   Verify output schema (kept cols + 3 new phace cols, meanstd dim = 2×D), `passed` absent,
   `nsites>1000` enforced + counted, join-back by `configuration_id`, phace ≠ old cols. Measure struct/s.
4. Scale to full manifest with chosen node count in 24 h resumable batches; babysit to drain.
5. `build_index.py` → `index.parquet` from sidecars; verify Σ n_out ≈ 756M − (nsites>1000); write README.
