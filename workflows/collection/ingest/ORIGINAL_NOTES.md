# mad-core

Pipeline for the nested-coresets ML-potentials work: featurize atomistic datasets with the
**PhACE m292** model, then select **FPS coresets**. Covers ~410 ColabFit datasets + several
large non-ColabFit ones (OMol25, OC20, OC25, ODAC, Alexandria, nomad, oc20neb, optimade).

> **Current state of every dataset is tracked in `datasets status.xlsx`** (download / filter /
> features / fps status per dataset; tabs: colabfit, non-colabfit, restricted licence, optimade;
> excluded datasets are red). **Check it first** to know what's done / running / excluded.

## Environment (CSCS Alps, GH200)
- `uenv` image `pytorch/v2.9.1:v2` + venv `venv-phace`. Activate:
  `uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "source venv-phace/bin/activate && <cmd>"`
- **vesin 0.5.8** (NL bug fixed; use `--nl-backend metatomic` for speed, `ase` also correct).
  Also: `ase_db_backends` (OMol/ODAC/OC25 aselmdb), no pymatgen (Alexandria parsed manually).
- SLURM: account `aa002`, partitions `normal` (12h) / `debug` (≤4 nodes, ≤90 node-min, 1 job).

Everything links by **`configuration_id`** (see `inference/DATASET_IDS.txt` for per-dataset id format).
- **features**: `store/.../${USER}/model_292/<variant>/<dataset>/<stem>_with_phace.parquet`
  — `configuration_id` + 256-d `phace_features_bb_mean` + predicted `phace_energy`/forces.
  (non-colabfit = LEAN: ids + features only; colabfit = full source schema passed through.)
- **structures** (non-colabfit only): `store/.../${USER}/nc_sources/<variant>/<dataset>/nc_passed/*.parquet`
  — geometry (atomic_numbers/positions/cell/pbc) + DFT energy/forces/stress. colabfit structures
  stay in the ColabFit source tree.
- **coresets** (FPS output): `store/.../${USER}/model_292/fps/<pool>/fps.parquet`
  — `fps_order, epsilon_sq, input_row, configuration_id, …`; first K' rows by fps_order = size-K' coreset.
- `<variant>` = `filtered` (passed@0.35) or `filtered_rej_passes_at_025` (rejected@0.35, passed@0.25).

## Layout
- `inference/run_inference.py` — featurizer (input→+3 phace cols; `--keep-cols` for lean output).
- `inference/nc_ingest/` — convert non-colabfit (extxyz/aselmdb/json/traj) → colabfit-schema parquet
  with content-addressed ids. `ids.py` (FROZEN hash params), readers/, `convert.py`, `_nc_one.sh`.
- `fps/run_fps.py` — multi-GPU NCCL FPS (needs CUDA; `coresets` editable pkg, compiled CUDA ext).
  `fps/analysis/` — join coreset↔features (`join_coreset_to_features.py`, `example_fps_to_features.py`).
- **Per-run committed scripts** (the convention): `fps/runs/<pool>/`, `inference/runs/<run>/`,
  `inference/nc_ingest/runs/<dataset>/` — each = `submit.sh` + `manifest.txt` + `logs/` (+NOTES).
- `datasets status.xlsx` — master status (colabfit / non-colabfit / restricted / optimade tabs).

## Key conventions / gotchas
- **Reproducibility**: every HPC run gets its own committed self-contained `submit.sh` (not ad-hoc).
- **Never `git commit`/`push`** — Sofiia does that; only `git add` + advise.
- **Long login-node jobs**: run detached (`setsid nohup … &`). Note CSCS rotates login nodes, so a
  nohup proc lives on one node and may be invisible from a later session (the SLURM jobs are unaffected).
- **FPS node-sizing**: CPU mem ∝ rows/rank = N/world_size; keep ≲12–15M rows/rank or it OOMs at load
  (1 node OOMs for N≳100M; use 2–4 nodes). Colabfit full-id pools need more nodes than lean ones.
- **Featurize OOM**: huge-structure shards (e.g. ODAC MOFs) need fewer ranks/node.
- Detailed docs: `inference/DATASET_IDS.txt`, `inference/runs/README.txt`, `STORAGE_REORG_PLAN.txt`
  (deferred: reorg to `phace_m292/<family>/{structures,features,coreset}`).

General rules:
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

Don't write comments in the code unless requested. No commit or push allowed.
Write code with clear spaces between the lines (logical) and give good variable names
Keep all imports on top
never mention anyone's names

${STORE_ROOT}/ - never modify anything without asking and examplaing what you do
