# MAD COSMO first interval export and upload

## Local public-file review on 2026-10-05

The user requested local changes and testing before giving any upload approval.
The review and upload folder is `matcloud-upload/mad-cosmo-v1/first/files/`,
containing the README, reader/example, dataset metadata and three HDF5 pairs.
The validated 2pow18-2pow19 and 2pow19-2pow20 pairs were added as hard links at the
user's request. Their four full-file SHA256 hashes were checked against the
passing validation reports, READY bindings and build records. The first three
intervals cover 11,184,761 structures and total 22.42 GB (20.88 GiB) of HDF5.
This is local staging only, not an upload or quota approval.
The separate review folder was consolidated here at the user's request, preserving
their README edits. The README combines a practical
quick start, filtering examples, ordering, assignments and HDF5 field guidance.
The reader now prints filter and duplicate flags, feature dimensions when
requested, and a selected-row count. Its default cell is the first in the file.

The proposed record keeps both first-interval HDF5 files and the metadata catalog
unchanged. It replaces the existing README and reader, and removes the separate
HDF5 guide, public manifest, source manifests, validation report and counts table.
The candidate manifest is under `first/internal/`, outside the upload folder.
`first/REVIEW.txt` contains exact local paths, hashes and test commands. The old
guide and packaged manifest were moved to `first/internal/MAD_COSMO_HDF5.md` and
`first/internal/packaged_manifest_20261003.json`. Earlier uploaded-file snapshots
are preserved there as well. The HDF5 files and validation reports are unchanged.

This proposal is NOT an upload approval. No remote mutation or production change
was made. Do not run the legacy finish/package/upload workflow unchanged: it
expects a public manifest and refuses replacements. After user review, prepare the corresponding internal
manifest workflow and recheck the exact remote inventory before any mutation.

## All-row packaging decision and filtered example, 2026-10-05

The agreed layout retains all structures in the existing HDF5 files. No discarded
supplement, physical split or re-export is required. The authoritative reader now
provides `iter_rows()` and a command-line example that omit collection rows with
nonzero filter codes or duplicate flags by default. `--include-discarded` opts in
to all stored collection rows. MAD-NEXT remains an explicitly separate unflagged
tier, included by default and removable with `--exclude-mad-next`. Raw `read(row)`,
file indices and `cell_rows()` retain their original meanings. The format guide
documents this policy and the distinction between stored and selected populations.

The real cell-1 smoke check returned only its retained centre by default, and
returned rejected rows with explicit opt-in. Ninety-three local unit tests passed,
including eight new filtering tests. These are candidate public helper/docs changes
only; nothing was uploaded or copied into a frozen production snapshot.

At 17:36, a Slurm DNS configuration lookup error stopped the login-node recovery
controller. Compute jobs continued. The same frozen controller was restarted from
its ledger after Slurm became reachable; its heartbeat and exclusive lock were
verified, with the same eight recovery submissions and no duplicate jobs. The
original production plan and all frozen worker hashes remain unchanged.

## Held-queue recovery on 2026-10-05, afternoon

The user authorized recovery and restart. Bulk `4982420` was released at about
16:20 Europe/Zurich. Its pending tasks now have eight-hour limits; Slurm refused
to extend already-running four-hour tasks. Eight initial recovery submissions
are `4984351` through `4984358`, for indices 10, 13, 15, 16, 28, 30, 31 and 34.
These use the existing frozen worker and `submit.sh`, with `part --resume`.
The original plan, gate requirements, exporter, validator and HDF5 are unchanged.

`recover_production.py` supervises this specific production-v2 run on Daint using
`lp133`. It runs detached on daint-ln003, not on the laptop and not on a compute
node. Its frozen copy and launcher are under `production-v2/recovery-20261005/code`.
Its state and log are `recovery-20261005/state.json` and `controller.log`.
`monitor_recovery.sh` is the reproducible login-node entry point. Do not launch a
second controller or another production launcher while it is active.

The controller allows only timeouts, the verified validation-checkpoint JSON
permission failure, and workers blocked by an earlier reviewed failure. It
preserves reviewed failure markers under `recovery-20261005/reviewed-failures`.
Completed HDF5 is reused and checkpoint payload hashes are checked on restore;
every recovered part repeats the full existing validation and reader tests.
The underlying intermittent permission problem is not proven resolved.

At most eight recovery tasks are queued at a time. Bulk is capped at 24 while
recovery is outstanding and returns to 32 when it is clear. New recovery
submissions also account for original tasks still draining from a higher cap.
The combined compute concurrency is at most 32. There are at most two retries
per part, 40 retry submissions total, eight hours per retry, and a 48-hour
controller deadline. Any unexpected error or exhausted limit holds pending
work and stops for review. These are safety ceilings, not expected runtime:
retry allocations alone cannot exceed 320 node-hours. Together with the 145
pending tasks extended from four to eight hours, this Daint run's theoretical
allocation ceiling rises from 785 to 1,685 node-hours; actual usage ends when
each task finishes. This does not verify the available project balance.

Audit `4982421` is intentionally held. Its dependency was changed to `afterany`
on the original bulk array, which now contains known failed tasks. The controller
releases the audit only after all 202 production parts pass READY verification
and all original/recovery workers have exited. The unchanged audit then checks
complete coverage. An audit hold is therefore expected, unlike a bulk hold.

The first controller attempts stopped on Slurm's nonzero partial-success
responses for array members that had already finished. These responses are now
handled explicitly; genuine scheduler permission errors still stop recovery.
Queue reasons are normalized because Slurm includes parentheses around them.
These initial controller failures did not run or modify any data-processing code.

The controller uses the existing immutable snapshot for all compute jobs. Changes
to the authoritative controller require stopping its recorded PID on its recorded
host, checking `recovery-submit.lock`, updating its separate controller snapshot,
then restarting; never edit production-v2/code or plan.json. The state ledger
prevents duplicate retry submission, including stopping on an uncertain sbatch
outcome. All changes are staged only, with no commit, push, upload or source-release
modification. See STATUS.txt for the dated Materials Cloud handoff.

## Current recovery on Daint / lp133, 2026-10-05

At 10:52, pending preparation `4982419` and audit `4982421` were moved to `debug`,
both with **30-minute limits**, at the user's request to use debug for short work.
Daint's live debug maximum is 30 minutes, not 1h30. Bulk `4982420` remains on
`normal`, up to 32 concurrent nodes and four hours per task. Account and
dependencies are unchanged. This is a recorded scheduling override in
`production-v2/scheduling-overrides.json`; the frozen plan/code are not edited.
The one-hour limits below describe the original submission before this override.

The active run is `matcloud-upload/mad-cosmo-v1/production-v2`, with immutable plan
SHA-256 `97cea7b7c56c83936a491cb5677d6bf5ce96c5a443a82b36fe28364d9e01612f`.
Use the existing authoritative launcher from this repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3.11 -B \
  release/build/matcloud/mad-cosmo-v1/submit_production.py --submit --max-nodes 32
```

The jobs were submitted at 10:47 Europe/Zurich on Daint, all under `lp133`:

- `4982419`: one preparation node rehashes/reuses the six validated parts, checks
  their provenance and timings, and runs reader tests. One-hour hard limit.
- `4982420`: the remaining 196 part tasks, dependent on preparation success,
  with at most 32 nodes concurrently and four hours maximum per task.
- `4982421`: final coverage audit, dependent on bulk success. One-hour hard limit.

The new timing allowance is three hours per full gate part. Validation is not
relaxed. The previous run's correct 2h36m part is accepted under this new recorded
resource policy; the old plan and its results remain unchanged. A gate certificate
is required before bulk work, and worker gate failures now enter the failure/hold
handler. Six completed parts are hard-linked with explicit old-validation bindings,
not recomputed. Eight old exports and four partial source-copy checkpoints are
reused through the existing worker paths. No upload is part of any job.

Read `REVIEW.txt` for the recovery review, node-hour estimate, checkpoint-retention
limits and manual resume procedure. The expected remaining work is about 380
node-hours, but 32-way throughput on Daint has not been measured. Queue delay is
additional: dry-run forecasts at submission were around 07:55 on 6 October;
the actual preparation job was pending for priority with no fixed start time.
After submission, `squeue --start` revised its estimate to **11:35 on 5 October**;
bulk and audit still had no start estimate. Scheduler forecasts can change.
Sixty-eight tests plus the separate study-reader test passed before submission.

## Repository cleanup on 2026-10-05

Recovery code, tests and documentation are staged, not committed. The completed
handover and Daint-check submission scripts are retained for run reproducibility.
`thread_probe.py` and `slurm/thread_probe/submit.sh` are also retained as the source
of completed read-performance probe `3578783`. That probe used the historical
Clariden `ab010` account and a 45-minute debug limit; its submission script is not
a current Daint launch command. Production read concurrency remains eight.

Generated Python bytecode was removed from the source tree. The runtime
`gate.lock` is ignored alongside the existing runtime lock rules, not deleted.
Frozen code, plans, validation evidence, data and checkpoints are preserved.
No upload or job-state change is part of this cleanup.

The sections below are historical and refer to earlier run versions and accounts.

## Fast production launcher on 2026-10-04

The full-size fast pilot passed in job `3577843` at 22:51 Europe/Zurich:
8,388,608 structures, 267,240,523 atoms, 4,007.47 worker seconds. Both whole-file
HDF5 SHA-256 hashes match the original baseline. Source validation and reader
tests passed. This establishes correctness and one-part performance, not optimal
I/O or linear scaling. Full evidence is in `pilot-fast-v1/PILOT_PASS.json`.

The user approved implementing production and using up to 16 nodes if this
improves completion time. The authoritative entry point is now
`submit_production.py --submit --max-nodes 16`, with `production.py` and
`slurm/production/submit.sh`. The work root is
`matcloud-upload/mad-cosmo-v1/production-v1`. See `REVIEW.txt` for the code review
brief, layout, evidence, assumptions, recovery controls and known limitations.

The launcher freezes eleven necessary runtime files and preserves all 202
existing part boundaries. Two new full-size parts, indices 20 and 120, run first.
Then indices 21, 22, 23 and 24 run at concurrency four. Each gate part must pass
full validation and finish within 2.25 hours normalized to a full part. Only then
can the other 196 parts start at concurrency 16. The bulk array also waits for
all legacy workers observed at submission to terminate. Old queues remain held;
this launcher does not cancel or release them. All jobs have a three-hour hard
limit and no requeue. The absolute cap is 606 part node-hours plus three for the
final audit; the provisional expected remaining work is about 220-450 node-hours,
not a measured whole-collection runtime or verified allocation balance.

New exports and full validation share one node. Existing HDF5 is revalidated and
hard-linked rather than re-encoded. Part zero can reuse the pilot validation after
rehashing the original files and checking the evidence binding. Legacy partial
staging is reusable under the existing run lock. Every part gets a conservative
64-GiB staging guard with 192 GiB allocated memory. RAM checkpoints are durable
and plan/code-bound; attempts use separate source-validation checkpoint namespaces.
Failure markers block new work, and a failing worker attempts to hold its array's
pending tasks. There is no automatic retry, repair, release or upload.

The 51 launcher/I/O/partition/upload tests and the separate study reader test
passed before preparation. Eight obsolete canonical files were deleted: the old
continuation and remaining drivers, their three submission/recovery launchers,
the legacy recovery test, and the two corresponding Slurm scripts. Frozen code
needed by active/held jobs remains intact. No backup archive was created. The
unrelated coverage-certification edit remains untouched; no commit or push.

Submitted at 23:27 Europe/Zurich on 2026-10-04:

- Canary array `3578271`, indices 20 and 120, started at 23:27:14 on
  `nid006315` and `nid006890`; hard stop 02:27:14 on 2026-10-05.
- Four-part ramp array `3578273`, dependent on canary success.
- Bulk array `3578274`, 196 parts at concurrency 16, dependent on ramp success
  and termination of legacy tasks `3576044_0`, `_2`, `_3`, `3576045_4`, `_6`,
  `_7`, `_8`.
- Final audit `3578275`, dependent on bulk success.

Plan SHA-256 is
`a35b971ac57e169a2c72a6472702751c80338290861586685505b9b78a36055b`.
All eleven frozen production files match the staged authoritative source; all
31 legacy and ten pilot snapshot hashes also still match. At launch the seven
legacy workers remained running and their pending queues remained held. A
separate request to retire those legacy jobs was awaiting user approval; this
submission did not cancel or release them.

## Retired runtime-script cleanup on 2026-10-04

The user requested removal of unnecessary scripts under
`matcloud-upload/mad-cosmo-v1`. Scheduler accounting confirmed that the first
validation, continuation intervals and study had finished, and the superseded
`remaining-v1` production jobs were cancelled. Their retired code snapshots were
archived and the loose copies removed:

- `node-code-1791026104`
- `continuation-v1/code`
- `remaining-v1/code`
- `study-v1/code`
- `pilot/internal/legacy-package/madcosmo.py`

This removed 77 scripts, six historical documentation copies and 13 regenerable
Python cache files. At the user's follow-up request, the temporary backup
`matcloud-upload/mad-cosmo-v1/internal/retired-code-20261004.tar.gz` was also
deleted. No cleanup backup remains for restoring those exact retired snapshots.
Earlier references in this log to these paths describe historical runs, not
currently available code. Current jobs do not depend on the removed archive.

`remaining-v2/code`, `remaining-v2/recovery-20261004/submit.sh`, and
`pilot-fast-v1/code` were preserved for active and held jobs. All 31 production
and 10 pilot snapshot hashes, and both plan hashes, still match after cleanup.
`first/internal/madcosmo.py` remains because the first-upload audit verifies it
against the historical upload manifest. Authoritative source scripts under
`release/build/matcloud/mad-cosmo-v1` were unchanged. No HDF5, staging arrays,
checkpoints, plans, result reports, logs or public package files were removed;
no scheduler state or upload was changed.

## Approved full-part fast pilot and cleanup on 2026-10-04

The user approved implementing the optimized converter, staging relevant Git
changes, cleaning unnecessary artifacts, and running one full-size pilot on
`ab010` with a maximum of **one node for two hours**. No commit, push, upload,
production queue release, or modification of the source release is authorized.

The pilot uses `pilot_fast.py` and `slurm/fast/submit.sh`. Its isolated run root is
`matcloud-upload/mad-cosmo-v1/pilot-fast-v1`. It targets part 0 of `remaining-v2`:
`madcosmo_cells_2pow20-2pow21_part0001-of-0002`, 8,388,608 structures. Existing
outputs and production snapshots are read-only comparison references. The pilot
requests 192 GiB RAM on one exclusively allocated node; this is still one node,
not an increase in the authorized node count. Slurm enforces the two-hour limit,
with no requeue or automatic retry. All earlier production/release holds remain.

Job `3577843` started on `nid006039` at **21:44:23 Europe/Zurich on 2026-10-04**.
Its hard stop is **23:44:23**. Follow `pilot-fast-v1/status.json` and
`logs/fast-pilot-3577843.log`; only `PILOT_PASS.json` certifies the full pilot.
The launch plan and code hashes are in `pilot-fast-v1/plan.json`. All 30 local
unit tests passed before submission, and the workflow changes were staged in Git.
The smoke gate passed in **43.98 seconds**: durable checkpoint restoration, all
golden decoded-field hashes and metadata, the optimized source validator, and the
frozen original source validator all passed on the image-rich fixture. The job
then proceeded to the full-part preflight. This is not a full-part PASS.

The exporter now accepts optional reader/checkpoint hooks; old callers retain
the mmap path. `block_reader.py` groups nearby source rows into bounded reads,
including `.npy` header offsets, arbitrary index order, repeated indices, and
exact dtype bits. Dense groups span at most about 1 MiB and at most 16 times
selected row bytes; sparse groups fall back to touched 64-KiB windows. A single
row can cross a window boundary. These are logical read bounds, not a claim
about physical filesystem read amplification. I/O concurrency is eight threads,
including the shared read pool used by the batched independent validator.

Reordering occurs in a unique node-local `/tmp` directory, bounded to a reference
staging size below 64 GiB, with additional memory for reads and validation. Every
five minutes at a completed batch boundary, and once before final packing,
`ram_checkpoint.py` sequentially copies and fsyncs the RAM arrays to scratch,
records each file's size and SHA-256, fsyncs the generation metadata, and atomically
publishes its pointer. The latest two committed generations are retained. Older
committed generations created by this pilot are pruned; uncommitted generations
are never treated as resumable state. Checkpoints bind the source report, code
snapshot and plan, completed sources, and within-source batch cursors.

Before the full part, the same job runs the image-rich 73,247-member cell-0 fixture,
interrupts after a committed batch, discards its RAM directory, restores into a
new one, and checks every decoded field against the original golden fixture.
Both the optimized validator and the frozen original validator must pass,
including more than 60,000 image sidecars. Only then does the full-size run start.
The full part must match all baseline decoded field hashes, shapes, dtypes, and
HDF5 attributes, pass the complete independent source validation, and pass the
reader tests before `PILOT_PASS.json` is emitted. No public `READY` marker or
upload approval is generated by the pilot.

`--resume` is an explicit manual recovery path, not an automatic retry. It requires
a new authorized allocation, a matching smoke-pass/plan/code binding and a valid
committed checkpoint. Partial HDF5 or old validation-checkpoint directories are
preserved under timestamped names before a new encoding/validation attempt.

Cleanup was restricted to this workflow: the two superseded standalone read-probe
scripts were removed from the authoritative source tree; their historical frozen
copies remain with old runs. The three deterministic 512-MiB synthetic study arrays
were deleted, reclaiming about 1.5 GiB; they can be regenerated by the study, and
all measurements and reports remain. The old cell-0 prototype package's seven
non-HDF5 files were moved to `pilot/internal/legacy-package`, not deleted. Golden
HDF5 fixtures remain in place. Frozen runtime code copies, locks and private
archives are kept on disk but excluded from Git; this does not ignore any public
non-HDF5 package file. The unrelated coverage-certification edit is untouched.

## Efficiency study and production hold on 2026-10-04

The user approved holding pending production work and a separate study capped at
one node-hour on `ab010`. No production restart or upload is authorized by this
study. The source release, existing HDF5 files, and frozen production code remain
unchanged.

At approximately 20:45 Europe/Zurich, all pending entries in `3573932`, `3573933`,
`3576045`, `3576047`, and audit `3573934` were confirmed `JobHeldUser`. Automatic
release jobs `3576046` and `3576058` were also held, so completion of running jobs
cannot release the production queues. The eight running tasks were preserved:
exports `3576045_4`, `_6`, `_7`, `_8`; validators `3576044_0`, `_2`, `_3`,
and `3573933_1`. Do not release any of these holds without a new production
decision. A held array can still contain already-running tasks.

Eight completed large exports averaged 7.960 hours copying/reordering and
13.644 minutes encoding, round-trip checking, and hashing HDF5. Example completed
jobs recorded about 19–20 CPU-minutes over 7–11 wall-hours, while reserving one
288-CPU/four-GPU node each. Thus changing HDF5 compression cannot address most
of the observed cost. The previous 17-day estimate extrapolated this implementation;
it is not an inherent cost of the requested layout. The actual remaining project
balance has not been verified.

The isolated `study/` code compares the existing sparse mmap reader, explicit
64-KiB-grouped reads, and sequential source windows on real production selections.
It uses disjoint windows with rotated method order and separately labelled warm
repeats; no global or source cache invalidation is performed. Concurrent production
and storage-server caches remain confounders. Features and ragged atom forces are
checked bitwise against the original source, including unit tests for NaN payloads,
duplicate and unsorted indices, and block boundaries.

It also compares scratch-backed scatter/flush with tmpfs and RAM-first sequential
output. Two 4,096-member slices from production parts 0 and 120 run the unchanged
full exporter on scratch and tmpfs, compare every staged field, encode HDF5, and
run the independent full source validator. A separate warm-array kernel measures
per-structure versus batched force validation and tests deliberate corruption;
this is not a replacement for the full validator or a full-run speedup estimate.

One `normal` job has a hard one-hour limit, no requeue, and phase subprocess limits
totalling 50 minutes. Artifacts belong under `matcloud-upload/mad-cosmo-v1/study-v1`,
not any public `files/` directory. Generated tmpfs fixtures are temporary; persistent
scratch fixtures and machine-readable measurements are retained for inspection.
Microbenchmarks alone cannot certify an optimized production exporter. Before a
restart, require a complete representative-part test, unchanged schema/order,
full source validation, bounded memory, restart tests, and an agreed allocation cap.

Study job `3577657` was submitted at 20:52:32 Europe/Zurich and started at 20:52:33
on `nid006264`, with `TimeLimit=01:00:00` and `Requeue=0`. Its frozen code is
`study-v1/code`, measurements are `study-v1/run/measurements.jsonl`, and its log is
`logs/study-3577657.log`. The local reader bitwise/order/boundary test passed before
submission. This is the only study allocation submitted under the one-node-hour cap.

### Completed study results

Job `3577657` completed successfully, exit `0:0`, in **8 minutes 39 seconds**:
0.1442 node-hours, equivalent to 0.5767 GPU-hours at four GPUs per node. Its measured
maximum RSS was about 3.2 GiB. All three phases passed. The exact observations are
in `study-v1/run/measurements.jsonl`, `summary.json`, and the derived `analysis.json`.
The source code for reproducing the derived summary is `study/summarize.py`.

There were 48 real-source read cases, each checked bitwise and repeated warm. All
passed. The two full-schema samples totalled **8,192 structures and 158,178 atoms**.
Every scratch/tmpfs staging field matched, HDF5 compression round trips passed,
and the existing independent validator passed against original source fields,
record identities, cell assignments, cell offsets, and within-cell FPS ordering.
Both validation kernels detected a deliberately changed force bit. The two real
samples did not include image-flag rows; they are not new coverage of that optional
sidecar or a replacement for the earlier image-rich pilot and restart tests.

First-touch-in-this-study feature payload throughput, aggregated over two disjoint
windows per method, was:

| Source | Existing sparse mmap | Grouped reads | Sequential window |
| --- | ---: | ---: | ---: |
| Alexandria | 3.92 MiB/s | 15.09 MiB/s | 28.40 MiB/s |
| OC20 S2EF | 3.29 MiB/s | 9.78 MiB/s | 17.52 MiB/s |
| GDB13 | 3.64 MiB/s | 12.21 MiB/s | 17.42 MiB/s |

These are small component tests, not total export speeds. Adjacent source windows
were assigned to different methods to reduce warm-cache bias, so their precise row
counts differ. OMol25 windows had especially different densities, from 19 to 421
structures; its method ratios are not a fair controlled speedup estimate. The
grouped reader also caused considerable read amplification: for the listed feature
tests, process read counters were about 52–83 times useful payload versus 15–26
times for sparse mmap. Sequential windows were about 41–64 times useful payload.
Process counters are not exact physical-server I/O measurements. Blindly reading
entire sources once per output part would multiply total traffic and is not the
recommended implementation.

In the isolated scatter test, 32 MiB of values were placed into a 512 MiB array:

| Strategy | Scatter plus persistence time |
| --- | ---: |
| Scratch mmap, flush each 4,096-row batch | 2.176 s |
| Scratch mmap, flush once at end | 0.660 s |
| RAM assembly, sequential save and fsync | 0.185 s |
| Tmpfs scatter and flush, not durable | 0.022 s |

The first strategy recorded about 508 MiB read and 830 MiB written for 32 MiB of
payload. This confirms the cost of scattered file-backed writes, but the synthetic
per-batch flush is deliberately a stress case: production flushes per source,
not per 4,096-row batch. The approximately 12-fold synthetic improvement must not
be presented as a full exporter speedup. Completed production staging arrays occupy
28.8–30.7 GiB per tested 8.4-million-member part, suggesting RAM staging is feasible
within a carefully measured job budget. Larger-atom-count parts still require their
own preflight memory bound; the 3.2 GiB study peak proves nothing about their peak.

Cache effects were large. The mixed-source 4,096-member sample took 276 seconds to
stage on scratch first and 0.74 seconds on tmpfs second. Reversing the order on the
GDB13-dominated sample gave 106 seconds on tmpfs first and 8.18 seconds on scratch
second. The second runs benefited from source caching. This rules out interpreting
276/0.74 as a tmpfs speedup. Warm full-source validation took 0.52 and 0.31 seconds;
it does not predict the cost of a separate-node, cold-source production validator.
The warm force-comparison kernel was 15–30 times faster batched, but this measures
only CPU overhead, not source reads, staging reads, identities, or other fields.
The small correctness slices are sparser than complete parts and should not be
linearly extrapolated to the entire population.

### Recommended next decision, not yet implemented or submitted

Keep the public contract unchanged: the same interval/part boundaries, MAD-CORE
cell order, within-cell FPS order, separate data/features HDF5, existing dtypes,
all metadata, and stable record IDs. Existing validated output remains reusable.
Do not solve this by reducing precision, dropping features, relaxing validation,
moving heavy work to login nodes, or adding more GPU nodes to the same I/O pattern.

The least disruptive candidate is an adaptive block reader plus RAM-backed part
assembly, followed by sequential persistence and batched independent validation.
Coalesce nearby rows with an explicit maximum span and over-read budget; retain a
sparse path for very isolated rows. Do not use the fixed 64-KiB benchmark reader
unchanged as a universal solution. Start with one part per node, no concurrency
increase. Reuse node-local caches for export and full validation where feasible,
but derive validation pointers independently from the immutable source index.

RAM staging alone is not restart-safe. A production candidate must commit durable,
checksum-bound checkpoints at bounded intervals, publish checkpoint metadata only
after persisted data is complete, restore from committed generations only, and
never mark a partial HDF5 ready. Avoid dumping every full RAM array after each
small source. This is a design requirement, not a feature proved by this study.

Before resuming production, compare a complete representative 8,388,608-member
part against its existing output and source, including peak memory, wall time,
I/O counters, export plus full validation, split-cell boundaries, and forced
interruption/restart. Keep new results separate from frozen production outputs.
Agree an explicit compute cap for that pilot first; no new allocation is submitted
by this study. A new whole-dataset ETA requires that full-part measurement and
representative source composition, not multiplication of component speedups.

If bounded coalescing still produces excessive read amplification, the next design
to study is source-major streaming into append-only destination-part buckets, then
sorting each bucket by the existing global member ordinal before HDF5 encoding.
That could amortize source reads across parts, but is more invasive and needs
coverage/uniqueness proofs, ragged-field alignment checks, and durable bucket
checkpoints. It is an alternative, not a validated replacement at this point.

Another alternative is an approved SSD working copy on Iopsstor. CSCS documents
Iopsstor as SSD-backed and Capstor as HDD-backed, with different random-access
characteristics. No such copy was made or benchmarked; capacity, transfer cost,
permissions and lifecycle must be checked first. See the
[CSCS storage guide](https://docs.cscs.ch/guides/storage/#iopsstor-vs-capstor).

At study completion, production holds and automatic-release holds remain in place.
The eight pre-existing running jobs were not interrupted. No upload, original
release modification, production-code modification, or production queue release
was performed. Only generated temporary node-local fixtures were automatically
cleaned up; persistent study fixtures remain in the study directory, not the
public package.

## User approval required before any further upload

Do not upload automatically after validation. Before every upload, show the user
the exact proposed filenames, their sizes, their purpose, and what is already in
the draft. Ask for explicit approval. Only after approval, record the approved
draft ID and filename-to-size/SHA-256 mapping in an approval file and pass it with
`--approved-files` to the uploader or completion workflow. These commands refuse
to run without that file; the uploader rejects files whose bytes changed after
approval. Never generate an approval file merely to get around this gate.

The queued continuation jobs only export and validate; they stop at `READY.json`.
Validation and source-manifest reports are internal evidence, not default public
downloads. The existing reader is a single-file, per-row prototype and requires
a usability review before further distribution. The counts table is optional;
keep it internal pending the user's choice. Preserve source attribution and
license information in an approved public metadata document.

Nine files were uploaded before this policy was clarified, including the two
internal JSON reports, the reader prototype and the counts table. They remain in
the unpublished draft until the user approves an exact cleanup list. Do not
delete remote files or replace the uploaded manifest/README without approval.
All non-HDF5 public package files under the MAD-COSMO `files/` directories must
remain visible to Git; HDF5 payloads and intermediate staging arrays stay ignored.

## Public folder cleanup and the initial 50 GiB target

`first/files/` is now a local public-package proposal containing only the two
HDF5 payloads, `MAD_COSMO_HDF5.md`, `dataset_metadata.tsv` and
`madcosmo_manifest.json`. The reader prototype, counts table, detailed source
manifests and validation copy were moved to `first/internal/`, not deleted.
The originally uploaded README and manifest are also preserved there under
`uploaded_*_20261003` names. The revised public README uses direct h5py examples
and does not require the rejected prototype reader. Neither revised metadata
file has been uploaded; the remote draft and original store copy still match
the archived uploaded package. The audit tool checks that archived upload, not
the unapproved local proposal.

`package.py` now writes counts and source-manifest reports under `internal/`
and rejects unexpected files in `files/`. The uploader takes an explicitly
specified private `--validation-report`; validation evidence is not implicitly
included in public files. No HDF5 payload was moved or changed during cleanup.

The user wants an initial upload up to 50 GiB, then to ask Materials Cloud for
more capacity. The uploader now defaults to a 50 GiB safety cap for the entire
draft, counting existing files as well as proposed additions. This is a local
safety limit, not a verified statement of the server's configured quota.
Approval is still required for the exact upload list. The user confirmed that
the initial target includes both data and feature files.
The current uploaded package is about 4.364 GiB including features. The first
data file alone is 1.162 GiB. Size projections from that first interval are rough
because later intervals have different source and atom-count compositions.

## Status on 2026-10-03

The first interval passed exhaustive validation in Slurm job `3571929` on
`ab010`. Its data and features files are 1,247,190,273 and 3,436,228,119 bytes.
The separate store copy was verified and all nine package files were uploaded
and remotely checksum-verified in the unpublished MAD-COSMO draft. See
`upload_state.json` and `completion.json` in the workspace run directory.

The initial upload stopped after creating an empty pending data-file entry:
the API returns `created` for pending entries and `file_id` for completed files,
not the previously assumed `id`. Recovery recognized only the exact empty entry
with its recorded creation timestamp; it did not delete or replace remote files.

The next two intervals are queued under `ab010`:

| Cell interval | Structures | Export job | Validation job |
| --- | ---: | --- | --- |
| `[2^18,2^19)` | 2,551,963 | `3573527_0` | `3573528_0` |
| `[2^19,2^20)` | 6,275,294 | `3573529_1` | `3573530_1` |

Their prerequisite pilot `3573525` passed parallel-copy bitwise comparison,
copy-checkpoint recovery, chunked source validation and validation-checkpoint
reuse. Each interval has its own export-to-validation dependency. At most two
continuation nodes are active. Queue resource limits still apply.

The immutable code snapshot, interval plan and submitted job IDs are under
`matcloud-upload/mad-cosmo-v1/continuation-v1`. Each interval has its own
`madcosmo_cells_<start>-<stop>/` directory there, with preflight metadata, staging,
HDF5 files and validation reports. `READY.json` means the entire interval passed
source validation and reader tests; it does not mean it was uploaded. These new
jobs currently stop at READY. They do not use the first-interval uploader's shared
metadata filenames for later intervals.

Copy checkpoints retain completed source groups. Validation uses 65,536-row
source chunks and checksum-bound checkpoints. A time-limit interruption still
requires resubmission; there is no unbounded automatic retry. An interrupted
encoding/allocation is preserved under an `*-incomplete-*` directory before a
new attempt. The original release and first-interval artifacts are unchanged.

The remaining intervals are prepared by `submit_remaining.py`. Their populations
are 15,817,789, 37,788,932, 97,490,362 and 1,532,220,511 structures. The worker and
submission evidence are under `matcloud-upload/mad-cosmo-v1/remaining-v2`; consult
its `submission.json` and Slurm before claiming that a task is queued or running.

The earlier 404-part plan under `remaining-v1` is superseded by the user's
request for parts up to twice as large. Its pilot `3573777` passed, but export
array `3573778`, matching validation array `3573779`, and audit `3573780` were
cancelled and replaced. Retain their snapshot, reports and partial staging
for provenance; do not upload or resume the old part layout. The frozen old
plan SHA-256 is `827294ea495044b5740c31ed8c2fdb767540b056f1443baf3149a5bec3b4c704`.
The earlier two unsplit continuation intervals are not changed by this replacement.

The replacement on 2026-10-04 recorded pilot `3573931`, export array `3573932`,
validation array `3573933`, and final audit `3573934`. The new pilot passed in
36 seconds. At 00:35 Zurich time the first four larger-part exports were running,
alongside the unchanged earlier exports `3573527_0` and `3573529_1`. Both new
arrays cover tasks 0 through 201 with a concurrency limit of four each.
The replacement plan SHA-256 is
`6be6186db5e95329b80c3ec5d9d46cd91fe60c456d57787ad1679efd4ae8e019`.
Each replacement member range is exactly the union of two adjacent old ranges;
coverage and ordering are unchanged. No old partial output was deleted or reused
as a complete larger part. The new layout restarts those four interrupted parts.

### Targeted node-failure recovery on 2026-10-04

Cluster launch cancellations interrupted export tasks 4 and 7 through 16;
export task 6 suffered a node failure. The latter retains ten completed source
groups covering 60,090 rows. Its incomplete source group must be copied again.
Validation tasks 0, 2 and 3 were cancelled before reaching the validator. The
affected inspected nodes reported failed ATOM GPU tests or nonresponse, and
the recovery excludes all 15 nodes associated with these failed jobs.

`recover_remaining.py --submit` recorded one bounded recovery under
`remaining-v2/recovery-20261004/submission.json`. Do not run it again blindly;
an existing journal requires inspection before any further attempt. The original
source snapshot and part plan are unchanged. No completed HDF5 is re-exported,
no source file is written, and no upload is submitted.

The recovery jobs are:

| Purpose | Job | Task indices |
| --- | --- | --- |
| Validate existing completed pairs | `3576044` | 0, 2, 3 |
| Resume or start interrupted exports | `3576045` | 4, 6 through 16 |
| Release queued original exports | `3576046` | one control job |
| Validate recovered exports | `3576047` | 4, 6 through 16 |
| Release queued original validations | `3576058` | one control job |

The original arrays `3573932` and `3573933` are temporarily held for pending
tasks only. Four already-running exports and one already-running validation
continue, as do both earlier continuation validators. The three validation-only
retries may use the other three remaining-interval validation slots immediately.
Initially the export retry array waited for the four running original exports
to terminate, then allowed at most four tasks. That conservative gate was relaxed
in the capacity update below. Its completion releases the original export queue.
The recovered-export validators depend on the corresponding recovered export
and wait for both the validation-only retry array and the running original
validator to terminate. At most four such validations run. The validation release
job waits for all recovery validations and that original validator before
releasing queued original validations. This preserves the four-export and
four-validation caps for remaining-v2 throughout the handover.

Audit job `3573934` now uses `afterany` on the original validation array and both
replacement validation arrays. A dependency on the failed original array with
`afterok` could never succeed. `afterany` does not relax data checks: the audit
still requires a checksum-bound PASS report and matching metadata for every one
of the 202 parts, and fails if any part remains incomplete. Release jobs also use
`afterany` so a failed retry does not unnecessarily hold unrelated queued work.
This is not an automatic retry loop; further failures or timeouts need inspection.

At approximately 19:47 Zurich time the user approved opening the three free
export slots. Only `3573932_17` was still running, with all other original pending
exports held. The retry array `3576045` was first reduced to a limit of three,
then its export dependency was cleared. Retry tasks 4, 6 and 7 started immediately,
bringing the combined active exports to four without restarting the original job.
Control job `3577447` uses `afterany:3573932_17` to restore the retry limit to four
only after the original export terminates. The original queued exports remain
held until the retry array terminates, preserving the same total export cap.
The validation gates, release jobs and final audit dependency are unchanged.
The separate `recovery-20261004/capacity-update-20261004.json` records this change;
the earlier submission journal is retained as the original submission evidence.

Each interval is partitioned into consecutive source-index member ranges with
at most 8,388,608 structures per part. This gives 2, 5, 12 and 183 parts,
respectively. Jobs are interleaved across intervals, with at most four exports
and four validations active in this new submission, plus the two existing
continuation jobs. Actual concurrency is subject to scheduler resources. The
four-array-task limit applies separately to export and validation.

Run `python3.11 release/build/matcloud/mad-cosmo-v1/submit_remaining.py` to inspect
the plan, and add `--submit` to submit the frozen plan. It records a checksum of
every snapshot file and refuses ambiguous or inconsistent resubmissions. It does
not change the earlier frozen continuation snapshot or submit any upload.

A new small compute-node pilot must pass before any remaining-interval export
starts. It checks partial cells and a cell boundary against the original source
and already validated first interval. Each validation depends only on its
corresponding export, not all exports. A final audit depends on all validations
and verifies that all part member ranges cover the remaining population exactly.
The audit checks validated file sizes, metadata and checksum-bound reports;
each preceding part validator checks every field and actual HDF5 checksum.

Parts retain MAD-CORE cell order and the original within-cell FPS ordering.
Large cells may cross file boundaries; FPS ranks are never reset. Each data file
has local `index/cell_offsets`, plus `index/global_cell_offsets` containing the
full source member bounds of its included cells. Attributes give parent interval,
part number/count, actual cell bounds and exact half-open global member bounds.
Concatenating the numbered parts reconstructs the interval without gaps or
duplicates. Data/features files remain paired by record ID and exact member range.

Every part stops at `READY.json` after full validation and internal reader tests.
The `files/` subdirectory contains only its data/features HDF5 pair. Reports,
preflight metadata, build manifests and checkpoints stay outside `files/`.
The parts are structure-count bounded, not guaranteed to fit a specific byte
limit; inspect measured file sizes before proposing an upload. Source files are
read-only, all new work stays on scratch, and all staging is retained. Timeouts
require an inspected checkpoint resume, not an unbounded automatic retry.

The target is the complete level-24 cell interval `[0,262144)`: 2,357,504
structures and 53,973,063 atoms, preserving the existing cell-major/local-FPS
index exactly. Geometry, predictions, provenance, diagnostics, all 25 assignments,
and image sidecars are in the data HDF5; the aligned 512-dimensional features are
in a separate HDF5. The source release is read-only throughout this workflow.

## Paths

- Source: `${STORE_ROOT}/mad-core/release`
- Workspace run: `matcloud-upload/mad-cosmo-v1`
- Final staging: `${STORE_ROOT}/mad-core/matcloud/mad-cosmo-v1`
- Flat upload files: the `files` subdirectory of final staging
- Unpublished destination: MAD-COSMO draft `${MC_DRAFT_MADCOSMO}`
- MAD-CORE draft `${MC_DRAFT_MADCORE}` must not be changed by this workflow

The source root manifest digest is
`fc7e02a9907ca8ed97446bca28fd80fe416db2e59a489f85ac0933959864e8a3`.
This pins the actual source snapshot, including its current source catalog;
an earlier study used an older root manifest digest.

## Verified pilot

Cell 0 was exported and exhaustively checked against its source. The validation
report is `matcloud-upload/mad-cosmo-v1/pilot/validation.json`: PASS for all 73,247
structures, 939,650 atoms and 67,930 image-flag records. The pair is 90,483,028 bytes.
The reader boundary, optional-feature and image reconstruction checks also passed.
Pilot files are local test artifacts, not the production interval upload.

## Active processes at launch

Both jobs were launched detached on `clariden-ln002`. The export parent PID was
132912; the guarded completion workflow PID was 31555. These are historical launch
identifiers, not proof of current liveness. Check the recorded host and live process
before deciding that a run has stopped or restarting it.

```sh
python3.11 release/build/matcloud/mad-cosmo-v1/launch.py first --status
python3.11 release/build/matcloud/mad-cosmo-v1/launch.py finish --status
```

The status commands need host process visibility, outside the command sandbox.
Source storage reads are slow; increasing read counters with a live exporter is
progress, even before a source-group completion line is written to its log.

## Completion gates

`submit.sh first` runs the exporter and then the independent source validator.
`finish.py --run` waits for that live process and a PASS report, runs reader tests,
packages the files, copies them to final staging without replacing existing files,
verifies all copied checksums, and invokes the additive draft uploader.

The completion workflow has been authorized to write only the new MAD-COSMO staging
copy and upload verified files to the unpublished MAD-COSMO draft. It does not
publish, replace completed uploads, remove remote files, edit the placeholder
README, or modify the original release or MAD-CORE draft.

Success evidence is all of:

- `first/validation.json` with PASS and the complete interval count
- complete `first/files/madcosmo_manifest.json` and checksum-matching HDF5 pair
- matching copies under the final staging `files` directory
- `upload_state.json` with every expected file remotely verified and complete
- an independent final remote inventory check, with the record still unpublished
- `completion.json` emitted by the guarded workflow

Do not claim that files are uploaded from an exporter or validator result alone.
Do not rerun `launch.py first` into the existing staging directory. If an actual
failure occurs, inspect the logs and preserve partial files until the exact recovery
step is understood. The uploader can skip checksum-matching completed files and
resume its own recognized uncommitted file; it refuses unknown pending files.

The public source catalog preserves its original dataset names. The counts table
adds `metadata_dataset_id` to map the original MAD-NEXT subset names to their
broader catalog families, without changing any HDF5 source identities.
