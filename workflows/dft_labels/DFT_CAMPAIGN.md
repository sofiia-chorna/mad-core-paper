# DFT campaign — rules, conventions and known gaps

The one place to look before running, extending, repairing or describing any DFT in this
project. Written 2026-09-14, consolidating `lda_pbe/campaign/README.md`, `docs/DECISIONS.md`,
`docs/CAMPAIGN.md`, `DENSITIES.md`, `common/bin/README.md`, `labelled/README.md` and the
r²SCAN `LYRA_NOTES.md`. Where those disagree with this file, they are older.

**The governing principle: every calculated structure carries the full output set — energies,
forces, Hamiltonian and overlap matrices, eigenstate info.** Not energies and forces alone.
Any run that drops the I/O keywords produces an incomplete record, and that is a gap to be
repaired, not a design choice. See §9 for where this has already gone wrong.

---

## 1. What exists

| campaign | functional(s) | arms | where |
|---|---|---|---|
| LDA/PBE | pw-LDA + PBE, both per frame | `madcore_25k`, `random_25k`, `voronoi_25k`, `random_test_6k` | `aa002/mad-core/lda_pbe/` |
| r²SCAN | r²SCAN only | 1,000-frame subsample of the PBE test | `aa002/mad-core/r2scan_random_test/` |

`madcore_25k` was computed by an earlier, separate toolchain (preserved verbatim in
`campaign/common/_upstream/`). The other three arms ran through the pipeline described here.
r²SCAN ran on SCITAS Lyra.

---

## 2. Physics is frozen

**No DFT parameter that can affect convergence is ever changed.** Decided 2026-08-22 after
measuring the alternatives; this is closed, do not reopen it to buy speed.

Fixed in every `control.in`, identical across arms and functionals — only the `xc` line differs:

```
xc                      pbe  |  pw-lda  |  libxc MGGA_X_R2SCAN+MGGA_C_R2SCAN
spin                    none
charge                  0.
relativistic            atomic_zora scalar
sc_iter_limit           200
sc_init_iter            199
basis_threshold         1d-5
override_illconditioning .true.
override_kgrid_checks   .true.
occupation_type         gaussian 0.05
charge_mix_param        0.02
sc_accuracy_etot        1e-6
sc_accuracy_forces      1e-4
sc_accuracy_rho         1e-5
k_grid_density          8          (periodic only; isolated has no k-grid at all)
```

Basis: `defaults_2020` **tight**, from `common/defaults_2020/tight/NN_El_default`.

Two accelerations were tested on real frames and **rejected**:

- **GPU offload is slower.** Same frame: 62 s CPU, 119 s GPU, identical SCF path (18 cycles),
  energies agreeing to 1e-9 eV. 72 MPI ranks land on one device and serialise. `use_gpu .false.`
  stays.
- **`charge_mix_param` 0.02 → 0.05 is a genuine 2× speedup and still rejected.** 98 cycles /
  1081 s versus 66 cycles / 521 s, energies agreeing to 15 significant figures. But mixing
  changes *which* frames converge, not only how fast, and the arms exist to be compared against
  each other — a different surviving set per arm would invalidate that comparison. 0.10 did
  worse than 0.05, so the behaviour is not even monotonic.

---

## 3. Required outputs — every frame, every functional

These I/O keywords are part of the protocol, not optional extras:

```
compute_forces          .true.
output_rs_matrices      HDF5          # rs_hamiltonian.h5 + rs_overlap.h5
output                  eigenstate_info
output_level            full
use_local_index         .false.
elsi_restart            read_and_write 10
```

`run_aims.sh` keeps exactly this set per frame and functional (`KEEP_FILES`):

```
aims.out  aims.err  basis-indices.out  rs_indices.out
rs_overlap.h5  rs_hamiltonian.h5  eigenstate_info.out
```

**`rs_indices.out` is periodic-only.** Its absence for an `isolated_fff` frame is correct and
is not a missing output. Everything else must be present for both functionals.

Cost of this policy: ~175 MB per structure, which is why the LDA/PBE campaign is 11 TB for
~66,000 structures. That is the accepted price.

`elsi_restart` additionally writes per-k-point `D_*.csc` density matrices — see §7.

---

## 4. Structure classification and routing

Two **independent** cuts decide which of the three pipelines a structure runs through. Getting
these confused has caused real errors.

### 4a. Which file a structure lands in — `scripts/xyz_write.py::classify`

```
n_periodic == 0                  -> fff        (molecule/cluster)
n_periodic <  3                  -> surfaces
n_periodic == 3, vacuum_axes 0   -> ttt        (bulk)
n_periodic == 3, vacuum_axes 3   -> fff        (molecule in a box)
otherwise                        -> surfaces
```

The vacuum test uses **`min_gap = 6.0 Å`**. Verified: `classify()` reproduces `selection.tsv`
for 25,000 of 25,000 rows at exactly that value.

### 4b. Which pipeline a *surface* runs through — `common/_upstream/get_true_surfaces.py`

```python
for d in data:
   if d.pbc[-1]:
      if d.cell[2,2] > 80:  data_ttf.append(d)   # dipole-corrected
      else:                 data_ttt.append(d)   # NOT dipole-corrected
   else:                    data_ttf.append(d)
```

A surface goes to the dipole-corrected `surfaces_ttf` only if its **third cell vector exceeds
80 Å**, or if `pbc[2]` is False. Everything else runs as ordinary `periodic_ttt`.

**This is not the 6 Å rule.** The 6 Å rule only chose the file; the 80 Å rule is a second cut
applied afterwards. For `random_25k` it sends 4 frames of 2,781 to `ttf` — the other **2,777
slabs run without a dipole correction**.

**Confirmed deliberate 2026-08-22. Do not change without redoing `madcore_25k`.** Consistency
across the merged dataset outweighs the marginal physics, and madcore was already computed
this way.

| pipeline | k-grid | stress | dipole correction | cell in geometry.in |
|---|---|---|---|---|
| `periodic_ttt` | `k_grid_density 8` | yes | no | yes |
| `surfaces_ttf` | `k_grid_density 8` | yes | **yes** | yes |
| `isolated_fff` | none | no | no | **no** |

### 4c. The molecule-in-a-box rule

13 of `random_25k`'s 17,642 `fff` rows carry a cell and `pbc="T T T"` — documented upstream
behaviour, marked `vacuum_axes=3` in the manifest, not corruption. Left alone, ASE writes
`lattice_vector` lines, the fff `control.in` has no k-grid, and aims aborts with
`Lattice vectors present but no k-grid specified` — silently losing the frame as FAILED.

**The fix is `atoms.set_pbc((False, False, False))` before writing, and nothing else.**

It has a known side effect. All released coordinates are folded into the cell, so a molecule on
a cell face is stored split across the boundary; dropping the cell leaves it fragmented. The
canonical case is `random_25k` frame 5004, methane:

```
C-H with the cell:      1.07  1.07  1.20  1.05    real CH4
C-H after set_pbc(F):   9.07  1.07  1.20  1.05    CH3 + a lone H
```

aims then converges cleanly on a meaningless system and flags nothing.

**Decision: run the plain fix anyway and exclude the affected frames** (madcore 5, random 1,
voronoi 1, test 0 — 7 of ~81,000). An identical DFT treatment across arms matters more than
repairing one frame per arm; repairing only the later arms would confound the very comparison
the arms exist to make. `prepare_frames.py` compares fragment counts with and without the cell
and prints any frame that breaks.

A bond-graph unwrap was considered and rejected: it modifies coordinates and diverges from
madcore, and exact recovery is unavailable (image flags exist for only ~70k rows, none of them
the boxed frames).

---

## 5. Run protocol

### 5a. LDA before PBE, always

PBE inherits the converged LDA density matrices via `elsi_restart`:
`mv ../pw-lda/D_* .`. This is a **dependency, not a preference** — and note `mv`, not `cp`:
starting PBE destroys the LDA density matrices.

### 5b. Size pools, not index ranges

A chunk mixing a 296-atom cell with small molecules pins the node to one slot, because the big
step's memory reservation leaves nothing for its neighbours.

| pool | atoms | slots × mem | ranks |
|---|---|---|---|
| small | ≤ 71 | 4 × 195G | 72 (36 isolated) |
| big | 72–250 | 2 × 390G | 48 |
| monster | > 250 | 2 × 390G | 32 |

### 5c. Rank floor for tiny systems

`TINY_FRAME_ATOMS = 4` → `TINY_FRAME_RANKS = 8`. `elsi_restart` aborts with
`Too many MPI tasks` when ranks exceed the basis size, so 1–4 atom frames must run on few
ranks. Rank count does not change the converged answer.

### 5d. srun flags that are load-bearing

```
srun --mpi=cray_shasta --exact --gres=none --mem=<step_mem> -n <ranks> -c 1
```

`--gres=none` matters: the job allocates `gpu:4`, and since Slurm 21.08 a step naming no GRES
implicitly takes **all** the job's GPUs exclusively, so without it step 2 waits on step 1
(`Requested nodes are busy`) and the node runs one calculation instead of four.

### 5e. Everything runs on /dev/shm

aims works in a per-frame tmpfs directory so the per-k-point `D_*.csc` never touch Lustre.
Only the kept outputs are copied back. The runner is also snapshotted to `/dev/shm` at job
start, verified with `bash -n` plus an `# EOF` marker (hundreds of nodes reading one small
Lustre file occasionally returns a partial read that bash reports as a syntax error, silently
skipping the frame).

---

## 6. Bookkeeping rules

Violating these has caused every incident so far.

- **Completeness is manifest-minus-log, never directory counts.** Sweeping deletes finished
  frame dirs, archiving deletes whole blocks, `gen_manifests.py` recreates bare stubs.
  `runs/` is **not** a progress signal — `results/` is.
- **Progress = unique (block, frame_id) across both archive levels**, `results/<a>/<k>/*_pbe.xyz`
  and `chunks/*_pbe.xyz`, deduped, `*_passN*` overlap handled.
- **Convergence logs are append-only.** Status per functional is the *last* line for that frame
  in `convergence_<functional>.log`. `CONVERGED`/`FAILED` are terminal; `INFRA` and `RECOMPUTE`
  are not. To reopen a frame append
  `frame_XXXXX: RECOMPUTE at=<epoch> reason=...` — to **both** logs if the pw-LDA side is
  affected, because `submit.sh` decides completeness from the PBE log alone. Never rewrite a
  log; appends are safe against concurrent writers, rewrites are not.
- **One frame must never run in two jobs at once.** It produces an interleaved, unparseable
  `aims.out` that still logs CONVERGED. The pool guard exists for this; do not bypass it.
- **`merge_dataset.py` keeps a row only while the frame's last log status is CONVERGED**, so
  appending RECOMPUTE also drops a bad archived row at merge — and temporarily drops a good one
  until the rerun lands.
- **`merged/` is the product; `results/` is an append-only ledger** with duplicates and
  superseded rows. Never train on `results/` directly.
- Known quirk: a few frames log CONVERGED for a functional whose `aims.out` is absent (a forced
  warm-start rerun never rewrites it). `sweep_chunk.sh` lets such a frame leave only if that row
  is already banked. A "CONVERGED without aims.out and not banked" line means **look**, do not
  recompute blindly.

### Data flow

```
datasets/<arm>/*.extxyz                       source structures
  -> runs/<arm>/<kind>/block_NN/frame_XXXXX/  prepared inputs, live work
  -> results/<arm>/<kind>/                    banked tars + xyz (the ledger)
  -> merged/                                  log-CONVERGED frames only
  -> product/labelled/<arm>/                  the dataset you train on
```

`prepare_frames.py` also writes `runs/<arm>/<kind>/frames.tsv` mapping frame_id → configuration_id,
because ASE's aims writer discards `atoms.info` and the calculation directory would otherwise be
impossible to join back to `selection.tsv`.

---

## 7. Density matrices

**Preservation is opt-in per frame and off by default.** `preserve_density()` returns
immediately unless `preserve_density.txt` exists in the block dir *and* names the frame
verbatim. At ~21 GB per frame, preserving all 56,000 would need petabytes.

What is kept: 519 unique periodic frames, both functionals for 508 of them, **5.9 TB**, banked
at `results/<arm>/periodic_ttt/density_matrices/frame_XXXXX_<functional>_density.tar`. Only
`periodic_ttt` has lists — isolated and surfaces frames have none.

**The selection is not reproducible from code.** The lists were produced by hand; treat
`preserve_density.txt` itself as the record.

Nothing extracts these tars. They are a deliverable, not a restart mechanism — a wall-killed
frame does **not** warm-start from them.

---

## 8. Energy convention

With Gaussian smearing (0.05 eV) FHI-aims prints three energies: the plain total energy `E`,
`E − TS/2` ("Total energy corrected", extrapolation towards zero smearing), and `E − TS`
("Electronic free energy"). They coincide for gapped closed-shell systems and differ for
metals, radicals and open-shell atoms.

**Use the free energy `E − TS`** — it is the quantity the printed forces are derivatives of.

The `madcore_25k` arm and the MAD isolated-atom references carry it natively. The other three
arms came from a pipeline that delivered `E − TS/2`, so the shift was applied at merge:
`scan_free_energies.py` streams every archived `aims.out`, matches each delivered frame to the
run whose corrected energy equals the delivered value to 1e-5 eV, and takes that run's free
energy. `half_ts_<functional>` in the manifest records the amount subtracted.

Before this fix the 11 madcore/voronoi overlap structures disagreed by up to 4.7 meV/atom;
after it they agree exactly. **Do not train on `computed/` or the pipeline's `merged/`
directly** — those still carry `E − TS/2`.

---

## 9. Known deviations and gaps

Everything in this section is a real hole in the record. None is a design choice to be
defended.

### 9a. r²SCAN: Hamiltonians recovered 2026-09-14, physics block still differs

**Resolved for the outputs.** The Lyra run wrote the matrices but they were not in the first
transfer; they were shipped separately on 2026-09-14 into
`r2scan_random_test/matrices/frame_XXXX/`, 49 GB.

Audited on arrival: **all 973 converged frames have the complete set** —
`rs_hamiltonian.h5`, `rs_overlap.h5`, `eigenstate_info.out`, `basis-indices.out` — with zero
missing and zero empty files. All 1,968 `.h5` files carry a valid HDF5 signature and hold a
`sparse_matrix` dataset, and for all 984 frames present the Hamiltonian and overlap have
**identical file size**, as a matched sparse pair must. 246 frames carry `rs_indices.out`
against 244 periodic frames; the two extras (`frame_0261`, `frame_0357`) are large isolated
molecules of 223 and 228 atoms computed with no lattice vectors and no k-grid, so the extra
file is benign and their `pbc="F F F"` labels are right. Of the 27 non-converged frames, 16
have no matrices and 11 are partial, which is expected.

The matrices arrived **with no checksums** — none from Lyra, none in the directory's own
`SHA256SUMS` (16 lines, none mentioning matrices), and they are not inside
`r2scan_results_lyra.tar.gz`. A manifest was generated on arrival at
`matrices/SHA256SUMS.txt`: **4,191 files, 0 read errors**, matching the 4,191 on disk exactly
(984 × {`rs_hamiltonian.h5`, `rs_overlap.h5`, `eigenstate_info.out`, `basis-indices.out`} plus
255 `rs_indices.out`). Verify any time with `cd matrices && sha256sum -c SHA256SUMS.txt`.
The directory's top-level `SHA256SUMS` was deliberately left untouched, since it is
script-generated; `matrices/` carries its own.

**Still different: the physics block.** This is not fixed by the transfer and is worth knowing
separately. Same as LDA/PBE: `spin none`, `charge 0.`, `relativistic atomic_zora scalar`,
`sc_iter_limit 200`, `basis_threshold 1d-5`, `override_illconditioning`, all three
`sc_accuracy_*`, `compute_forces`, tight basis. Different:

| | LDA/PBE | r²SCAN |
|---|---|---|
| smearing / mixing | `occupation_type gaussian 0.05` + `charge_mix_param 0.02` | `adjust_scf always 2` + `adjust_scf_param {lowgap,largegap} occupation_width 0.05` |
| `sc_init_iter` | 199 | 99 |

So r²SCAN used adaptive SCF adjustment rather than fixed smearing with slow mixing. Do not
treat the two campaigns as the same protocol.

### 9b. `madcore_25k` has no run tree

Computed by the earlier toolchain; its working directory was deleted from scratch and is on
neither `aa002` nor `lp133`. Missing per frame: `aims.out`, `geometry.in`,
`convergence_pbe.log`, SCF cycle counts. The only surviving raw record is
`product/computed/madcore_25k-{pbe,pwlda}.xyz`.

Mitigation, all verified 2026-09-14: its `control.in` set in `common/_upstream/control/` is
**byte-identical** to the campaign's (6 of 6 files); it is cross-validated against the campaign
toolchain on **20 structures** to 0.000116 meV/atom; and its 15 tail structures were rerun from
scratch and reproduced their stored energies to 0.000115 meV/atom.

### 9c. 2,777 slabs ran without a dipole correction

Consequence of §4b. An asymmetric slab under 3D PBC produces a spurious field in the vacuum and
a slab-image interaction decaying only as 1/L — typically tens of meV plus contaminated forces
near the surface. **There is no measured number here; do not quote one.** An attempt to
identify which slabs are asymmetric enough to matter failed (the proxy scored a symmetric
Pt(111) slab higher than an asymmetric Cu/Pt one) and was discarded. Answering it needs test
calculations with and without the correction.

### 9d. Periodic arms are truncated at 250 atoms

Frames above 250 atoms are not computed (decided 2026-08-28, on the pipeline author's advice —
they exceed the 195G/72-rank sizing and routinely fail to finish LDA+PBE inside the 12 h wall).

| arm | periodic frames | above 250 | share |
|---|---|---|---|
| random_25k | 7,354 | 126 | 1.71% |
| random_test_6k | 1,725 | 23 | 1.33% |
| voronoi_25k | 23,658 | 14 | 0.06% |

Two things to state plainly whenever the dataset is described: the periodic arms are complete
only up to 250 atoms and should be called truncated, not a full sample; and the truncation is
**uneven across arms**, so they are not censored identically.

### 9e-bis. voronoi's higher PBE failure rate is composition, not geometry

Measured 2026-09-29 (`clariden_handoff/completion_20260929/WHY_VORONOI_FAILS_MORE.md`).
voronoi loses **5.9%** of LDA-converged frames at the PBE step against 1.0% for random and 1.1%
for madcore. It is **not** the "aggressive geometries" the earlier records claimed: madcore has
10x the force aggression (|F|max median 18.99 eV/Å, 28.7% above 50) at a 1.1% loss, and within
voronoi the failed frames are *milder* than the paired ones. The cause is what each arm is made
of — isolated molecules essentially never fail (0.1%), and within the same source the arms fail
identically (OMat24: random 7.8%, voronoi 6.9%; Alexandria: random 5.6%, voronoi 7.4%). voronoi
is 94% periodic and dominated by OMat24 and Alexandria small-gap inorganic bulk; random is 72%
isolated (shares of LDA-converged frames). **Similar source-level rates explain the aggregate
difference but do NOT establish unbiased missingness** — whether the structures lost within a
source are a random sample of it is unmeasured, and failure plausibly correlates with chemistry.
Do not subsample the archival product to equalise arms; handle matched sizes in the experiment's
own training-set construction instead.

### 9e. `voronoi_25k` is a half-run arm

Only ttt rows 0–10,999 of 22,623 and 16 of 1,035 surface slabs were ever submitted — it ran out
of node-hours. 11,163 of 25,000 have PBE. The remainder was bundled for Lyra, not computed.

### 9f. 194 converged runs were never banked

Present in `campaign/runs_unbanked/`, `xyz=no` in the census, on no known-bad list. All are
**half-pairs** — the partner functional never ran, hit INFRA, or needs RECOMPUTE — so none
yields a usable LDA+PBE row. Their `aims.out` is preserved; their h5 is not.

### 9g. The r²SCAN molecule-in-a-box convention differs

`madcore_r2scan.xyz` computed all 27 boxed rows as **periodic**, cells kept — the opposite of
the `set_pbc(False)` fix in §4c. That sidesteps the folding problem entirely. Not adopted for
LDA/PBE because consistency with `madcore_25k` governs that experiment. Open question which
convention the project wants long term.

### 9h. Other recorded incidents

`interleaved_outputs.tsv`, `stale_outputs.tsv`, `lost_frames.tsv`, `failed_in_archives.tsv`,
`RERUN_FROM_SCRATCH.tsv` in `campaign/`. The last of these — 22 frames that converged but whose
h5 were later removed from disk — was repaired on 2026-09-14; see
`lda_pbe/rerun_hamiltonians/reports/RESULTS.md`.

---

## 10. Cross-cluster policy

The same aims build number **260622** is used everywhere. Clariden runs a GNU/ARM build
(commit `a4a81ffa7`); Kuma and Lyra run the Jed/Ice-Lake Intel build (commit `350705b63`) on
AMD hardware, which prints

```
*** WARNING: CPU mismatch detected ... can produce incorrect results
```

**This warning is expected and accepted**, because it was tested rather than assumed:

| comparison | frames | agreement |
|---|---|---|
| Clariden vs Kuma | 83 | 4.2e-07 eV max total-energy difference |
| Clariden vs Lyra | 1 (23-atom molecule) | 1.26e-08 eV; max force difference 1.73e-05 eV/Å |
| Lyra 1 rank vs 32 ranks | 1 | 3.0e-10 eV |

Treat results from either side as interchangeable. Record every cross-cluster run's binary,
commit, CPU and compiler, as `LYRA_NOTES.md` does.

---

## 11. Cost anchors (measured, clariden, CHF 2.87/node-h)

| pool | node-h per frame |
|---|---|
| isolated | ~0.012 |
| small | 0.26 |
| big | 3.42 |
| monster | ~11 |

PBE costs the same as LDA per SCF iteration. GH200 node: 288 cores, 4 GPUs, 800G; aims runs
CPU-only. `normal` 12 h wall; `debug` 1.5 h, starts instantly, use for smoke tests.

---

## 12. Where things live

```
aa002/mad-core/lda_pbe/
  product/            28 GB   labelled/ computed/ references/ share/ sets/ scripts/
                              madcore_recheck/ rerun_hamiltonians/
  campaign/           11 TB   results/ from_kuma/ runs_unbanked/ merged/ datasets/
                              runs/ reports/ common/ docs/ rerun_hamiltonians_20260914/
aa002/mad-core/r2scan_random_test/   294 MB
```

Live pipeline tree (scratch, purged after 30 days unread):
`${DEV_ROOT}/lda_pbe/`
Sync to the store with `mad-core-valid/dft/lda_pbe/scripts/sync_to_aa002.sh --go` (idempotent,
additive, dry-run by default). Inventory and counts: `mad-core-valid/dft/lda_pbe/INVENTORY.md`.

One entry point for everything:

```
./common/bin/dft.sh status | queue | sweep | census | merge
./common/bin/dft.sh submit ARM KIND small|big|monster [--go]
```

All destructive or submitting actions are dry-run unless `--go`.
