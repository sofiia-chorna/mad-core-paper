# LDA/PBE recompute — the four frozen selections

The universe-level selection experiment: pick unlabelled structures straight out of the
**full 1.7B release pool** by different strategies, recompute them at a cheap DFT level
(**LDA and PBE**, MAD settings recipe), train fresh pet-mad heads on each, and compare. This is
the experiment that answers the question the r2SCAN work cannot: *top-K of the hierarchy vs
random from the whole pool*.

Source pool: the frozen release at `${STORE_ROOT}/mad-core/release`
(read-only). Coordinates are the released folded coordinates. Everything joins back to the
release by `configuration_id`.

## Status 2026-10-03

DFT is complete for three arms; Voronoi has 22,695 pairs with ~300 structures still open on
other clusters (see `completion_20260929/STATUS_20260929.md`). **The snapshot to train on is
`labelled_20260929b/`** (checksummed, `SHA256SUMS`). Energy convention, references, units,
force filter and the paired-vs-single-functional distinction are in `TRAINING_HANDOFF.md`.

| arm | selected | PBE | LDA | paired | pair-eligible (fmax ≤ 100) |
|---|---:|---:|---:|---:|---:|
| madcore_25k | 25,000 | 24,514 | 24,719 | 24,447 | 22,276 |
| random_25k | 25,000 | 24,295 | 24,516 | 24,283 | 24,281 |
| voronoi_25k | 25,000 | 22,712 | 24,116 | 22,695 | 22,558 |
| random_test_6k | 6,000 | 5,860 | 5,887 | 5,848 | 5,847 |

## Layout (working copy on scratch)

    labelled_20260929b/   CURRENT snapshot: <arm>/{pbe,lda}.xyz + manifest.tsv, SHA256SUMS, merge log
    labelled/             original 2026-09-06 build; the stage-1 models were trained from it (provenance)
    computed/             raw collected xyz straight from the DFT; madcore_25k-{pbe,pwlda}.xyz is the
                          only surviving raw record of the MAD-CORE arm. Carries E - TS/2: never train on it
    sets/<arm>/           frozen selections: selection.tsv, release_ids.tsv, {ttt,surfaces,fff}.extxyz
    references/           monomers_{pbe,lda}.xyz (102 elements), free_energies*.tsv, free_energy_scan/,
                          monomers_run/ (scripts and templates only; the 22 GB of run outputs are on aa002)
    scripts/              selection, splitting and label-building scripts; sync_to_aa002.sh
    madcore_recheck/      2026-09-14 rerun of 30 MAD-CORE structures: reports/ + scripts (run dirs on aa002)
    rerun_hamiltonians/   the 22-frame Hamiltonian rerun: lists, reports, submit scripts
    completion_20260929/  Voronoi completion campaign: status, handovers, ledgers, submit scripts
    import_2026*/         merge logs and the new_*.xyz of each label import (staged raw outputs removed)
    share/                build_share.py and the 2026-09-09 collaborator bundle record (xyz on aa002)
    INVENTORY.md          what exists on aa002, where, how much
    TRAINING_HANDOFF.md   how to train from the snapshot

Everything removed from scratch on 2026-10-03 (superseded `labelled_*` and `computed_*`
snapshots, staged raw outputs, monomer run outputs, logs) is byte-identical on aa002 under
`${STORE_ROOT}/mad-core/lda_pbe/product/` or inside the checksummed
delivery tars under `${STORE_ROOT}/mad-core/voronoi_remaining/`.
Rebuilding a snapshot in place needs its `computed_<date>/` restored from aa002 first.

## The four arms (`sets/<arm>/`)

Selections were frozen 2026-08-12/13. Arms are sized **25k** so that ~20k survive DFT
non-convergence.

| arm | what it is | total | ttt | fff | surfaces | elements | atoms | mean nat |
|---|---|---|---|---|---|---|---|---|
| `madcore_25k` | first 25,000 eligible picks of the mad-core coreset in FPS/accrete order | 25,000 | 22,683 | 1,777 | 540 | 102 | 176,947 | 7.1 |
| `random_25k` | uniform over the full pool (1,694,502,355 rows), seed 20260812 | 25,000 | 4,577 | 17,642 | 2,781 | 102 | 956,930 | 38.3 |
| `voronoi_25k` | level-24 Voronoi cells drawn with prob ∝ population^(1/4), seed 20260813 | 25,000 | 22,623 | 1,342 | 1,035 | 102 | 358,052 | 14.3 |
| `random_test_6k` | held-out test, same uniform sampler, seed 20260814 | 6,000 | 1,071 | 4,275 | 654 | 89 | 230,200 | 38.4 |

Verified 2026-08-19: every arm has the stated row count, no duplicate ids inside an arm, and
all pairs disjoint by `configuration_id` **except** `madcore ∩ voronoi = 21` (by design; listed
in `sets/voronoi_25k/madcore_intersection.txt`, computed once and the labels shared).

- **madcore_25k**: eligible = MAD row (`dataset_group == mad-next`, no atom cap) or any other
  row with < 300 atoms. Picks 0..25,008 consumed; 9 oversized non-MAD rows skipped.
  `fps_order` is kept on every frame.
- **random_25k**: rejects release-filtered rows, duplicates, ≥ 300 atoms (MAD tier exempt) and
  `madcore_25k` ids. 11 elements missing from the plain draw were topped up by
  `ensure_elements.py` (9 swaps, logged in `element_topup.tsv`). 57% gdb13 molecules, which is
  what uniform-over-the-pool means.
- **voronoi_25k**: 25,000 of the 16,777,216 level-24 cells drawn without replacement with
  probability ∝ population^alpha, alpha = 1/4 (Gumbel top-k), one structure per cell = first
  eligible member of the within-cell FPS ordering.
- **random_test_6k**: disjoint from all training arms, 89 elements, deliberately not topped up.

## Files in each arm

- `ttt.extxyz` — true bulk: fully periodic, no vacuum
- `surfaces.extxyz` — slabs/2D (vacuum along 1 lattice direction) and wires/1D (2), stored
  fully periodic with their real cells
- `fff.extxyz` — molecules/clusters: non-periodic flags, plus periodic molecule-in-a-box rows
- `selection.tsv` — one row per structure: `configuration_id`, provenance, `natoms`, pbc flags,
  `vacuum_axes`, `split_file`; madcore carries `fps_order`, the random arms `row_in_group`,
  voronoi `release_folder / row_in_folder / cell / cell_population / member_rank`

Vacuum = largest empty gap along a lattice direction exceeds 6 Å. `vacuum_axes`: 0 bulk,
1 slab, 2 wire, 3 box-molecule, -1 not flag-TTT. Any bucket can be re-cut from the manifest
with `resplit_by_vacuum.py`.

## Scripts (`scripts/`)

Selection: `select_madcore_25k.py`, `select_random_25k.py`, `select_random_test6k.py` (wrappers
over `random_sampler.py`), `select_voronoi_25k.py`; element guarantee `check_elements.py`,
`ensure_elements.py`; splitting `xyz_write.py`, `resplit_by_vacuum.py`. The `*.log` files are
the original sampler runs. Paths inside the selection scripts are `Path(__file__).parent / "out"
/ <arm>`; to reproduce the chain, first `ln -s ../sets scripts/out`.

Labels: `merge_labels.py` is the builder; each dated `merge_labels_staged_<date>.py` is the
exact script that produced `labelled_<date>/` from `computed_<date>/` and the matching
`references/free_energies*.tsv`. `scan_free_energies.py` builds the free-energy table from the
raw `aims.out` files. `sync_to_aa002.sh` is the additive sync to the durable copy.

## Provenance

Selections copied 2026-08-19 from `mad/madcore_vs_random_20k/` (byte-identical). The MAD-CORE
arm was computed by an earlier toolchain whose working directory was deleted; only the
collected xyz survives (see `TRAINING_HANDOFF.md` §6). All other arms have the full output set
on aa002.
