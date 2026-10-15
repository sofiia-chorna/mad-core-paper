# LDA/PBE DFT data on aa002 — what exists, where, how much

Root: `${STORE_ROOT}/mad-core/lda_pbe/`
Two trees: `product/` (28 GB, what you train and evaluate on) and `campaign/` (11 TB, the raw
DFT ledger). Verified in sync with scratch on 2026-09-14.

## product/ — the dataset (28 GB)

| path | size | what |
|---|---|---|
| `labelled/<arm>/{pbe,lda}.xyz` + `manifest.tsv` | 391 MB | **the dataset.** Energies, forces, atomization energies, both functionals, one free-energy convention |
| `references/` | 27 GB | isolated-atom references: `monomers_run/` raw outputs, `free_energies.tsv`, `monomers_{pbe,lda}.xyz` |
| `computed/` | 315 MB | raw collected xyz straight from the DFT, before merging. `madcore_25k-{pbe,pwlda}.xyz` is the only surviving raw record of the MAD-CORE arm |
| `share/lda_pbe_2026-09-09/` | 369 MB | the collaborator bundle: `madcore_{pbe,lda}_all.xyz`, monomers, `CHECKSUMS.sha256` |
| `sets/` | 95 MB | selection tables and source extxyz per arm |
| `madcore_recheck/` | 88 MB | 2026-09-14 rerun of 30 MAD-CORE structures (convergence records) |
| `rerun_hamiltonians/` | 140 KB | scripts and reports for the 22-frame Hamiltonian rerun |
| `scripts/` | 120 KB | build and sync tooling, incl. `sync_to_aa002.sh` |

### Counts

| arm | selected | PBE | LDA | both | atoms (PBE) |
|---|---|---|---|---|---|
| madcore_25k | 25,000 | 24,514 | 24,719 | 24,447 | 159,357 |
| random_25k | 25,000 | 24,295 | 24,516 | 24,283 | 885,643 |
| voronoi_25k | 25,000 | 11,163 | 11,803 | 11,153 | 124,870 |
| random_test_6k | 6,000 | 5,860 | 5,887 | 5,848 | 215,562 |
| **total** | **81,000** | **65,832** | **66,925** | **65,731** | **1,385,432** |

`voronoi_25k` is short because that arm ran out of node-hours: only ttt rows 0–10,999 and 16
slabs were ever submitted. The rest is bundled for Lyra, not computed.

## campaign/ — the raw DFT ledger (11 TB)

| path | size | what |
|---|---|---|
| `results/<arm>/<kind>/` | 11 TB | **the bulk.** Per-block `block_NN_outputs.tar` (full aims output incl. `aims.out`, `geometry.in`, Hamiltonians) plus `block_NN_{pbe,pw-lda}.xyz` and convergence logs. random_25k 6.5 TB, random_test_6k 2.6 TB, voronoi_25k 1.4 TB |
| `from_kuma/` | 8.8 GB | 96 frames with full Hamiltonians + density matrices computed on Kuma |
| `runs_unbanked/` | 4.1 GB | text outputs of every prepared frame dir (26,786 `control.in`, 412 `aims.out`) — the 194 converged-but-never-banked runs live here |
| `rerun_hamiltonians_20260914/` | 3.8 GB | the 22 frames whose h5 were lost, recomputed 2026-09-14; 88 h5 files |
| `merged/` | 261 MB | intermediate: `results/` filtered to log-CONVERGED frames. `labelled/` was built from this |
| `datasets/`, `sets/` | 84 MB | source structures per arm and kind |
| `reports/` | 81 MB | `census.tsv` (85,785 rows, authoritative status), `tar_inventory.tsv`, audits |
| `runs/` | 161 MB | per-block `manifest.txt` (frame census) and convergence logs |
| `to_lyra/`, `to_kuma/`, `archive/`, `logs/`, `common/`, `docs/` | 1 GB | transfer bundles, pipeline code, job logs |

**No MAD-CORE run tree exists.** That arm was computed by an earlier, separate toolchain whose
working directory was deleted. Its only surviving raw record is `product/computed/madcore_25k-*.xyz`.
Its settings are preserved in `campaign/common/_upstream/control/` and are byte-identical to the
campaign's, and the arm is cross-validated against the campaign toolchain on 20 structures to
0.000116 meV/atom.

## Is `merge` needed?

**No.** `merged/` already matches the labelled product exactly:

| arm | merged PBE | labelled PBE | |
|---|---|---|---|
| random_25k | 24,295 | 24,295 | match |
| random_test_6k | 5,860 | 5,860 | match |
| voronoi_25k | 11,153 | 11,163 | +10, the documented `madcore ∩ voronoi` frames that carry the MAD-CORE label |

The 22 frames reopened for the Hamiltonian rerun are all logged `CONVERGED` again, and their
energies reproduce the stored values to 0.000000 meV/atom, so a merge would rebuild the same
rows. The new Hamiltonians are not part of `merged/` (which holds xyz only) — they are kept in
`campaign/rerun_hamiltonians_20260914/`.

## Other DFT in aa002/mad-core

`r2scan_random_test/` — **49 GB**, the 973-structure r²SCAN test set computed on Lyra:

| item | size | what |
|---|---|---|
| `matrices/frame_XXXX/` | 49 GB | Hamiltonians, overlaps, eigenstate info, basis indices — 1,000 frame dirs, complete for all 973 converged (transferred 2026-09-14) |
| `r2scan_results_lyra.tar.gz` | 286 MB | as returned from Lyra: the 1,000 raw `aims.out` plus Lyra's `results/` |
| `results/test_1000_r2scan.xyz` | 3.6 MB | 973 frames, the labels to train and evaluate on |
| `r2scan_random_test_1000.tar.gz` | 3.6 MB | the input bundle as sent: geometries, control.in, PBE labels |
| `lyra_record/`, `scripts/`, `test_1000.tsv`, `SELECTION.json` | 400 KB | execution record, membership, provenance |
