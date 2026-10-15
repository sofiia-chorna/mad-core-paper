# Snapshot check against LDA, PBE and r2SCAN single points

## FHI-aims single points

| snapshot | functional | state | scf cycles | free energy, eV |
|---|---|---|---|---|
| event_after | pw-lda | converged | 21 | -66201.28124065 |
| event_after | pbe | converged | 22 | -66603.5725863828 |
| event_after | r2scan | converged | 20 | -66639.6188975977 |
| event_before | pw-lda | converged | 23 | -66200.6772440556 |
| event_before | pbe | converged | 23 | -66603.5479503919 |
| event_before | r2scan | converged | 20 | -66639.9945025785 |
| event_during | pw-lda | converged | 21 | -66201.2127880589 |
| event_during | pbe | converged | 25 | -66603.6493122661 |
| event_during | r2scan | converged | 21 | -66639.7829138012 |
| late_baseline_seed1_step140000 | pw-lda | converged | 27 | -66196.4771600023 |
| late_baseline_seed1_step140000 | pbe | converged | 26 | -66603.6002994693 |
| late_baseline_seed1_step140000 | r2scan | converged | 20 | -66640.9391793408 |
| late_baseline_seed3_step200000 | pw-lda | converged | 21 | -66196.2596907347 |
| late_baseline_seed3_step200000 | pbe | converged | 23 | -66603.3393765127 |
| late_baseline_seed3_step200000 | r2scan | converged | 20 | -66640.5764363669 |
| late_lda_corrected_seed1_step140000 | pw-lda | converged | 24 | -66200.714566295 |
| late_lda_corrected_seed1_step140000 | pbe | converged | 24 | -66604.2516285122 |
| late_lda_corrected_seed1_step140000 | r2scan | converged | 20 | -66640.349211866 |
| late_lda_corrected_seed3_step200000 | pw-lda | converged | 21 | -66200.7746833399 |
| late_lda_corrected_seed3_step200000 | pbe | converged | 24 | -66602.830030183 |
| late_lda_corrected_seed3_step200000 | r2scan | converged | 20 | -66639.0610765075 |
| late_pbe_corrected_seed1_step140000 | pw-lda | converged | 25 | -66198.9648628324 |
| late_pbe_corrected_seed1_step140000 | pbe | converged | 21 | -66604.9463819676 |
| late_pbe_corrected_seed1_step140000 | r2scan | converged | 20 | -66641.6365592774 |
| late_pbe_corrected_seed3_step200000 | pw-lda | converged | 23 | -66198.7030589823 |
| late_pbe_corrected_seed3_step200000 | pbe | converged | 23 | -66604.7319130831 |
| late_pbe_corrected_seed3_step200000 | r2scan | converged | 19 | -66641.3839194419 |

9 of 9 snapshots have all three functionals converged; the comparisons below use those.

## Forces pooled over the complete snapshots, MAE / RMSE / max in meV/A per Cartesian component

| comparison | H | O | all | RMS of the reference, H / O |
|---|---|---|---|---|
| A baseline PET-MAD-1.6-M against r2SCAN | 10 / 13 / 63 | 14 / 18 / 56 | 11 / 15 / 63 | 712 / 991 |
| B1 delta_LDA against F_r2SCAN minus F_LDA | 22 / 28 / 132 | 25 / 32 / 120 | 23 / 29 / 132 | 438 / 478 |
| B2 delta_PBE against F_r2SCAN minus F_PBE | 29 / 35 / 89 | 22 / 28 / 107 | 27 / 33 / 107 | 226 / 272 |
| C1 approx. LDA (baseline minus delta_LDA) against LDA | 25 / 31 / 162 | 31 / 39 / 154 | 27 / 34 / 162 | 678 / 935 |
| C2 approx. PBE (baseline minus delta_PBE) against PBE | 33 / 40 / 123 | 28 / 35 / 104 | 31 / 38 / 123 | 659 / 930 |
| D1 context, baseline alone against LDA | 377 / 441 / 895 | 408 / 480 / 1017 | 388 / 454 / 1017 | 678 / 935 |
| D2 context, baseline alone against PBE | 195 / 229 / 478 | 233 / 274 / 618 | 208 / 245 / 618 | 659 / 930 |
| D3 context, DFT r2SCAN against DFT LDA | 375 / 438 / 878 | 407 / 478 / 1000 | 386 / 452 / 1000 | 678 / 935 |
| D4 context, DFT r2SCAN against DFT PBE | 193 / 226 / 445 | 231 / 272 / 621 | 206 / 243 / 621 | 659 / 930 |

## Per snapshot force RMSE in meV/A, H / O

| snapshot | A base vs r2SCAN | B1 dLDA | B2 dPBE | C1 approx LDA vs LDA | C2 approx PBE vs PBE | D3 r2SCAN vs LDA | D4 r2SCAN vs PBE |
|---|---|---|---|---|---|---|---|
| event_after | 16 / 21 | 32 / 41 | 31 / 25 | 33 / 46 | 37 / 32 | 406 / 442 | 206 / 244 |
| event_before | 13 / 18 | 31 / 36 | 33 / 27 | 33 / 42 | 39 / 35 | 430 / 464 | 216 / 256 |
| event_during | 13 / 20 | 30 / 35 | 31 / 22 | 32 / 43 | 36 / 33 | 409 / 455 | 207 / 255 |
| late_baseline_seed1_step140000 | 11 / 17 | 26 / 28 | 38 / 35 | 31 / 36 | 44 / 40 | 467 / 525 | 247 / 301 |
| late_baseline_seed3_step200000 | 11 / 14 | 22 / 24 | 37 / 31 | 26 / 28 | 41 / 37 | 459 / 518 | 244 / 301 |
| late_lda_corrected_seed1_step140000 | 13 / 18 | 30 / 32 | 34 / 24 | 32 / 37 | 38 / 30 | 433 / 449 | 222 / 260 |
| late_lda_corrected_seed3_step200000 | 15 / 21 | 31 / 35 | 29 / 22 | 35 / 44 | 35 / 28 | 410 / 448 | 205 / 246 |
| late_pbe_corrected_seed1_step140000 | 11 / 16 | 26 / 26 | 40 / 32 | 30 / 34 | 44 / 38 | 462 / 505 | 242 / 291 |
| late_pbe_corrected_seed3_step200000 | 11 / 15 | 25 / 25 | 40 / 33 | 28 / 33 | 44 / 40 | 458 / 491 | 241 / 287 |

## Decomposition of the approximate potential error, RMS over all components in meV/A

approx minus target = (baseline minus r2SCAN) + (r2SCAN minus target minus delta). The two terms and their sum per snapshot.

| snapshot | LDA: baseline term | LDA: correction term | LDA: sum | PBE: baseline term | PBE: correction term | PBE: sum |
|---|---|---|---|---|---|---|
| event_after | 18 | 35 | 38 | 18 | 29 | 35 |
| event_before | 15 | 32 | 36 | 15 | 31 | 37 |
| event_during | 16 | 32 | 36 | 16 | 28 | 35 |
| late_baseline_seed1_step140000 | 13 | 27 | 33 | 13 | 37 | 43 |
| late_baseline_seed3_step200000 | 12 | 23 | 27 | 12 | 35 | 40 |
| late_lda_corrected_seed1_step140000 | 15 | 31 | 34 | 15 | 31 | 36 |
| late_lda_corrected_seed3_step200000 | 17 | 32 | 38 | 17 | 27 | 33 |
| late_pbe_corrected_seed1_step140000 | 13 | 26 | 31 | 13 | 37 | 42 |
| late_pbe_corrected_seed3_step200000 | 13 | 25 | 30 | 13 | 38 | 43 |

## Relative energies in meV/atom, each source minus its own mean over the complete snapshots

| snapshot | DFT LDA | DFT PBE | DFT r2SCAN | PET-MAD-1.6-M | approx. LDA | approx. PBE | DFT r2SCAN minus LDA | delta_LDA model | DFT r2SCAN minus PBE | delta_PBE model |
|---|---|---|---|---|---|---|---|---|---|---|
| event_after | -19.1 | +2.7 | +7.8 | +8.3 | -19.8 | +5.5 | +26.9 | +28.1 | +5.2 | +2.9 |
| event_before | -12.8 | +2.9 | +3.9 | +4.2 | -14.3 | +4.4 | +16.7 | +18.5 | +1.0 | -0.2 |
| event_during | -18.3 | +1.9 | +6.1 | +6.6 | -20.4 | +3.5 | +24.5 | +27.0 | +4.2 | +3.1 |
| late_baseline_seed1_step140000 | +31.0 | +2.4 | -5.9 | -6.2 | +32.8 | -0.6 | -36.9 | -39.1 | -8.3 | -5.6 |
| late_baseline_seed3_step200000 | +33.3 | +5.1 | -2.1 | -2.6 | +36.3 | +2.3 | -35.4 | -38.8 | -7.2 | -4.8 |
| late_lda_corrected_seed1_step140000 | -13.2 | -4.4 | +0.2 | +0.4 | -13.1 | -3.1 | +13.4 | +13.4 | +4.6 | +3.5 |
| late_lda_corrected_seed3_step200000 | -13.8 | +10.4 | +13.6 | +14.1 | -15.7 | +12.4 | +27.4 | +29.8 | +3.2 | +1.7 |
| late_pbe_corrected_seed1_step140000 | +5.1 | -11.6 | -13.2 | -13.8 | +6.2 | -13.2 | -18.2 | -20.0 | -1.5 | -0.7 |
| late_pbe_corrected_seed3_step200000 | +7.8 | -9.4 | -10.5 | -10.9 | +8.0 | -11.2 | -18.3 | -18.9 | -1.2 | +0.3 |

| model against reference | MAE of relative energies, meV/atom | max | spread of the reference (max minus min) |
|---|---|---|---|
| PET-MAD-1.6-M vs DFT r2SCAN | 0.4 | 0.7 | 26.8 |
| approx. LDA vs DFT LDA | 1.4 | 3.0 | 52.3 |
| approx. PBE vs DFT PBE | 2.0 | 3.0 | 22.0 |
| delta_LDA model vs DFT r2SCAN minus LDA | 1.8 | 3.4 | 64.3 |
| delta_PBE model vs DFT r2SCAN minus PBE | 1.7 | 2.7 | 13.5 |

## The proton transfer event, energies relative to the before snapshot in meV/atom (eV per cell in brackets)

| source | during | after |
|---|---|---|
| DFT LDA | -5.6 (-0.536) | -6.3 (-0.604) |
| DFT PBE | -1.1 (-0.101) | -0.3 (-0.025) |
| DFT r2SCAN | +2.2 (+0.212) | +3.9 (+0.376) |
| PET-MAD-1.6-M | +2.4 (+0.232) | +4.2 (+0.401) |
| approx. LDA | -6.1 (-0.585) | -5.5 (-0.527) |
| approx. PBE | -0.9 (-0.084) | +1.1 (+0.104) |

## Forces on the transferring proton and its two oxygens, component along the donor to acceptor axis in eV/A (positive pushes toward the acceptor)

hydrogen 41, donor oxygen 39, acceptor oxygen 54, run lda_corrected_seed3. For the oxygens the same axis is used, so a positive value moves the oxygen toward the acceptor side.

| snapshot | source | H along axis | H magnitude | donor O along axis | acceptor O along axis | donor H, acceptor H distance, A |
|---|---|---|---|---|---|---|
| event_before | DFT LDA | -1.01 | 1.28 | +0.50 | +0.20 | 1.051, 1.545 |
| event_before | DFT PBE | -1.36 | 1.59 | +0.57 | +0.42 | 1.051, 1.545 |
| event_before | DFT r2SCAN | -1.69 | 1.92 | +0.84 | +0.58 | 1.051, 1.545 |
| event_before | PET-MAD-1.6-M | -1.68 | 1.90 | +0.82 | +0.57 | 1.051, 1.545 |
| event_before | approx. LDA | -0.96 | 1.23 | +0.44 | +0.20 | 1.051, 1.545 |
| event_before | approx. PBE | -1.39 | 1.62 | +0.57 | +0.45 | 1.051, 1.545 |
| event_during | DFT LDA | -0.75 | 0.85 | -0.17 | +0.77 | 1.195, 1.207 |
| event_during | DFT PBE | -0.81 | 0.92 | -0.25 | +1.00 | 1.195, 1.207 |
| event_during | DFT r2SCAN | -0.84 | 0.95 | -0.22 | +1.10 | 1.195, 1.207 |
| event_during | PET-MAD-1.6-M | -0.81 | 0.93 | -0.23 | +1.08 | 1.195, 1.207 |
| event_during | approx. LDA | -0.73 | 0.84 | -0.19 | +0.80 | 1.195, 1.207 |
| event_during | approx. PBE | -0.79 | 0.89 | -0.27 | +0.99 | 1.195, 1.207 |
| event_after | DFT LDA | -0.97 | 0.99 | +0.10 | +1.18 | 1.653, 0.989 |
| event_after | DFT PBE | -0.55 | 0.57 | -0.23 | +1.02 | 1.653, 0.989 |
| event_after | DFT r2SCAN | -0.15 | 0.21 | -0.50 | +0.72 | 1.653, 0.989 |
| event_after | PET-MAD-1.6-M | -0.13 | 0.19 | -0.53 | +0.72 | 1.653, 0.989 |
| event_after | approx. LDA | -0.99 | 1.00 | +0.02 | +1.27 | 1.653, 0.989 |
| event_after | approx. PBE | -0.47 | 0.48 | -0.28 | +1.00 | 1.653, 0.989 |

## Consistency of the stored MD forces with the float64 models

largest difference between the forces i-PI stored and baseline minus correction recomputed in float64, over the nine snapshots: 3.9e-05 eV/A

