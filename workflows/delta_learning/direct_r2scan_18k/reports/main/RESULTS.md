# direct_r2scan_18k, stage main: direct r2SCAN fine tune on the shared 18,000 structure split, next to the two corrections

Energy |dE|/N in meV/atom pooled over structures, forces as Cartesian components pooled over all atoms in meV/A, against the r2SCAN references. MAE and RMSE are the mean and root mean square of the absolute errors, p8 is the L8 norm (mean of |e|^8)^(1/8), SUP is the largest |e|, E bias is the mean signed per atom energy error. The direct model predicts r2SCAN atomization energies and forces on its own and needs no DFT calculation at inference; the corrected rows (from delta_lda_pbe_18k, same structures) are the LDA or PBE DFT result plus the learned correction and need that DFT calculation. All three models share the training, validation and test structures, the PET-OMat-S checkpoint, the head replacement, the hyperparameters and seed 42.

Checkpoint selection rule, as metatrain computes it: the product of the validation metric entries whose name contains MAE (energy per atom and force components), lowest wins, evaluated every epoch on the 1,000 validation structures, applied to this model's direct predictions.

### test_madcore: 1,881 structures, 10,966 atoms, 32,898 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| direct r2SCAN (standalone) | no DFT | 143.2 / 209.0 | -6.9 | 382 / 903 | 629.6 | 5905 | 1483.3 | 17858 |
| LDA + learned delta_LDA | LDA DFT | 28.5 / 58.8 | +1.0 | 62 / 153 | 479.5 | 1308 | 1227.8 | 4281 |
| PBE + learned delta_PBE | PBE DFT | 24.0 / 56.7 | -0.8 | 51 / 133 | 464.8 | 1221 | 1176.3 | 3958 |

### test_madcore, nonperiodic: 137 structures, 401 atoms, 1,203 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| direct r2SCAN (standalone) | no DFT | 313.1 / 424.6 | -46.0 | 1379 / 2403 | 862.0 | 8361 | 1483.3 | 17858 |
| LDA + learned delta_LDA | LDA DFT | 72.4 / 112.0 | +4.9 | 217 / 397 | 372.5 | 1931 | 687.3 | 4281 |
| PBE + learned delta_PBE | PBE DFT | 65.1 / 109.9 | +11.7 | 180 / 361 | 440.3 | 1789 | 814.1 | 3958 |

### test_madcore, periodic: 1,744 structures, 10,565 atoms, 31,695 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| direct r2SCAN (standalone) | no DFT | 129.9 / 181.5 | -3.8 | 344 / 793 | 477.2 | 5305 | 948.3 | 17533 |
| LDA + learned delta_LDA | LDA DFT | 25.1 / 52.4 | +0.7 | 56 / 136 | 483.5 | 1055 | 1227.8 | 3280 |
| PBE + learned delta_PBE | PBE DFT | 20.8 / 50.2 | -1.8 | 46 / 116 | 466.4 | 1019 | 1176.3 | 3175 |

### test_random: 973 structures, 34,009 atoms, 102,027 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| direct r2SCAN (standalone) | no DFT | 61.6 / 78.0 | +46.8 | 266 / 391 | 202.9 | 1163 | 431.5 | 4311 |
| LDA + learned delta_LDA | LDA DFT | 20.3 / 24.1 | +17.2 | 41 / 60 | 80.6 | 252 | 174.0 | 1028 |
| PBE + learned delta_PBE | PBE DFT | 6.2 / 12.9 | -2.9 | 24 / 36 | 70.6 | 220 | 159.4 | 889 |

### test_random, nonperiodic: 729 structures, 20,539 atoms, 61,617 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| direct r2SCAN (standalone) | no DFT | 51.6 / 57.7 | +49.9 | 280 / 407 | 130.0 | 1091 | 293.3 | 3428 |
| LDA + learned delta_LDA | LDA DFT | 18.9 / 19.9 | +18.5 | 43 / 61 | 56.3 | 166 | 128.3 | 608 |
| PBE + learned delta_PBE | PBE DFT | 3.1 / 3.9 | -1.4 | 22 / 32 | 9.6 | 137 | 18.8 | 504 |

### test_random, periodic: 244 structures, 13,470 atoms, 40,410 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| direct r2SCAN (standalone) | no DFT | 91.3 / 119.8 | +37.5 | 244 / 366 | 240.6 | 1235 | 431.5 | 4311 |
| LDA + learned delta_LDA | LDA DFT | 24.4 / 33.7 | +13.2 | 39 / 58 | 95.3 | 283 | 174.0 | 1028 |
| PBE + learned delta_PBE | PBE DFT | 15.3 / 24.8 | -7.1 | 27 / 41 | 83.9 | 246 | 159.4 | 889 |

## Initialisation and selection of direct_r2scan--omat-s--madcore_shared18000--s42

- best epoch 113 of 150, fresh r2scan energy head, pretrained energy head pruned, baseline least squares of the r2scan atomization energies on this training set, scale 6.98750 eV, steps per epoch [281], optimizer updates 42150 in the final model (150 epochs logged, 0 redone after a restart)

## Checks

- PASS direct_r2scan--omat-s--madcore_shared18000--s42: corrected package
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: training finished
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: 150 epochs in the final model as configured (150), 150 logged
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: constant steps per epoch [281]
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: heads belong to the r2scan target only, pretrained energy head pruned
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: head freshly initialised
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: composition baseline equals an independent least-squares fit of the r2scan targets on every training structure (max 3.9e-14 eV/atom)
- PASS direct_r2scan--omat-s--madcore_shared18000--s42: scale 6.98750 eV equals the RMS residual per atom 6.98750 or per structure 54.99172
- PASS test_madcore: predictions in the order of the test file (positions within 7.9e-07 A), finite
- PASS test_madcore: the shared corrections were evaluated on the same 1,881 structures
- PASS test_madcore, nonperiodic: the shared corrections were evaluated on the same 137 structures
- PASS test_madcore, periodic: the shared corrections were evaluated on the same 1,744 structures
- PASS test_random: predictions in the order of the test file (positions within 0.0e+00 A), finite
- PASS test_random: the shared corrections were evaluated on the same 973 structures
- PASS test_random, nonperiodic: the shared corrections were evaluated on the same 729 structures
- PASS test_random, periodic: the shared corrections were evaluated on the same 244 structures
