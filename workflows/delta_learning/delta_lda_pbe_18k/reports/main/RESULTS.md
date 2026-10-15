# delta_lda_pbe_18k, stage main: r2SCAN-LDA and r2SCAN-PBE corrections on one shared split

Energy |dE|/N in meV/atom pooled over structures, forces as Cartesian components pooled over all atoms in meV/A, against the r2SCAN references. MAE and RMSE are the mean and root mean square of the absolute errors, p8 is the L8 norm (mean of |e|^8)^(1/8), SUP is the largest |e|, E bias is the mean signed per atom energy error (prediction minus reference). Corrected rows are reconstructed as the DFT energy and forces of the respective functional plus the learned correction, so each corrected prediction requires that functional's DFT calculation at inference. Both corrections are evaluated on identical structures. No test offset is subtracted and no checkpoint is chosen on test data.

Checkpoint selection rule, as metatrain computes it: at every epoch the validation metric entries whose name contains MAE (energy per atom and force components) are multiplied, the epoch with the lowest product is kept (best_model_metric mae_prod), applied separately to each model on the same 1,000 validation structures.

### test_madcore: 1,881 structures, 10,966 atoms, 32,898 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| LDA, no correction | LDA DFT | 945.0 / 1090.7 | -936.5 | 217 / 460 | 1561.8 | 1926 | 2601.5 | 6346 |
| LDA + learned delta_LDA | LDA DFT | 28.5 / 58.8 | +1.0 | 62 / 153 | 479.5 | 1308 | 1227.8 | 4281 |
| PBE, no correction | PBE DFT | 633.3 / 693.9 | +630.6 | 111 / 245 | 927.9 | 1283 | 1845.5 | 4098 |
| PBE + learned delta_PBE | PBE DFT | 24.0 / 56.7 | -0.8 | 51 / 133 | 464.8 | 1221 | 1176.3 | 3958 |

### test_madcore, nonperiodic: 137 structures, 401 atoms, 1,203 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| LDA, no correction | LDA DFT | 397.2 / 462.2 | -329.5 | 758 / 1069 | 667.8 | 2878 | 994.6 | 6346 |
| LDA + learned delta_LDA | LDA DFT | 72.4 / 112.0 | +4.9 | 217 / 397 | 372.5 | 1931 | 687.3 | 4281 |
| PBE, no correction | PBE DFT | 375.3 / 460.0 | +346.5 | 295 / 479 | 767.4 | 1846 | 1179.3 | 4098 |
| PBE + learned delta_PBE | PBE DFT | 65.1 / 109.9 | +11.7 | 180 / 361 | 440.3 | 1789 | 814.1 | 3958 |

### test_madcore, periodic: 1,744 structures, 10,565 atoms, 31,695 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| LDA, no correction | LDA DFT | 988.0 / 1125.3 | -984.2 | 197 / 420 | 1576.6 | 1439 | 2601.5 | 3868 |
| LDA + learned delta_LDA | LDA DFT | 25.1 / 52.4 | +0.7 | 56 / 136 | 483.5 | 1055 | 1227.8 | 3280 |
| PBE, no correction | PBE DFT | 653.6 / 709.1 | +652.9 | 104 / 231 | 934.8 | 1121 | 1845.5 | 3113 |
| PBE + learned delta_PBE | PBE DFT | 20.8 / 50.2 | -1.8 | 46 / 116 | 466.4 | 1019 | 1176.3 | 3175 |

### test_random: 973 structures, 34,009 atoms, 102,027 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| LDA, no correction | LDA DFT | 57.5 / 120.8 | -42.9 | 193 / 263 | 406.1 | 509 | 919.3 | 1361 |
| LDA + learned delta_LDA | LDA DFT | 20.3 / 24.1 | +17.2 | 41 / 60 | 80.6 | 252 | 174.0 | 1028 |
| PBE, no correction | PBE DFT | 437.2 / 448.5 | +437.0 | 111 / 158 | 566.1 | 429 | 976.8 | 1305 |
| PBE + learned delta_PBE | PBE DFT | 6.2 / 12.9 | -2.9 | 24 / 36 | 70.6 | 220 | 159.4 | 889 |

### test_random, nonperiodic: 729 structures, 20,539 atoms, 61,617 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| LDA, no correction | LDA DFT | 16.8 / 23.4 | -3.4 | 233 / 297 | 64.5 | 525 | 137.1 | 1194 |
| LDA + learned delta_LDA | LDA DFT | 18.9 / 19.9 | +18.5 | 43 / 61 | 56.3 | 166 | 128.3 | 608 |
| PBE, no correction | PBE DFT | 408.1 / 409.1 | +408.1 | 124 / 166 | 415.2 | 429 | 515.7 | 1134 |
| PBE + learned delta_PBE | PBE DFT | 3.1 / 3.9 | -1.4 | 22 / 32 | 9.6 | 137 | 18.8 | 504 |

### test_random, periodic: 244 structures, 13,470 atoms, 40,410 force components

| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |
|---|---|---|---|---|---|---|---|---|
| LDA, no correction | LDA DFT | 178.9 / 237.8 | -160.9 | 132 / 200 | 482.8 | 475 | 919.3 | 1361 |
| LDA + learned delta_LDA | LDA DFT | 24.4 / 33.7 | +13.2 | 39 / 58 | 95.3 | 283 | 174.0 | 1028 |
| PBE, no correction | PBE DFT | 524.0 / 549.8 | +523.3 | 90 / 146 | 667.5 | 429 | 976.8 | 1305 |
| PBE + learned delta_PBE | PBE DFT | 15.3 / 24.8 | -7.1 | 27 / 41 | 83.9 | 246 | 159.4 | 889 |

## Initialisation and selection

- delta_lda--omat-s--madcore_shared18000--s42: best epoch 102 of 150, fresh delta head, pretrained energy head pruned, baseline least squares on this training set, scale 0.45985 eV, steps per epoch [281], optimizer updates 42150 in the final model (155 epochs logged, 5 redone after a restart)
- delta_pbe--omat-s--madcore_shared18000--s42: best epoch 101 of 150, fresh delta head, pretrained energy head pruned, baseline least squares on this training set, scale 0.14191 eV, steps per epoch [281], optimizer updates 42150 in the final model (174 epochs logged, 24 redone after a restart)

## Checks

- PASS train: lda + delta target reproduces r2scan (E 0.0e+00 eV, F 4.4e-16 eV/A)
- PASS train: pbe + delta target reproduces r2scan (E 0.0e+00 eV, F 2.2e-16 eV/A)
- PASS test_madcore: lda + delta target reproduces r2scan (E 0.0e+00 eV, F 2.2e-16 eV/A)
- PASS test_madcore: pbe + delta target reproduces r2scan (E 0.0e+00 eV, F 1.1e-16 eV/A)
- PASS test_random: lda + delta target reproduces r2scan (E 0.0e+00 eV, F 2.2e-16 eV/A)
- PASS test_random: pbe + delta target reproduces r2scan (E 0.0e+00 eV, F 1.1e-16 eV/A)
- PASS delta_lda--omat-s--madcore_shared18000--s42: corrected package
- PASS delta_lda--omat-s--madcore_shared18000--s42: training finished
- PASS delta_lda--omat-s--madcore_shared18000--s42: 150 epochs in the final model as configured (150), 155 logged
- PASS delta_lda--omat-s--madcore_shared18000--s42: constant steps per epoch [281]
- PASS delta_lda--omat-s--madcore_shared18000--s42: heads belong to the delta target only
- PASS delta_lda--omat-s--madcore_shared18000--s42: delta head freshly initialised
- PASS delta_lda--omat-s--madcore_shared18000--s42: composition baseline equals an independent least-squares fit on every training structure (max 4.0e-15 eV/atom)
- PASS delta_lda--omat-s--madcore_shared18000--s42: scale 0.45985 eV equals the RMS residual per atom 0.45985 or per structure 3.07526
- PASS delta_pbe--omat-s--madcore_shared18000--s42: corrected package
- PASS delta_pbe--omat-s--madcore_shared18000--s42: training finished
- PASS delta_pbe--omat-s--madcore_shared18000--s42: 150 epochs in the final model as configured (150), 174 logged
- PASS delta_pbe--omat-s--madcore_shared18000--s42: constant steps per epoch [281]
- PASS delta_pbe--omat-s--madcore_shared18000--s42: heads belong to the delta target only
- PASS delta_pbe--omat-s--madcore_shared18000--s42: delta head freshly initialised
- PASS delta_pbe--omat-s--madcore_shared18000--s42: composition baseline equals an independent least-squares fit on every training structure (max 3.3e-15 eV/atom)
- PASS delta_pbe--omat-s--madcore_shared18000--s42: scale 0.14191 eV equals the RMS residual per atom 0.14191 or per structure 0.84485
- PASS both runs read the same training and validation files
- PASS test_madcore delta_lda--omat-s--madcore_shared18000--s42: predictions in the order of the test file (positions within 7.9e-07 A), finite
- PASS test_madcore delta_pbe--omat-s--madcore_shared18000--s42: predictions in the order of the test file (positions within 7.9e-07 A), finite
- PASS test_madcore delta_lda--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_madcore delta_pbe--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_madcore, nonperiodic delta_lda--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 2.8e-14)
- PASS test_madcore, nonperiodic delta_pbe--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_madcore, periodic delta_lda--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_madcore, periodic delta_pbe--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_random delta_lda--omat-s--madcore_shared18000--s42: predictions in the order of the test file (positions within 0.0e+00 A), finite
- PASS test_random delta_pbe--omat-s--madcore_shared18000--s42: predictions in the order of the test file (positions within 0.0e+00 A), finite
- PASS test_random delta_lda--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_random delta_pbe--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_random, nonperiodic delta_lda--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 7.1e-15)
- PASS test_random, nonperiodic delta_pbe--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_random, periodic delta_lda--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
- PASS test_random, periodic delta_pbe--omat-s--madcore_shared18000--s42: reconstructed r2SCAN error equals the delta target error (gap 0.0e+00)
