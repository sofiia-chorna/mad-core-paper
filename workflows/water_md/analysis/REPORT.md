# Water production, automatic summary

Written by `scripts/analyse.py`. 32 H2O in a fixed periodic cube of 9.855480 Å, the Cookbook start with its molecular centres scaled to 1.0 g/cm³, density 1.0000 g/cm³. NVT at 300 K.

Means are over the velocity-seeded runs of a condition, ± one standard error over runs. Analysis window 5 ps to the end of each analysed trajectory, which ends only at a confirmed numerical failure. Production blocks 5-20 ps, 20-35 ps, 35-50 ps, plus the discarded 0-5 ps for reference.

Read the equilibration and convergence section before the structure tables. Removing the first 5 ps does not by itself establish equilibrium.

## Runs

A confirmed numerical failure is a non-finite position, velocity or force, or a crashed run. Threshold crossings (any distance below 0.6 Å, any force above 36.0 eV/Å, kinetic temperature above 600.0 K) are listed separately, flagged for inspection, and do not cut the trajectory.

| run | complete | analysed to (ps) | confirmed numerical failure | threshold crossings | integration warning | T (K) | T_H (K) | T_O (K) | conserved drift (meV/atom/ps) | max F (eV/Å) | min d (Å) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_seed0 | yes | 50.00 | none | none | no | 299.5 | 300.0 | 298.3 | +0.0010 | 5.97 | 0.882 |
| baseline_seed1 | yes | 50.00 | none | none | no | 299.4 | 299.0 | 300.2 | -0.0003 | 6.43 | 0.876 |
| baseline_seed2 | yes | 50.00 | none | none | no | 301.1 | 300.3 | 302.7 | +0.0009 | 6.29 | 0.874 |
| baseline_seed3 | yes | 50.00 | none | none | no | 299.7 | 299.6 | 299.9 | +0.0002 | 6.10 | 0.882 |
| baseline_seed4 | yes | 50.00 | none | none | no | 299.8 | 300.2 | 298.8 | +0.0004 | 6.23 | 0.876 |
| lda_corrected_seed0 | yes | 50.00 | none | none | no | 299.6 | 301.1 | 296.4 | -0.0001 | 5.98 | 0.890 |
| lda_corrected_seed1 | yes | 50.00 | none | none | no | 299.5 | 296.0 | 306.6 | -0.0001 | 5.68 | 0.903 |
| lda_corrected_seed2 | yes | 50.00 | none | none | no | 300.7 | 299.5 | 303.3 | +0.0006 | 6.09 | 0.903 |
| lda_corrected_seed3 | yes | 50.00 | none | none | no | 299.5 | 299.9 | 298.7 | -0.0002 | 6.01 | 0.903 |
| lda_corrected_seed4 | yes | 50.00 | none | none | no | 299.5 | 304.4 | 289.6 | +0.0004 | 5.62 | 0.898 |
| pbe_corrected_seed0 | yes | 50.00 | none | none | no | 299.4 | 295.4 | 307.7 | -0.0011 | 6.48 | 0.888 |
| pbe_corrected_seed1 | yes | 50.00 | none | none | no | 299.4 | 301.6 | 294.9 | +0.0010 | 5.82 | 0.892 |
| pbe_corrected_seed2 | yes | 50.00 | none | none | no | 301.1 | 296.6 | 310.3 | -0.0001 | 6.99 | 0.881 |
| pbe_corrected_seed3 | yes | 50.00 | none | none | no | 299.7 | 295.6 | 308.2 | -0.0002 | 6.20 | 0.888 |
| pbe_corrected_seed4 | yes | 50.00 | none | none | no | 299.8 | 296.9 | 305.7 | -0.0004 | 6.41 | 0.886 |

## Threshold crossings

No frame of any run crosses a threshold.

## Frame bookkeeping

Positions, velocities and forces are paired by step, the latest occurrence of a step wins. Restart duplicates count the superseded occurrences over the three files. All trajectory files are kept in full.

| run | segments | paired frames (of 4001) | restart duplicates | damaged frames | damaged property rows | missing steps | unpaired steps |
|---|---|---|---|---|---|---|---|
| baseline_seed0 | 6 | 4001 | 249 | 0 | 0 | 0 | 0 |
| baseline_seed1 | 6 | 4001 | 318 | 0 | 0 | 0 | 0 |
| baseline_seed2 | 7 | 4001 | 582 | 0 | 0 | 0 | 0 |
| baseline_seed3 | 7 | 4001 | 315 | 0 | 0 | 0 | 0 |
| baseline_seed4 | 7 | 4001 | 324 | 0 | 0 | 0 | 0 |
| lda_corrected_seed0 | 6 | 4001 | 309 | 0 | 0 | 0 | 0 |
| lda_corrected_seed1 | 6 | 4001 | 279 | 0 | 0 | 0 | 0 |
| lda_corrected_seed2 | 6 | 4001 | 414 | 0 | 0 | 0 | 0 |
| lda_corrected_seed3 | 6 | 4001 | 357 | 0 | 0 | 0 | 0 |
| lda_corrected_seed4 | 6 | 4001 | 273 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed0 | 6 | 4001 | 252 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed1 | 6 | 4001 | 267 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed2 | 6 | 4001 | 216 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed3 | 6 | 4001 | 309 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed4 | 6 | 4001 | 339 | 0 | 0 | 0 | 0 |

## Equilibration and convergence

### Kinetic temperatures by block

From the saved velocities, each subset over its share of 3N-3 degrees of freedom. Per-picosecond means are in `curves/temperatures_1ps.tsv` and `figures/temperature_water.png`.

| condition | block (ps) | T (K) | T_H (K) | T_O (K) | T_H - T_O (K) |
|---|---|---|---|---|---|
| PET-MAD-1.6-M | 0-5 (discarded) | 299.6 ± 1.0 | 294.0 ± 2.1 | 311.3 ± 2.3 | -17.3 ± 4.1 |
| PET-MAD-1.6-M | 5-20 | 300.2 ± 0.5 | 300.6 ± 1.6 | 299.6 ± 2.8 | 1.0 ± 4.3 |
| PET-MAD-1.6-M | 20-35 | 299.4 ± 0.3 | 299.1 ± 1.2 | 299.8 ± 3.2 | -0.7 ± 4.4 |
| PET-MAD-1.6-M | 35-50 | 300.0 ± 0.3 | 299.8 ± 1.3 | 300.5 ± 3.3 | -0.8 ± 4.6 |
| PET-MAD-1.6-M | last minus first production block | -0.2 ± 0.5 | -0.8 ± 2.8 | 1.0 ± 5.8 | -1.8 ± 8.6 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 0-5 (discarded) | 300.6 ± 0.9 | 281.4 ± 3.0 | 340.0 ± 5.3 | -58.6 ± 8.2 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 5-20 | 300.2 ± 0.6 | 300.9 ± 3.2 | 298.8 ± 6.5 | 2.1 ± 9.5 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 20-35 | 299.3 ± 0.3 | 299.6 ± 1.6 | 298.6 ± 2.7 | 1.1 ± 4.2 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 35-50 | 299.8 ± 0.4 | 300.0 ± 1.5 | 299.4 ± 3.5 | 0.6 ± 4.9 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | last minus first production block | -0.4 ± 0.8 | -0.9 ± 2.7 | 0.6 ± 4.6 | -1.5 ± 7.1 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 0-5 (discarded) | 299.8 ± 0.9 | 294.4 ± 2.1 | 310.8 ± 3.3 | -16.4 ± 5.0 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 5-20 | 300.3 ± 0.6 | 296.1 ± 1.5 | 308.7 ± 2.7 | -12.6 ± 4.1 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 20-35 | 299.3 ± 0.3 | 296.5 ± 2.4 | 305.0 ± 5.1 | -8.4 ± 7.4 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 35-50 | 300.1 ± 0.3 | 299.0 ± 1.6 | 302.4 ± 3.3 | -3.4 ± 4.8 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | last minus first production block | -0.2 ± 0.6 | 2.8 ± 2.8 | -6.4 ± 4.3 | 9.2 ± 7.0 |

### O-O structure, potential energy and O-H bond by block

| condition | block (ps) | first peak r (Å) | first peak g | first min g | second peak g | n_OO(3.3 Å) | potential (meV/molecule) | nearest O-H (Å) |
|---|---|---|---|---|---|---|---|---|
| PET-MAD-1.6-M | 0-5 (discarded) | 2.75 ± 0.01 | 2.92 ± 0.06 | 0.76 ± 0.02 | 1.25 ± 0.01 | 4.36 ± 0.02 | -15725.0 ± 2.2 | 0.9780 ± 0.0003 |
| PET-MAD-1.6-M | 5-20 | 2.76 ± 0.01 | 3.05 ± 0.03 | 0.70 ± 0.01 | 1.27 ± 0.03 | 4.33 ± 0.01 | -15733.5 ± 1.7 | 0.9788 ± 0.0002 |
| PET-MAD-1.6-M | 20-35 | 2.74 ± 0.00 | 3.16 ± 0.06 | 0.68 ± 0.03 | 1.29 ± 0.02 | 4.33 ± 0.01 | -15736.1 ± 2.5 | 0.9790 ± 0.0002 |
| PET-MAD-1.6-M | 35-50 | 2.75 ± 0.01 | 3.15 ± 0.08 | 0.67 ± 0.04 | 1.30 ± 0.03 | 4.33 ± 0.02 | -15737.1 ± 3.4 | 0.9792 ± 0.0003 |
| PET-MAD-1.6-M | last minus first production block | -0.00 ± 0.01 | 0.10 ± 0.09 | -0.04 ± 0.04 | 0.03 ± 0.04 | 0.00 ± 0.02 | -3.6 ± 3.5 | 0.0004 ± 0.0003 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 0-5 (discarded) | 2.58 ± 0.00 | 4.43 ± 0.09 | 0.29 ± 0.03 | 1.46 ± 0.04 | 4.18 ± 0.05 | -15690.6 ± 6.1 | 1.0202 ± 0.0007 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 5-20 | 2.57 ± 0.01 | 4.83 ± 0.13 | 0.19 ± 0.03 | 1.58 ± 0.06 | 4.14 ± 0.05 | -15711.7 ± 4.0 | 1.0234 ± 0.0013 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 20-35 | 2.57 ± 0.01 | 4.84 ± 0.19 | 0.18 ± 0.02 | 1.61 ± 0.03 | 4.13 ± 0.02 | -15717.6 ± 2.5 | 1.0232 ± 0.0006 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 35-50 | 2.59 ± 0.00 | 4.81 ± 0.08 | 0.19 ± 0.03 | 1.62 ± 0.04 | 4.16 ± 0.04 | -15718.0 ± 1.9 | 1.0230 ± 0.0004 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | last minus first production block | 0.01 ± 0.01 | -0.02 ± 0.06 | 0.00 ± 0.02 | 0.04 ± 0.05 | 0.02 ± 0.06 | -6.4 ± 2.4 | -0.0003 ± 0.0013 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 0-5 (discarded) | 2.75 ± 0.01 | 3.08 ± 0.03 | 0.58 ± 0.01 | 1.36 ± 0.02 | 4.12 ± 0.01 | -14536.1 ± 1.6 | 0.9889 ± 0.0002 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 5-20 | 2.74 ± 0.01 | 3.27 ± 0.06 | 0.49 ± 0.01 | 1.45 ± 0.02 | 4.09 ± 0.02 | -14549.7 ± 1.9 | 0.9902 ± 0.0003 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 20-35 | 2.73 ± 0.01 | 3.48 ± 0.14 | 0.40 ± 0.05 | 1.50 ± 0.05 | 4.09 ± 0.02 | -14560.2 ± 6.0 | 0.9913 ± 0.0007 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 35-50 | 2.72 ± 0.00 | 3.53 ± 0.15 | 0.37 ± 0.06 | 1.55 ± 0.06 | 4.08 ± 0.02 | -14563.6 ± 7.1 | 0.9917 ± 0.0008 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | last minus first production block | -0.02 ± 0.01 | 0.26 ± 0.16 | -0.12 ± 0.06 | 0.10 ± 0.05 | -0.01 ± 0.03 | -13.9 ± 5.7 | 0.0014 ± 0.0007 |

Drift flags, a last-minus-first change larger than twice its standard error over runs:

- PET-MAD-1.6-M: none
- approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k): first_min_r_A, potential_mean_meV_per_molecule
- approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k): first_peak_r_A, first_min_g, peak_to_min_ratio, potential_mean_meV_per_molecule, nearest_oh_mean_A

### RDF convergence across blocks

Integrated absolute difference of two g(r), summed over r ≥ 1.3 Å in Å, which leaves out the covalent O-H peak. Within-run columns compare successive blocks of the same run. The seed-to-seed column compares different runs in the same production block, the size of the difference expected from finite sampling alone. A ratio above one means a block-to-block change larger than that spread, a ratio well below one means the runs differ from each other more than each run changes over time, a sign that they sample different states. Neither alone proves convergence, read it with the drift flags above.

| pair | condition | 0-5 vs 5-20 ps | 5-20 vs 20-35 ps | 20-35 vs 35-50 ps | seed to seed | largest production change / seed spread |
|---|---|---|---|---|---|---|
| O-O | PET-MAD-1.6-M | 0.185 ± 0.026 | 0.131 ± 0.018 | 0.137 ± 0.037 | 0.170 ± 0.016 | 0.80 |
| O-O | approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 0.293 ± 0.038 | 0.193 ± 0.037 | 0.119 ± 0.021 | 0.330 ± 0.018 | 0.58 |
| O-O | approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 0.269 ± 0.039 | 0.252 ± 0.017 | 0.143 ± 0.019 | 0.285 ± 0.035 | 0.89 |
| O-H | PET-MAD-1.6-M | 0.128 ± 0.012 | 0.078 ± 0.005 | 0.090 ± 0.018 | 0.104 ± 0.007 | 0.86 |
| O-H | approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 0.198 ± 0.017 | 0.122 ± 0.022 | 0.087 ± 0.015 | 0.239 ± 0.013 | 0.51 |
| O-H | approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 0.156 ± 0.023 | 0.133 ± 0.014 | 0.091 ± 0.009 | 0.176 ± 0.015 | 0.76 |
| H-H | PET-MAD-1.6-M | 0.127 ± 0.011 | 0.076 ± 0.005 | 0.086 ± 0.018 | 0.109 ± 0.007 | 0.79 |
| H-H | approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 0.200 ± 0.018 | 0.136 ± 0.024 | 0.097 ± 0.016 | 0.206 ± 0.009 | 0.66 |
| H-H | approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 0.171 ± 0.024 | 0.137 ± 0.025 | 0.091 ± 0.009 | 0.159 ± 0.014 | 0.86 |

## O-O structure, 5-50 ps

RDF grid 0.02 Å bins to 4.92 Å, below half the cell width, minimum image, ideal-gas normalisation at the density of this cell. O-H and H-H RDFs include intramolecular pairs.

| condition | first peak r (Å) | first peak g | first min r (Å) | first min g | second peak r (Å) | second peak g | peak/min | n_OO(3.3 Å) | n_OO(own first min) |
|---|---|---|---|---|---|---|---|---|---|
| PET-MAD-1.6-M | 2.75 ± 0.01 | 3.10 ± 0.05 | 3.34 ± 0.02 | 0.69 ± 0.02 | 4.45 ± 0.02 | 1.27 ± 0.01 | 4.50 ± 0.21 | 4.33 ± 0.01 | 4.50 ± 0.07 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 2.58 ± 0.00 | 4.79 ± 0.13 | 3.09 ± 0.01 | 0.19 ± 0.03 | 4.25 ± 0.03 | 1.59 ± 0.04 | 27.19 ± 3.82 | 4.14 ± 0.03 | 3.93 ± 0.01 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 2.73 ± 0.01 | 3.39 ± 0.10 | 3.28 ± 0.01 | 0.43 ± 0.04 | 4.38 ± 0.02 | 1.49 ± 0.04 | 8.19 ± 0.89 | 4.09 ± 0.01 | 4.08 ± 0.04 |

## O-O structure, 20-50 ps, supplementary

Added after the block analysis showed the LDA-corrected structure still relaxing through 5-20 ps. The 5-50 ps table above is the frozen protocol, this later window leaves out that relaxation for all three conditions alike.

| condition | first peak r (Å) | first peak g | first min r (Å) | first min g | second peak r (Å) | second peak g | peak/min | n_OO(3.3 Å) | potential (meV/molecule) | nearest O-H (Å) |
|---|---|---|---|---|---|---|---|---|---|---|
| PET-MAD-1.6-M | 2.75 ± 0.00 | 3.14 ± 0.06 | 3.34 ± 0.02 | 0.68 ± 0.03 | 4.43 ± 0.02 | 1.29 ± 0.02 | 4.70 ± 0.30 | 4.33 ± 0.01 | -15736.6 ± 2.6 | 0.9791 ± 0.0003 |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 2.57 ± 0.01 | 4.81 ± 0.14 | 3.09 ± 0.01 | 0.19 ± 0.03 | 4.21 ± 0.03 | 1.61 ± 0.04 | 27.89 ± 4.08 | 4.15 ± 0.03 | -15717.8 ± 2.2 | 1.0231 ± 0.0004 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 2.73 ± 0.01 | 3.48 ± 0.14 | 3.28 ± 0.02 | 0.39 ± 0.06 | 4.37 ± 0.03 | 1.51 ± 0.05 | 9.62 ± 1.43 | 4.08 ± 0.01 | -14561.9 ± 6.6 | 0.9915 ± 0.0008 |

## Thermodynamics and molecules, 5-50 ps

Nearest O-H distance: for every H, the distance to its nearest O. Pressure is the virial pressure of a 96-atom cell, a diagnostic only.

| condition | T (K) | T_H (K) | T_O (K) | potential (meV/molecule) | pressure (bar) | nearest O-H (Å) | H-O-H (deg) | correction energy (meV/molecule) |
|---|---|---|---|---|---|---|---|---|
| PET-MAD-1.6-M | 299.9 ± 0.3 | 299.8 ± 0.3 | 300.0 ± 0.8 | -15735.6 ± 2.0 | -260 ± 235 | 0.9790 ± 0.0002 | 105.59 ± 0.03 | n/a |
| approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) | 299.8 ± 0.2 | 300.2 ± 1.4 | 298.9 ± 2.9 | -15715.8 ± 2.7 | -7571 ± 983 | 1.0232 ± 0.0006 | 107.21 ± 0.18 | 3.3 ± 3.3 |
| approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) | 299.9 ± 0.3 | 297.2 ± 1.1 | 305.4 ± 2.7 | -14557.8 ± 4.9 | 6970 ± 443 | 0.9911 ± 0.0006 | 105.46 ± 0.07 | -1192.4 ± 0.5 |

## Coordination changes, possible proton-transfer events

Each H is assigned to its nearest O in every frame. A frame with any O that owns other than two H, or an H whose nearest O changes, is a coordination change and a possible proton-transfer event. It is reported and never counted as a numerical failure. Persistently on another O means away from the initial O for at least 0.5 ps in a row.

| run | frames with a coordination change (all / after 5 ps) | longest stretch (ps) | nearest-O changes between frames | H on another O at end | H persistently on another O | frames with H-H < 1 Å | frames with O-O < 2 Å |
|---|---|---|---|---|---|---|---|
| baseline_seed0 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| baseline_seed1 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| baseline_seed2 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| baseline_seed3 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| baseline_seed4 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| lda_corrected_seed0 | 2221 / 2173 | 4.1375 | 2403 | 32 | 42 | 0 | 0 |
| lda_corrected_seed1 | 358 / 166 | 2.1625 | 431 | 5 | 13 | 0 | 0 |
| lda_corrected_seed2 | 245 / 236 | 0.2250 | 443 | 9 | 10 | 0 | 0 |
| lda_corrected_seed3 | 345 / 331 | 0.2625 | 571 | 18 | 27 | 0 | 0 |
| lda_corrected_seed4 | 643 / 584 | 0.6625 | 1078 | 24 | 29 | 0 | 0 |
| pbe_corrected_seed0 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed1 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed2 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed3 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| pbe_corrected_seed4 | 0 / 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
