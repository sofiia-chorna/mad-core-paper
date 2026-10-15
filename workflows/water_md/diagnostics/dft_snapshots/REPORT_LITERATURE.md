# Literature comparison of the O-O radial distribution functions

Our values are the five-run means over 20 to 50 ps as stored by the production analysis (spread as given there); n_OO within 3.28 A is integrated here from the stored mean curves with the DiStasio cutoff and the analysis convention (unique pairs, density (N minus 1) over V). Literature values are copied from the papers' tables (Todorova Table 4 in pm, DiStasio Table I).

| curve | first peak r, A | first peak g | FWHM, A | first minimum r, A | first minimum g | second peak r, A | second peak g | n_OO |
|---|---|---|---|---|---|---|---|---|
| PET-MAD-1.6-M, 300 K | 2.75 | 3.14 ± 0.06 | 0.38 | 3.34 | 0.68 ± 0.03 | 4.43 | 1.29 | 4.27 ± 0.01 (3.28 A), 4.33 (3.3 A) |
| approx. LDA (18k), 300 K | 2.57 | 4.81 ± 0.14 | 0.28 | 3.09 | 0.19 ± 0.03 | 4.21 | 1.61 | 4.12 ± 0.03 (3.28 A), 4.15 (3.3 A) |
| approx. PBE (18k), 300 K | 2.73 | 3.48 ± 0.14 | 0.34 | 3.28 | 0.39 ± 0.06 | 4.37 | 1.51 | 4.05 ± 0.01 (3.28 A), 4.08 (3.3 A) |
| LDA, Todorova 2006, 350 K | 2.59 | 5.07 | 0.21 | 3.12 | 0.23 |  |  |  |
| PBE, Todorova 2006, 350 K | 2.70 | 2.99 | 0.37 | 3.29 | 0.47 |  |  |  |
| PBE, DiStasio 2014, 300 K | 2.69 | 3.28 |  | 3.28 | 0.37 | 4.44 | 1.44 | 4.03 |
| experiment, Skinner 2013, 295 K (as quoted by DiStasio) | 2.80 | 2.57 |  | 3.45 | 0.84 | 4.50 | 1.12 | 4.3 |
| experiment, Soper and Benmore 2008, 298 K (as quoted by DiStasio) | 2.76 | 2.62 |  | 3.42 | 0.82 | 4.43 | 1.14 | 4.67 |

## Protocol differences

| quantity | this work | Todorova 2006 (LDA and PBE rows) | DiStasio 2014 (PBE row) |
|---|---|---|---|
| potential | PET-MAD-1.6-M minus a learned r2SCAN minus LDA or PBE correction, machine-learned approximation | Kohn-Sham DFT, LDA in the Pade parametrisation, PBE; Troullier-Martins pseudopotentials, 70 Ry plane waves | Kohn-Sham DFT, PBE; Troullier-Martins pseudopotentials, 71 Ry plane waves |
| molecules, cell | 32, cubic 9.855 A | 32, cubic 9.939 A | 64, cubic 12.4 A |
| density | 1.000 g/cm3 | 0.975 g/cm3 | experimental ambient density |
| temperature | 300 K | 350 K | 300 K |
| dynamics | Born-Oppenheimer with the ML potential, velocity Verlet, 0.25 fs | Car-Parrinello, electron mass 600 a.u., 5 a.u. (0.12 fs) time step | Car-Parrinello, electron mass 300 a.u., 4 a.u. (0.1 fs) time step |
| nuclei | classical, hydrogen mass | classical, deuterium mass for hydrogen | classical, deuterium mass for hydrogen |
| thermostat | stochastic velocity rescaling, tau 10 fs, global | Nose-Hoover chains of length 4 on ions and electrons | massive Nose-Hoover chains of length 4 |
| sampling | 5 trajectories, 20 to 50 ps each, 150 ps in total | 10 ps equilibration, 10 ps averages | about 2 ps equilibration, at least 20 ps averages |
| dispersion | none beyond what the functionals contain | none | none in the plain PBE row |
| periodic box treatment | fixed cell, pressures between minus 8 and plus 7 kbar | fixed cell | fixed cell |

## Reading

- Qualitative consistency, not validation: the approximate LDA potential reproduces the direction and the size class of the LDA overstructuring reported at 350 K (first peak near 2.58 A with g about 5, first minimum near 0.2, narrow peak), and the approximate PBE potential sits between the two PBE references and above experiment, as PBE does in both papers.
- The references were obtained at other temperatures (350 K for the LDA row), another density (0.975 g/cm3), with deuterium masses, Car-Parrinello dynamics and pseudopotential plane-wave DFT, so a quantitative match is not expected and would not be meaningful at the 0.1 to 0.3 level in g.
- The approximate PBE first peak moved from 3.06 (10k correction) to 3.48 (18k correction) at identical MD protocol, which is larger than its distance to either PBE reference; the correction, not the physics, dominates that number.
- Neither paper reports proton transfer or ionic defects for LDA water, so the coordination changes seen with the approximate LDA potential are not a known LDA feature and are the point most in need of the direct DFT check.
