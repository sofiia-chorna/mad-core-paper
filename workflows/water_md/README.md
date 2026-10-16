The water simulations of SI Fig. 3. 32 molecules in a cubic box of 9.855480 A, 1.000
g/cm3, NVT at 300 K with a stochastic velocity rescaling thermostat (10 fs), 0.25 fs steps,
200,000 steps, five seeds per potential, the same starts for all. i-PI runs two clients at
every step, PET-MAD-1.6-M with weight +1 and a correction with weight -1
(scripts/metatomic_output_driver.py, runs/production/*/input.xml). The baseline is an
export of PET-MAD-M v1.6.0 with a fixed attention mask (baseline_model, the two exports agree
to float32 rounding). scripts/analyse.py computes the radial distribution functions over 20
to 50 ps in 0.02 A bins, the curves are in analysis/curves. diagnostics/dft_snapshots
checks nine snapshots at LDA, PBE and r2SCAN: PET-MAD gives 11 meV/A against r2SCAN, the
corrected potentials 27 and 31 meV/A against LDA and PBE.
