# Figure captions

## rdf_water

O–O, O–H and H–H radial distribution functions, 5–50 ps of each run. 32 H2O in a fixed periodic 9.8555 Å cube, the Atomistic Cookbook thermostat start with its molecular centres scaled to density 1.0000 g/cm³, NVT at 300 K with an SVR thermostat (τ = 10 fs) and a 0.25 fs timestep. Runs per condition: PET-MAD-1.6-M 5, approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) 5, approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) 5. Faint lines are individual velocity-seeded runs, thick lines their arithmetic mean. PET-MAD-1.6-M is PET-MAD-M v1.6.0 alone (the corrected-attention-mask export). The approximate LDA and PBE potentials are PET-MAD-1.6-M minus the learned r²SCAN−LDA or r²SCAN−PBE correction trained on 18,000 MAD-CORE structures (validation 1,000), evaluated on the same coordinates at every step; they are machine-learned approximations, not DFT trajectories. Bins of 0.02 Å up to 4.92 Å, below half the cell width, normalised to the ideal gas at the cell density. The O–H and H–H functions include intramolecular pairs, and the O–H axis is cut below the intramolecular peak.

## oh_distance_water

Nearest and second-nearest O–H distances, 5–50 ps. For every H in every frame, the distance to its nearest O and to its second-nearest O under the minimum image, as probability densities. 32 H2O in a fixed periodic 9.8555 Å cube, the Atomistic Cookbook thermostat start with its molecular centres scaled to density 1.0000 g/cm³, NVT at 300 K with an SVR thermostat (τ = 10 fs) and a 0.25 fs timestep. Runs per condition: PET-MAD-1.6-M 5, approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) 5, approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) 5. Faint lines are individual velocity-seeded runs, thick lines their arithmetic mean. PET-MAD-1.6-M is PET-MAD-M v1.6.0 alone (the corrected-attention-mask export). The approximate LDA and PBE potentials are PET-MAD-1.6-M minus the learned r²SCAN−LDA or r²SCAN−PBE correction trained on 18,000 MAD-CORE structures (validation 1,000), evaluated on the same coordinates at every step; they are machine-learned approximations, not DFT trajectories.

## rdf_water_20_50ps and oh_distance_water_20_50ps

The same as rdf_water and oh_distance_water over 20–50 ps of each run, a supplementary window added after the block analysis showed the LDA-corrected structure still relaxing through 5–20 ps. Density 1.0000 g/cm³. The O–H and H–H functions include intramolecular pairs.

## temperature_water

Kinetic temperature of all atoms, of H and of O in one-picosecond bins, from the saved velocities, each subset over its share of the 3N−3 degrees of freedom. The dashed line marks the end of the discarded first 5 ps, the grey line the 300 K bath. 32 H2O in a fixed periodic 9.8555 Å cube, the Atomistic Cookbook thermostat start with its molecular centres scaled to density 1.0000 g/cm³, NVT at 300 K with an SVR thermostat (τ = 10 fs) and a 0.25 fs timestep. Runs per condition: PET-MAD-1.6-M 5, approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k) 5, approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k) 5. Faint lines are individual velocity-seeded runs, thick lines their arithmetic mean. PET-MAD-1.6-M is PET-MAD-M v1.6.0 alone (the corrected-attention-mask export). The approximate LDA and PBE potentials are PET-MAD-1.6-M minus the learned r²SCAN−LDA or r²SCAN−PBE correction trained on 18,000 MAD-CORE structures (validation 1,000), evaluated on the same coordinates at every step; they are machine-learned approximations, not DFT trajectories.

## rdf_oo_blocks_water

O–O radial distribution function of each time block, the mean over the runs of a condition: the discarded first 5 ps and the production blocks 5–20 ps, 20–35 ps, 35–50 ps. 32 H2O in a fixed periodic 9.8555 Å cube, the Atomistic Cookbook thermostat start with its molecular centres scaled to density 1.0000 g/cm³, NVT at 300 K with an SVR thermostat (τ = 10 fs) and a 0.25 fs timestep.

## rdf_oo_oh_hh_runs_20_50ps

O–O, O–H and H–H radial distribution functions over 20–50 ps of each run. Thick lines are the arithmetic mean of the five velocity-seeded runs per potential, thin lines the individual runs. The O–H panel starts at zero so the covalent peak is shown in full, and its inset repeats 1.3–4.9 Å on a smaller g(r) scale for the hydrogen-bond peaks. The O–H and H–H functions include intramolecular pairs. 32 H2O in a fixed periodic 9.8555 Å cube at 1.0000 g/cm³, NVT at 300 K with an SVR thermostat (τ = 10 fs) and a 0.25 fs timestep. PET-MAD-1.6-M is PET-MAD-M v1.6.0 alone; LDA-corrected and PBE-corrected are PET-MAD-1.6-M minus the learned r²SCAN−LDA or r²SCAN−PBE correction trained on 18,000 MAD-CORE structures (validation 1,000), machine-learned approximations, not DFT trajectories. Bins of 0.02 Å up to 4.92 Å, normalised to the ideal gas at the cell density.
