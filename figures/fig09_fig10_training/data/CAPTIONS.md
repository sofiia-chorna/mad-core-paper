# Captions, round 3 pooled-test figures (2026-10-08)

Common to all figures below. PET-S trained from scratch, one seed (42). The figures show six models, MAD-CORE
and RANDOM training at 5k, 10k and 20k structures; the 2k models are in the metric tables only. Each training set also contains the same 102 isolated-atom reference
structures (one per element, atomization energy 0, forces 0); the training-size axis and the size labels count only
the ordinary structures. Checkpoints are the validation-selected ones (best native mae_prod on the common
1,250-structure validation set, 1,000 RANDOM + 250 MAD-CORE), the same checkpoints as in the separate-test figures;
nothing was retrained or reselected. Errors come from the stored test predictions of those checkpoints.

## Evaluation populations

- **Full union (`union_7847`)**: all 5,847 structures of the frozen RANDOM test set plus all 2,000 structures of the
  frozen MAD-CORE test set; 7,847 structures, 225,959 atoms, 677,877 force components. The RANDOM test set contributes
  74.5 % of the structures and 94.7 % of the atoms and force components.
- **Equal structure counts (`balanced_4000`)**: 2,000 structures drawn uniformly without replacement from the frozen
  RANDOM test set (seed 20261008, by id only, before any error was read), plus all 2,000 MAD-CORE test structures;
  4,000 structures, 85,166 atoms, 255,498 force components. This version is balanced by structure count, not by atom
  or force-component count: the RANDOM part has 36.6 atoms per structure against 6.0 for the MAD-CORE part, so it
  still contributes 86.0 % of the atoms and force components.

The two tests share no structure in either version (configuration id, frozen id, geometry hash and invariant
fingerprint all checked).

## `learning_curves_union_7847.{png,pdf}` and `learning_curves_balanced_4000.{png,pdf}`

Test error against training-set size, MAD-CORE training (orange squares) and RANDOM training (blue circles).
Rows: energy (meV/atom) and force (eV/Å). Columns: MAE, RMSE, p8 = (mean e^8)^(1/8) and SUP = max. The test set of each
figure is the population named in the file (see above); the two tests are pooled into one population, not averaged.
Energy: absolute error per atom of each structure, |ΔE| / N, pooled over the structures of the population. Forces:
absolute Cartesian force-component errors, pooled over all atoms and components of the population (metatrain
definition); normalisation recomputed for the population, so a structure with many atoms weighs more in the force
metrics. Each panel has the same y range in both versions. Equal test-structure counts do not equalize the
force-component counts: in the balanced version 86.0 % of the force components still come from the RANDOM test part.
Every model's largest energy and force error lies in the MAD-CORE test set, which both versions contain in full, so SUP
is identical in the two versions.

## `sketchmap_energy_<population>.{png,pdf}` and `sketchmap_force_<population>.{png,pdf}`

Mean test error on the sketch map of the library, for exactly the structures of the named population. Rows: RANDOM
and MAD-CORE training; columns: 5k, 10k and 20k training structures. Projection coordinates, bin boundaries and
colour limits are those of the earlier maps (colour limits = 1st and 99th percentile of the round 1 bin values:
energy 35.7 to 9,496 meV/atom, force 0.112 to 10.9 eV/Å); bins outside these limits are drawn in the end colours.
Bin values and counts are computed again for each population. Aggregation per structure: each structure contributes
one value, |ΔE| / N for energy and the mean absolute Cartesian force-component error over its 3N components for
forces; a bin shows the Gaussian-smoothed (sigma 0.9 bins) sum of these values divided by the smoothed structure count.
Bins with a smoothed count below 1.5, and isolated patches smaller than 6 bins, are blank, so support shrinks with the
smaller population (559 bins with 7,624 of 7,847 structures in the full union, 544 bins with 3,768 of 4,000 structures
in the balanced version). The map weighs structures equally while the global force metrics pool components, so the
global force values need not equal an average of the bin values. Last column: library density (grey) with its
contours, which are repeated on every panel, and the number of evaluated structures per bin (blue), on a count scale
shared by both versions.
