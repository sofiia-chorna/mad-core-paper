Training of the ecumetric, a SPACE model (metatrain), on 677,454 structures of MAD-1.5,
MAD-NEXT and MAD-SOL (17.9 M atoms) with r2SCAN energies, non conservative forces and five
per atom heads that regress the last layer features of PET-MAD-1.5, MACE-OMat, UMA-OMat,
DPA3.2-OMat and DPA3-OMol. Huber losses, batch 32, learning rate 3e-3, cutoff 7.5 A, two
message passing layers, 9.7 M parameters (options.yaml, options_restart.yaml).

prep_features.py builds the training memmaps from the three xyz files and the teacher
features (build_stats.json has the counts and the teacher scales), submit.sh runs metatrain
on one node with four GPUs, about 15 minutes per epoch. The run stopped at epoch 92
(run_2026-06-26_03-09-50/train.log, validation MAE 29.07 meV/atom and 85.4 meV/A, split
594,769 / 41,338 / 41,347 in indices/). MAD-NEXT is not public yet, the memmaps are only on
the project store.
