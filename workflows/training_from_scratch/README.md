The eight PET models of Figs. 9 and 10, trained from scratch on 2k, 5k, 10k and 20k
structures from the top of MAD-CORE or drawn at random, each set with the 102 isolated atoms
added (scripts/build_sets.py, manifests/). PET-OMat-S size, MSE loss, Adam, batch 64,
learning rate 3e-4, cosine schedule, seed 42, 187,200 updates for every model
(runs/PROVENANCE.json, runs/*/options.yaml). The checkpoint is chosen on 1,000 random and
250 MAD-CORE validation structures. The test is 2,000 random and 2,000 MAD-CORE structures
(reports/populations/manifests, drawn with seed 20261008), scripts/evaluate_final.py
evaluates, scripts/population_figures.py writes the figure data. The checkpoints are in the
models archive.
