The map of Fig. 8. derive_params_2p20.py takes sigma from the pair weighted distance
histogram of the first 2^20 MAD-CORE structures (1.24). fit_dist.py fits an eight
dimensional sketch-map with A 4, B 2, a 2, b 2, PCA start, 400 iterations, each point
weighted by its level 20 cell population to the power 0.25, on two nodes in 79 minutes.
train_projector.py trains the MLP 512, 512, 256, 128, 8 with SiLU that extends the map
(projector/, validation R2 0.998), apply_projector.py projects all 2^24. The fit needs the
torch sketch-map of the scikit-matter fork, which is not published yet. Overlaps in the first
two dimensions are not overlaps in eight.
