Feature reconstruction errors between the ecumetric and six universal potentials.
union/gfre_lfre_union.py takes the MAD-NEXT test split, 15,000 structures stratified by
subset, 4,000 to evaluate and 11,000 to fit a ridge regression with two fold cross
validation (scikit-matter), the LFRE neighbourhood equal to the larger feature dimension.
Its results are the inputs of Fig. 3. random_test does the same on the 6,000 random
MAD-COSMO structures, 5,905 usable, 4,330 to fit and 1,575 to evaluate, with one
featurisation script per model and environment, and eval_r2scan.py gives the ecumetric
errors on the 973 r2SCAN structures (16.4 meV/atom, 109 meV/A). teacher_features holds the
scripts that extracted the last layer features of the teacher models. The per structure
features are in the results archive.
