# workflows

The code and small records of each step, as they ran. Nothing here was rerun for the release.
Cluster paths are replaced by placeholders, ${DEV_ROOT} the development workspace,
${STORE_ROOT} the project store, ${HOME} the home with the virtual environments, the rest
other clusters of the original runs.

    collection/           downloads, license policy, the covalent radius prefilter, ingestion to parquet, GDB-13 from SMILES
    ecumetric_training/   training of the SPACE model, its options, data statistics, training log and splits
    featurize_collection/ featurisation of all structures with the ecumetric model
    fps/                  the scaler of the metric, FPS of MAD-NEXT, per family FPS of the large datasets
    accretion/            anchor (accretion into MAD-NEXT), fps_fill (to 10M and 25M), cut_2pow24
    voronoi/              assignment of every structure to its nearest MAD-CORE structure, cell FPS
    sketchmap/            sketch-map fit and projector
    gfre_lfre/            feature reconstruction errors (Fig. 3, SI Fig. 1), teacher features
    dft_labels/           FHI-aims protocol, selections, references, the r2SCAN test set
    training_from_scratch/ the eight PET models of Figs. 9 and 10
    finetune/, delta_learning/  the SI fine-tune and delta-learning tables
    water_md/             the water simulations of SI Fig. 3

The collection was prefiltered on close contacts (pyykko_filter_parquet.py, scale 0.35, or
0.25 for structures rejected at 0.35) before the ZBL filter of the paper. The sketch-map fit
needs the torch sketch-map of the scikit-matter fork, not published yet.
