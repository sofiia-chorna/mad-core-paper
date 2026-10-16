Featurisation of every structure of the collection with the ecumetric model.
run_inference.py evaluates a parquet shard and adds the energy, the non conservative forces
and the 512 dimensional structure feature (mean and standard deviation of the 256 per atom
features, --meanstd). Structures with more than 1000 atoms are skipped. build_manifest.py
lists the shards, the Slurm harness strides the list over all GPUs, build_index.py collects
the row counts. 6,744 shards, 756 million structures, 16 nodes, about 2,000 GPU hours.
