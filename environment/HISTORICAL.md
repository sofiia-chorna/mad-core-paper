# Versions evidenced in the sources

These are the versions recorded in logs, PDF producers, checkpoint metadata and the
experiment notes. They describe what produced the submitted results; they are not pins
that were re-installed for this release (the tested pins are in the two requirements files).

| component | version | evidence |
|---|---|---|
| Python | 3.12 (uenv `pytorch/v2.9.1:v2`) | venv layouts |
| torch | 2.9.1 | uenv image, every Slurm script |
| matplotlib | 3.10.9 (figures of the papers repository, July to October 2026); 3.11.1 (training figures, 2026-10-10 to 14); 3.11.2 (water RDF, 2026-10-15) | `Producer` field of the submitted PDFs |
| numpy, pyarrow, pandas, scipy | 2.4.4, 24.0.0, 3.0.3, 1.17.1 | figure environment of the papers repository |
| metatrain | `0.2.0+phace.parquet.prod` (editable, PhACE-era fork; trained the ecumetric model and featurised the collection); `2026.4` (fine-tune, delta, direct and scratch trainings, evaluations, exports) | experiment READMEs, `workflows/ecumetric_export` |
| metatomic-torch | 0.1.11 (featurisation), 0.1.17 (exports, molecular dynamics) | model files, experiment READMEs |
| metatensor-torch | 0.8.5 (featurisation), 0.10.3 (MD) | venv layouts |
| vesin | 0.5.8 (featurisation), 0.6.1 (MD) | venv layouts, experiment READMEs |
| ase | 3.28.0 (featurisation), 3.29.0 (training, MD) | venv layouts |
| i-PI | 3.3.0 | `workflows/water_md/README_original.md` |
| chemiscope | 1.0.3 | venv layout |
| scikit-matter | 0.1.dev448+gef56135ff (editable fork with the torch sketch-map, uncommitted at the time of the release) | venv layout, `workflows/sketchmap` |
| coresets | git `develop`, commits up to 67a932d (2026-07-10) for the July runs | `workflows/accretion`, repository history |
| RDKit | version not recorded (GDB-13 conversion, June 2026) | `workflows/collection/ingest/runs/gdb13/NOTES.txt` |
| FHI-aims | 260622 | SI, `workflows/dft_labels/DFT_CAMPAIGN.md` |
| orb-models 0.7.0, fairchem-core 2.19.0, deepmd-kit 3.2.0, MACE (LLPR fork) | random-test featurisation | `workflows/gfre_lfre/random_test/README.md` |
| Ghostscript, ImageMagick 7 | pixel checks | `scripts/check_figures.sh` |
