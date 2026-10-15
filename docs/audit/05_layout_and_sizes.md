# Proposed repository and Zenodo archive layout

Modest by design: one README, one reproduction guide, scripts grouped by scientific purpose,
small numerical data in the repo, large results in one optional archive, models and datasets
referenced externally. No package, no plugin or configuration layer; a single shared
`plotting/` module and a `--release-root` convention for the scripts that read the data records.

```
mad-core-paper/                         git repo, frozen as mad-core-paper-<version>-src.tar.gz
  README.md                             what the release is, what can be reproduced at which level, links
  REPRODUCTION.md                       the one authoritative map: every figure, table and number ->
                                        inputs, command, environment, output, numerical check, level, status
  LICENSE                               question Q2
  CITATION.cff                          paper, records, models
  environment/
    plot-requirements.txt               tested: matplotlib 3.11.1, numpy, pyarrow, pandas, ase
    ml-requirements.txt                 tested: torch 2.9.1, metatrain 2026.4, metatomic-torch 0.1.17, i-PI, chemiscope
    HISTORICAL.md                       versions evidenced in the sources (uenv, venvs, producers), not pins
  plotting/                             epfl.py palette, periodic-table layout, shared heatmap helpers (orig)
  figures/                              one folder per figure or figure group, plot.py reads only data/
    fig01_nested_kset/                  synthetic (L1)
    fig02_accrete/                      README only, conceptual, PDF in reference_figures
    fig03_gfre_lfre/   siS1_gfre_lfre_random/
    fig04_madnext_subsets/  fig05_madnext_elements/
    fig06_madcore_decomposition/  fig07_madcore_elements/
    fig08_sketchmap/                    data/smap_points.npz (17 MB)
    fig09_fig10_training/               curves (L1 verified), force map (L1 blocked by G1)
    siS2_training_parts/                shares data with fig09
    siS3_water_rdf/                     curves TSV (L1 verified)
  tables/                               check_tables.py and the TSV/JSON sources of the four SI tables and Table I counts
  claims/                               scripts behind the numbers in the text, with their committed outputs in claims/out/
  reference_figures/                    submitted PDFs (papers repo) and the dev builds, with md5 lists, never overwritten
  workflows/                            upstream code and small records, by purpose (level 2 and 3, documented, not rerun):
    ecumetric_training/  featurize_collection/  madnext_fps/  family_fps/  accretion/
    coverage_certificate/  voronoi/  sketchmap/  ecumetric_export/  gfre_lfre/
    dft_labels/  training_from_scratch/  finetune/  delta_learning/  water_md/  data_records/
  data/                                 small shared inputs: ecumetric_scaler.npz, levels.json, (projector 1.7 MB: Q3)
  scripts/                              check_figures.sh, compare_pdf.sh, build_archive.sh, verify_checksums.sh
  examples/                             toy nested-coreset demonstration (optional, G26)
  MANIFEST/                             SHA256SUMS of the repo files, external_assets.tsv (name, version, url, sha256, destination)
```

Zenodo deposit (manual, after the Materials Cloud records are published):

| file | content | estimated size |
|---|---|---|
| `mad-core-paper-<version>-src.tar.gz` | the git repository at the tagged commit (figures, claims, workflows, reference figures, data/) | 95 to 125 MB (figure data 19 MB, claim outputs 2.5 MB, reference PDFs 2.3 MB, workflow code and id manifests 45 to 75 MB depending on how many id lists are kept, store records of the SPACE run 4.7 MB) |
| `mad-core-paper-<version>-results.tar.gz` (optional, question Q3) | `voronoi_population_record_level24.npy` (134 MB), `descriptors_seed0.npz` (1.1 GB), random-test features (1.6 GB), predictions of the evaluated models | 0.15 to 3 GB depending on the choice |
| `mad-core-paper-<version>-models.tar.gz` (optional) | the 13 trained PET checkpoints (`model_best.pt`), with options.yaml and sha256 | 1.3 GB |
| `SHA256SUMS`, `README.md` (top level, extraction layout) | | < 50 KB |

Without the optional archives the deposit stays near 100 MB and supports L1 of every figure
except the force map (G1), all SI tables, all stated numbers that have claim outputs, and
documents L2/L3 fully. With the results archive, L2 of Figs. 3, 8 and SI 1 becomes possible
offline; with the models archive, L2 of Figs. 9, 10, the three SI tables and L3 of the water MD.
