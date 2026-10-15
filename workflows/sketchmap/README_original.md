# Sketch-map pipeline

```
src/         production scripts
slurm/       one submit per script (debug partition unless a GPU fit is needed)
docs/        design notes from the exploration phase (sigma rule, 2^20 design, sweeps)
logs/        job output
results/     embeddings/ (p20d8 production + rand20d8 comparison) and metrics/
projector/   the MLP that extends the map to unseen structures
figures/     the figures that go in the paper
```

## Where things are

| what | where |
|---|---|
| production figure | `figures/voronoi_groups_d8_p20d8.pdf` |
| single-panel variant | `figures/voronoi_families_d8_p20d8.pdf` |
| 2^24 projected panels | `figures/coreset24_panels_d8_p20d8.pdf` |
| random-2^20 comparison | `figures/random_2p20_d8_rand20d8.png` |
| the fitted 2^20 map | `results/embeddings/embedding_l20_d8_p20d8.npy` |
| projector weights + scalers | `projector/projector_d8_p20d8.pt` |
| projector validation | `projector/projector_d8_p20d8.json` |
| all 2^24 points projected | `projector/projection_2p24_d8_p20d8.npy` (537 MB) |

## Pipeline

| step | script | submit |
|---|---|---|
| 1. sigma from pair-weighted histogram | `derive_params_2p20.py` | run on a login/debug node |
| 2. fit the map (2 nodes, 8 GPU) | `fit_dist.py` | `submit_2p20.sh` |
| 3. score it | `metrics_2p20.py` | `submit_score_rand.sh` |
| 4. train projector | `train_projector.py` | `submit_projector.sh` |
| 5. project 2^24 | `apply_projector.py` | `submit_project24.sh` |
| 6. production figure | `plot_voronoi_groups.py` | `sbatch submit_plot.sh plot_voronoi_groups.py --tag p20d8 --d 8` |

`submit_plot.sh` is the generic debug-partition runner for any CPU plotting script
(`plot_voronoi_groups.py`, `plot_voronoi_families.py`, `plot_coreset24_panels.py`), taking
the script name and its flags as arguments. The random-baseline fit is
`submit_random2p20_d8.sh` (tag `rand20d8`), plotted by `plot_random_2p20.py` via
`submit_plot_rand.sh`. Chemiscope export: `make_chemiscope.py` + `make_bundle.py` via
`submit_chemiscope.sh` (regenerate from the production embedding — the old 150-iteration
bundle was deleted).

## Production parameters (tag p20d8)

```
d = 8, sigma = 1.24, A = 4, B = 2, a_low = 2, b_low = 2,
wgamma = 0.25, center = False, init = PCA(8), 400 iterations
```

Fit on the 2^20 coreset points with per-point weights = (Voronoi cell population)^0.25.
sigma comes from the **pair-weighted** distance histogram, not the bare one, because the
stress sums over `w_i w_j`. Use `--gtol 1e-12`: with near-uniform weights the default
`1e-8` stops after nine iterations as a units artefact. wgamma = 0.25 keeps the effective
sample size (sum w)^2 / sum w^2 at 698,224 points; 0.75 collapses it to 156 and 1.0 to 4.

## The production figure

`plot_voronoi_groups.py`: all 1,048,576 fitted points in SMAP dims 1-2, grey background
in every panel; colour = the point's own source dataset; marker area = the number of
library structures in its Voronoi cell (populations sum to 1,695,275,276). Six panels:
MAD1.5 + MAD-NEXT / Bulk materials / Repositories / Molecular / Catalysis and
interfaces / Other datasets.

Colour is the centre's own provenance — the structures inside a cell can come from
anywhere. gdb13 holds 933M library structures but only 2 coloured points at 2^20; its
largest cell is centred on a MAD-union point.

## The projector

`512 -> 512 -> 256 -> 128 -> 8`, ReLU, standard scalers on features and projections,
following `MLPProjector` in `lab-cosmo/upet`. Trained on the 2^20 (feature, embedding)
pairs, 10% held out, Adam 1e-3, early stopping.

Mean R^2 **0.9970**, held-out pairwise-distance correlation **0.9987**. Re-applied to the
2^20 it reproduces the fit to a median 0.0104 against an embedding scale of 0.6118.

**Not done:** wrapping it as a `metatomic` model that takes `System` objects and returns a
`TensorMap`. That needs the PhACE featurizer inside `forward()`. `body.tex` promises it.

## Caveat worth keeping

Apparent overlaps in SMAP dims 1-2 (for example Alexandria and OMol25) are a projection
artefact: in the full 8-d embedding the two are ~99% separated by nearest-neighbour
membership, in dims 1-2 only ~27% of that separation survives. Do not read 2-d overlap
as chemical similarity.
