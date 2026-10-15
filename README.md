# Code and supporting results for "Hierarchical universal open structure dataset for atomistic modeling"

The paper introduces MAD-COSMO, a collection of about 1.7 billion structures from public
atomistic datasets, and MAD-CORE, its nested coreset of 2^24 structures selected by farthest
point sampling and accretion in the space of the ecumetric, a learned structural descriptor.
The structures, features and DFT labels are in the MAD-CORE and MAD-COSMO records of the
Materials Cloud Archive, the ecumetric model at https://huggingface.co/lab-cosmo/ecumetric.
This repository holds what is needed to regenerate the figures, tables and numbers of the
paper, and the code of the workflows behind them.

| folder | content |
|---|---|
| `figures/` | one folder per figure, `plot.py` draws it from `data/`, `prepare_data.py` rebuilds `data/` from the large sources |
| `tables/` | the source files of the SI tables and `check_tables.py` |
| `claims/` | the scripts behind the numbers stated in the text, with their outputs in `claims/out/` |
| `reference_figures/` | the submitted figure files, never overwritten |
| `plotting/`, `data/` | shared plotting code, the scaler of the metric and the sketch-map projector |
| `workflows/` | the upstream code and run records, by purpose, with a README |
| `docs/` | `reproduction.md` (every figure, table and number with its inputs, command, check and status), `verification.md`, `zenodo.json`, the audit |
| `provenance/` | source commits and the original path and hash of every copied file |
| `scripts/` | `check_figures.sh` (every figure against the submitted PDF), `build_archives.sh` |
| `environment/` | tested plotting environment, recorded ML environment |

```
pip install -r environment/plot-requirements.txt
bash scripts/check_figures.sh python
python tables/check_tables.py
```

The release files are the source archive of this repository, a supporting-results archive
(level-2 inputs, unpack as `results_archive/`) and a models archive (the trained PET
checkpoints, unpack as `models_archive/`), with `SHA256SUMS`. Code is BSD-3-Clause
(`LICENSE`), figures and results CC BY 4.0 (`LICENSE-DATA`). The paper's authors acknowledge
Claude (Anthropic) and ChatGPT (OpenAI) for assistance with code development and manuscript
preparation; this release was prepared with Claude under the authors' direction.
