# mad-core-paper

Code and results behind the paper "Hierarchical universal open structure dataset for
atomistic modeling".

```
figures/            one folder per figure, plot.py draws it from data/, prepare_data.py rebuilds data/
tables/             source files of the SI tables
claims/             scripts behind the numbers in the text
reference_figures/  the submitted pdfs
plotting/, data/    shared plotting code, the scaler of the metric, the sketch-map projector
workflows/          code and run records of the pipeline
scripts/            draw_figures.sh, build_archives.sh, zenodo_upload.py
environment/        package versions
```

To run:

```
pip install -r environment/plot-requirements.txt
bash scripts/draw_figures.sh python
```

The structures are in the MAD-CORE (https://doi.org/10.24435/materialscloud:rs-tt) and
MAD-COSMO (https://doi.org/10.24435/materialscloud:58-wq) records on the Materials Cloud
Archive.

The ecumetric model on Hugging Face: https://huggingface.co/lab-cosmo/ecumetric.

The FPS and accretion code at https://github.com/sofiia-chorna/coresets.
