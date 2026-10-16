# mad-core-paper

Code and results behind the paper "Hierarchical universal open structure dataset for
atomistic modeling" (MAD-COSMO, a collection of 1.7 billion structures from public datasets,
and MAD-CORE, its nested coreset of 2^24 structures). The structures are in the MAD-CORE and
MAD-COSMO records on the Materials Cloud Archive, the ecumetric model on Hugging Face
(lab-cosmo/ecumetric), the FPS and accretion code at github.com/sofiia-chorna/coresets.

    figures/            one folder per figure, plot.py draws it from data/, prepare_data.py rebuilds data/
    tables/             source files of the SI tables
    claims/             scripts behind the numbers in the text, outputs in claims/out
    reference_figures/  the submitted pdfs
    plotting/, data/    shared plotting code, the scaler of the metric, the sketch-map projector
    workflows/          code and run records of the pipeline, not rerun
    provenance/         where every file came from (extracted_files.csv, archive_files.csv)
    scripts/            draw_figures.sh, build_archives.sh, zenodo_upload.py
    environment/        package versions

    pip install -r environment/plot-requirements.txt
    bash scripts/draw_figures.sh python

Figures 1 and 3 to 10 and the three SI figures come out equal to the submitted files
(matplotlib 3.10.9 for the first eight, 3.11.1 for the rest). Figure 2 is drawn by hand.
The large level-2 inputs and the trained models are in the results and models archives of the
release, to be unpacked as results_archive/ and models_archive/ next to this folder.

Sources: papers repository at 28cfe43, development repository at 97bec6f. Cluster paths in
workflows/ are replaced by ${DEV_ROOT}, ${STORE_ROOT} and ${HOME}. Code is BSD-3-Clause
(LICENSE), figures and results CC BY 4.0 (LICENSE-DATA).
