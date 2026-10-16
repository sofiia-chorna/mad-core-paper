#!/bin/bash
# draws every figure of the paper into build/figures

cd "$(dirname "$0")/.."
python=${1:-python}
mkdir -p build/figures
here=$(pwd)

for folder in fig01_nested_kset fig03_gfre_lfre siS1_gfre_lfre_random fig04_madnext_subsets fig05_madnext_elements fig06_madcore_decomposition fig07_madcore_elements fig08_sketchmap siS3_water_rdf; do
    cd $here/figures/$folder
    $python plot.py
done

cd $here/figures/fig09_fig10_training
$python plot_training.py curves
$python plot_training.py force_map
$python plot_training.py separate_parts
