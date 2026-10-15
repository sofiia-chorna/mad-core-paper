#!/bin/bash
# draws every figure into build/check and counts the pixels that differ from the submitted pdf
# usage: bash scripts/check_figures.sh [python]
root=$(cd "$(dirname "$0")/.." && pwd)
python=${1:-python}
out=$root/build/check
mkdir -p $out/png

cases="
fig01_nested_kset plot.py - fig1.pdf
fig03_gfre_lfre plot.py - gfre_lfre_heatmap_union.pdf
siS1_gfre_lfre_random plot.py - gfre_lfre_heatmap_random_test.pdf
fig04_madnext_subsets plot.py - mad-kset-1col.pdf
fig05_madnext_elements plot.py - mad-kset-counts.pdf
fig06_madcore_decomposition plot.py - coreset224-kset.pdf
fig07_madcore_elements plot.py - coreset224-elements.pdf
fig08_sketchmap plot.py - voronoi_groups24_d8_p20d8.pdf
fig09_fig10_training plot_training.py curves train_curves_balanced_4000.pdf
fig09_fig10_training plot_training.py force_map train_force_map_balanced_4000.pdf
fig09_fig10_training plot_training.py separate_parts si_train_curves_balanced_parts.pdf
siS3_water_rdf plot.py - si_rdf_water_20_50ps.pdf
"

echo "$cases" | while read -r folder script arg pdf; do
    if [ -z "$folder" ]; then
        continue
    fi
    if [ "$arg" = "-" ]; then
        arg=""
    fi
    name=${pdf%.pdf}
    (cd $root/figures/$folder && $python $script $arg $out/$pdf > $out/$name.log 2>&1) || echo "$pdf failed, see $out/$name.log"
    gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r100 -o $out/png/$name.reference.png $root/reference_figures/submitted/$pdf
    gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r100 -o $out/png/$name.new.png $out/$pdf
    pixels=$(magick compare -metric AE $out/png/$name.reference.png $out/png/$name.new.png $out/png/$name.diff.png 2>&1)
    echo "$pdf $pixels differing pixels"
done
