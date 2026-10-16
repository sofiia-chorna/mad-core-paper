#!/bin/bash
# draws each figure again and counts the pixels that differ from the submitted pdf
# usage  bash scripts/check_figures.sh [python]

cd "$(dirname "$0")/.."
python=${1:-python}
mkdir -p build/check/png

# folder, script, argument, output file
cases="
fig01_nested_kset           plot.py           -               fig1.pdf
fig03_gfre_lfre             plot.py           -               gfre_lfre_heatmap_union.pdf
siS1_gfre_lfre_random       plot.py           -               gfre_lfre_heatmap_random_test.pdf
fig04_madnext_subsets       plot.py           -               mad-kset-1col.pdf
fig05_madnext_elements      plot.py           -               mad-kset-counts.pdf
fig06_madcore_decomposition plot.py           -               coreset224-kset.pdf
fig07_madcore_elements      plot.py           -               coreset224-elements.pdf
fig08_sketchmap             plot.py           -               voronoi_groups24_d8_p20d8.pdf
fig09_fig10_training        plot_training.py  curves          train_curves_balanced_4000.pdf
fig09_fig10_training        plot_training.py  force_map       train_force_map_balanced_4000.pdf
fig09_fig10_training        plot_training.py  separate_parts  si_train_curves_balanced_parts.pdf
siS3_water_rdf              plot.py           -               si_rdf_water_20_50ps.pdf
"

here=$(pwd)

echo "$cases" | while read -r folder script argument pdf; do

    if [ -z "$folder" ]; then
        continue
    fi

    if [ "$argument" = "-" ]; then
        argument=""
    fi

    name=${pdf%.pdf}
    new=$here/build/check/$pdf
    reference=$here/reference_figures/submitted/$pdf

    cd $here/figures/$folder
    $python $script $argument $new > $here/build/check/$name.log 2>&1
    cd $here

    gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r100 -o build/check/png/$name.reference.png $reference
    gs -q -dBATCH -dNOPAUSE -sDEVICE=png16m -r100 -o build/check/png/$name.new.png $new

    pixels=$(magick compare -metric AE build/check/png/$name.reference.png build/check/png/$name.new.png build/check/png/$name.diff.png 2>&1)
    echo "$pdf  $pixels differing pixels"

done
