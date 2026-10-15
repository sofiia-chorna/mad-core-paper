#!/bin/bash
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps

while read -r nodes chains cells; do
    ckpt=4000000
    [ "$chains" -gt 1 ] && ckpt=2000000

    dep=""
    for ((i = 0; i < chains; i++)); do
        job=$(sbatch --parsable -N "$nodes" $dep \
              --export=ALL,CELLS="$cells",CKPT_EVERY="$ckpt" submit_mega_cells.sh)
        echo "submitted $job nodes=$nodes chain=$((i + 1))/$chains cells: $cells"
        dep="--dependency=afterany:$job"
    done
done < mega_job_plan.txt
