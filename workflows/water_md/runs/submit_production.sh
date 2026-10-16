#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --job-name=water18k-prod

# 30 minute segments, four runs per node, resubmitted until every run has 200,000 steps
here=$SLURM_SUBMIT_DIR

unfinished=""
for run in $(ls $here/production); do
    steps=$(grep -o "<step>[^<]*</step>" $here/production/$run/RESTART 2> /dev/null | tr -dc 0-9)
    if [ "${steps:-0}" -lt 200000 ]; then
        unfinished="$unfinished $run"
    fi
done

if [ -z "$unfinished" ]; then
    exit 0
fi

sbatch --dependency=afterany:$SLURM_JOB_ID $SLURM_SUBMIT_DIR/submit_production.sh

set -- $unfinished
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    for gpu in 0 1 2 3; do
        if [ -n "${1:-}" ]; then
            srun --nodes=1 --ntasks=1 --cpus-per-task=72 --nodelist=$node uenv run pytorch/v2.9.1:v2 --view=default -- bash $here/run.sh $1 $gpu &
            shift
        fi
    done
done

wait
