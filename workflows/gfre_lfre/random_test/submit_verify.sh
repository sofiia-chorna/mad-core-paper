#!/bin/bash
#SBATCH --job-name=verify-mad15
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/random_test/logs/%x_%j.out

set -uo pipefail
ROOT=${DEV_ROOT}
HERE=$ROOT/paper/figures/mad1.5_gfre_lfre_compare/random_test
VENVS=$ROOT/venvs_feats

export OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1
export DP_BACKEND=pytorch

N=${N:-40}
START=${START:-150000}

echo "STARTING $* n $N start $START $(date) on $(hostname)"

for MODEL in "$@"; do
    case $MODEL in
        pet-mad)
            PY=${HOME}/venv-metatrain/bin/python
            ;;
        mace-omat)
            PY=$VENVS/mace/bin/python
            ;;
        uma-omat)
            PY=$VENVS/uma2/bin/python
            ;;
        dpa-omat|dpa-omol)
            PY=$VENVS/dpa/bin/python
            ;;
        *)
            echo "unknown model $MODEL"
            continue
            ;;
    esac

    echo "=== $MODEL $(date)"
    cd $HERE
    $PY -u verify_mad15.py $MODEL --n $N --start $START
    echo "=== $MODEL exit $? $(date)"
done

echo "ENDING $(date)"
