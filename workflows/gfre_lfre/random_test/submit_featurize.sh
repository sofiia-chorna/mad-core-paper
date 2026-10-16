#!/bin/bash
#SBATCH --job-name=feat-random-test
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/random_test/logs/%x_%j.out

set -uo pipefail
ROOT=${DEV_ROOT}
HERE=$ROOT/paper/figures/mad1.5_gfre_lfre_compare/random_test
VENVS=$ROOT/venvs_feats

export OMP_NUM_THREADS=16
export CACHED_PATH_CACHE_ROOT=$ROOT/orb_ckpt_cache
export HF_HUB_OFFLINE=1
export DP_BACKEND=pytorch

EXTRA=""
if [ -n "${LIMIT:-}" ]; then
    EXTRA="--limit $LIMIT"
fi

for MODEL in "$@"; do
    case $MODEL in
        phace-atomic)
            PY=$ROOT/phace-atomic/huggingface/ecumetric_work/venv-ecumetric/bin/python
            SCRIPT=featurize_metatomic.py
            ;;
        pet-mad)
            PY=${HOME}/venv-metatrain/bin/python
            SCRIPT=featurize_metatomic.py
            ;;
        orb-v3-omat)
            PY=$VENVS/orb/bin/python
            SCRIPT=featurize_orb.py
            ;;
        mace-omat)
            PY=$VENVS/mace/bin/python
            SCRIPT=featurize_mace.py
            ;;
        uma-omat)
            PY=$VENVS/uma2/bin/python
            SCRIPT=featurize_uma.py
            ;;
        dpa-omat|dpa-omol)
            PY=$VENVS/dpa/bin/python
            SCRIPT=featurize_dpa.py
            ;;
        *)
            echo "unknown model $MODEL"
            continue
            ;;
    esac

    cd $HERE
    $PY -u $SCRIPT $MODEL $EXTRA
done

