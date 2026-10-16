#!/bin/bash
#SBATCH --job-name=claim_madcosmo_populations
#SBATCH --account=ab010
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --output=${DEV_ROOT}/paper/claims/logs/%x_%j.out

set -euo pipefail

cd ${DEV_ROOT}/paper/claims
source ${HOME}/venv-phace/bin/activate

python madcosmo_populations.py 32
