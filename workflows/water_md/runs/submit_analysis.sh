#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --job-name=water18k-analysis
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg

D=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k
PY=${HOME}/venv-ipi-petmad/bin/python

cd $D
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/check_runs.py production > logs/check_runs_production.log 2>&1
echo "check_runs exit $?"

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/analyse.py > logs/analyse_production.log 2>&1
echo "analyse exit $?"

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/plot.py > logs/plot_production.log 2>&1
echo "plot exit $?"

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/plot_rdf_20_50ps.py > logs/plot_rdf_20_50ps.log 2>&1
echo "three-panel figure exit $?"

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/plot_rdf_panels_18k.py > logs/plot_rdf_panels_18k.log 2>&1
echo "two-panel figure exit $?"

uenv run pytorch/v2.9.1:v2 --view=default -- $PY diagnostics/proton_transfer/inspect_events.py > diagnostics/proton_transfer/run.log 2>&1
echo "hydrogen-transfer diagnostic exit $?"

echo "WATER ANALYSIS DONE"
