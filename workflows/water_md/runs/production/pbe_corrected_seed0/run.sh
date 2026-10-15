#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"

VENV=${HOME}/venv-ipi-petmad/bin
GPU=${GPU:-0}

# the step stored in an i-PI restart file, empty for an empty file
step_of() {
  if [ -s "$1" ]; then
    grep -o "<step>[^<]*</step>" "$1" | head -n 1 | tr -dc '0-9'
  fi
}

# a checkpoint cut by a kill cannot be parsed and is set aside, the other one is used instead
for f in RESTART sim.chk; do
  if [ -s "$f" ] && ! $VENV/python -c "import sys, xml.etree.ElementTree as E; E.parse(sys.argv[1])" "$f" 2>/dev/null; then
    mv "$f" "$f.unreadable.$(date +%s)"
    echo "$(date +%F_%T) $f unreadable, set aside" >> segments.log
  fi
done

# resume from whichever of the final and the periodic checkpoint is furthest, through a copy
restart_step=$(step_of RESTART)
chk_step=$(step_of sim.chk)
INPUT=input.xml
done_steps=0

if [ -n "$restart_step" ] && [ "$restart_step" -ge "${chk_step:-0}" ]; then
  cp RESTART resume.xml
  INPUT=resume.xml
  done_steps=$restart_step
elif [ -n "$chk_step" ]; then
  cp sim.chk resume.xml
  INPUT=resume.xml
  done_steps=$chk_step
fi

if [ "$done_steps" -ge 200000 ]; then
  echo "already finished at step $done_steps"
  exit 0
fi

# a fresh start over existing outputs would mix two trajectories, so it is refused
if [ "$INPUT" = input.xml ] && [ -s sim.out ]; then
  echo "$(date +%F_%T) outputs exist but no readable checkpoint, not restarted" >> segments.log
  exit 3
fi

# sockets left by an interrupted segment of this run would block i-PI
rm -f /tmp/ipi_pk_prdp0_base /tmp/ipi_pk_prdp0_corr

echo "$(date +%F_%T) start from $INPUT at step $done_steps on $(hostname) GPU $GPU job ${SLURM_JOB_ID:-none}" >> segments.log
$VENV/i-pi $INPUT >> ipi.log 2>&1 &
IPI=$!
sleep 15

CUDA_VISIBLE_DEVICES=$GPU $VENV/i-pi-py_driver -u -a pk_prdp0_base -m custom -P ${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/scripts/metatomic_output_driver.py -o template=init.extxyz,model=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_minus_lda_delta/models/pet-mad-m-v1.6.0.fixed.pt,device=cuda,output=energy,geometry_log=geometry_baseline.jsonl,log_calls=20 >> driver_baseline.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU $VENV/i-pi-py_driver -u -a pk_prdp0_corr -m custom -P ${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/scripts/metatomic_output_driver.py -o template=init.extxyz,model=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/delta_lda_pbe_shared/runs/main/delta_pbe--omat-s--madcore_shared18000--s42/model_best.pt,device=cuda,output=mtt::delta_energy,geometry_log=geometry_correction.jsonl,log_calls=20 >> driver_correction.log 2>&1 &

wait $IPI
status=$?
echo "$(date +%F_%T) i-PI exit $status" >> segments.log

# the drivers only serve this i-PI process, they are stopped when it exits
kill $(jobs -p) 2>/dev/null
wait
exit $status
