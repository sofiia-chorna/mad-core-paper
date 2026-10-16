#!/bin/bash

# one production run on one GPU, started from input.xml or resumed from the RESTART file
run=$1
gpu=$2

venv=${HOME}/venv-ipi-petmad/bin
scripts=$(cd $(dirname $0)/../scripts && pwd)
baseline=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_minus_lda_delta/models/pet-mad-m-v1.6.0.fixed.pt
corrections=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/delta_lda_pbe_shared/runs/main

cd $(dirname $0)/production/$run

input=input.xml
if [ -s RESTART ]; then
    cp RESTART resume.xml
    input=resume.xml
fi

set -- $(grep -o "<address>[^<]*</address>" input.xml | sed 's/<[^>]*>//g')

$venv/i-pi $input >> ipi.log 2>&1 &
ipi=$!
sleep 15

CUDA_VISIBLE_DEVICES=$gpu $venv/i-pi-py_driver -u -a $1 -m custom -P $scripts/metatomic_output_driver.py \
    -o template=init.extxyz,model=$baseline,device=cuda,output=energy >> driver_baseline.log 2>&1 &

if [ -n "${2:-}" ]; then
    case $run in
        lda*) delta=$corrections/delta_lda--omat-s--madcore_shared18000--s42/model_best.pt ;;
        pbe*) delta=$corrections/delta_pbe--omat-s--madcore_shared18000--s42/model_best.pt ;;
    esac
    CUDA_VISIBLE_DEVICES=$gpu $venv/i-pi-py_driver -u -a $2 -m custom -P $scripts/metatomic_output_driver.py \
        -o template=init.extxyz,model=$delta,device=cuda,output=mtt::delta_energy >> driver_correction.log 2>&1 &
fi

wait $ipi
kill $(jobs -p) 2> /dev/null
