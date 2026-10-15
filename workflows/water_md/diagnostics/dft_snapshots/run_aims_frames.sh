#!/bin/bash

# shared by submit_probe.sh and submit.sh, runs the listed frame/functional pairs in slots, skips finished ones
AIMS_BIN=/capstor/store/cscs/userlab/lp133/FHIaims/clariden/aims.260622.hdf5.scalapack.mpi.x

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

run_one () {
  local work=$1

  if [ -f "$work/aims.out" ] && grep -q "Have a nice day." "$work/aims.out"; then
    echo "$work: already done"
    return
  fi

  echo "$work: starting at $(date +%H:%M:%S)"
  (
    cd "$work" && srun --mpi=cray_shasta --exact --gres=none \
      --mem=$STEP_MEM -n $RANKS -c 1 \
      "$AIMS_BIN" < control.in > aims.out 2> aims.err
  ) || true

  rm -f "$work"/D_*.csc

  if grep -q "Have a nice day." "$work/aims.out" 2>/dev/null; then
    echo "$work: OK at $(date +%H:%M:%S), $(grep -c 'Begin self-consistency iteration' "$work/aims.out") scf cycles"
  else
    echo "$work: FAILED at $(date +%H:%M:%S)"
    tail -5 "$work/aims.err" "$work/aims.out" 2>/dev/null | cut -c1-200
  fi
}

run_list () {
  local slots=$1
  shift
  local works=("$@")

  echo "pairs: ${#works[@]}, slots: $slots, ranks per step: $RANKS, memory per step: $STEP_MEM"

  for slot in $(seq 0 $((slots - 1))); do
    (
      for index in "${!works[@]}"; do
        if [ $((index % slots)) -eq $slot ]; then
          run_one "${works[$index]}"
        fi
      done
    ) &
  done
  wait
}
