#!/bin/bash
# daint copy of the driver (account lp133): evaluations on daint, no production submission; it stays passive until the
# marker markers/clariden_driver_stopped exists (created by hand once the clariden driver has been killed)
# detached driver for the isolated atom reference round: mixed batch smoke check (fixed package, random init, native
# selection recomputed, references with zero neighbours and finite predictions), production chains, monitoring with
# isolated atom tracking and per run initialization checks, evaluation, report, dated 08:00 status copy
# marker based, restartable, no duplicate launches

S=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3
P=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k
R=$S/runs
M=$S/markers
LOCK=$S/driver_daint.lock
LOG=$S/logs/driver.log
MAX_SECONDS=172800
POLL=300
UENV=pytorch/v2.9.1:v2
FIXPY=${HOME}/venv-metatrain-fix/bin/python
export UENV_WARN_MIGRATE=1

mkdir -p "$M" "$S/logs" "$S/reports"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date +%F_%T) another driver holds $LOCK, exiting" >> "$LOG"
  exit 1
fi
echo "$(date +%F_%T) driver started pid $$ on $(hostname)" >> "$LOG"
echo $$ > "$S/driver_daint.pid"

say() {
  echo "$(date +%F_%T) [daint] $*" >> "$LOG"
}

job_running() {
  squeue -j "$1" -h -o %T 2>/dev/null | grep -q .
}

job_final() {
  sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | tr -d ' '
}

submit() {
  local out
  out=$(cd "$(dirname "$1")" && sbatch "$1" 2>&1)
  if echo "$out" | grep -q Submitted; then
    echo "$out" | awk '{print $4}'
  else
    say "sbatch refused $1: $out"
  fi
}

finished() {
  grep -q "Training finished!" "$R/$1/train.log" 2>/dev/null
}

RUNS="scratchv3-s--madcore_2k--s42 scratchv3-s--madcore_5k--s42 scratchv3-s--madcore_10k--s42 scratchv3-s--madcore_20k--s42 scratchv3-s--random_2k--s42 scratchv3-s--random_5k--s42 scratchv3-s--random_10k--s42 scratchv3-s--random_20k--s42"

smoke_ok() {
  [ -f "$M/smoke.ok" ] && return 0
  if [ ! -f "$M/smoke.jobid" ]; then
    jid=$(submit "$R/smoke/submit_smoke.sh")
    [ -n "$jid" ] && echo "$jid" > "$M/smoke.jobid" && say "smoke job submitted: $jid"
    return 1
  fi
  jid=$(cat "$M/smoke.jobid")
  job_running "$jid" && return 1
  [ -f "$M/smoke.checked" ] && return 1
  touch "$M/smoke.checked"
  say "smoke job $jid ended with state $(job_final "$jid")"
  log="$R/smoke/scratchv3-s--smoke_mixed--s42/train.log"
  if grep -q "Training finished!" "$log" 2>/dev/null && grep -q "Package directory: ${HOME}/venv-metatrain-fix/" "$log" && ! grep -q "Package directory: ${HOME}/venv-metatrain/" "$log" \
     && grep -q "RANDOM INIT CONFIRMED" "$R/smoke/ssv3-smoke-$jid.out" 2>/dev/null && grep -q "SELECTION CHECK PASSED" "$R/smoke/ssv3-smoke-$jid.out" 2>/dev/null && grep -q "REFERENCES CHECK PASSED" "$R/smoke/ssv3-smoke-$jid.out" 2>/dev/null; then
    touch "$M/smoke.ok"
    say "smoke PASSED (2 mixed batch epochs, fixed package imported, random initialization confirmed, native mae_prod selection recomputed, references have zero neighbours and finite predictions)"
    return 0
  fi
  say "smoke FAILED, production is not launched"
  touch "$M/smoke.failed"
  return 1
}

clariden_driver_alive() {
  [ ! -f "$M/clariden_driver_stopped" ]
}

production() {
  for run in $RUNS; do
    if grep -q "Package directory: ${HOME}/venv-metatrain/" "$R/$run/train.log" 2>/dev/null && [ ! -f "$M/wrong_package_$run" ]; then
      touch "$M/wrong_package_$run"
      say "ERROR: $run imported the original package, see its train.log"
    fi
  done
}

evaluation() {
  all=1
  finished_list=""
  for run in $RUNS; do
    if finished "$run"; then
      finished_list="$finished_list $run"
    else
      all=0
    fi
  done
  # incremental: as soon as a run is finished and no evaluation job is running, evaluate its selected and final
  # checkpoints (the job evaluates every finished run whose predictions are missing)
  if [ "$all" = "0" ]; then
    if [ ! -f "$M/partial_eval.jobid" ] || ! job_running "$(cat "$M/partial_eval.jobid")"; then
      newly=""
      for run in $finished_list; do
        [ -f "$M/evaluated_$run" ] || newly="$newly $run"
      done
      if [ -n "$newly" ]; then
        jid=$(submit "$S/scripts/submit_eval_partial_daint.sh")
        if [ -n "$jid" ]; then
          echo "$jid" > "$M/partial_eval.jobid"
          for run in $newly; do
            touch "$M/evaluated_$run"
          done
          say "partial evaluation submitted: $jid for$newly"
        fi
      fi
    fi
  fi
  # the full evaluation (tests, validation populations, maps) once every run is finished
  if [ "$all" = "1" ] && [ ! -f "$M/eval.jobid" ]; then
    if [ -f "$M/partial_eval.jobid" ] && job_running "$(cat "$M/partial_eval.jobid")"; then
      return
    fi
    jid=$(submit "$S/scripts/submit_eval_daint.sh")
    [ -n "$jid" ] && echo "$jid" > "$M/eval.jobid" && say "evaluation submitted: $jid"
  fi
  if [ -f "$M/eval.jobid" ] && ! job_running "$(cat "$M/eval.jobid")" && [ ! -f "$M/eval.done" ]; then
    if grep -q "EVAL DONE" "$S/logs/ssv3d-eval-$(cat "$M/eval.jobid").out" 2>/dev/null; then
      touch "$M/eval.done"
      say "evaluation done"
    else
      say "evaluation job ended without EVAL DONE, resubmitting once"
      [ -f "$M/eval.retried" ] && touch "$M/eval.done" || { touch "$M/eval.retried"; rm -f "$M/eval.jobid"; }
    fi
  fi
}

report() {
  (cd "$S" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 RAYON_NUM_THREADS=2 uenv run $UENV --view=default -- $FIXPY scripts/iso_track.py > "$S/logs/iso_track.log" 2>&1) || say "isolated atom tracking failed, see logs/iso_track.log"
  for run in $RUNS; do
    if [ ! -f "$R/$run/init_check.json" ] && find "$R/$run/outputs" -name "model_*.ckpt" -print -quit 2>/dev/null | grep -q .; then
      (cd "$S" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uenv run $UENV --view=default -- $FIXPY scripts/check_scratch_init.py "$R/$run" > "$S/logs/init_check_$run.log" 2>&1) && say "initialization check written for $run"
    fi
  done
  (cd "$S" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uenv run $UENV --view=default -- $FIXPY scripts/build_report.py > "$S/logs/report.log" 2>&1) && say "report rebuilt"
  if [ "$(date +%Y%m%d%H%M)" -ge 202610070800 ] && [ ! -f "$M/report_0800.done" ]; then
    cp "$S/reports/STATUS.md" "$S/reports/STATUS_20261007_0800.md" && touch "$M/report_0800.done" && say "08:00 status report written to reports/STATUS_20261007_0800.md"
  fi
}

start=$(date +%s)
last_report=0
while true; do
  now=$(date +%s)
  if [ $((now - start)) -gt $MAX_SECONDS ]; then
    say "time budget exhausted, stopping"
    break
  fi
  if [ -f "$M/smoke.failed" ]; then
    say "stopped before production because the smoke check failed"
    break
  fi
  if smoke_ok; then
    production
    if clariden_driver_alive; then
      [ -f "$M/daint_waiting" ] || { say "daint driver: clariden driver still active, evaluations left to it"; touch "$M/daint_waiting"; }
    else
      [ -f "$M/daint_waiting" ] && { rm -f "$M/daint_waiting"; say "daint driver: clariden driver quiet, taking over evaluations"; }
      evaluation
    fi
    if [ -f "$M/eval.done" ]; then
      report
      say "all steps complete"
      break
    fi
  fi
  if [ $((now - last_report)) -gt 1800 ]; then
    report
    last_report=$now
  fi
  sleep $POLL
done
say "driver finished"
