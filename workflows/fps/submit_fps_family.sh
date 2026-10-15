#!/bin/bash
#SBATCH --job-name=fps-fam
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/fps/logs/%x_%j.out

set -euo pipefail

family="$1"
k="${2:-1000000}"
proj=${DEV_ROOT}/phace-atomic
scaler="$proj/fps/scaler_mad15_phaceatomic_meanstd.pkl"

if [[ "$family" == "gdb13" ]]; then
    store=${STORE_LP133}/mad-core/datasets/phace_atomic_ep92
else
    store=${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92
fi
core=${STORE_LP133}/mad-core/coresets/phace_atomic_ep92/$family
mkdir -p "$core/checkpoints" "$core/logs" "$proj/fps/logs"

# freeze the input shard list (defines the run; copied next to the result by run_fps too)
find "$store/$family" -name '*.parquet' | sort > "$core/manifest.txt"
n_shards=$(wc -l < "$core/manifest.txt")
git_commit=$(git -C "$proj" rev-parse HEAD 2>/dev/null || echo unknown)
scaler_md5=$(md5sum "$scaler" | cut -d' ' -f1)

# the exact run_fps invocation, shared by this run AND the frozen repro.sh
run_fps_cmd=(python "$proj/fps/run_fps.py"
    --manifest "$core/manifest.txt"
    --scaler "$scaler"
    --K "$k"
    --max-force 200 --max-energy-per-atom 50 --max-atomic-energy 100
    --emit-features
    --output "$core/fps.parquet"
    --checkpoint-dir "$core/checkpoints/ckpt" --checkpoint-every 100000
    --notes "${family}_K${k}_filtered_emitfeat")

# --- write the frozen, self-contained reproducer ---
cat > "$core/repro.sh" <<EOF
#!/bin/bash
# === Reproducible FPS: ${family}  K=${k} ===
# generated : $(date -Is)
# git commit: ${git_commit}   (repo: ${proj})
# store     : ${store}/${family}   (${n_shards} shards, frozen in manifest.txt)
# scaler    : ${scaler}  (md5 ${scaler_md5})
# filters   : max_force=200  energy_per_atom=50  max_atomic_energy=100
# Re-run identically:   sbatch -N<nodes> ${core}/repro.sh
#SBATCH --job-name=fps-${family}-repro
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${core}/logs/repro_%j.out
set -euo pipefail
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # long GIL-heavy resume rebuild starves the NCCL heartbeat watchdog (gdb13 933M-row pool); collective timeout below still bounds real hangs
source ${HOME}/venv-phace/bin/activate
srun -ul --ntasks=\$((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) --gpus-per-task=1 \\
    ${run_fps_cmd[*]}
EOF
chmod +x "$core/repro.sh"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # long GIL-heavy resume rebuild starves the NCCL heartbeat watchdog (gdb13 933M-row pool); collective timeout still bounds real hangs
source ${HOME}/venv-phace/bin/activate


resume_args=()
if [[ -f "$core/checkpoints/ckpt_latest.txt" ]]; then
    resume_args=(--resume-from "$core/checkpoints/ckpt")
    echo "RESUMING from $core/checkpoints/ckpt ($(cat "$core/checkpoints/ckpt_latest.txt" 2>/dev/null))"
fi

echo "start $(date -Is)  family=$family  shards=$n_shards  K=$k  nodes=$SLURM_NNODES  git=$git_commit"
echo "froze reproducer -> $core/repro.sh"
srun -ul --ntasks="$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))" --gpus-per-task=1 "${run_fps_cmd[@]}" "${resume_args[@]}"
echo "finish $(date -Is)"
