#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --cpus-per-task=1
#SBATCH --mem=800G
#SBATCH --job-name=water18k-dft-snapshots
#SBATCH --uenv prgenv-gnu/24.11:v1
#SBATCH --view default

# the nine snapshots at the three functionals, four calculations of 72 ranks at a time
export OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR

n=0
for functional in r2scan pw-lda pbe; do
    for frame in aims/*/; do
        (
            cd $frame/$functional
            srun --mpi=cray_shasta --exact -n 72 -c 1 --mem=195G aims.260622.hdf5.scalapack.mpi.x < control.in > aims.out 2> aims.err
        ) &
        n=$((n + 1))
        if [ $((n % 4)) = 0 ]; then
            wait
        fi
    done
done

wait
