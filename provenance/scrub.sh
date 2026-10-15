#!/bin/bash
# replaces the cluster paths and the private draft ids of the Materials Cloud records by placeholders
# usage: DRAFT_MADCORE=... DRAFT_MADCOSMO=... bash provenance/scrub.sh
root=$(cd "$(dirname "$0")/.." && pwd)
cd $root

grep -rlI -e /capstor/ -e /users/schorna -e schorna -e /home/chorna -e /work/cosmo/chorna -e $DRAFT_MADCORE -e $DRAFT_MADCOSMO \
    --exclude-dir=.git --exclude-dir=docs --exclude-dir=build --exclude-dir=dist --exclude-dir=provenance \
    --exclude-dir=out --exclude-dir=reference_figures --exclude-dir=results_archive --exclude-dir=models_archive . \
    | xargs -d '\n' sed -i \
    -e 's|/capstor/scratch/cscs/schorna/merge-datasets|${DEV_ROOT}|g' \
    -e 's|/capstor/scratch/cscs/schorna|${SCRATCH_ROOT}|g' \
    -e 's|/capstor/store/cscs/userlab/lp133/schorna|${STORE_LP133}|g' \
    -e 's|/capstor/store/cscs/userlab/lp133/cmalosso|${COLLABORATOR_STORE}|g' \
    -e 's|/capstor/store/cscs/swissai/aa002/schorna|${STORE_USER_ROOT}|g' \
    -e 's|/capstor/store/cscs/swissai/aa002|${STORE_ROOT}|g' \
    -e 's|/users/schorna|${HOME}|g' \
    -e 's|/home/schorna|${HOME}|g' \
    -e 's|/transfer/schorna|${TRANSFER_ROOT}|g' \
    -e 's|/iopsstor/scratch/cscs/schorna|${IOPS_SCRATCH_ROOT}|g' \
    -e 's|/scratch/cscs/schorna|${SCRATCH_ROOT}|g' \
    -e 's|/scratch/schorna|${KUMA_SCRATCH_ROOT}|g' \
    -e 's|/work/[a-z]*/schorna|${KUMA_WORK_ROOT}|g' \
    -e 's|/work/cosmo/chorna/merge-datasets|${KUMA_WORK_ROOT}|g' \
    -e 's|/work/cosmo/chorna/carto-speedrun|${OTHER_WORKSPACE}|g' \
    -e 's|/home/chorna/workspace/carto-speedrun|${OTHER_WORKSPACE}|g' \
    -e 's|schorna@|${USER}@|g' \
    -e "s|$DRAFT_MADCORE|\${MC_DRAFT_MADCORE}|g" \
    -e "s|$DRAFT_MADCOSMO|\${MC_DRAFT_MADCOSMO}|g"
