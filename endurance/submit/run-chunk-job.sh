#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -J chunkDriver
#SBATCH -o /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/chunkDriver_%j.out
#SBATCH -e /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/chunkDriver_%j.out

# Usage, from a shell with the LSST stack and bps_parsl_sites set up:
#   START=singleFrame STAGES=singleFrame sbatch run-chunk-job.sh 0
#   START=template sbatch run-chunk-job.sh 0
# or both in one go, each stage with its own 2-day limit; the second waits for
# the first to exit 0, which run_chunk.sh only does if the single-frame gate passes:
#   J=$(START=singleFrame STAGES=singleFrame sbatch --parsable run-chunk-job.sh 0)
#   START=template sbatch --dependency=afterok:$J run-chunk-job.sh 0
#
# Runs run_chunk.sh as a Slurm job instead of on a login node. Login nodes cap
# each user at 30 GiB, and building chunk 0's single-frame quantum graph (463k
# quanta) went past that and was OOM-killed on 2026-10-31; the template graph
# is ~4x bigger. Hence a whole node (all 512 GB): on 2026-10-31 the scheduler
# also put a shared-QOS job about a month out against ~3 days for regular,
# the same wait as the stage's own worker blocks.
#
# This job is the workflow driver: its bps process hosts the TaskVine manager,
# so it has to outlive the workflow, while the worker blocks bps asks for are
# separate Slurm jobs. sbatch passes on the submitting shell's environment, so
# the stack is whatever that shell set up, and STAGES and START pass through.

set -euo pipefail
CHUNK=${1:?usage: sbatch run-chunk-job.sh <chunk>}
# Import check rather than `command -v bps`: /usr/bin/bps is an unrelated tool.
python -c 'import lsst.ctrl.bps, bps_parsl_sites' 2> /dev/null \
    || { echo "Set up the LSST stack and bps_parsl_sites before sbatch." >&2; exit 1; }
echo "chunk ${CHUNK} driver: job ${SLURM_JOB_ID:-none} on $(hostname), STAGES=${STAGES:-all} START=${START:-overscan}"

# The worker blocks are sbatch'ed from here, inherit this environment, and start
# their workers with srun, which would take this job's SLURM_* settings
# (cpus-per-task, mem, ...) as its own; drop them so the blocks start clean.
unset ${!SLURM_@}

cd /global/u1/d/descdm/seanmacb/decam_template_tools/endurance/submit
exec ./run_chunk.sh "$CHUNK"
