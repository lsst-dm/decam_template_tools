#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -J aggregateGraph
#SBATCH -o /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/aggregateGraph_%j.out
#SBATCH -e /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/aggregateGraph_%j.out

# Registers the outputs of a bps run whose driver died before final_job.bash ran,
# so a later submit can resume it with --skip-existing-in <run>. Quantum-backed
# workers write their outputs to the datastore but only `butler aggregate-graph`
# registers them and creates the output RUN and CHAINED collections.
#
# Usage, from a shell with the LSST stack set up:
#   sbatch run-aggregate-graph.sh <chunk> <run timestamp>
#   DRY_RUN=1 sbatch run-aggregate-graph.sh 0 20261101T200346Z   # scan only, no ingest
# or interactively on a node from `salloc -q interactive -C cpu -A m1727 -t 4:00:00`:
#   ./run-aggregate-graph.sh 0 20261101T200346Z
#
# Runs with --incomplete: the run did not finish, and the default (--complete)
# treats the unfinished quanta as permanent failures and deletes datasets it
# decides are no longer needed. PROCS sets the worker count (default 16, as
# final_job.bash uses).

set -euo pipefail
CHUNK=${1:?usage: run-aggregate-graph.sh <chunk> <run timestamp>}
RUN=${2:?usage: run-aggregate-graph.sh <chunk> <run timestamp>}
: "${PSCRATCH:?PSCRATCH must be set}"
REPO=${REPO:-/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo}
PROCS=${PROCS:-16}
python -c 'import lsst.daf.butler' 2> /dev/null \
    || { echo "Set up the LSST stack before running." >&2; exit 1; }
OPERATOR=$(python -c 'import getpass; print(getpass.getuser())')

COLL="u/${OPERATOR}/DECam/endurance/template/chunk${CHUNK}"
QG="${PSCRATCH}/endurance_submit/${COLL}/${RUN}/u_${OPERATOR}_DECam_endurance_template_chunk${CHUNK}_${RUN}.qg"
[ -f "$QG" ] || { echo "No quantum graph at ${QG}" >&2; exit 1; }
echo "aggregate-graph: job ${SLURM_JOB_ID:-none} on $(hostname), ${COLL}/${RUN}, DRY_RUN=${DRY_RUN:-0}"

# The run's workers are gone, but keep this out of any inherited SLURM_* settings.
unset ${!SLURM_@}

ARGS=(--incomplete --register-dataset-types --update-output-chain -j "$PROCS")
[ -z "${DRY_RUN:-}" ] || ARGS+=(--dry-run)
exec butler --long-log --log-level=VERBOSE aggregate-graph "$QG" "$REPO" "${ARGS[@]}"
