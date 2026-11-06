#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -J melonNersc

# Steps 1-3 of the melon workflow (create butler, register instrument, copy from
# the decam repo) as a Slurm job. Run it directly, from a shell with the LSST stack
# set up; it checks its inputs and then sbatches itself.
#   MAIL_TO=you@example.org ./run-nersc-job.sh u/descdm/DECam/endurance/template/chunk0 chunk0_template
# Slurm options go in front of the script through SBATCH_* variables or by running
# sbatch yourself, e.g. a smaller allocation:
#   SBATCH_QOS=shared SBATCH_TIMELIMIT=48:00:00 MAIL_TO=... ./run-nersc-job.sh ...
#
# The job counts template_coadd datasets in the source repo and in the new sqlite
# butler, and mails MAIL_TO only if they match; a failed copy or a count mismatch
# exits non-zero and sends nothing (Slurm's own --mail-type=END would also mail on
# failure). The job output is $BUTLER_PARENT/logs/melonNersc_<jobid>.out.
#
# Defaults that differ from common.sh: SOURCE_REPO is the CFS decam repo, since the
# `decam` alias isn't defined for descdm, and BUTLER_PARENT is on $PSCRATCH, since
# the butler holds the datasets themselves (chunk 0's coadds are tens of TB).
# The remaining settings (COLLECTIONS, WHERE, ...) are common.sh's.

set -euo pipefail
export MELON_DIR=${MELON_DIR:-/global/u1/d/descdm/seanmacb/decam_template_tools/melon}
export SOURCE_REPO=${SOURCE_REPO:-/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo}
export BUTLER_PARENT=${BUTLER_PARENT:-${PSCRATCH:?PSCRATCH must be set}/melon}
source "$MELON_DIR/common.sh"
melon_parse_args "$@"      # sets CHAIN, NAME, REPO, COLLECTIONS or exits with usage
melon_require_butler

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Submit side: fail here rather than after days in the queue.
    [[ -n "${MAIL_TO:-}" ]] || melon_die "set MAIL_TO to the address for the success mail."
    [[ ! -e "$REPO" ]] || melon_die "$REPO already exists; step 1 would refuse to overwrite it."
    mkdir -p "$BUTLER_PARENT/logs"
    exec sbatch -o "$BUTLER_PARENT/logs/melonNersc_%j.out" -e "$BUTLER_PARENT/logs/melonNersc_%j.out" \
        "$MELON_DIR/run-nersc-job.sh" "$@"
fi

# Job side. The stack comes from the submitting shell's environment.
echo "melon ${CHAIN} -> ${REPO}: job ${SLURM_JOB_ID} on $(hostname), source ${SOURCE_REPO}"
start=$SECONDS

"$MELON_DIR/run_nersc_steps.sh" "$@"

# query-datasets prints a few header lines before one row per dataset, and each
# row starts with the dataset type, so count rows rather than lines.
count() { butler query-datasets "$1" "$DATASET_TYPE" --collections "$2" --where "$WHERE" --limit 0 | grep -c "^${DATASET_TYPE}"; }
src=$(count "$SOURCE_REPO" "$COLLECTIONS")
# The sqlite butler has the runs the datasets came from, not the chain (see melon_repo_runs).
dst=$(count "$REPO/butler.yaml" "$(melon_repo_runs "$REPO")")
echo "${DATASET_TYPE} datasets: source ${src}, sqlite butler ${dst}"
[[ "$src" == "$dst" ]] || melon_die "counts differ (source ${src}, sqlite butler ${dst}); not sending mail."

size=$(du -sh "$REPO" | cut -f1)
elapsed=$(( (SECONDS - start) / 60 ))
mail -s "melon NERSC transfer done: ${NAME}" "$MAIL_TO" <<EOF
Steps 1-3 finished for ${CHAIN}.
${dst} ${DATASET_TYPE} datasets copied (matches the source), ${size} in ${REPO}.
Took ${elapsed} min; job ${SLURM_JOB_ID}.
Next: globus ${REPO} to USDF, then 05_transfer_to_main.sh there.
EOF
