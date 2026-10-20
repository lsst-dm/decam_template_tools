#!/bin/bash
# Overscan -> single-frame -> template jobs for one sky chunk, then the template
# PSF QA job.
#   ./run_chunk.sh <chunk>
# The sky extent for each chunk id comes from chunk_grid.py (chunk 0 = RA 300-330,
# dec 0..30; ids step down in dec, then move to the next RA column). Every
# stage takes all observing dates.
#
# Needs the LSST stack set up (see ~/startup_desc_commands.sh). bps submit dirs
# go to $PSCRATCH/endurance_submit/<run>, set as submitPath in each yaml.
#
# Each stage's bps job writes u/<operator>/<payloadName>, a CHAINED collection of its
# run plus its inputs, and the next stage reads from that chain:
#   overscanRaws/chunkN -> singleFrame/chunkN -> template/chunkN
# so no manual collection-chain is needed. The psfQa stage then submits
# ../../verify/run-chunk-psf-qa.sh as a Slurm job on template/chunkN; it writes
# its CSVs to $LOGDIR/templatePsf/chunkN/ (LOGDIR defaults to $REPO/logs).
#
# Stage selection (both optional, they combine):
#   STAGES  which stages to run, space or comma separated; default all four.
#   START   skip every stage before this one; default overscan.
# e.g. STAGES="overscan singleFrame" stops before the template job,
# START=singleFrame resumes from there, and START=psfQa reruns only the QA.
# Stages always run in pipeline order.
set -euo pipefail

export CHUNK=${1:?chunk id}
: "${PSCRATCH:?PSCRATCH must be set; the yamls put their bps submit dirs under it}"
REPO=${REPO:-/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo}
ALL_STAGES=(overscan singleFrame template psfQa)
STAGES=${STAGES:-${ALL_STAGES[*]}}
STAGES=${STAGES//,/ }
START=${START:-overscan}
# calibrateImage legitimately fails on some star-poor detectors and the
# downstream tasks skip them, so single-frame only needs most of them.
SF_MIN_FRACTION=${SF_MIN_FRACTION:-0.9}
# The user bps names its output collections after (u/<operator>/...); the yamls
# read the previous stage's chain through {operator}, and the gates below look
# at the same names, so use the same call bps does (ctrl_bps initialize.py).
OPERATOR=$(python -c 'import getpass; print(getpass.getuser())')
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
EXP_CONSTRAINT=$(python chunk_grid.py "$CHUNK")
[ "$CHUNK" -lt 48 ] || echo "Warning: chunks 48-51 repeat the sky of chunks 0-3." >&2
echo "chunk ${CHUNK}: ${EXP_CONSTRAINT}"

# Cut the exclude lists out of the chunk (see delve_exclude.py): the DELVE
# exclude list, and the raw CCDs whose files cfitsio cannot decompress.
# EXP_CONSTRAINT drops the exposures they reject outright and feeds every stage;
# DET_CONSTRAINT also drops their single CCDs and feeds single-frame onwards, since
# crosstalk in the overscan job still reads those CCDs. So a corrupt CCD still
# fails in the overscan job on every rerun, but the overscan gate below goes by
# SF_WHERE and doesn't count it. DELVE_EXCLUDE picks another make_exclude_list.py
# output and CORRUPT_RAWS another EXPNUM,CCDNUM table; set either empty to skip it.
DELVE_EXCLUDE=${DELVE_EXCLUDE-../data/delve_exclude_20260101.fits}
CORRUPT_RAWS=${CORRUPT_RAWS-../data/corrupt_raws.csv}
EXCLUDE_LISTS=()
[ -z "$DELVE_EXCLUDE" ] || EXCLUDE_LISTS+=("$DELVE_EXCLUDE")
[ -z "$CORRUPT_RAWS" ] || EXCLUDE_LISTS+=("$CORRUPT_RAWS")
DET_CONSTRAINT=$EXP_CONSTRAINT
if [ ${#EXCLUDE_LISTS[@]} -gt 0 ]; then
    EXP_CONSTRAINT=$(python delve_exclude.py "$REPO" "$EXP_CONSTRAINT" "${EXCLUDE_LISTS[@]}")
    DET_CONSTRAINT=$(python delve_exclude.py --detectors "$REPO" "$EXP_CONSTRAINT" "${EXCLUDE_LISTS[@]}")
fi
export EXP_CONSTRAINT DET_CONSTRAINT

# Raws each stage is expected to produce an output for. Keep in sync with the
# dataQuery in the matching bps_build_chunk_*.yaml. Detector 2 (S30) is cut
# before 2016-12-29, when it began working again (NOIRLab DECam Known Problems).
SF_WHERE="instrument='DECam' AND exposure.observation_type='science' AND detector NOT IN (61,31) AND NOT (detector=2 AND exposure.day_obs<20161229) AND ${DET_CONSTRAINT}"

stage_index() { # position of $1 in the pipeline, or exit 2 if it is not a stage
    local i
    for i in "${!ALL_STAGES[@]}"; do [ "${ALL_STAGES[$i]}" = "$1" ] && { echo "$i"; return; }; done
    echo "Unknown stage '$1'; stages are: ${ALL_STAGES[*]}" >&2
    exit 2
}

START_INDEX=$(stage_index "$START")
for st in $STAGES; do stage_index "$st" > /dev/null; done

want() { # selected in STAGES and not before START?
    [[ " $STAGES " == *" $1 "* ]] && [ "$(stage_index "$1")" -ge "$START_INDEX" ]
}

# `bps submit` blocks until every job has finished but exits 0 even if jobs
# failed, so each stage is followed by an output check before the next starts.

if want overscan; then
    bps submit bps_build_chunk_overscanRaws.yaml
    python check_complete.py "$REPO" "u/${OPERATOR}/DECam/endurance/overscanRaws/chunk${CHUNK}" \
        overscanRaw "$SF_WHERE" 1.0 \
      || { echo "Overscan incomplete for chunk ${CHUNK}; stopping." >&2; exit 1; }
fi

if want singleFrame; then
    # calibrateImage runs per visit and ingest defines none, so define them for
    # the chunk's science exposures that still lack one (see missing_visits.py);
    # a rerun finds none left and skips this.
    NEW_VISITS=$(python missing_visits.py "$REPO" \
        "instrument='DECam' AND exposure.observation_type='science' AND ${EXP_CONSTRAINT}")
    if [ -n "$NEW_VISITS" ]; then
        butler define-visits "$REPO" DECam --collections DECam/calib/curated --where "$NEW_VISITS"
    fi
    bps submit bps_build_chunk_singleFrame.yaml
    python check_complete.py "$REPO" "u/${OPERATOR}/DECam/endurance/singleFrame/chunk${CHUNK}" \
        preliminary_visit_image "$SF_WHERE" "$SF_MIN_FRACTION" \
      || { echo "Single-frame incomplete for chunk ${CHUNK}; stopping." >&2; exit 1; }
fi

if want template; then
    bps submit bps_build_chunk_template.yaml
fi

# A full chunk's PSF measurement takes hours on many workers, so it runs as its
# own Slurm job rather than on this login node; sbatch returns once it is queued.
if want psfQa; then
    sbatch "$HERE/../../verify/run-chunk-psf-qa.sh" "$CHUNK"
fi
