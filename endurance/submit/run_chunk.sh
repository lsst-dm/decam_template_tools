#!/bin/bash
# Overscan -> single-frame -> template jobs for one sky chunk.
#   ./run_chunk.sh <chunk>
# The sky extent for each chunk id comes from chunk_grid.py (chunk 0 = RA 300-330,
# dec 0..30; ids step down in dec, then move to the next RA column; 48 chunks, 0-47).
#
# Must be run from this directory with the LSST stack set up
# (see ~/startup_desc_commands.sh), so that ./submit points at the scratch dir.
#
# Each stage's bps job writes u/$USER/<payloadName>, a CHAINED collection of its
# run plus its inputs, and the next stage reads from that chain:
#   overscanRaws/chunkN -> singleFrame/chunkN -> template/chunkN
# so no manual collection-chain is needed.
#
# Stage selection (both optional, they combine):
#   STAGES  which stages to run, space or comma separated; default all three.
#   START   skip every stage before this one; default overscan.
# e.g. STAGES="overscan singleFrame" stops before the template job, and
# START=singleFrame resumes from there. Stages always run in pipeline order.
set -euo pipefail

export CHUNK=${1:?chunk id}
REPO=${REPO:-/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo}
ALL_STAGES=(overscan singleFrame template)
STAGES=${STAGES:-${ALL_STAGES[*]}}
STAGES=${STAGES//,/ }
START=${START:-overscan}
# calibrateImage legitimately fails on some star-poor detectors and the
# downstream tasks skip them, so single-frame only needs most of them.
SF_MIN_FRACTION=${SF_MIN_FRACTION:-0.9}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
EXP_CONSTRAINT=$(python chunk_grid.py "$CHUNK")
export EXP_CONSTRAINT
echo "chunk ${CHUNK}: ${EXP_CONSTRAINT}"

# Raws each stage is expected to produce an output for. Keep in sync with the
# dataQuery in the matching bps_build_chunk_*.yaml.
SF_WHERE="instrument='DECam' AND exposure.observation_type='science' AND detector NOT IN (61,31) AND (detector!=2 AND exposure.day_obs<20160000) AND ${EXP_CONSTRAINT}"

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
    # Re-running a chunk resumes: skip quanta already done in its chain.
    OVERSCAN_CHAIN="u/${USER}/DECam/endurance/overscanRaws/chunk${CHUNK}"
    if butler query-collections "$REPO" "$OVERSCAN_CHAIN" > /dev/null 2>&1; then
        export SKIP_EXISTING_OPT="--skip-existing-in ${OVERSCAN_CHAIN}"
        echo "Overscan: resuming, skipping quanta already in ${OVERSCAN_CHAIN}"
    else
        export SKIP_EXISTING_OPT=""
    fi
    bps submit bps_build_chunk_overscanRaws.yaml
    python check_complete.py "$REPO" "$OVERSCAN_CHAIN" \
        overscanRaw "$SF_WHERE" 1.0 \
      || { echo "Overscan incomplete for chunk ${CHUNK}; stopping." >&2; exit 1; }
fi

if want singleFrame; then
    bps submit bps_build_chunk_singleFrame.yaml
    python check_complete.py "$REPO" "u/${USER}/DECam/endurance/singleFrame/chunk${CHUNK}" \
        preliminary_visit_image "$SF_WHERE" "$SF_MIN_FRACTION" \
      || { echo "Single-frame incomplete for chunk ${CHUNK}; stopping." >&2; exit 1; }
fi

if want template; then
    bps submit bps_build_chunk_template.yaml
fi
