#!/usr/bin/env bash
# Step 3: copy the run collection's datasets from the NERSC decam repo into the sqlite butler.
#   ./03_transfer_from_nersc.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40 [short-name]
# Transfers <chain>/template by default; override with COLLECTIONS=...
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
melon_parse_args "$@"
melon_require_butler

[[ -f "$REPO/butler.yaml" ]] || melon_die "no butler at $REPO; run 01/02 first."

# Run from the repo so the destination is the relative `butler.yaml`, as in the reference command.
cd "$REPO"
butler transfer-datasets \
    --collections "$COLLECTIONS" \
    --where "$WHERE" \
    -d "$DATASET_TYPE" \
    --register-dataset-types \
    -t copy \
    --limit 0 \
    "$SOURCE_REPO" butler.yaml

echo "Transferred $DATASET_TYPE from $COLLECTIONS ($SOURCE_REPO) into $REPO"
echo "Next: globus-transfer $REPO to USDF (step 4, manual), then run 05_transfer_to_main.sh there."
