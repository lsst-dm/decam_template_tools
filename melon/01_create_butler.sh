#!/usr/bin/env bash
# Step 1: create a new sqlite butler named after the chained collection.
#   ./01_create_butler.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40 [short-name]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
melon_parse_args "$@"
melon_require_butler

[[ ! -e "$REPO" ]] || melon_die "$REPO already exists; refusing to overwrite. Remove it or pass a different short name."

mkdir -p "$BUTLER_PARENT"
# Default `butler create` config is sqlite (gen3.sqlite3 + file datastore under the repo root).
butler create "$REPO"
echo "Created sqlite butler: $REPO/butler.yaml"
