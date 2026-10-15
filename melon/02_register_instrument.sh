#!/usr/bin/env bash
# Step 2: register the DECam instrument in the new butler.
#   ./02_register_instrument.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40 [short-name]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
melon_parse_args "$@"
melon_require_butler

[[ -f "$REPO/butler.yaml" ]] || melon_die "no butler at $REPO; run 01_create_butler.sh first."

butler register-instrument "$REPO" "$INSTRUMENT"
echo "Registered $INSTRUMENT in $REPO"
