#!/usr/bin/env bash
# Steps 1-3 in one go (NERSC). Step 4 (globus) is manual; step 5 runs at USDF.
#   ./run_nersc_steps.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40 [short-name]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -euo pipefail
"$here/01_create_butler.sh" "$@"
"$here/02_register_instrument.sh" "$@"
"$here/03_transfer_from_nersc.sh" "$@"
