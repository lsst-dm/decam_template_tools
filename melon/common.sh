#!/usr/bin/env bash
# Shared helpers for the melon transfer workflow. Source this, don't run it.
#
# Inputs (positional args are parsed by each script via melon_parse_args):
#   CHAIN   full chained collection path, e.g. u/seanmacb/DECam/calib/ELAIS_pilot_w40
#   NAME    optional short name for the sqlite butler (default: last path component
#           of CHAIN, e.g. ELAIS_pilot_w40)
#
# Environment overrides (defaults in brackets):
#   BUTLER_PARENT   directory that holds the new butler repo     [<melon>/pilot]
#   SOURCE_REPO     repo to copy datasets from (NERSC)           [decam]
#   COLLECTIONS     run collection(s) to transfer                [<CHAIN>/template]
#   DATASET_TYPE    dataset type to transfer                     [template_coadd]
#   WHERE           data ID constraint                           [skymap='lsst_cells_v2']
#   INSTRUMENT      instrument class to register                 [lsst.obs.decam.DarkEnergyCamera]
#   MAIN_REPO       destination repo at USDF                     [main]
#   MAIN_TRANSFER   transfer mode into main                      [copy]

set -euo pipefail

MELON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUTLER_PARENT="${BUTLER_PARENT:-$MELON_DIR/pilot}"
SOURCE_REPO="${SOURCE_REPO:-decam}"
DATASET_TYPE="${DATASET_TYPE:-template_coadd}"
WHERE="${WHERE:-skymap='lsst_cells_v2'}"
INSTRUMENT="${INSTRUMENT:-lsst.obs.decam.DarkEnergyCamera}"
MAIN_REPO="${MAIN_REPO:-main}"
MAIN_TRANSFER="${MAIN_TRANSFER:-copy}"

melon_die() { echo "ERROR: $*" >&2; exit 1; }

melon_require_butler() {
    command -v butler >/dev/null 2>&1 \
        || melon_die "'butler' not on PATH; set up the LSST stack first (e.g. source loadLSST.bash; setup lsst_distrib)."
}

# Usage: melon_parse_args "$@"   -> sets CHAIN, NAME, REPO, COLLECTIONS
melon_parse_args() {
    CHAIN="${1:-}"
    [[ -n "$CHAIN" ]] || melon_die "usage: $(basename "$0") <chained-collection> [short-name]"
    CHAIN="${CHAIN%/}"
    NAME="${2:-${CHAIN##*/}}"
    # The short name becomes a directory, so it must not be the full (slashed) path.
    [[ "$NAME" != */* ]] || melon_die "short name '$NAME' must not contain '/'"
    REPO="$BUTLER_PARENT/$NAME"
    COLLECTIONS="${COLLECTIONS:-$CHAIN/template}"
}
