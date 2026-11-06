#!/usr/bin/env bash
# Step 5 (run at USDF): transfer the datasets in the globus-ed sqlite butler into main.
#   ./05_transfer_to_main.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40 [short-name]
# The sqlite butler is expected at $BUTLER_PARENT/<short-name>; set REPO_OVERRIDE=/abs/path
# if globus put it elsewhere. Set DRY_RUN=1 to only count what would be transferred.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
melon_parse_args "$@"
melon_require_butler

REPO="${REPO_OVERRIDE:-$REPO}"
[[ -f "$REPO/butler.yaml" ]] || melon_die "no butler at $REPO (set REPO_OVERRIDE=/abs/path to change)."

count() {  # count <repo> <collections>: number of matching datasets (one query-datasets row each)
    butler query-datasets "$1" "$DATASET_TYPE" --collections "$2" --where "$WHERE" --limit 0 | grep -c "^${DATASET_TYPE}" || true
}

# The sqlite butler holds only the runs step 3 copied, not the chain they came from.
RUNS=$(melon_repo_runs "$REPO")
echo "Runs in sqlite butler: $RUNS"
echo "$DATASET_TYPE datasets in sqlite butler ($REPO): $(count "$REPO" "$RUNS")"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: not transferring to $MAIN_REPO."
    exit 0
fi

butler transfer-datasets \
    --collections "$RUNS" \
    --where "$WHERE" \
    -d "$DATASET_TYPE" \
    --register-dataset-types \
    -t "$MAIN_TRANSFER" \
    --limit 0 \
    "$REPO/butler.yaml" "$MAIN_REPO"

echo "$DATASET_TYPE datasets in $MAIN_REPO (same runs): $(count "$MAIN_REPO" "$RUNS")"
echo "(The two counts should match; the 'main' count includes only these runs.)"
