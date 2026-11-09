#!/usr/bin/env bash
# Step 5 (run at USDF): transfer the datasets in the globus-ed sqlite butler into main.
#   ./05_transfer_to_main.sh u/seanmacb/DECam/calib/ELAIS_pilot_w40 [short-name]
# The sqlite butler is expected at $BUTLER_PARENT/<short-name>; set REPO_OVERRIDE=/abs/path
# if globus put it elsewhere. Set DRY_RUN=1 to only count what would be transferred.
#
# Step 3 copies only RUN collections into the sqlite butler (chains are not carried), so the
# chain name does not exist here. COLLECTIONS is instead a comma-separated list of names or
# globs matched against the sqlite butler's RUN collections; the default '*' means everything
# step 3 brought over. It is expanded to explicit run names before transferring.
_user_collections="${COLLECTIONS:-}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
melon_parse_args "$@"
melon_require_butler

REPO="${REPO_OVERRIDE:-$REPO}"
[[ -f "$REPO/butler.yaml" ]] || melon_die "no butler at $REPO (set REPO_OVERRIDE=/abs/path to change)."

# Resolve the RUN collections, and check that main's datastore will let us create the files.
# transfer-datasets keeps each file's path relative to the datastore root, i.e. <run>/..., so
# a run under someone else's u/<user>/ needs that directory to exist and be writable in main.
COLLECTIONS="$(python - "$REPO" "${_user_collections:-*}" "$MAIN_REPO" <<'EOF'
import os
import sys

from lsst.daf.butler import Butler, CollectionType

src, pattern, main = sys.argv[1:]
runs = Butler.from_config(src).collections.query(
    pattern.split(","), collection_types=CollectionType.RUN
)
if not runs:
    sys.exit(f"ERROR: no RUN collections in {src} match {pattern!r}.")

try:
    root = Butler.from_config(main)._datastore.root.ospath
except Exception as e:  # e.g. a non-POSIX or chained datastore
    print(f"WARNING: skipping write-access check for {main}: {e}", file=sys.stderr)
    root = None

blocked = []
for run in runs if root else []:
    d = os.path.join(root, run)
    while not os.path.exists(d):
        d = os.path.dirname(d)
    if not os.access(d, os.W_OK):
        blocked.append(f"  {run}\n    -> {os.path.join(root, run)} would be created under {d}, which you cannot write")
if blocked:
    sys.exit(
        f"ERROR: cannot write the copied files into {main}'s datastore:\n"
        + "\n".join(blocked)
        + "\nAsk a repo admin to create that directory with write access for you."
    )
print(",".join(runs))
EOF
)"
echo "Run collections to transfer:"
printf '  %s\n' ${COLLECTIONS//,/ }

count() {  # count <repo>: number of matching datasets (query-datasets prints a header + one row each)
    butler query-datasets "$1" "$DATASET_TYPE" --collections "$COLLECTIONS" --where "$WHERE" --limit 0 | wc -l
}

# Assign before echoing: a failure inside "echo $(...)" would be ignored and print 0.
n_src="$(count "$REPO")"
echo "Lines of query-datasets output in sqlite butler ($REPO): $n_src"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: not transferring to $MAIN_REPO."
    exit 0
fi

butler transfer-datasets \
    --collections "$COLLECTIONS" \
    --where "$WHERE" \
    -d "$DATASET_TYPE" \
    --register-dataset-types \
    -t "$MAIN_TRANSFER" \
    --limit 0 \
    "$REPO/butler.yaml" "$MAIN_REPO"

n_main="$(count "$MAIN_REPO")"
echo "Lines of query-datasets output in $MAIN_REPO: $n_main"
echo "(The two counts should match; the 'main' count includes only these collections.)"
