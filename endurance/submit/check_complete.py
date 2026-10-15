"""Gate between stages: did enough of a chunk's raws produce `dataset_type`?

    check_complete.py REPO CHAIN DATASET_TYPE RAW_WHERE [MIN_FRACTION]

Expected = (exposure, detector) of the raws matching RAW_WHERE. Done = those
that have DATASET_TYPE (keyed by exposure or visit; they are equal on DECam)
in CHAIN. Exit 0 if done/expected >= MIN_FRACTION (default 1.0), else 1.

`bps submit` with ParslService blocks until the workflow ends but returns 0
even when jobs failed, so run_chunk.sh gates each stage on this instead.
"""
import sys

from lsst.daf.butler import Butler

repo, chain, dataset_type, where = sys.argv[1:5]
min_fraction = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

butler = Butler.from_config(repo)
raws = {
    (r.dataId["exposure"], r.dataId["detector"])
    for r in butler.query_datasets("raw", collections="DECam/raw/all", where=where, limit=None, explain=False)
}
# The chain is per chunk, so no `where` needed here; intersecting with `raws`
# drops anything outside the chunk.
done = {
    (r.dataId.get("exposure", r.dataId.get("visit")), r.dataId["detector"])
    for r in butler.query_datasets(dataset_type, collections=chain, limit=None, explain=False)
} & raws

fraction = len(done) / len(raws) if raws else 0.0
print(f"{dataset_type}: {len(done)}/{len(raws)} raws done ({fraction:.1%}), need {min_fraction:.1%}")
sys.exit(0 if raws and fraction >= min_fraction else 1)
