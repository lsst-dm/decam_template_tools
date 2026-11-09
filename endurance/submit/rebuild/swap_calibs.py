"""Swap certifications in DECam/calib/{flat,fringe} from the current datasets to the
rebuilt *_v2 runs, one (year, band) at a time, atomically.

    DRY_RUN=1 python swap_calibs.py flat 2014:g 2014:r ...     # default: report only
    DRY_RUN=0 python swap_calibs.py fringe 2014 2016 ...        # do it
    python swap_calibs.py --rollback rebuild/swap_flat_<stamp>.json

Each swap decertifies exactly the 62 datasets certified for [YYYY-01-01, YYYY+1-01-01)
TAI (that band only, for flats) and certifies the 62 from the v2 chain for the same window,
inside one transaction. A resumed build only holds the quanta it re-ran, so each detector takes
its dataset from the newest run in the chain that has one. The old dataset IDs and runs are written to
a JSON record that --rollback uses to restore them. --rollback also undoes certify_new.py and
decertify.py records, using the validity window stored in the record when there is one.
"""
import datetime
import json
import os
import sys

from astropy.time import Time
from lsst.daf.butler import Butler, CollectionType, Timespan

REPO = os.environ.get("REPO", "/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo")
PREFIX = "u/descdm/DECam/calib/ELAIS_pilot"


def year_span(year):
    return Timespan(Time(f"{year}-01-01T00:00:00", scale="tai"),
                    Time(f"{int(year) + 1}-01-01T00:00:00", scale="tai"))


def window_span(begin, end):
    """[begin, end) TAI from ISO dates or datetimes."""
    return Timespan(Time(begin, scale="tai"), Time(end, scale="tai"))


def current(b, dstype, ts, band):
    coll = f"DECam/calib/{dstype}"
    return [a.ref for a in b.registry.queryDatasetAssociations(dstype, collections=[coll])
            if a.timespan == ts and (band is None or a.ref.dataId["band"] == band)]


def chain_runs(b, chain):
    """RUN collections of the chain, newest first."""
    return [c.name for c in b.collections.query_info(chain, collection_types=[CollectionType.RUN],
                                                     flatten_chains=True)]


def plan_item(b, dstype, item):
    year, _, band = item.partition(":")
    band = band or None
    ts = year_span(year)
    chain = f"{PREFIX}/{dstype}_v2/{year}" + (f"/{band}_band" if band else "")
    runs = chain_runs(b, chain)
    new = [r for r in b.registry.queryDatasets(dstype, collections=runs, findFirst=True)
           if band is None or r.dataId["band"] == band]
    new_runs = sorted({r.run for r in new}, key=runs.index)
    old = current(b, dstype, ts, band)
    dets_new, dets_old = {r.dataId["detector"] for r in new}, {r.dataId["detector"] for r in old}
    ok = len(new) == len(old) == 62 and dets_new == dets_old
    print(f"{dstype} {item}: {len(old)} certified now from {sorted({r.run for r in old})}"
          f" -> {len(new)} from {new_runs}  [{'OK' if ok else 'MISMATCH: not swapping'}]")
    return dict(year=year, band=band, ok=ok, new_runs=new_runs, new=new, old=old, ts=ts)


def swap(dstype, items, dry):
    b = Butler.from_config(REPO, writeable=not dry)
    plans = [plan_item(b, dstype, it) for it in items]
    if dry or not all(p["ok"] for p in plans):
        print("dry run: nothing changed" if dry else "not swapping: fix the mismatches first")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path = f"rebuild/swap_{dstype}_{stamp}.json"
    record = [dict(year=p["year"], band=p["band"], new_runs=p["new_runs"],
                   new_ids=[str(r.id) for r in p["new"]], old_ids=[str(r.id) for r in p["old"]],
                   old_runs=sorted({r.run for r in p["old"]})) for p in plans]
    with open(record_path, "w") as f:   # written before any change, for --rollback
        json.dump(dict(dataset_type=dstype, items=record), f, indent=1)
    coll = f"DECam/calib/{dstype}"
    for p in plans:
        with b.transaction():
            b.registry.decertify(coll, dstype, p["ts"], dataIds=[r.dataId for r in p["old"]])
            b.registry.certify(coll, p["new"], p["ts"])
        print(f"swapped {dstype} {p['year']} {p['band'] or ''}")
    print(f"rollback record: {record_path}")


def rollback(path, dry):
    rec = json.load(open(path))
    dstype, coll = rec["dataset_type"], f"DECam/calib/{rec['dataset_type']}"
    b = Butler.from_config(REPO, writeable=not dry)
    for it in rec["items"]:
        # only that window: a full-year decertify would also drop the other part of a split year
        ts = window_span(it["begin"], it["end"]) if it.get("begin") else year_span(it["year"])
        new = [b.registry.getDataset(i) for i in it["new_ids"]]
        old = [b.registry.getDataset(i) for i in it["old_ids"]]
        print(f"restore {dstype} {it['year']} {it['band'] or ''} [{ts.begin.isot[:10]}, {ts.end.isot[:10]}): "
              f"{len(new)} decertified, {len(old)} restored from {it['old_runs']}")
        if not dry:
            with b.transaction():
                if new:   # a decertify.py record has none; don't pass decertify an empty list
                    b.registry.decertify(coll, dstype, ts, dataIds=[r.dataId for r in new])
                if old:
                    b.registry.certify(coll, old, ts)
    print("dry run: nothing changed" if dry else "rolled back")


if __name__ == "__main__":
    dry = os.environ.get("DRY_RUN", "1") != "0"
    if sys.argv[1] == "--rollback":
        rollback(sys.argv[2], dry)
    else:
        swap(sys.argv[1], sys.argv[2:], dry)
