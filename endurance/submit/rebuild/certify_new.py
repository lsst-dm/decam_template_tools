"""Certify a year that has nothing certified yet in DECam/calib/{bias,flat,fringe}.

    DRY_RUN=1 python rebuild/certify_new.py fringe 2022 u/descdm/DECam/calib/ELAIS_pilot/fringe_v2/2022
    DRY_RUN=1 python rebuild/certify_new.py fringe 2015 u/descdm/DECam/calib/ELAIS_pilot/fringe_v2/2016 --skip 2
    DRY_RUN=0 python rebuild/certify_new.py flat 2023:g <chain or RUN>        # do it
    DRY_RUN=1 python rebuild/certify_new.py flat 2024:g <chain> --begin 2024-01-01 --end 2024-04-01
    DRY_RUN=0 python rebuild/swap_calibs.py --rollback rebuild/certify_<type>_<stamp>.json

swap_calibs.py replaces existing certifications; this adds them where the
[YYYY-01-01, YYYY+1-01-01) TAI window is empty, or the [--begin, --end) TAI window for a calib
that changed within a year (e.g. the 2024 g flat). The record keeps that window, and --rollback
decertifies only it. As in swap_calibs.py, each detector takes its
dataset from the newest RUN under the source chain that has one, so a resumed build is handled.
The source may be another year's build, to reuse a neighbouring template for a year without
enough data of its own. --skip leaves detectors out (e.g. 2, S30, off 2013-11-30 to 2016-12-29).
The rollback record has swap_calibs.py's format.
"""
import argparse
import datetime
import json
import os

from lsst.daf.butler import Butler, MissingCollectionError

from swap_calibs import REPO, chain_runs, window_span, year_span


def plan(b, dstype, item, source, skip, window=None):
    year, _, band = item.partition(":")
    band = band or None
    ts = window_span(*window) if window else year_span(year)
    try:
        runs = [r for r in chain_runs(b, source) if r == source or r.startswith(source + "/")]
    except MissingCollectionError:
        print(f"{dstype} {item}: {source} does not exist  [MISMATCH: not certifying]")
        return dict(ok=False)
    new = [r for r in b.registry.queryDatasets(dstype, collections=runs, findFirst=True)
           if r.dataId["detector"] not in skip and (band is None or r.dataId["band"] == band)]
    new_runs = sorted({r.run for r in new}, key=runs.index)
    dets = {r.dataId["detector"] for r in new}
    clash = [a for a in b.registry.queryDatasetAssociations(dstype, collections=[f"DECam/calib/{dstype}"])
             if a.timespan.overlaps(ts) and (band is None or a.ref.dataId["band"] == band)]
    missing = sorted(set(range(1, 63)) - dets - skip)
    ok = len(new) == len(dets) == 62 - len(skip) and not clash
    print(f"{dstype} {item} [{ts.begin.isot[:10]}, {ts.end.isot[:10]}): {len(new)} from {new_runs}, "
          f"missing {missing}, skipped {sorted(skip)}, already certified in the window {len(clash)}  "
          f"[{'OK' if ok else 'MISMATCH: not certifying'}]")
    return dict(year=year, band=band, ok=ok, new_runs=new_runs, new=new, ts=ts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dstype", choices=["bias", "flat", "fringe"])
    p.add_argument("item", help="YEAR, or YEAR:BAND for flats")
    p.add_argument("source", help="chain or RUN holding the datasets")
    p.add_argument("--skip", default="", help="detectors to leave out, e.g. 2 or 2,61")
    p.add_argument("--begin", help="start of the validity window (TAI, e.g. 2024-04-01) instead of YEAR-01-01")
    p.add_argument("--end", help="end of the validity window (TAI, exclusive) instead of YEAR+1-01-01")
    args = p.parse_args()
    if bool(args.begin) != bool(args.end):
        p.error("give both --begin and --end, or neither")
    window = (args.begin, args.end) if args.begin else None
    dry = os.environ.get("DRY_RUN", "1") != "0"
    skip = {int(d) for d in args.skip.split(",") if d}
    b = Butler.from_config(REPO, writeable=not dry)
    pl = plan(b, args.dstype, args.item, args.source, skip, window)
    if dry or not pl["ok"]:
        print("dry run: nothing changed" if dry else "not certifying: fix the mismatch first")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path = f"rebuild/certify_{args.dstype}_{stamp}.json"
    with open(record_path, "w") as f:   # written before any change, for swap_calibs.py --rollback
        json.dump(dict(dataset_type=args.dstype, items=[dict(
            year=pl["year"], band=pl["band"], new_runs=pl["new_runs"],
            new_ids=[str(r.id) for r in pl["new"]], old_ids=[], old_runs=[],
            begin=pl["ts"].begin.isot, end=pl["ts"].end.isot)]), f, indent=1)
    with b.transaction():
        b.registry.certify(f"DECam/calib/{args.dstype}", pl["new"], pl["ts"])
    print(f"certified {args.dstype} {args.item}: {len(pl['new'])}")
    print(f"rollback record: {record_path}")


if __name__ == "__main__":
    main()
