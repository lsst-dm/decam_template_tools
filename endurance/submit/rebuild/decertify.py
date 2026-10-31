"""Decertify single detectors from DECam/calib/{bias,flat,fringe}, with nothing certified in
their place, e.g. det-2 templates built only from frames taken while S30 was off.

    DRY_RUN=1 python rebuild/decertify.py fringe 2014 2016 --detectors 2      # default: report only
    DRY_RUN=0 python rebuild/decertify.py fringe 2014 2016 --detectors 2      # do it
    DRY_RUN=0 python rebuild/swap_calibs.py --rollback rebuild/decertify_<type>_<stamp>.json

Each item removes the named detectors certified for the [YYYY-01-01, YYYY+1-01-01) TAI window
(that band only, for YEAR:BAND), all items in one transaction. The record has swap_calibs.py's
format, with the removed datasets as old_ids and no new_ids, so --rollback recertifies them.
"""
import argparse
import datetime
import json
import os

from lsst.daf.butler import Butler

from swap_calibs import REPO, current, year_span


def plan(b, dstype, item, dets):
    year, _, band = item.partition(":")
    band = band or None
    ts = year_span(year)
    old = [r for r in current(b, dstype, ts, band) if r.dataId["detector"] in dets]
    ok = {r.dataId["detector"] for r in old} == dets and len(old) == len(dets)
    print(f"{dstype} {item}: detectors {sorted(r.dataId['detector'] for r in old)} certified from "
          f"{sorted({r.run for r in old})}  [{'OK' if ok else 'MISMATCH: not decertifying'}]")
    return dict(year=year, band=band, ok=ok, old=old, ts=ts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dstype", choices=["bias", "flat", "fringe"])
    p.add_argument("items", nargs="+", help="YEAR, or YEAR:BAND for flats")
    p.add_argument("--detectors", required=True, help="detectors to decertify, e.g. 2 or 2,61")
    args = p.parse_args()
    dry = os.environ.get("DRY_RUN", "1") != "0"
    dets = {int(d) for d in args.detectors.split(",") if d}
    b = Butler.from_config(REPO, writeable=not dry)
    plans = [plan(b, args.dstype, it, dets) for it in args.items]
    if dry or not all(pl["ok"] for pl in plans):
        print("dry run: nothing changed" if dry else "not decertifying: fix the mismatches first")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path = f"rebuild/decertify_{args.dstype}_{stamp}.json"
    with open(record_path, "w") as f:   # written before any change, for swap_calibs.py --rollback
        json.dump(dict(dataset_type=args.dstype, items=[dict(
            year=pl["year"], band=pl["band"], new_runs=[], new_ids=[],
            old_ids=[str(r.id) for r in pl["old"]], old_runs=sorted({r.run for r in pl["old"]}),
            begin=pl["ts"].begin.isot, end=pl["ts"].end.isot) for pl in plans]), f, indent=1)
    coll = f"DECam/calib/{args.dstype}"
    with b.transaction():
        for pl in plans:
            b.registry.decertify(coll, args.dstype, pl["ts"], dataIds=[r.dataId for r in pl["old"]])
    print(f"decertified {args.dstype} detectors {sorted(dets)} for {', '.join(args.items)}")
    print(f"rollback record: {record_path}")


if __name__ == "__main__":
    main()
