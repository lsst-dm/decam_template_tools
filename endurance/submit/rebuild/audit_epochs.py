"""Did the inputs of each flat/fringe in COLLECTION use the epoch-correct calibrations?

    python audit_epochs.py flat|fringe COLLECTION

For every RUN holding flats/fringes in COLLECTION (a certified CALIBRATION collection or
a v2 build chain), reads the ISR headers of the first, middle and last input on detectors
10, 35 and 59 and compares each calibration used (LSST CALIB RUN *) with what the
certified DECam/calib/{bias,flat} or DECam/calib/curated give at that exposure's time.
Also tallies the inputs' observation types. Exits 1 if anything mismatches.
"""
import os
import re
import sys
from collections import Counter, defaultdict

from lsst.daf.butler import Butler, CollectionType

REPO = os.environ.get("REPO", "/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo")
DETS = (10, 35, 59)
INPUT = {"flat": "cpFlatIsrExp", "fringe": "cpFringeIsrExp"}
LOOKUP = {"bias": "DECam/calib/bias", "flat": "DECam/calib/flat"}
dstype, collection = sys.argv[1], sys.argv[2]
b = Butler.from_config(REPO)
cache = {}


def expected(kind, data_id, ts):
    key = (kind, data_id["detector"], data_id.get("physical_filter"), ts.begin.isot[:10])
    if key not in cache:
        kw = dict(instrument="DECam", detector=data_id["detector"])
        if kind == "flat":
            kw["physical_filter"] = data_id["physical_filter"]
        ref = b.find_dataset(kind, collections=LOOKUP.get(kind, "DECam/calib/curated"),
                             timespan=ts, **kw)
        cache[key] = ref.run if ref else "none certified"
    return cache[key]


if b.collections.get_info(collection).type == CollectionType.CALIBRATION:
    runs = {a.ref.run for a in b.registry.queryDatasetAssociations(dstype, collections=[collection])}
else:
    runs = {r.run for r in b.registry.queryDatasets(dstype, collections=collection)}
bad = 0
for run in sorted(runs):
    verdicts, obstypes, tally_det = defaultdict(Counter), Counter(), None
    for det in DETS:
        refs = sorted(b.registry.queryDatasets(INPUT[dstype], collections=run,
                                               where=f"instrument='DECam' AND detector={det}").expanded(),
                      key=lambda r: r.dataId["exposure"])
        if refs and not obstypes:
            obstypes.update(r.dataId.records["exposure"].observation_type for r in refs)
            tally_det = det
        for ref in {r.id: r for r in (refs[:1] + refs[len(refs) // 2:][:1] + refs[-1:])}.values():
            md = b.get(ref.makeComponentRef("metadata")).toDict()
            ts = ref.dataId.records["exposure"].timespan
            for k, used in md.items():
                m = re.fullmatch(r"LSST CALIB RUN (\w+)", k)
                kind = m and m.group(1).lower()
                if not kind or kind in ("camera", "ccdexposure"):
                    continue
                exp = expected(kind, ref.dataId, ts)
                verdicts[kind][("ok" if used == exp else "MISMATCH", used, exp)] += 1
    print(f"\n{run}\n   inputs (detector {tally_det}): {dict(obstypes)}")
    for kind, c in sorted(verdicts.items()):
        for (v, used, exp), n in sorted(c.items()):
            print(f"   {kind:10s} {v:8s} n={n:2d} used={used}" + ("" if v == "ok" else f"  expected={exp}"))
            bad += v != "ok"
print("\nALL OK" if not bad else f"\n{bad} MISMATCHES")
sys.exit(1 if bad else 0)
