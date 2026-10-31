"""Which of a chunk's exposures still need a visit?

    missing_visits.py REPO EXPOSURE_WHERE

Prints `exposure IN (...)` for the exposures matching EXPOSURE_WHERE that have no
visit yet, for run_chunk.sh to hand to `butler define-visits --where`, or nothing
when they all have one. Ingest defines no visits, and calibrateImage runs per
visit, so without them the single-frame graph comes out empty. Visits are
one-to-one on DECam, so a visit id is its exposure id.
"""
import sys

from lsst.daf.butler import Butler

repo, where = sys.argv[1:3]

butler = Butler.from_config(repo)
exposures = {r.id for r in butler.query_dimension_records("exposure", where=where, limit=None, explain=False)}
visits = {r.id for r in butler.query_dimension_records("visit", where="instrument='DECam'", limit=None, explain=False)}
missing = sorted(exposures - visits)
print(f"{len(missing)} of {len(exposures)} exposures need a visit", file=sys.stderr)
if missing:
    print(f"exposure IN ({','.join(map(str, missing))})")
