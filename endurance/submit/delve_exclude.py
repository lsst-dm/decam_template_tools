"""Cut the DELVE exclude list (and any other CCD lists) out of a chunk's registry `where` clause.

    python delve_exclude.py [--detectors] REPO WHERE EXCLUDE_LIST [EXCLUDE_LIST ...]

Prints WHERE with the DECam data on the exclude lists removed:
  default      drop the exposures it rejects outright (every CCD listed);
  --detectors  also drop the single CCDs it lists for the other exposures.
run_chunk.sh applies the first to every stage and the second from single-frame
on, because crosstalk in the overscan job reads every CCD of an exposure.

The list (https://github.com/delve-survey/exclude_list) has ~9e4 (EXPNUM, CCDNUM)
rows, which spelled out is more than the pipetask command line can carry. So
only the exposures matching WHERE are looked up, the CCD cuts are grouped into
one NOT (detector=D AND exposure IN (...)) term per detector, and consecutive
exposures collapse into a..b ranges. On DECam, exposure == EXPNUM and
detector == CCDNUM.

run_chunk.sh reads EXCLUDE_LIST from ../data/delve_exclude_20260101.fits. To
refresh it, in a clone of the repo above:

    python bin/make_exclude_list.py y*/*.txt -o delve_exclude_YYYYMMDD.fits

Several lists are merged. run_chunk.sh also passes ../data/corrupt_raws.csv:
raw CCDs whose .fits.fz HDU cfitsio cannot decompress, so every task reading
them fails. Add a row (EXPNUM,CCDNUM,REASON) when a new one turns up.
"""
import argparse
import collections
import sys

from astropy.table import Table
from lsst.daf.butler import Butler

CCDNUMS = frozenset(range(1, 63))
# bps passes the dataQuery to pipetask as a single argument, which Linux caps at
# 128 KiB; leave room for the rest of the query.
MAX_LEN = 100_000


def read_exclude_list(*paths):
    """{EXPNUM: set of excluded CCDNUMs} merged from FITS or CSV tables with EXPNUM and CCDNUM columns.

    A row without a CCD (make_exclude_list.py writes its NaN ccdnum as 0)
    excludes the whole exposure.
    """
    excluded = collections.defaultdict(set)
    for path in paths:
        t = Table.read(path)
        for expnum, ccdnum in zip(t["EXPNUM"].tolist(), t["CCDNUM"].tolist()):
            excluded[expnum] |= {ccdnum} if ccdnum in CCDNUMS else CCDNUMS
    return excluded


def id_list(ids):
    """Comma-separated ids, with runs of consecutive ids folded into a..b ranges."""
    runs = []
    for i in sorted(ids):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ",".join(str(a) if a == b else f"{a}..{b}" for a, b in runs)


def exclude(where, exposures, excluded, detectors=False):
    """`where` minus the excluded part of `exposures`, and how many exposures and CCDs that drops."""
    listed = {e: excluded[e] for e in exposures & excluded.keys()}
    whole = [e for e, ccds in listed.items() if ccds == CCDNUMS]
    by_detector = collections.defaultdict(list)
    if detectors:
        for e, ccds in listed.items():
            if ccds != CCDNUMS:
                for c in ccds:
                    by_detector[c].append(e)

    terms = [f"exposure NOT IN ({id_list(whole)})"] if whole else []
    terms += [f"NOT (detector={d} AND exposure IN ({id_list(es)}))" for d, es in sorted(by_detector.items())]
    clause = " AND ".join([f"({where})", *terms]) if terms else where
    return clause, len(whole), sum(map(len, by_detector.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", action="store_true", help="also drop single excluded CCDs")
    parser.add_argument("repo")
    parser.add_argument("where")
    parser.add_argument(
        "exclude_list", nargs="+", help="make_exclude_list.py output or another EXPNUM,CCDNUM table (FITS or CSV)"
    )
    args = parser.parse_args()

    butler = Butler.from_config(args.repo)
    exposures = {
        r.id
        for r in butler.query_dimension_records(
            "exposure", instrument="DECam", where=args.where, limit=None, explain=False
        )
    }
    clause, n_exposures, n_ccds = exclude(args.where, exposures, read_exclude_list(*args.exclude_list), args.detectors)
    print(
        f"Exclude lists: dropping {n_exposures} whole exposures"
        + (f" and {n_ccds} single CCDs" if args.detectors else "")
        + f" of the {len(exposures)} exposures in the chunk",
        file=sys.stderr,
    )
    if len(clause) > MAX_LEN:
        sys.exit(f"Exclusion clause is {len(clause)} characters, over the {MAX_LEN} pipetask can take; use smaller chunks.")
    print(clause)
