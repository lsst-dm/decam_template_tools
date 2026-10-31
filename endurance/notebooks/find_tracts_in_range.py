#!/usr/bin/env python
"""Find the tracts of a skymap that overlap an RA/Dec range.

The scheme
----------
1. **Candidate tracts.**
   Lay a grid (`grid_step_deg`) over the query box, padded on all sides by
   `patch_margin_deg`, and call `skymap.findTractIdArray()` -- vectorized,
   exact point-in-tract membership -- on every grid point; the set of
   distinct tract ids returned is the candidate list. (`skymap.
   findTractPatchList()`, which takes just the box's 4 corners, looks like
   the obvious tool for this but is unreliable in practice: for a
   `RingsSkyMap` it can silently miss whole interior rings of genuinely
   overlapping tracts -- confirmed against `lsst_cells_v2`, where it
   dropped the entire middle dec-ring for a real query box. The grid is
   the exact substitute: since a skymap's tracts tile the sky with no
   gaps, any tract with non-negligible overlap contains at least one grid
   point once the step is finer than the tract size, and the margin
   catches tracts that only overlap through their patch-border padding.)

2. **Exact filter.**
   Each candidate's *outer* sky region (patch-border-padded, so a tract
   that only brushes the query box at its patch overlap still counts) is
   tested against the query box with sphgeom's `isDisjointFrom`. Only
   tracts that are genuinely not disjoint survive.

3. **Overlap accounting.**
   Because both the query region and each tract's *inner* sky region are
   axis-aligned lon/lat boxes, their intersection area has a closed form
   (`delta_lon_rad * (sin(dec_hi) - sin(dec_lo))`), so overlap area,
   fraction-of-tract, and fraction-of-query are computed exactly rather
   than by rasterizing -- no grid needed for this script.

Usage
-----
From a shell with the LSST stack set up (`source loadLSST.bash && setup
lsst_distrib`)::

    python find_tracts_in_range.py \\
        --ra-min 305 --ra-max 309 --dec-min -53 --dec-max -50 \\
        --skymap lsst_cells_v2 --repo main

or from a notebook::

    from find_tracts_in_range import find_tracts_in_range
    from tract_group_delve_coverage import get_skymap
    skymap = get_skymap(skymap_name="lsst_cells_v2")
    tracts_df = find_tracts_in_range(305, 309, -53, -50, skymap)
"""

from __future__ import annotations

import argparse
import math
import warnings

import numpy as np
import pandas as pd

from tract_group_delve_coverage import DEG2_PER_SR, get_skymap


def _split_ra_range(ra_min, ra_max):
    """Return a list of non-wrapping (lo, hi) RA sub-ranges in [0, 360)."""
    ra_min, ra_max = ra_min % 360.0, ra_max % 360.0
    if ra_min < ra_max:
        return [(ra_min, ra_max)]
    if ra_min > ra_max:
        # wraps through 0/360
        return [(ra_min, 360.0), (0.0, ra_max)]
    # ra_min == ra_max: treat as the full circle
    return [(0.0, 360.0)]


def _box_overlap_area_deg2(ra_lo1, ra_hi1, dec_lo1, dec_hi1, ra_lo2, ra_hi2, dec_lo2, dec_hi2):
    """Exact overlap area (deg^2) of two axis-aligned lon/lat boxes, already unwrapped/non-wrapping."""
    ra_lo, ra_hi = max(ra_lo1, ra_lo2), min(ra_hi1, ra_hi2)
    dec_lo, dec_hi = max(dec_lo1, dec_lo2), min(dec_hi1, dec_hi2)
    if ra_hi <= ra_lo or dec_hi <= dec_lo:
        return 0.0
    return (
        math.radians(ra_hi - ra_lo)
        * (math.sin(math.radians(dec_hi)) - math.sin(math.radians(dec_lo)))
        * DEG2_PER_SR
    )


def find_tracts_in_range(ra_min, ra_max, dec_min, dec_max, skymap, grid_step_deg=0.05, patch_margin_deg=0.1):
    """Find every tract of `skymap` overlapping the given RA/Dec range (all in deg).

    `grid_step_deg` sets the candidate-search grid resolution (deg); it
    should be well under the skymap's tract size (a few deg for
    `lsst_cells_v2`). `patch_margin_deg` pads the grid beyond the query box
    so tracts that only overlap through their patch-border padding are
    still found.

    Returns a DataFrame with one row per overlapping tract: tract_id, its
    inner-region RA/Dec bounds, its full inner area, the exact overlap area
    with the query range, and that overlap as a fraction of the tract and
    of the query range.
    """
    import lsst.sphgeom as sphgeom

    warnings.filterwarnings("ignore", category=FutureWarning, module="lsst")

    dec_min, dec_max = max(-90.0, min(dec_min, dec_max)), min(90.0, max(dec_min, dec_max))
    ra_subranges = _split_ra_range(ra_min, ra_max)
    query_area_deg2 = sum(
        math.radians(hi - lo) * (math.sin(math.radians(dec_max)) - math.sin(math.radians(dec_min))) * DEG2_PER_SR
        for lo, hi in ra_subranges
    )

    # -- pass 1: exact candidate generation via a padded point-membership grid --
    padded_dec_min = max(-90.0, dec_min - patch_margin_deg)
    padded_dec_max = min(90.0, dec_max + patch_margin_deg)
    dec_grid = np.arange(padded_dec_min + grid_step_deg / 2, padded_dec_max, grid_step_deg)
    candidates = {}
    for ra_lo, ra_hi in ra_subranges:
        padded_ra_lo = ra_lo - patch_margin_deg
        padded_ra_hi = ra_hi + patch_margin_deg
        ra_grid = np.arange(padded_ra_lo + grid_step_deg / 2, padded_ra_hi, grid_step_deg)
        if len(ra_grid) == 0 or len(dec_grid) == 0:
            continue
        RA, DEC = np.meshgrid(ra_grid % 360.0, dec_grid)
        ids = skymap.findTractIdArray(RA.ravel(), DEC.ravel(), degrees=True)
        for tract_id in np.unique(ids):
            candidates[int(tract_id)] = skymap[int(tract_id)]

    # -- pass 2: exact disjointness test + overlap accounting --
    rows = []
    for tract_id, tract in candidates.items():
        box = tract.inner_sky_region
        # Use the interval's own lower bound as the unwrap reference (not the
        # A/B midpoint): if the tract's inner box itself straddles 0/360
        # (A > B), averaging A and B lands the reference on the opposite side
        # of the sky and corrupts the unwrap for every other point.
        ra_ref = box.getLon().getA().asDegrees()
        t_ra_lo = ra_ref
        t_ra_hi = ra_ref + (((box.getLon().getB().asDegrees() - ra_ref + 180.0) % 360.0) - 180.0)
        t_dec_lo = box.getLat().getA().asDegrees()
        t_dec_hi = box.getLat().getB().asDegrees()
        tract_area_deg2 = box.getArea() * DEG2_PER_SR

        overlap_deg2 = 0.0
        is_disjoint = True
        for ra_lo, ra_hi in ra_subranges:
            q_ra_lo = ra_ref + (((ra_lo - ra_ref + 180.0) % 360.0) - 180.0)
            q_ra_hi = q_ra_lo + (ra_hi - ra_lo)
            query_box = sphgeom.Box.fromDegrees(ra_lo, dec_min, ra_hi, dec_max)
            if not tract.outer_sky_polygon.isDisjointFrom(query_box):
                is_disjoint = False
            overlap_deg2 += _box_overlap_area_deg2(
                t_ra_lo, t_ra_hi, t_dec_lo, t_dec_hi, q_ra_lo, q_ra_hi, dec_min, dec_max
            )

        if is_disjoint and overlap_deg2 <= 0.0:
            continue

        rows.append(dict(
            tract_id=tract_id,
            ra_lo=box.getLon().getA().asDegrees() % 360.0,
            ra_hi=box.getLon().getB().asDegrees() % 360.0,
            dec_lo=t_dec_lo,
            dec_hi=t_dec_hi,
            tract_area_deg2=tract_area_deg2,
            overlap_area_deg2=overlap_deg2,
            frac_of_tract=overlap_deg2 / tract_area_deg2 if tract_area_deg2 > 0 else np.nan,
            frac_of_query=overlap_deg2 / query_area_deg2 if query_area_deg2 > 0 else np.nan,
        ))

    df = pd.DataFrame(rows).sort_values("tract_id").reset_index(drop=True)
    df.attrs["query"] = dict(
        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max,
        skymap=skymap.config.name if hasattr(skymap.config, "name") else type(skymap).__name__,
        query_area_deg2=query_area_deg2,
    )
    return df


def _print_report(df):
    q = df.attrs["query"]
    print(f"RA [{q['ra_min']:.4f}, {q['ra_max']:.4f}] x Dec [{q['dec_min']:.4f}, {q['dec_max']:.4f}] "
          f"({q['query_area_deg2']:.4f} deg^2), skymap={q['skymap']}")
    print(f"{len(df)} overlapping tract(s):\n")
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ra-min", type=float, required=True)
    parser.add_argument("--ra-max", type=float, required=True)
    parser.add_argument("--dec-min", type=float, required=True)
    parser.add_argument("--dec-max", type=float, required=True)
    parser.add_argument("--skymap", default="lsst_cells_v2", help="Skymap name (default: lsst_cells_v2).")
    parser.add_argument("--repo", default="main", help="Butler repo/alias (default: main).")
    parser.add_argument("--grid-step", type=float, default=0.05,
                         help="Candidate-search grid step in deg (default: 0.05).")
    parser.add_argument("--patch-margin", type=float, default=0.1,
                         help="Grid padding beyond the query box in deg (default: 0.1).")
    parser.add_argument("--out", default=None, help="Optional path to write the tract list csv.")
    args = parser.parse_args()

    skymap = get_skymap(repo=args.repo, skymap_name=args.skymap)
    df = find_tracts_in_range(
        args.ra_min, args.ra_max, args.dec_min, args.dec_max, skymap,
        grid_step_deg=args.grid_step, patch_margin_deg=args.patch_margin,
    )
    _print_report(df)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
