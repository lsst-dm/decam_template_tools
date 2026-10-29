#!/usr/bin/env python
"""Estimate DELVE per-band single-epoch input coverage for a group of tracts.

The scheme
----------
1. **Sky area of the tract group.**
   Pull `TractInfo` for each requested tract id from a skymap (default
   `lsst_cells_v2`, from the `skymaps` collection of a butler repo, default
   `main`). Use each tract's *inner* sky region (a lon/lat `Box` in sphgeom
   terms), not the outer, patch-border-padded region. Inner regions of a
   `RingsSkyMap` tile the sky edge-to-edge with zero-area overlap, so the
   union sky area of *any* set of tracts -- adjacent or not -- is exactly the
   sum of their inner areas. No polygon union / de-duplication step needed.

   Tract ids can be given directly (a list of ints), or ingested from
   `find_tracts_in_range.py`'s output -- its returned DataFrame, or a csv
   written via its `--out`. When ingesting that output, tracts whose
   `overlap_area_deg2` is <= `min_overlap_deg2` (default 0.0) are dropped:
   those are boundary-touching tracts with zero actual overlap with the
   queried range, not tracts you'd want included in the group. Note the
   group still uses each surviving tract's *whole* inner area (not the
   clipped overlap slice) -- a tract group is always whole tracts.

2. **Match DELVE inputs to the group.**
   Each row of the DELVE csv (e.g. `delve_dr3qr_combo_v10.csv`) is one
   single-epoch exposure catalog: a pointing center (`ra`, `dec`) + `band`.
   There is no per-exposure footprint polygon in the file, so each exposure
   is approximated as a spherical `Circle` of radius `DECAM_FOV_RADIUS_DEG`.
   This defaults to 1.3 / 1.2 = 1.083 deg, matching the convention already
   used in this repo (`obs_decam/config/SkyBotEphemerisQuery.py` sets its
   query radius to "DECam FoV radius + 20% margin" = 1.3 deg).
   An exposure "hits" the group if its circle is not disjoint from the
   *outer* sky region of any tract in the group (outer, not inner, because a
   real input that only touches a tract's patch-overlap border still
   legitimately contributes to that tract's coadd/template).
   A cheap angular-separation prefilter around the group centroid is applied
   first so the ~2e5-row DELVE csv doesn't need an exact sphgeom test against
   every tract for every row.

3. **Per-band coverage / completeness.**
   Rasterize the group's bounding box on a plain RA/Dec grid (cos(dec)
   area-weighted, not equal-area, but simple and adequate for a bounded
   region -- see `grid_step_deg`). For each band: the fraction of "in the
   group" grid cells that fall within `DECAM_FOV_RADIUS_DEG` of at least one
   hit exposure gives the covered area and completeness fraction; the mean
   number of hit exposures covering each covered cell gives a depth proxy
   (average epoch count).

4. **Report.**
   `summarize_tract_group()` returns one row per band with: n_inputs,
   exptime_total, mean psf_fwhm / t_eff, covered_area_deg2,
   completeness_frac, mean_epochs. The group's tract ids and exact area are
   attached as a `.attrs["group"]` dict on the returned DataFrame.

Usage
-----
From a shell with the LSST stack set up (`source loadLSST.bash && setup
lsst_distrib`)::

    python tract_group_delve_coverage.py \\
        --tracts 8434 8435 8436 8556 8557 8558 \\
        --skymap lsst_cells_v2 --repo main \\
        --delve-csv ../data/delve_dr3qr_combo_v10.csv

or chained with `find_tracts_in_range.py`::

    python find_tracts_in_range.py --ra-min 305 --ra-max 309 --dec-min -53 --dec-max -50 \\
        --skymap lsst_cells_v2 --repo main --out tracts.csv
    python tract_group_delve_coverage.py --tract-range-csv tracts.csv \\
        --skymap lsst_cells_v2 --repo main --delve-csv ../data/delve_dr3qr_combo_v10.csv

or from a notebook::

    from find_tracts_in_range import find_tracts_in_range
    from tract_group_delve_coverage import summarize_tract_group, get_skymap
    skymap = get_skymap(skymap_name="lsst_cells_v2")
    tracts_df = find_tracts_in_range(305, 309, -53, -50, skymap)
    summary = summarize_tract_group(tracts_df, delve_csv="...", skymap=skymap)
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BANDS = ("g", "r", "i", "z")
# obs_decam/config/SkyBotEphemerisQuery.py: "DECam FoV radius + 20% margin" = 1.3 deg
DECAM_FOV_RADIUS_DEG = 1.3 / 1.2
RAD_PER_DEG = math.pi / 180.0
DEG2_PER_SR = (180.0 / math.pi) ** 2


# --------------------------------------------------------------------------
# Step 1: tract group geometry
# --------------------------------------------------------------------------

@dataclass
class TractGroup:
    tract_ids: list
    skymap_name: str
    tracts: list = field(default_factory=list)  # TractInfo objects, same order as tract_ids
    area_deg2: float = 0.0                      # exact, from inner regions
    ra_center: float = 0.0                      # deg, for RA-wrap unwrapping
    ra_min: float = 0.0
    ra_max: float = 0.0
    dec_min: float = 0.0
    dec_max: float = 0.0
    inner_boxes: list = field(default_factory=list)  # (ra_lo, ra_hi, dec_lo, dec_hi) deg, unwrapped


def get_skymap(repo="main", skymap_name="lsst_cells_v2", collection="skymaps"):
    import lsst.daf.butler as dafButler

    butler = dafButler.Butler(repo)
    return butler.get("skyMap", skymap=skymap_name, collections=collection)


def _unwrap_deg(lon_deg, ref_deg):
    """Shift lon_deg by +-360 so it's within 180 deg of ref_deg (handles RA wrap)."""
    d = (lon_deg - ref_deg + 180.0) % 360.0 - 180.0
    return ref_deg + d


def tract_ids_from_range_output(source, min_overlap_deg2=0.0):
    """Extract a tract id list from `find_tracts_in_range.py`'s DataFrame or csv output.

    `source` is either the DataFrame `find_tracts_in_range()` returns, or a
    path to a csv written from it (e.g. via its `--out`). Rows whose
    `overlap_area_deg2` is <= `min_overlap_deg2` are dropped -- those are
    tracts that only touch the queried range at a zero-area boundary, not
    ones with real overlap.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if "overlap_area_deg2" in df.columns:
        df = df[df["overlap_area_deg2"] > min_overlap_deg2]
    return df["tract_id"].astype(int).tolist()


def _normalize_tract_ids(tract_ids, min_overlap_deg2=0.0):
    """Accept either a plain list of tract ids or find_tracts_in_range()'s DataFrame/csv output."""
    if isinstance(tract_ids, pd.DataFrame) or isinstance(tract_ids, (str, Path)):
        return tract_ids_from_range_output(tract_ids, min_overlap_deg2=min_overlap_deg2)
    return [int(t) for t in tract_ids]


def build_tract_group(tract_ids, skymap, skymap_name="lsst_cells_v2", min_overlap_deg2=0.0):
    """Step 1: pull tract geometry and compute the exact union sky area."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="lsst")

    tract_ids = _normalize_tract_ids(tract_ids, min_overlap_deg2=min_overlap_deg2)
    tracts = [skymap[int(t)] for t in tract_ids]
    area_deg2 = sum(t.inner_sky_region.getArea() for t in tracts) * DEG2_PER_SR

    # Reference RA for unwrapping = first tract's inner-box lower bound (not
    # the A/B midpoint: if that tract's own box straddles 0/360, averaging A
    # and B lands the reference on the opposite side of the sky).
    box0 = tracts[0].inner_sky_region
    ra_ref = box0.getLon().getA().asDegrees()

    inner_boxes = []
    for t in tracts:
        box = t.inner_sky_region
        ra_lo = _unwrap_deg(box.getLon().getA().asDegrees(), ra_ref)
        ra_hi = _unwrap_deg(box.getLon().getB().asDegrees(), ra_ref)
        if ra_hi < ra_lo:
            ra_lo, ra_hi = ra_hi, ra_lo
        dec_lo = box.getLat().getA().asDegrees()
        dec_hi = box.getLat().getB().asDegrees()
        inner_boxes.append((ra_lo, ra_hi, dec_lo, dec_hi))

    ra_min = min(b[0] for b in inner_boxes)
    ra_max = max(b[1] for b in inner_boxes)
    dec_min = min(b[2] for b in inner_boxes)
    dec_max = max(b[3] for b in inner_boxes)

    return TractGroup(
        tract_ids=tract_ids,
        skymap_name=skymap_name,
        tracts=tracts,
        area_deg2=area_deg2,
        ra_center=ra_ref,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        inner_boxes=inner_boxes,
    )


# --------------------------------------------------------------------------
# Step 2: match DELVE inputs to the group
# --------------------------------------------------------------------------

def _angular_sep_deg(ra1, dec1, ra2, dec2):
    """Vectorized great-circle separation (deg) via the haversine formula."""
    ra1, dec1, ra2, dec2 = (np.radians(x) for x in (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2
    return np.degrees(2 * np.arcsin(np.clip(np.sqrt(a), 0, 1)))


def load_delve_csv(path, bands=DEFAULT_BANDS):
    df = pd.read_csv(path)
    if bands:
        df = df[df["band"].isin(bands)].reset_index(drop=True)
    return df


def match_inputs_to_group(delve_df, group, fov_radius_deg=DECAM_FOV_RADIUS_DEG, margin_deg=0.1):
    """Step 2: exact per-exposure hit test against the tract group.

    Returns delve_df filtered down to rows whose FoV circle is not disjoint
    from the outer sky region of at least one tract in the group.
    """
    import lsst.sphgeom as sphgeom

    warnings.filterwarnings("ignore", category=FutureWarning, module="lsst")

    # -- cheap prefilter: angular separation from the group's bbox center --
    ra_c = 0.5 * (group.ra_min + group.ra_max)
    dec_c = 0.5 * (group.dec_min + group.dec_max)
    half_diag_deg = 0.5 * math.hypot(
        (group.ra_max - group.ra_min) * math.cos(math.radians(dec_c)),
        group.dec_max - group.dec_min,
    )
    radius_cut = fov_radius_deg + half_diag_deg + margin_deg

    ra = delve_df["ra"].to_numpy(dtype=float)
    dec = delve_df["dec"].to_numpy(dtype=float)
    ra_unwrapped = ra_c + (((ra - ra_c + 180.0) % 360.0) - 180.0)
    sep = _angular_sep_deg(ra_unwrapped, dec, ra_c, dec_c)
    candidate_mask = sep <= radius_cut
    candidates = delve_df.loc[candidate_mask].copy()

    # -- exact sphgeom test on the (small) candidate set --
    outer_regions = [t.outer_sky_polygon for t in group.tracts]
    hit = np.zeros(len(candidates), dtype=bool)
    for i, (r, d) in enumerate(zip(candidates["ra"].to_numpy(), candidates["dec"].to_numpy())):
        center = sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(r, d))
        circ = sphgeom.Circle(center, sphgeom.Angle.fromDegrees(fov_radius_deg))
        hit[i] = any(not region.isDisjointFrom(circ) for region in outer_regions)

    return candidates.loc[hit].reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 3: per-band covered area / completeness via a RA/Dec grid
# --------------------------------------------------------------------------

def _points_in_group(ra, dec, group):
    """Vectorized point-in-any-tract-inner-box test."""
    ra_unwrapped = group.ra_center + (((ra - group.ra_center + 180.0) % 360.0) - 180.0)
    inside = np.zeros(ra.shape, dtype=bool)
    for ra_lo, ra_hi, dec_lo, dec_hi in group.inner_boxes:
        inside |= (
            (ra_unwrapped >= ra_lo) & (ra_unwrapped <= ra_hi)
            & (dec >= dec_lo) & (dec <= dec_hi)
        )
    return inside


def band_coverage(hit_df, group, band, fov_radius_deg=DECAM_FOV_RADIUS_DEG, grid_step_deg=0.02):
    """Step 3: covered area, completeness fraction, and mean depth for one band."""
    band_df = hit_df[hit_df["band"] == band]

    dec_centers = np.arange(group.dec_min + grid_step_deg / 2, group.dec_max, grid_step_deg)
    ra_centers = np.arange(group.ra_min + grid_step_deg / 2, group.ra_max, grid_step_deg)
    if len(dec_centers) == 0 or len(ra_centers) == 0:
        return dict(covered_area_deg2=0.0, group_area_pixelized_deg2=0.0,
                    completeness_frac=np.nan, mean_epochs=0.0)

    RA, DEC = np.meshgrid(ra_centers, dec_centers)
    ra_flat, dec_flat = RA.ravel(), DEC.ravel()
    cell_area_deg2 = grid_step_deg**2 * np.cos(np.radians(dec_flat))

    in_group = _points_in_group(ra_flat, dec_flat, group)
    group_area_pixelized_deg2 = cell_area_deg2[in_group].sum()

    if len(band_df) == 0:
        return dict(covered_area_deg2=0.0, group_area_pixelized_deg2=group_area_pixelized_deg2,
                     completeness_frac=0.0, mean_epochs=0.0)

    exp_ra = group.ra_center + (((band_df["ra"].to_numpy() - group.ra_center + 180.0) % 360.0) - 180.0)
    exp_dec = band_df["dec"].to_numpy()

    # epoch_count[j] = number of hit exposures covering grid cell j (only where it matters)
    pts_ra = ra_flat[in_group]
    pts_dec = dec_flat[in_group]
    epoch_count = np.zeros(pts_ra.shape[0], dtype=int)
    # chunk over exposures to bound memory (n_pixels x n_exposures matrix)
    chunk = max(1, int(2_000_000 / max(1, pts_ra.shape[0])))
    for start in range(0, exp_ra.shape[0], chunk):
        er = exp_ra[start:start + chunk]
        ed = exp_dec[start:start + chunk]
        sep = _angular_sep_deg(pts_ra[:, None], pts_dec[:, None], er[None, :], ed[None, :])
        epoch_count += (sep <= fov_radius_deg).sum(axis=1)

    covered = epoch_count > 0
    covered_area_deg2 = cell_area_deg2[in_group][covered].sum()
    completeness_frac = (
        covered_area_deg2 / group_area_pixelized_deg2 if group_area_pixelized_deg2 > 0 else np.nan
    )
    mean_epochs = float(epoch_count[covered].mean()) if covered.any() else 0.0

    return dict(
        covered_area_deg2=covered_area_deg2,
        group_area_pixelized_deg2=group_area_pixelized_deg2,
        completeness_frac=completeness_frac,
        mean_epochs=mean_epochs,
    )


# --------------------------------------------------------------------------
# Step 4: orchestration / report
# --------------------------------------------------------------------------

def summarize_tract_group(
    tract_ids,
    delve_csv,
    skymap=None,
    skymap_name="lsst_cells_v2",
    repo="main",
    bands=DEFAULT_BANDS,
    fov_radius_deg=DECAM_FOV_RADIUS_DEG,
    grid_step_deg=0.02,
    min_overlap_deg2=0.0,
):
    """Run the full scheme for one group of tracts and return a summary DataFrame.

    `tract_ids` accepts a plain list of tract ids, or the DataFrame/csv
    output of `find_tracts_in_range()` -- see `tract_ids_from_range_output`.
    `min_overlap_deg2` only applies in the latter case.
    """
    if skymap is None:
        skymap = get_skymap(repo=repo, skymap_name=skymap_name)

    group = build_tract_group(tract_ids, skymap, skymap_name=skymap_name, min_overlap_deg2=min_overlap_deg2)

    delve_df = delve_csv if isinstance(delve_csv, pd.DataFrame) else load_delve_csv(delve_csv, bands=bands)
    hit_df = match_inputs_to_group(delve_df, group, fov_radius_deg=fov_radius_deg)

    rows = []
    for band in bands:
        band_hits = hit_df[hit_df["band"] == band]
        cov = band_coverage(hit_df, group, band, fov_radius_deg=fov_radius_deg, grid_step_deg=grid_step_deg)
        rows.append(dict(
            band=band,
            n_inputs=len(band_hits),
            exptime_total_s=float(band_hits["exptime"].sum()) if len(band_hits) else 0.0,
            mean_psf_fwhm=float(band_hits["psf_fwhm"].mean()) if len(band_hits) else np.nan,
            mean_t_eff=float(band_hits["t_eff"].mean()) if len(band_hits) else np.nan,
            **cov,
        ))

    summary = pd.DataFrame(rows)
    summary.attrs["group"] = dict(
        tract_ids=group.tract_ids,
        skymap=group.skymap_name,
        area_deg2=group.area_deg2,
    )
    return summary


def _print_report(summary):
    group = summary.attrs["group"]
    print(f"Tract group ({group['skymap']}): {group['tract_ids']}")
    print(f"  union sky area (exact, from inner regions): {group['area_deg2']:.4f} deg^2\n")
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tract_source = parser.add_mutually_exclusive_group(required=True)
    tract_source.add_argument("--tracts", type=int, nargs="+", help="Explicit tract ids to group.")
    tract_source.add_argument("--tract-range-csv",
                               help="Path to a tract-list csv from find_tracts_in_range.py's --out.")
    parser.add_argument("--min-overlap", type=float, default=0.0,
                         help="With --tract-range-csv, drop tracts whose overlap_area_deg2 is <= this "
                              "(deg^2, default: 0.0 -- drops zero-area boundary touches only).")
    parser.add_argument("--skymap", default="lsst_cells_v2", help="Skymap name (default: lsst_cells_v2).")
    parser.add_argument("--repo", default="main", help="Butler repo/alias (default: main).")
    parser.add_argument("--delve-csv", required=True, help="Path to the DELVE input csv.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--fov-radius", type=float, default=DECAM_FOV_RADIUS_DEG,
                         help=f"DECam FoV radius in deg (default: {DECAM_FOV_RADIUS_DEG:.4f}).")
    parser.add_argument("--grid-step", type=float, default=0.02, help="RA/Dec grid step in deg (default: 0.02).")
    parser.add_argument("--out", default=None, help="Optional path to write the summary csv.")
    args = parser.parse_args()

    tract_ids = args.tracts if args.tracts is not None else args.tract_range_csv
    summary = summarize_tract_group(
        tract_ids, args.delve_csv, skymap_name=args.skymap, repo=args.repo,
        bands=args.bands, fov_radius_deg=args.fov_radius, grid_step_deg=args.grid_step,
        min_overlap_deg2=args.min_overlap,
    )
    _print_report(summary)
    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
