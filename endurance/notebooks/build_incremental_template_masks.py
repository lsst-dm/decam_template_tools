#!/usr/bin/env python
"""Build per-band HEALPix coverage maps from the incremental-template healsparse outputs.

Reads the sharded `INCREMENTAL_TEMPLATE_DIR` healsparse maps (same source as Step 3 of
`template_priority_masks.ipynb`) and, per band, unions all shards into a single coverage map giving
the fraction of fine HEALPix pixels with `n >= N_TEMPLATE_THRESHOLD` independent incremental
templates within each output-resolution pixel. Unlike the notebook, this does not apply the
Galactic-plane/dec/airmass cuts -- it only characterizes where incremental-template coverage already
exists, so the resulting maps can be reused later (e.g. to pick RA/Dec ranges or tracts still needing
templates) independent of any particular priority weighting.

Intended to be run as a batch job, e.g.:

    python build_incremental_template_masks.py \\
        --incremental-dir /home/e/ebellm/u/workspace/incremental_templates/template_tools/output_incremental_test \\
        --out-dir incremental_template_masks

Outputs, per band, under --out-dir:
    {band}_template_fraction.npy  float32 HEALPix map (RING, nside=NSIDE), fraction of fine pixels
                                   over threshold in each output pixel (0.0 where no coverage).
    {band}_template_mask.npy      bool HEALPix map, fraction > --exist-threshold.
and one `metadata.json` recording the parameters used to build them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import healpy as hp
import healsparse as hsp
import numpy as np
from tqdm import tqdm

BANDS = ["g", "r", "i", "z"]

INCREMENTAL_FILE_PATTERN = "8_*/incremental_template_inputs_8_*{band}_inputs.hsp"


def load_incremental_template_count(m, valid, vector_reduction):
    """Per-valid-pixel scalar template count, handling wide bit masks and plain vector maps."""
    values = m.get_values_pix(valid)
    if m.is_wide_mask_map:
        # Each pixel is a bank of individual bits (e.g. one per incremental processing run/visit);
        # the template count is the number of bits set, i.e. how many increments touched this pixel.
        packed = values.view(np.uint8).reshape(values.shape[0], -1)
        return np.unpackbits(packed, axis=1).sum(axis=1).astype(np.float64)
    if values.ndim > 1:
        # Plain (non-mask) multi-column map, e.g. counts split across incremental batches.
        return getattr(values, vector_reduction)(axis=-1).astype(np.float64)
    return values.astype(np.float64)


def load_incremental_template_fraction(
    band, nside_out, incremental_dir, file_pattern, count_key, n_template_threshold, vector_reduction
):
    """Fraction of nside_out pixels' fine sub-pixels with >= n_template_threshold incremental templates.

    The incremental template coverage is sharded across many per-tile (nside=8) files, so every
    matching shard must be read and unioned into one combined map -- reading only the first match
    silently drops coverage for all but one small tile of sky.
    """
    pattern = os.path.join(incremental_dir, file_pattern.format(band=band))
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No incremental template map found for band '{band}' matching {pattern}")

    thresholded = None
    n_valid_total = 0
    count_min, count_max = np.inf, -np.inf
    for path in tqdm(matches, desc=f"{band}-band shards"):
        m = hsp.HealSparseMap.read(path)
        if m.is_rec_array:
            if count_key is None:
                raise ValueError(f"{path} is a recarray map ({m.dtype.names}); set --count-key")
            m = m[count_key]

        if thresholded is None:
            thresholded = hsp.HealSparseMap.make_empty(
                m.nside_coverage, m.nside_sparse, dtype=np.float64, sentinel=hp.UNSEEN
            )

        valid = m.valid_pixels
        if valid.size == 0:
            continue
        counts = load_incremental_template_count(m, valid, vector_reduction)
        thresholded.update_values_pix(valid, (counts >= n_template_threshold).astype(np.float64))
        n_valid_total += valid.size
        count_min = min(count_min, counts.min())
        count_max = max(count_max, counts.max())

    print(
        f"  {band}: {len(matches)} shard files, {n_valid_total} total valid fine pixels, "
        f"count range=[{count_min:.0f}, {count_max:.0f}]"
    )

    frac = thresholded.generate_healpix_map(nside=nside_out, reduction="mean", nest=False)
    return np.where(frac == hp.UNSEEN, 0.0, frac)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--incremental-dir",
        default="/sdf/home/e/ebellm/u/workspace/incremental_templates/template_tools/output_incremental_test",
        help="INCREMENTAL_TEMPLATE_DIR containing the per-tile shard files (default: %(default)s)",
    )
    p.add_argument(
        "--file-pattern",
        default=INCREMENTAL_FILE_PATTERN,
        help="glob pattern for shard files within --incremental-dir, with a {band} placeholder "
        "(default: %(default)s)",
    )
    p.add_argument("--bands", nargs="+", default=BANDS, help="bands to process (default: %(default)s)")
    p.add_argument("--nside", type=int, default=64, help="output HEALPix nside, RING order (default: %(default)s)")
    p.add_argument(
        "--n-template-threshold",
        type=int,
        default=5,
        help="minimum independent incremental templates for a fine pixel to count as covered "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--exist-threshold",
        type=float,
        default=0.0,
        help="output-pixel coverage fraction above which a pixel is marked covered in the boolean "
        "mask (default: %(default)s)",
    )
    p.add_argument(
        "--count-key",
        default=None,
        help="field name to use if a shard map is a recarray rather than a plain count map "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--vector-reduction",
        default="sum",
        help="reduction applied via getattr(values, ...)(axis=-1) when a shard map stores more than "
        "one value per pixel and is not a wide bit mask (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir", default="incremental_template_masks", help="output directory (default: %(default)s)"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    fraction_maps = {}
    for band in args.bands:
        fraction_maps[band] = load_incremental_template_fraction(
            band,
            args.nside,
            args.incremental_dir,
            args.file_pattern,
            args.count_key,
            args.n_template_threshold,
            args.vector_reduction,
        )

    metadata = {
        "incremental_dir": args.incremental_dir,
        "file_pattern": args.file_pattern,
        "bands": args.bands,
        "nside": args.nside,
        "n_template_threshold": args.n_template_threshold,
        "exist_threshold": args.exist_threshold,
        "count_key": args.count_key,
        "vector_reduction": args.vector_reduction,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    for band in args.bands:
        frac = fraction_maps[band].astype(np.float32)
        mask = frac > args.exist_threshold

        frac_path = os.path.join(args.out_dir, f"{band}_template_fraction.npy")
        mask_path = os.path.join(args.out_dir, f"{band}_template_mask.npy")
        np.save(frac_path, frac)
        np.save(mask_path, mask)

        print(
            f"{band}: wrote {frac_path} and {mask_path} "
            f"({mask.mean():.2%} of sky already covered at nside={args.nside})"
        )

    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'metadata.json')}")


if __name__ == "__main__":
    sys.exit(main())
