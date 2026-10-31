"""Sky grid of processing chunks: 30 x 30 deg cells, RA columns outer, dec rows inner.

chunk 0 is RA 300-330, dec 0..30; chunk 1 is the next cell down in dec
(-30..0), and so on to dec -90..-60 (chunk 3), after which chunk 4 starts the
next RA column (330-360). Twelve RA columns x four dec rows = 48 chunks (0-47);
the last column is RA 270-300. Nothing above dec +30 is covered.

    python chunk_grid.py 5     # print the exposure constraint for chunk 5
    python chunk_grid.py       # list every chunk
    python chunk_grid.py --plot [out.png]   # map of which chunk is where
"""
import sys

import numpy as np

step = 30
# RA where chunk 0 starts. The existing u/<operator>/DECam/endurance/*/chunk0
# collections were built for RA 300-330, so moving this renumbers them.
ra0 = 300

ra_min, ra_max = np.arange(ra0, 360 + ra0 + 1, step=step) % 360, np.arange(ra0 + step, 360 + ra0 + 1 + step, step=step) % 360
dec_min, dec_max = np.arange(30 - step, -61 - step, step=-step), np.arange(30, -61, step=-step)

ras, decs = np.array([ra_min, ra_max]).T, np.array([dec_min, dec_max]).T

patches = [
    dict(patch_id=i, ra_min=float(r[0]), ra_max=float(r[1]), dec_min=float(d[0]), dec_max=float(d[1]))
    for i, (r, d) in enumerate((r, d) for r in ras for d in decs)
]


PAD = 1.0  # deg on the sky, added on every side of a chunk


def exposure_constraint(chunk, pad=PAD):
    """Registry `where` clause for the raws whose pointing is within `pad` of the chunk.

    The pad reaches exposures just outside the chunk whose detectors still land
    in it. Neighbouring chunks therefore share their edge exposures.
    """
    p = patches[chunk]
    # ra_max is reduced mod 360, so the 330-360 column arrives as ra_max == 0.
    ra_lo, ra_hi = p["ra_min"], (360.0 if p["ra_max"] == 0 else p["ra_max"])
    dec_lo, dec_hi = max(p["dec_min"] - pad, -90.0), min(p["dec_max"] + pad, 90.0)

    # A pad of `pad` on the sky is pad / cos(dec) in RA, widest at the most
    # polar edge of the padded band; at the pole that covers every RA.
    cos_dec = np.cos(np.radians(max(abs(dec_lo), abs(dec_hi))))
    ra_pad = pad / cos_dec if cos_dec > 1e-9 else 360.0
    lo, hi = ra_lo - ra_pad, ra_hi + ra_pad

    dec_clause = f"exposure.tracking_dec >= {dec_lo:g} AND exposure.tracking_dec < {dec_hi:g}"
    if hi - lo >= 360:
        return dec_clause
    if lo < 0:  # wraps below RA 0
        ra_clause = f"(exposure.tracking_ra >= {lo + 360:.4f} OR exposure.tracking_ra < {hi:.4f})"
    elif hi > 360:  # wraps above RA 360
        ra_clause = f"(exposure.tracking_ra >= {lo:.4f} OR exposure.tracking_ra < {hi - 360:.4f})"
    else:
        ra_clause = f"exposure.tracking_ra >= {lo:.4f} AND exposure.tracking_ra < {hi:.4f}"
    return f"{ra_clause} AND {dec_clause}"


def plot_grid(out="chunk_grid.png"):
    """Map of chunk ids on the RA/dec plane (RA increases to the right, wrapping at 360)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5.5))
    cmap = plt.get_cmap("tab20")
    for p in patches:
        lo, hi = p["ra_min"], (360.0 if p["ra_max"] == 0 else p["ra_max"])
        col = p["patch_id"] // len(dec_min)
        ax.add_patch(plt.Rectangle((lo, p["dec_min"]), hi - lo, p["dec_max"] - p["dec_min"],
                                   facecolor=cmap(col % 20), alpha=0.5, edgecolor="k"))
        ax.text((lo + hi) / 2, (p["dec_min"] + p["dec_max"]) / 2, str(p["patch_id"]),
                ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 30)
    ax.set_xticks(range(0, 361, step))
    ax.set_yticks(range(-90, 31, step))
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Processing chunks: chunk 0 = RA 300-330, dec 0..30; ids step down in dec, then east in RA")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot_grid(*sys.argv[2:3])
    elif len(sys.argv) > 1:
        print(exposure_constraint(int(sys.argv[1])))
    else:
        print(f"{len(patches)} chunks, {step} deg x {step} deg")
        for p in patches:
            print(p["patch_id"], exposure_constraint(p["patch_id"]))
