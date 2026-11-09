"""Sky map of a chunk's template_coadd patches, one panel per band.

  python map_template_coadds.py <chunk> <run timestamp> [out.png]

Patches with an assembleTemplateCoadd quantum but no template_coadd are drawn in
red; the quantum ran and wrote metadata but produced no coadd. Needs the LSST stack.
"""
import collections
import getpass
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lsst.daf.butler import Butler

REPO = "/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo"
SKYMAP = "lsst_cells_v2"
BANDS = "griz"

chunk, stamp = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else f"template_coadd_map_chunk{chunk}.png"
run = f"u/{getpass.getuser()}/DECam/endurance/template/chunk{chunk}/{stamp}"

butler = Butler(REPO, collections=[run, "skymaps"])
skymap = butler.get("skyMap", skymap=SKYMAP)


def patches(dataset_type):
    refs = butler.query_datasets(dataset_type, collections=run, limit=None, explain=False)
    return {(r.dataId["tract"], r.dataId["patch"], r.dataId["band"]) for r in refs}


have = patches("template_coadd")
ran = patches("assembleTemplateCoadd_metadata")
print(f"{len(have)} template_coadd patches, {len(ran)} assembleTemplateCoadd quanta")

tracts = {}


def centre(tract, patch):
    info = tracts.get(tract)
    if info is None:
        info = tracts[tract] = skymap[tract]
    pinfo = info.getPatchInfo(patch)
    sky = info.getWcs().pixelToSky(pinfo.getInnerBBox().getCenter())
    return sky.getRa().asDegrees(), sky.getDec().asDegrees()


coords = {}
for key in ran | have:
    coords[key] = centre(key[0], key[1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
counts = collections.OrderedDict()
for ax, band in zip(axes.flat, BANDS):
    got = [coords[k] for k in have if k[2] == band]
    miss = [coords[k] for k in ran - have if k[2] == band]
    counts[band] = (len(got), len(miss))
    if miss:
        ax.scatter(*zip(*miss), s=1, c="red", label=f"no coadd ({len(miss)})")
    if got:
        ax.scatter(*zip(*got), s=1, c="tab:blue", label=f"template_coadd ({len(got)})")
    ax.set_title(f"{band} band")
    ax.legend(markerscale=8, loc="upper right")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.invert_xaxis()
fig.suptitle(f"template_coadd patches, chunk {chunk}, run {stamp}")
fig.tight_layout()
fig.savefig(out, dpi=130)
for band, (g, m) in counts.items():
    print(f"{band}: {g} coadds, {m} patches with a quantum but no coadd")
print(f"wrote {out}")
