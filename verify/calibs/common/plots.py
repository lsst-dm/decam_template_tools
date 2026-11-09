"""Diagnostic figures for the calibration QA tables. Every function writes one file.

Focal-plane positions come from calio's layout (camera via the Butler, or
amp_geometry.yaml); without a layout, detectors are drawn on a plain grid by id.
The backend is left to the caller (the run_*_qa scripts select Agg; notebooks keep
their inline backend).
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon

from .stats import mad_sigma

# One-hue sequential ramp (light -> dark) and reserved status colours. Status is never
# shown by colour alone: flagged CCDs also get a hatch and a W/F label.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "calqa_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SERIES = "#2a78d6"
STATUS = {"warn": "#fab219", "fail": "#d03b3b"}
MUTED = "#8a8a86"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _grid_corners(detectors, ncols=8):
    out = {}
    for i, det in enumerate(sorted(detectors)):
        r, c = divmod(i, ncols)
        out[det] = [[c, -2 * r], [c + 0.9, -2 * r], [c + 0.9, -2 * r + 1.8], [c, -2 * r + 1.8]]
    return out


def _worst_flag(flags):
    for f in ("fail", "warn"):
        if f in set(flags):
            return f
    return None


def focal_plane_mosaic(df, metric, layout, path, title=None):
    """CCDs coloured by the median over amps of `metric` (one calib per call)."""
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    per_det = sub.groupby("detector").agg(value=("value", "median"),
                                          flag=("flag", _worst_flag))
    corners = ({d: e["fp_corners"] for d, e in layout.items()} if layout
               else _grid_corners(per_det.index))
    values = per_det["value"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = matplotlib.colors.Normalize(vmin, vmax)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    for det, pts in corners.items():
        pts = np.asarray(pts)
        if det in per_det.index and np.isfinite(per_det.at[det, "value"]):
            face = SEQUENTIAL(norm(per_det.at[det, "value"]))
        else:
            face = "none"
        flag = per_det.at[det, "flag"] if det in per_det.index else None
        ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor="white", lw=1))
        if flag:
            ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=STATUS[flag],
                                 hatch="///", lw=2))
        cx, cy = pts.mean(axis=0)
        label = f"{det}" + (f"\n{flag[0].upper()}" if flag else "")
        dark = face != "none" and np.dot(face[:3], [0.299, 0.587, 0.114]) < 0.5
        ax.text(cx, cy, label, ha="center", va="center", fontsize=6,
                color="white" if dark else "#1a1a19")
    allpts = np.concatenate([np.asarray(p) for p in corners.values()])
    pad = 0.03 * np.ptp(allpts, axis=0)
    ax.set_xlim(allpts[:, 0].min() - pad[0], allpts[:, 0].max() + pad[0])
    ax.set_ylim(allpts[:, 1].min() - pad[1], allpts[:, 1].max() + pad[1])
    ax.set_aspect("equal")
    ax.set_xlabel("focal plane x [mm]" if layout else "")
    ax.set_ylabel("focal plane y [mm]" if layout else "")
    unit = sub["unit"].iloc[0]
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL), ax=ax, shrink=0.8,
                 label=f"{metric} [{unit}]" if unit else metric)
    ax.legend(handles=[Patch(facecolor="none", edgecolor=STATUS[f], hatch="///",
                             label=f"{f} ({f[0].upper()})") for f in ("warn", "fail")],
              loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title or metric)
    return _save(fig, path)


def metric_histogram(df, metric, path, title=None):
    """Distribution over (detector, amp) with the thresholds when they are uniform."""
    sub = df[(df["metric"] == metric) & np.isfinite(df["value"])]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub["value"], bins=min(60, max(10, len(sub) // 3)), histtype="step",
            color=SERIES, lw=2)
    ax.grid(alpha=0.2)
    if sub["threshold"].nunique() == 1 and sub["threshold"].iloc[0]:
        for part in sub["threshold"].iloc[0].split():
            name, rng = part.split(":")
            for edge in rng.strip("[]").split(","):
                if "inf" not in edge:
                    ax.axvline(float(edge), color=STATUS[name],
                               ls="--" if name == "warn" else "-", lw=1.5, label=name)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False)
    unit = sub["unit"].iloc[0]
    ax.set_xlabel(f"{metric} [{unit}]" if unit else metric)
    ax.set_ylabel("N (detector, amp)")
    counts = sub["flag"].value_counts().to_dict()
    ax.set_title(title or metric)
    ax.text(0.98, 0.95, "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    return _save(fig, path)


def trend_plot(trend, value_col, path, x_col="exposure", title=None, ylabel=None):
    """Per-amp small multiples of `value_col` minus each detector's median, one thin
    muted line per (detector, calib) and the median over detectors as points on top.

    The y range is clipped to +-10 MAD-sigma so one bad input does not flatten the rest;
    clipped points are drawn as triangles on the edge and counted in the panel.
    """
    if trend is None or trend.empty or trend[value_col].isna().all():
        return None
    amps = sorted(trend["amp"].unique())
    fig, axes = plt.subplots(len(amps), 1, figsize=(8, 2.6 * len(amps)), sharex=True,
                             squeeze=False)
    for ax, amp in zip(axes[:, 0], amps):
        sub = trend[(trend["amp"] == amp) & trend[value_col].notna()].copy()
        keys = ["detector", "calib_id"]
        sub["delta"] = sub[value_col] - sub.groupby(keys)[value_col].transform("median")
        delta = sub["delta"].to_numpy(dtype=float)
        lim = 10 * mad_sigma(delta)
        lim = lim if np.isfinite(lim) and lim > 0 else np.nanmax(np.abs(delta)) or 1.0
        for _, g in sub.groupby(keys):
            g = g.sort_values(x_col)
            ax.plot(g[x_col], g["delta"].clip(-lim, lim), color=MUTED, lw=0.5, alpha=0.4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, (_, g) in enumerate(sub.groupby("calib_id")):
                med = g.groupby(x_col)["delta"].median()
                ax.plot(med.index, med.to_numpy().clip(-lim, lim), ".", color=SERIES, ms=4,
                        label="median over detectors" if i == 0 else None)
        off = sub[np.abs(sub["delta"]) > lim]
        if len(off):
            ax.plot(off[x_col], np.sign(off["delta"]) * lim * 0.97, "^", color=STATUS["fail"],
                    ms=8, ls="none", label="off scale")
            ax.text(0.01, 0.95, f"{len(off)} off-scale points, max |delta| "
                    f"{np.abs(off['delta']).max():.4g}", transform=ax.transAxes, va="top",
                    fontsize=8, color="#1a1a19")
        ax.set_ylim(-1.05 * lim, 1.05 * lim)
        ax.set_ylabel(f"amp {amp}")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel(x_col)
    axes[0, 0].set_title(title or f"{ylabel or value_col} minus per-detector median")
    return _save(fig, path)


def _zscale(values):
    from astropy.visualization import ZScaleInterval

    return ZScaleInterval().get_limits(values)


def thumbnail(array, path, title=None):
    """Grey-scale image with a zscale stretch (bright columns don't set the range)."""
    if array is None:
        return None
    a = np.asarray(array, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    lo, hi = _zscale(finite)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(a, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "")
    return _save(fig, path)


def focal_plane_image(binned, layout, path, title=None):
    """Thumbnail mosaic from per-detector binned images {detector: 2-D array}.

    Each CCD is placed at its focal-plane box; pixel x/y are mapped onto the box axes
    (transposed/flipped as the corners require). Fallback when cp_pipe's
    <type>Mosaic64 is not available.
    """
    if not binned:
        return None
    corners = ({d: e["fp_corners"] for d, e in layout.items()} if layout
               else _grid_corners(binned))
    finite = np.concatenate([b[np.isfinite(b)].ravel() for b in binned.values()])
    if finite.size == 0:
        return None
    lo, hi = _zscale(finite)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for det, img in binned.items():
        if det not in corners:
            continue
        p0, px, _, py = np.asarray(corners[det])  # pixel (0,0), (w,0), (w,h), (0,h)
        img = np.asarray(img, dtype=float)
        dx, dy = px - p0, py - p0
        if abs(dx[0]) < abs(dx[1]):  # pixel x runs along focal-plane y
            img = img.T
            dx, dy = dy, dx
        if dx[0] < 0:
            img = img[:, ::-1]
        if dy[1] < 0:
            img = img[::-1]
        xs = sorted([p0[0], p0[0] + dx[0]])
        ys = sorted([p0[1], p0[1] + dy[1]])
        ax.imshow(img, origin="lower", extent=[xs[0], xs[1], ys[0], ys[1]], cmap="gray",
                  vmin=lo, vmax=hi, interpolation="nearest")
    allpts = np.concatenate([np.asarray(p) for p in corners.values()])
    ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
    ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
    ax.set_aspect("equal")
    ax.set_title(title or "")
    return _save(fig, path)


def standard_figures(df, trend, layout, out_dir, prefix, key_metrics, thumbs=None,
                     binned=None):
    """Mosaics of `key_metrics` and thumbnails per calib, histograms per metric, trends.

    thumbs: {label: 2-D array} (e.g. cp_pipe mosaics); binned: {label: {det: array}}.
    """
    fig_dir = Path(out_dir) / "figures"
    written = []
    for label, g in df.groupby(["filter", "validity_begin", "collection"], dropna=False):
        tag = "_".join(str(x)[:10] for x in label[:2] if x) or "all"
        for metric in key_metrics:
            written.append(focal_plane_mosaic(
                g, metric, layout, fig_dir / f"{prefix}_fp_{metric}_{tag}.png",
                title=f"{prefix} {metric} {tag}"))
    for metric in sorted(df["metric"].unique()):
        written.append(metric_histogram(df, metric, fig_dir / f"{prefix}_hist_{metric}.png"))
    if trend is not None and len(trend):
        for col in [c for c in trend.columns if c.endswith(("_adu", "_e", "_ratio", "_frac"))]:
            written.append(trend_plot(trend, col, fig_dir / f"{prefix}_trend_{col}.png",
                                      x_col="mjd" if trend["mjd"].notna().any() else "exposure"))
    for label, array in (thumbs or {}).items():
        written.append(thumbnail(array, fig_dir / f"{prefix}_thumb_{label}.png",
                                 title=f"{prefix} {label}"))
    for label, per_det in (binned or {}).items():
        written.append(focal_plane_image(per_det, layout,
                                         fig_dir / f"{prefix}_thumb_{label}.png",
                                         title=f"{prefix} {label}"))
    written = [w for w in written if w is not None]
    print(f"Wrote {len(written)} figures to {fig_dir}")
    return written
