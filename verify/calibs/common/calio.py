"""Calibration I/O. This is the only module that talks to the Butler or to FITS files.

Metric code sees CalibFrame/AmpGeom only: an image array, a boolean bad-pixel array, and
per-amp geometry/gain. Two sources give the same interface:

ButlerSource(repo, collection)   primary; `lsst.daf.butler` is imported lazily
FitsSource(fits_dir)             fallback; astropy only, no LSST stack needed

Where the amp geometry and gain come from (nothing DECam-specific is hard-coded):
- Butler: the exposure's own Detector, i.e. the obs_decam camera. Assembled frames use
  amp.getBBox(); raws use getRawDataBBox()/getRawSerialOverscanBBox(). Gain, read
  noise and saturation are amp.getGain()/getReadNoise()/getSaturation().
- FITS: DATASEC<amp>/BIASSEC<amp>/GAIN<amp>/RDNOISE<amp>/SATURAT<amp> headers (the DECam
  raw convention, amps named by the key suffix). Frames without those headers (anything
  assembled by ISR) need an amp_geometry.yaml in the FITS directory, written from the
  Butler camera once with dump_amp_geometry().

FITS directory layout:
    <fits_dir>/<dataset_type>/*.fits           combined calibs, e.g. bias/
    <fits_dir>/<dataset_type>_inputs/*.fits    frames that went into them (optional)
    <fits_dir>/defects/*.fits                  bad pixel masks, image (nonzero = bad)
                                               or table with x0, y0, width, height
    <fits_dir>/amp_geometry.yaml               optional, see above
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Pixels in these mask planes are ignored by every metric (plus non-finite pixels).
MASK_PLANES = ("BAD", "SAT", "NO_DATA", "SUSPECT", "UNMASKEDNAN")

DEFAULT_CURATED_COLLECTION = "DECam/calib/curated"
DEFAULT_CAMERA_COLLECTION = "DECam/calib/curated/unbounded"
DEFAULT_RAW_COLLECTION = "DECam/raw/all"


@dataclass
class AmpGeom:
    """One amplifier. Boxes are (y0, y1, x0, x1), half-open indices into the image array."""

    name: str
    bbox: tuple
    gain: float = np.nan  # e-/ADU
    read_noise: float = np.nan  # e-
    saturation: float = np.nan  # ADU
    overscan: tuple | None = None  # only for raw (untrimmed) frames
    flip_x: bool = False  # raw readout order vs assembled orientation
    flip_y: bool = False

    @property
    def slices(self):
        y0, y1, x0, x1 = self.bbox
        return np.s_[y0:y1, x0:x1]

    @property
    def overscan_slices(self):
        if self.overscan is None:
            return None
        y0, y1, x0, x1 = self.overscan
        return np.s_[y0:y1, x0:x1]


@dataclass
class CalibInfo:
    """Identity of one dataset. `handle` is a DatasetRef or a path; metrics never use it."""

    collection: str
    dataset_type: str
    detector: int
    filter: str = ""
    calib_id: str = ""
    run: str = ""
    validity_begin: str = ""
    validity_end: str = ""
    exposure: int | None = None
    mjd: float = np.nan
    handle: object = None
    extra: dict = field(default_factory=dict)


@dataclass
class CalibFrame:
    image: np.ndarray
    bad: np.ndarray
    amps: list
    metadata: dict
    info: CalibInfo
    is_raw: bool = False

    def amp(self, name):
        return next(a for a in self.amps if a.name == name)

    def amp_pixels(self, amp):
        """(image, bad) of one amp's data region in assembled orientation (views)."""
        if isinstance(amp, str):
            amp = self.amp(amp)
        image, bad = self.image[amp.slices], self.bad[amp.slices]
        if amp.flip_x:
            image, bad = image[:, ::-1], bad[:, ::-1]
        if amp.flip_y:
            image, bad = image[::-1], bad[::-1]
        return image, bad


def parse_section(text):
    """IRAF section '[x1:x2,y1:y2]' (1-based, inclusive, either order) -> (y0, y1, x0, x1)."""
    m = re.fullmatch(r"\s*\[\s*(\d+):(\d+)\s*,\s*(\d+):(\d+)\s*\]\s*", str(text))
    if not m:
        raise ValueError(f"not an IRAF section: {text!r}")
    x1, x2, y1, y2 = (int(v) for v in m.groups())
    return min(y1, y2) - 1, max(y1, y2), min(x1, x2) - 1, max(x1, x2)


def _box(bbox):
    """lsst.geom.Box2I -> (y0, y1, x0, x1)."""
    return bbox.getMinY(), bbox.getMaxY() + 1, bbox.getMinX(), bbox.getMaxX() + 1


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def amps_from_header(header):
    """AmpGeom list from DATASEC<amp> etc. headers, or [] if there are none."""
    amps = []
    for key in sorted(header):
        m = re.fullmatch(r"DATASEC(\w+)", key)
        if not m:
            continue
        s = m.group(1)
        overscan = header.get(f"BIASSEC{s}")
        amps.append(AmpGeom(
            name=s,
            bbox=parse_section(header[key]),
            gain=_float(header.get(f"GAIN{s}")),
            read_noise=_float(header.get(f"RDNOISE{s}")),
            saturation=_float(header.get(f"SATURAT{s}")),
            overscan=parse_section(overscan) if overscan else None,
        ))
    return amps


def amps_from_layout(entry):
    """AmpGeom list from one detector entry of amp_geometry.yaml (assembled frames)."""
    return [AmpGeom(name=a["name"], bbox=tuple(a["bbox"]), gain=_float(a.get("gain")),
                    read_noise=_float(a.get("read_noise")),
                    saturation=_float(a.get("saturation")))
            for a in entry["amps"]]


def amps_from_detector(detector, raw=False):
    amps = []
    for amp in detector:
        amps.append(AmpGeom(
            name=amp.getName(),
            bbox=_box(amp.getRawDataBBox() if raw else amp.getBBox()),
            gain=float(amp.getGain()),
            read_noise=float(amp.getReadNoise()),
            saturation=float(amp.getSaturation()),
            overscan=_box(amp.getRawSerialOverscanBBox()) if raw else None,
            flip_x=bool(amp.getRawFlipX()) if raw else False,
            flip_y=bool(amp.getRawFlipY()) if raw else False,
        ))
    return amps


def select_detectors(layout, requested=None, include_wavefront=False):
    """Detector ids to process: `requested` if given, else the science CCDs in `layout`.

    Returns None (= everything found) when neither is available.
    """
    if requested:
        return sorted(requested)
    if not layout:
        return None
    keep = {"SCIENCE"} | ({"WAVEFRONT", "GUIDER", "FOCUS"} if include_wavefront else set())
    return sorted(int(d) for d, e in layout.items() if e.get("type", "SCIENCE") in keep)


def _window_key(timespan):
    """Hashable (begin, end) ISO strings of a Timespan; None ends stay None."""
    return tuple(None if t is None else t.isot for t in (timespan.begin, timespan.end))


def merge_calib_entries(certified, in_runs):
    """[(ref, timespan or None)]: one entry per (dataset, validity window).

    certified: (ref, timespan) pairs from CALIBRATION collections; a dataset certified in
    several windows gets one entry per window, and the same certification seen through
    two collections counts once. in_runs: refs from RUN collections, kept (with no
    window) only for datasets that are not certified anywhere in the collection.
    """
    out, seen, certified_ids = [], set(), set()
    for ref, timespan in certified:
        key = (ref.id, _window_key(timespan))
        if key not in seen:
            seen.add(key)
            certified_ids.add(ref.id)
            out.append((ref, timespan))
    for ref in in_runs:
        if ref.id not in certified_ids and (ref.id, None) not in seen:
            seen.add((ref.id, None))
            out.append((ref, None))
    return out


def provenance_inputs(metadata, dataset_type):
    """{run: {dataset id}} of the `dataset_type` inputs listed in a calib's Butler
    provenance header (LSST BUTLER INPUT <n> DATASETTYPE / RUN / ID)."""
    out = {}
    for k, v in metadata.items():
        m = re.fullmatch(r"LSST BUTLER INPUT (\d+) DATASETTYPE", k)
        if m and v == dataset_type:
            run = metadata.get(f"LSST BUTLER INPUT {m.group(1)} RUN")
            ref_id = metadata.get(f"LSST BUTLER INPUT {m.group(1)} ID")
            if run and ref_id:
                out.setdefault(run, set()).add(str(ref_id))
    return out


def _match_filter(info, filters):
    return not filters or info.filter in filters or info.extra.get("physical_filter") in filters


# ---------------------------------------------------------------------------- Butler


def frame_from_exposure(exp, info, raw=False, mask_planes=MASK_PLANES):
    image = np.array(exp.image.array, dtype=np.float32)
    mask = exp.mask
    planes = mask.getMaskPlaneDict()
    bits = 0
    for p in mask_planes:
        if p in planes:
            bits |= mask.getPlaneBitMask(p)
    bad = ((mask.array & bits) != 0) | ~np.isfinite(image)
    return CalibFrame(image=image, bad=bad, amps=amps_from_detector(exp.getDetector(), raw),
                      metadata=exp.getMetadata().toDict(), info=info, is_raw=raw)


class ButlerSource:
    kind = "butler"

    def __init__(self, repo, collection, curated_collection=DEFAULT_CURATED_COLLECTION,
                 camera_collection=DEFAULT_CAMERA_COLLECTION,
                 raw_collection=DEFAULT_RAW_COLLECTION, instrument="DECam",
                 mask_planes=MASK_PLANES):
        from lsst.daf.butler import Butler

        self.butler = Butler.from_config(repo)
        self.repo = repo
        self.collection = collection
        self.curated_collection = curated_collection
        self.camera_collection = camera_collection
        self.raw_collection = raw_collection
        self.instrument = instrument
        self.mask_planes = tuple(mask_planes)
        self._layout = None
        self._defect_assoc = None

    def _info(self, ref, timespan=None, collection=None):
        data_id = dict(ref.dataId.mapping)
        info = CalibInfo(
            collection=collection or self.collection,
            dataset_type=ref.datasetType.name,
            detector=int(data_id["detector"]),
            filter=str(data_id.get("band") or data_id.get("physical_filter") or ""),
            calib_id=str(ref.id),
            run=ref.run,
            exposure=data_id.get("exposure"),
            handle=ref,
        )
        if "physical_filter" in data_id:
            info.extra["physical_filter"] = data_id["physical_filter"]
        if timespan is not None:
            info.extra["timespan"] = timespan
            info.validity_begin = timespan.begin.isot if timespan.begin is not None else ""
            info.validity_end = timespan.end.isot if timespan.end is not None else ""
        if ref.dataId.hasRecords() and "exposure" in data_id:
            rec = ref.dataId.records["exposure"]
            if rec is not None and rec.timespan is not None:
                info.mjd = float(rec.timespan.begin.mjd)
        return info

    def list_calibs(self, dataset_type, detectors=None, filters=None):
        """Every `dataset_type` dataset in the collection (CALIBRATION, RUN or CHAINED),
        one CalibInfo per (dataset, validity window); see merge_calib_entries.

        CALIBRATION collections give validity ranges; RUNs leave them blank.
        """
        from lsst.daf.butler import CollectionType

        leaves = self.butler.collections.query(self.collection, flatten_chains=True)
        certified, in_runs = [], []
        for leaf in leaves:
            ctype = self.butler.collections.get_info(leaf).type
            if ctype == CollectionType.CALIBRATION:
                certified.extend((assoc.ref, assoc.timespan) for assoc in
                                 self.butler.registry.queryDatasetAssociations(
                                     dataset_type, collections=[leaf]))
            elif ctype == CollectionType.RUN:
                in_runs.extend(self.butler.registry.queryDatasets(dataset_type, collections=leaf))
        infos = [self._info(ref, ts) for ref, ts in merge_calib_entries(certified, in_runs)]
        infos = [i for i in infos if (detectors is None or i.detector in detectors)
                 and _match_filter(i, filters)]
        return sorted(infos, key=lambda i: (i.detector, i.filter, i.validity_begin, i.run))

    def _where(self, info):
        where = f"instrument='{self.instrument}' AND detector={info.detector}"
        if info.extra.get("physical_filter"):
            where += f" AND physical_filter='{info.extra['physical_filter']}'"
        return where

    def inputs(self, info, dataset_type):
        """The `dataset_type` frames in the calib's own RUN for the same detector/filter.

        A calib made by a RUN resumed with --skip-existing-in has its inputs in the earlier
        RUN; when its own RUN has none, they are found from its provenance header instead.
        """
        refs = list(self.butler.registry.queryDatasets(
            dataset_type, collections=info.run, where=self._where(info)).expanded())
        if not refs:
            for run, ids in provenance_inputs(self.load_metadata(info), dataset_type).items():
                refs.extend(r for r in self.butler.registry.queryDatasets(
                    dataset_type, collections=run, where=self._where(info)).expanded()
                    if str(r.id) in ids)
        out = [self._info(ref, collection=ref.run) for ref in refs]
        return sorted(out, key=lambda i: (i.exposure or 0))

    def raw_inputs(self, info, observation_type, limit=None):
        """Raws of `observation_type` inside the calib's validity range (fallback only)."""
        timespan = info.extra.get("timespan")
        if timespan is None:
            return []
        where = (self._where(info) + f" AND exposure.observation_type='{observation_type}'"
                 " AND exposure.timespan OVERLAPS ts")
        refs = self.butler.registry.queryDatasets(
            "raw", collections=self.raw_collection, where=where,
            bind={"ts": timespan}).expanded()
        out = sorted((self._info(r, collection=self.raw_collection) for r in refs),
                     key=lambda i: (i.exposure or 0))
        return out[:limit] if limit else out

    def frames_in(self, info, dataset_type, collection):
        """`dataset_type` frames from another collection for the same detector/filter,
        inside the calib's validity range when it has one (e.g. held-out science frames)."""
        where, bind = self._where(info), {}
        timespan = info.extra.get("timespan")
        if timespan is not None:
            where += " AND exposure.timespan OVERLAPS ts"
            bind = {"ts": timespan}
        refs = self.butler.registry.queryDatasets(
            dataset_type, collections=collection, where=where, bind=bind).expanded()
        out = [self._info(ref, collection=collection) for ref in refs]
        return sorted(out, key=lambda i: (i.exposure or 0))

    def load(self, info, mask_planes=None):
        """CalibFrame for `info`; mask_planes overrides the source's planes for this frame
        (e.g. adding DETECTED for science exposures)."""
        exp = self.butler.get(info.handle)
        return frame_from_exposure(exp, info, raw=info.dataset_type == "raw",
                                   mask_planes=mask_planes or self.mask_planes)

    def load_metadata(self, info):
        return self.butler.get(info.handle.makeComponentRef("metadata")).toDict()

    def defects_mask(self, info, shape):
        """Union of the curated defects whose validity overlaps the calib's, as a boolean
        array; None if there are none. Without a validity range (RUN input) every
        curated defect set for the detector is used."""
        if self._defect_assoc is None:
            self._defect_assoc = list(self.butler.registry.queryDatasetAssociations(
                "defects", collections=[self.curated_collection]))
        timespan = info.extra.get("timespan")
        refs = [a.ref for a in self._defect_assoc
                if a.ref.dataId["detector"] == info.detector
                and (timespan is None or a.timespan is None or a.timespan.overlaps(timespan))]
        if not refs:
            return None
        mask = np.zeros(shape, dtype=bool)
        for ref in refs:
            for defect in self.butler.get(ref):
                y0, y1, x0, x1 = _box(defect.getBBox())
                mask[y0:y1, x0:x1] = True
        return mask

    def focal_plane_layout(self):
        """{detector: {name, type, shape, fp_corners, amps}} from the camera, in mm."""
        if self._layout is None:
            import lsst.geom as geom
            from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

            camera = self.butler.get("camera", instrument=self.instrument,
                                     collections=self.camera_collection)
            layout = {}
            for det in camera:
                bbox = det.getBBox()
                w, h = bbox.getWidth(), bbox.getHeight()
                transform = det.getTransform(PIXELS, FOCAL_PLANE)
                corners = [transform.applyForward(geom.Point2D(x, y))
                           for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
                layout[int(det.getId())] = dict(
                    name=det.getName(), type=det.getType().name, shape=[h, w],
                    fp_corners=[[float(p.getX()), float(p.getY())] for p in corners],
                    amps=[dict(name=a.name, bbox=list(a.bbox), gain=a.gain,
                               read_noise=a.read_noise, saturation=a.saturation)
                          for a in amps_from_detector(det)],
                )
            self._layout = layout
        return self._layout

    def thumbnail(self, info):
        """cp_pipe's focal-plane mosaic (<type>Mosaic64) from the calib's RUN, or None."""
        try:
            refs = list(self.butler.registry.queryDatasets(
                f"{info.dataset_type}Mosaic64", collections=info.run))
            return np.array(self.butler.get(refs[0]).array) if refs else None
        except Exception:
            return None


def dump_amp_geometry(repo, path, camera_collection=DEFAULT_CAMERA_COLLECTION):
    """Write amp_geometry.yaml for FitsSource from the Butler camera."""
    import yaml

    source = ButlerSource(repo, camera_collection, camera_collection=camera_collection)
    with open(path, "w") as f:
        yaml.safe_dump({"detectors": source.focal_plane_layout()}, f, sort_keys=False)


# ------------------------------------------------------------------------------ FITS


def _header_dict(*headers):
    out = {}
    for h in headers:
        for k in h:
            if k and k not in ("COMMENT", "HISTORY"):
                out[k] = h[k]
    return out


def _first(header, *keys):
    for k in keys:
        if header.get(k) not in (None, ""):
            return header[k]
    return None


class FitsSource:
    kind = "fits"

    def __init__(self, fits_dir, mask_planes=MASK_PLANES):
        import yaml

        self.root = Path(fits_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"--fits-dir {fits_dir} is not a directory")
        self.collection = str(self.root)
        self.mask_planes = tuple(mask_planes)
        geom_path = self.root / "amp_geometry.yaml"
        self._layout = None
        if geom_path.exists():
            with open(geom_path) as f:
                self._layout = {int(k): v for k, v in yaml.safe_load(f)["detectors"].items()}

    def _read_header(self, path):
        from astropy.io import fits

        with fits.open(path) as hdul:
            headers = [hdul[0].header]
            if len(hdul) > 1 and hdul[0].data is None:
                headers.append(hdul[1].header)
            return _header_dict(*headers)

    def _info(self, path, dataset_type):
        hdr = self._read_header(path)
        det = _first(hdr, "CCDNUM", "LSST BUTLER DATAID DETECTOR", "DETECTOR")
        if det is None:
            raise ValueError(f"{path}: no CCDNUM/DETECTOR header")
        physical = _first(hdr, "LSST BUTLER DATAID PHYSICAL_FILTER", "FILTER") or ""
        band = _first(hdr, "LSST BUTLER DATAID BAND") or (str(physical).split() or [""])[0]
        info = CalibInfo(
            collection=self.collection, dataset_type=dataset_type, detector=int(det),
            filter=str(band), calib_id=path.name, run=str(path.parent),
            validity_begin=str(_first(hdr, "CALIB_VALID_BEGIN") or ""),
            validity_end=str(_first(hdr, "CALIB_VALID_END") or ""),
            exposure=_first(hdr, "EXPNUM", "LSST BUTLER DATAID EXPOSURE"),
            mjd=_float(_first(hdr, "MJD-OBS")), handle=path,
        )
        if physical:
            info.extra["physical_filter"] = str(physical)
        return info

    def _scan(self, subdir, dataset_type):
        d = self.root / subdir
        if not d.is_dir():
            return []
        return [self._info(p, dataset_type) for p in sorted(d.glob("*.fits*"))]

    def list_calibs(self, dataset_type, detectors=None, filters=None):
        infos = [i for i in self._scan(dataset_type, dataset_type)
                 if (detectors is None or i.detector in detectors) and _match_filter(i, filters)]
        return sorted(infos, key=lambda i: (i.detector, i.filter, i.calib_id))

    def inputs(self, info, dataset_type):
        out = [i for i in self._scan(f"{info.dataset_type}_inputs", dataset_type)
               if i.detector == info.detector and (not info.filter or i.filter == info.filter)]
        return sorted(out, key=lambda i: (i.exposure or 0, i.calib_id))

    def raw_inputs(self, info, observation_type, limit=None):
        return []  # FITS inputs are taken as they are, raw or not

    def frames_in(self, info, dataset_type, collection):
        """FITS frames in directory `collection` (absolute, or under --fits-dir) for the
        same detector and filter."""
        d = Path(collection)
        d = d if d.is_absolute() else self.root / d
        out = [i for i in (self._info(p, dataset_type) for p in sorted(d.glob("*.fits*")))
               if i.detector == info.detector and (not info.filter or i.filter == info.filter)]
        return sorted(out, key=lambda i: (i.exposure or 0, i.calib_id))

    def load(self, info, mask_planes=None):
        from astropy.io import fits

        with fits.open(info.handle) as hdul:
            names = [h.name for h in hdul]
            if "IMAGE" in names:
                img_hdu = hdul["IMAGE"]
            else:
                img_hdu = next(h for h in hdul if h.data is not None and h.data.ndim == 2)
            image = np.array(img_hdu.data, dtype=np.float32)
            header = _header_dict(hdul[0].header, img_hdu.header)
            bad = ~np.isfinite(image)
            if "MASK" in names:
                mhdr = hdul["MASK"].header
                bits = 0
                for p in mask_planes or self.mask_planes:
                    if f"MP_{p}" in mhdr:
                        bits |= 1 << int(mhdr[f"MP_{p}"])
                bad |= (np.asarray(hdul["MASK"].data).astype(np.int64) & bits) != 0
            if "BPM" in names:
                bad |= np.asarray(hdul["BPM"].data) != 0
        amps = amps_from_header(header)
        is_raw = any(a.overscan is not None for a in amps)
        if not amps and self._layout and info.detector in self._layout:
            amps = amps_from_layout(self._layout[info.detector])
        if not amps:
            raise ValueError(
                f"{info.handle}: no DATASEC<amp> headers and no amp_geometry.yaml in "
                f"{self.root}; write one with calio.dump_amp_geometry(repo, path)")
        return CalibFrame(image=image, bad=bad, amps=amps, metadata=header, info=info,
                          is_raw=is_raw)

    def load_metadata(self, info):
        return self._read_header(info.handle)

    def defects_mask(self, info, shape):
        from astropy.io import fits

        d = self.root / "defects"
        if not d.is_dir():
            return None
        for path in sorted(d.glob("*.fits*")):
            with fits.open(path) as hdul:
                hdr = _header_dict(*(h.header for h in hdul[:2]))
                det = _first(hdr, "CCDNUM", "LSST BUTLER DATAID DETECTOR", "DETECTOR")
                if det is None or int(det) != info.detector:
                    continue
                for h in hdul:
                    if h.data is None:
                        continue
                    if getattr(h, "columns", None) is not None:
                        mask = np.zeros(shape, dtype=bool)
                        for row in h.data:
                            x0, y0 = int(row["x0"]), int(row["y0"])
                            mask[y0:y0 + int(row["height"]), x0:x0 + int(row["width"])] = True
                        return mask
                    if h.data.ndim == 2:
                        return np.asarray(h.data) != 0
        return None

    def focal_plane_layout(self):
        return self._layout

    def thumbnail(self, info):
        return None


def source_spec(args, mask_planes=MASK_PLANES):
    """Picklable description of the source chosen by --fits-dir / --repo + --collection.

    Worker processes rebuild the source from this with open_source(), so each one gets
    its own Butler instead of a copy of the parent's database connection.
    """
    if getattr(args, "fits_dir", None):
        return dict(kind="fits", fits_dir=args.fits_dir, mask_planes=list(mask_planes))
    if not getattr(args, "repo", None):
        raise ValueError("need --fits-dir, or --repo (or $REPO) and --collection")
    return dict(kind="butler", repo=args.repo, collection=args.collection,
                mask_planes=list(mask_planes))


def open_source(spec):
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "fits":
        return FitsSource(**spec)
    return ButlerSource(**spec)
