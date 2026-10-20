"""FITS fallback source and header geometry parsing (no LSST stack needed)."""

import numpy as np
import pytest
from astropy.io import fits

import synthetic as syn
from common import calio


def test_parse_section():
    assert calio.parse_section("[1081:2104,51:4146]") == (50, 4146, 1080, 2104)
    assert calio.parse_section("[2104:1081, 4146:51]") == (50, 4146, 1080, 2104)
    with pytest.raises(ValueError):
        calio.parse_section("1:2,3:4")


def test_amps_from_decam_raw_header():
    header = {"DATASECA": "[1081:2104,51:4146]", "DATASECB": "[57:1080,51:4146]",
              "BIASSECA": "[2105:2154,51:4146]", "BIASSECB": "[7:56,51:4146]",
              "GAINA": 3.881, "GAINB": 4.036, "RDNOISEA": 6.2, "SATURATB": 46155.0,
              "DATASEC": "[57:2104,51:4146]"}  # unsuffixed key is not an amp
    amps = {a.name: a for a in calio.amps_from_header(header)}
    assert set(amps) == {"A", "B"}
    assert amps["A"].bbox == (50, 4146, 1080, 2104)
    assert amps["A"].overscan == (50, 4146, 2104, 2154)
    assert amps["A"].gain == 3.881 and amps["A"].read_noise == 6.2
    assert np.isnan(amps["A"].saturation) and amps["B"].saturation == 46155.0


def test_provenance_inputs():
    md = {"LSST BUTLER INPUT 0 DATASETTYPE": "cpFringeMeasuredIsrExp",
          "LSST BUTLER INPUT 0 RUN": "u/x/fringe/run1", "LSST BUTLER INPUT 0 ID": "a1",
          "LSST BUTLER INPUT 1 DATASETTYPE": "cpFringeMeasuredIsrExp",
          "LSST BUTLER INPUT 1 RUN": "u/x/fringe/run1", "LSST BUTLER INPUT 1 ID": "a2",
          "LSST BUTLER INPUT 12 DATASETTYPE": "cpFringeMeasuredIsrExp",
          "LSST BUTLER INPUT 12 RUN": "u/x/fringe/run0", "LSST BUTLER INPUT 12 ID": "b1",
          "LSST BUTLER INPUT 2 DATASETTYPE": "camera",
          "LSST BUTLER INPUT 2 RUN": "DECam/calib/unbounded", "LSST BUTLER INPUT 2 ID": "c1",
          "LSST BUTLER INPUT 3 DATASETTYPE": "cpFringeMeasuredIsrExp"}  # no RUN/ID: skipped
    assert calio.provenance_inputs(md, "cpFringeMeasuredIsrExp") == {
        "u/x/fringe/run1": {"a1", "a2"}, "u/x/fringe/run0": {"b1"}}
    assert calio.provenance_inputs(md, "cpFlatIsrExp") == {}
    assert calio.provenance_inputs({}, "cpFringeMeasuredIsrExp") == {}


def test_select_detectors():
    layout = {1: {"type": "SCIENCE"}, 2: {"type": "SCIENCE"}, 63: {"type": "FOCUS"}}
    assert calio.select_detectors(layout) == [1, 2]
    assert calio.select_detectors(layout, include_wavefront=True) == [1, 2, 63]
    assert calio.select_detectors(layout, [5, 3]) == [3, 5]
    assert calio.select_detectors(None) is None


@pytest.fixture
def fits_dir(tmp_path):
    truth = syn.write_bias_dir(tmp_path, detectors=(1, 2), defects={1: [(10, 20, 3, 4)]})
    return tmp_path, truth


def test_fits_source_lists_loads_and_finds_inputs(fits_dir):
    root, truth = fits_dir
    src = calio.FitsSource(root)
    infos = src.list_calibs("bias")
    assert [i.detector for i in infos] == [1, 2]
    assert src.list_calibs("bias", detectors=[2])[0].detector == 2
    frame = src.load(infos[0])
    np.testing.assert_array_equal(frame.image, truth[1][0].image)
    assert [a.name for a in frame.amps] == ["A", "B"]
    assert frame.amp("A").gain == syn.GAINS["A"]
    assert not frame.is_raw

    inputs = src.inputs(infos[0], "cpBiasIsrExp")
    assert [i.exposure for i in inputs] == [1000 + k for k in range(6)]
    md = src.load_metadata(inputs[0])
    assert md["LSST ISR OVERSCAN SERIAL MEDIAN A"] == pytest.approx(2300, abs=10)
    assert src.focal_plane_layout()[1]["fp_corners"][0] == [0.0, 0.0]


def test_fits_mask_planes_and_defects(fits_dir):
    root, _ = fits_dir
    mask = np.zeros((syn.NY, syn.NX), dtype=np.int32)
    mask[5, 6] = 1 << 0  # BAD
    mask[7, 8] = 1 << 5  # some plane not in MASK_PLANES
    syn.write_exposure_fits(root / "bias" / "bias_01.fits", np.ones((syn.NY, syn.NX)),
                            {"CCDNUM": 1}, mask=mask)
    src = calio.FitsSource(root)
    info = src.list_calibs("bias", detectors=[1])[0]
    frame = src.load(info)
    assert frame.bad[5, 6] and not frame.bad[7, 8] and frame.bad.sum() == 1
    defects = src.defects_mask(info, frame.image.shape)
    assert defects.sum() == 12 and defects[20:24, 10:13].all()
    assert src.defects_mask(src.list_calibs("bias", detectors=[2])[0], (4, 4)) is None


def test_fits_raw_frame_uses_header_geometry(tmp_path):
    hdr = fits.Header({"CCDNUM": 3, "DATASECA": "[21:36,1:64]", "BIASSECA": "[37:40,1:64]",
                       "DATASECB": "[5:20,1:64]", "BIASSECB": "[1:4,1:64]", "GAINA": 4.0,
                       "GAINB": 4.0})
    (tmp_path / "bias").mkdir()
    fits.PrimaryHDU(np.zeros((64, 40), dtype=np.float32), header=hdr).writeto(
        tmp_path / "bias" / "raw.fits")
    src = calio.FitsSource(tmp_path)
    frame = src.load(src.list_calibs("bias")[0])
    assert frame.is_raw and frame.amp("A").overscan == (0, 64, 36, 40)


def test_fits_without_geometry_fails_clearly(tmp_path):
    syn.write_exposure_fits(tmp_path / "bias" / "b.fits", np.zeros((8, 8)), {"CCDNUM": 1})
    src = calio.FitsSource(tmp_path)
    with pytest.raises(ValueError, match="amp_geometry.yaml"):
        src.load(src.list_calibs("bias")[0])


def test_missing_detector_header_fails(tmp_path):
    syn.write_exposure_fits(tmp_path / "bias" / "b.fits", np.zeros((8, 8)), {})
    with pytest.raises(ValueError, match="CCDNUM"):
        calio.FitsSource(tmp_path).list_calibs("bias")


def test_amp_pixels_applies_flips():
    image = np.arange(12, dtype=float).reshape(3, 4)
    amp = calio.AmpGeom("A", (0, 3, 0, 4), flip_x=True, flip_y=True)
    frame = calio.CalibFrame(image, np.zeros_like(image, bool), [amp], {}, syn.make_info())
    np.testing.assert_array_equal(frame.amp_pixels("A")[0], image[::-1, ::-1])


class _Time:
    def __init__(self, isot):
        self.isot = isot


class _Span:
    def __init__(self, begin, end):
        self.begin, self.end = _Time(begin), (None if end is None else _Time(end))


class _Ref:
    def __init__(self, id_):
        self.id = id_


def test_merge_calib_entries_one_per_dataset_and_window():
    a, b, c = _Ref("a"), _Ref("b"), _Ref("c")
    w14, w15 = _Span("2014-01-01", "2015-01-01"), _Span("2015-01-01", "2016-01-01")
    certified = [(a, w14), (a, w15),                   # one dataset, two windows: two entries
                 (b, w14), (b, _Span("2014-01-01", "2015-01-01")),  # same window twice: once
                 (c, _Span("2016-01-01", None))]       # open-ended window
    in_runs = [a, _Ref("d"), _Ref("d")]                # a is certified; d only in a RUN
    out = calio.merge_calib_entries(certified, in_runs)
    keys = [(ref.id, None if ts is None else (ts.begin.isot, ts.end and ts.end.isot))
            for ref, ts in out]
    assert keys == [("a", ("2014-01-01", "2015-01-01")), ("a", ("2015-01-01", "2016-01-01")),
                    ("b", ("2014-01-01", "2015-01-01")), ("c", ("2016-01-01", None)),
                    ("d", None)]
