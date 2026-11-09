# calibs: QA of DECam calibration products

Per-detector, per-amp metrics of combined `bias`, `flat` and `fringe` frames, flagged
pass/warn/fail against thresholds in a YAML config. Reads a
Butler collection by default, or a directory of FITS files without the LSST stack.

| Cal type | Script | Metric functions | Status |
|----------|--------|------------------|--------|
| bias | `run_bias_qa.py` | `bias_metrics.py` | implemented |
| flat | `run_flat_qa.py` | `flat_metrics.py` | implemented (photometric closure: stub) |
| fringe | `run_fringe_qa.py` | `fringe_metrics.py` | implemented |

`common/` holds the shared pieces: `calio.py` (the only code that touches the Butler or
FITS), `stats.py` (robust, mask-aware statistics), `qa.py` (config, thresholds, tables,
worker pool, baselines) and `plots.py`. `display_calib_qa.ipynb` shows a run's results.

```bash
source ~/startup_desc_commands.sh   # LSST stack, $REPO, $LOGDIR
cd ~/decam_template_tools/verify/calibs

# every certified bias (all validity periods), science CCDs only
python run_bias_qa.py --collection DECam/calib/bias --out $SCRATCH/calqa/bias
# a quick look at two detectors
python run_bias_qa.py --detectors 10,11 --out $SCRATCH/calqa/bias_10_11
# one year's processing chain (RUN inputs, no validity range)
python run_bias_qa.py --collection u/seanmacb/DECam/calib/ELAIS_pilot/bias/2016 --out ...
# derive per-amp references/thresholds from a collection you trust, then use them
python run_bias_qa.py --collection DECam/calib/bias --out ... --build-baseline bias_baseline.yaml
python run_bias_qa.py --collection <new> --config bias_baseline.yaml --out ...
# every certified flat; each is compared with the previous period of its detector/band
python run_flat_qa.py --collection DECam/calib/flat --out $SCRATCH/calqa/flat
# or against a fixed reference set, one band only
python run_flat_qa.py --filters r --reference-collection <known-good flats> --out ...
# every certified fringe template; each is fitted to another period's inputs
python run_fringe_qa.py --collection DECam/calib/fringe --out $SCRATCH/calqa/fringe
# or to held-out ISR'd science exposures inside each validity range
python run_fringe_qa.py --science-collection <collection> --science-dataset-type post_isr_image --out ...
# on a compute node
sbatch run-calib-qa.sh bias certified --collection DECam/calib/bias
sbatch --time=4:00:00 --cpus-per-task=32 --mem=64GB run-calib-qa.sh flat certified --collection DECam/calib/flat
sbatch --cpus-per-task=32 --mem=96GB run-calib-qa.sh fringe certified --collection DECam/calib/fringe
```

Options shared by every script: `--repo` (default `$REPO`), `--collection`, `--fits-dir`,
`--detectors` (`1-62`, `10,11`), `--filters` (bands or physical filters), `--config`,
`--out`, `--workers` (default `$SLURM_CPUS_PER_TASK` or all cores), `--build-baseline`,
`--include-wavefront`, `--max-calibs` (smoke tests) and `--no-plots`. `run_flat_qa.py`
adds `--reference-collection` and `--closure` (not implemented yet; prints a note);
`run_fringe_qa.py` adds `--reference-collection`, `--science-collection` and
`--science-dataset-type`.

## Outputs (in `--out`)

- `<type>_qa.parquet` and `.csv`: one row per (calib, detector, amp, metric) with the
  columns `collection, dataset_type, filter, calib_id, validity_begin, validity_end,
  detector, amp, metric, value, unit, reference, threshold, flag, error`. `calib_id` is
  the dataset UUID (Butler) or the file name (FITS). `flag` is `pass`/`warn`/`fail`;
  `info` means the metric has no thresholds; `nan` means it could not be measured (the
  reason is in `error`); `skip` marks amps listed under `known_bad`. `error` also carries
  notes on measured values, e.g. when no curated defects were found. Whole-CCD metrics
  have `amp` = `""`, amp-pair metrics `amp` = `"A/B"`. There is one entry per
  (dataset, validity window): a dataset certified for two epochs is measured under each,
  and its later window uses its earlier one as the reference.
- `<type>_qa_trend.parquet`: one row per (input exposure, amp). For bias it holds the
  overscan level and noise from ISR metadata, an `outlier` flag, and `frame_sigma_e` for
  the frames read in full; for flats, each input's level and saturated-pixel fraction;
  for fringes, each input's sky level (`BGMEAN`) and, for the fitted exposures, the
  scale, its error and the rms before/after subtraction.
- `<type>_qa_summary.json`: pass/warn/fail/nan/info/skip counts per metric and in total,
  the same counts per calibration epoch under `epochs` (keyed by validity start date, with
  its `validity_end`; uncertified inputs under `"no validity range"`), the first 200
  failures, and the run settings.
- `figures/`: a focal-plane mosaic of each key metric per calib period, a histogram per
  metric (thresholds drawn when uniform), trends across the input sequence, and a
  thumbnail of each master (cp_pipe's `<type>Mosaic64` if present, otherwise built
  from binned detectors).

## Bias metrics

Inputs per master: the master itself, the ISR metadata of every `cpBiasIsrExp` in the
master's own RUN (cheap, a header read), and `n_noise_frames` (6) of those frames read in
full, spread over the sequence. If the RUN has no `cpBiasIsrExp`, raw zeros inside the
validity range are used and overscan-corrected here.

| metric | unit | definition |
|--------|------|------------|
| `overscan_level_adu`, `overscan_stability_adu` | adu | median and MAD-sigma over the inputs of `LSST ISR OVERSCAN SERIAL MEDIAN <amp>` |
| `n_outlier_inputs` | exposures | inputs whose overscan level is > max(10 sigma, 20 adu) off the median, or whose overscan noise is > 1.5x the median |
| `bias_median_adu` | adu | median of the master |
| `read_noise_e` | e- | sigma(frame_i - frame_j) / sqrt(2) x gain, median over 3 pairs |
| `read_noise_overscan_e` | e- | median of `LSST ISR OVERSCAN RESIDUAL SERIAL STDEV <amp>` x gain |
| `single_bias_sigma_e` | e- | sigma of a single frame x gain (read noise + fixed pattern) |
| `col_profile_rms_adu`, `row_profile_rms_adu` | adu | MAD-sigma of the clipped-mean column / row profile of the master |
| `poly2d_ptp_adu` | adu | peak-to-peak of an order-2 Legendre fit to the 32x32-binned master |
| `pickup_peak_ratio`, `pickup_peak_freq` | -, cycle/pixel | strongest narrow line in the row (serial) power spectrum of frame - master, relative to its running-median continuum, f >= 0.01 |
| `interamp_row_corr` | - | Pearson r of the row-mean profiles of frame - master between amps that share rows. A proxy for correlated pickup, **not** a crosstalk coefficient |
| `n_hot_pix` | pixel | master pixels above the column + row + smooth structure by max(7 sigma, 3 adu) |
| `n_bad_cols` | column | column medians off a 31-column running median by max(7 sigma, 1 adu) |
| `n_new_defect_pix` | pixel | hot or bad-column pixels not in the curated `defects` or the master's mask planes |
| `master_noise_ratio` | - | master sigma after removing that structure / (read noise / sqrt(N)); N from the master's `LSST BUTLER INPUT * DATASETTYPE` headers |

Frames are compared against the master (frame - master), so read noise, pickup and the
inter-amp correlation see temporal noise only. Noise amplitudes ("sigma") are 4-sigma
clipped standard deviations corrected for the truncation, not MAD: raw frames are
integer ADU and their MAD is quantized. Level and sequence statistics use median/MAD.

## Flat metrics

Inputs per master: the master, the ISR metadata of every `cpFlatIsrExp` in its RUN
(`LSST ISR FINAL MEDIAN <amp>`, `LSST ISR MASK SAT <amp>`), `n_noise_frames` (2) well
exposed inputs read in full, and a reference flat: the previous validity period of the
same detector and band, or the matching flat in `--reference-collection`. Fractions are
relative to the local flat level. DECam CCD edges are bright (outer ~10 px at 1.3-2x the
interior, outermost columns dead), so pixel metrics skip `edge_margin` (30) px at the
CCD border and the edges get their own metric. High-pass images are
master / (binned median model, bilinearly interpolated) - 1, per amp.

| metric | unit | definition |
|--------|------|------------|
| `input_level_adu`, `input_level_scatter_frac` | adu, - | median and MAD/median of the inputs' levels |
| `n_low_count_inputs` | exposures | inputs below `low_count_adu` (5000) |
| `max_level_sat_frac`, `max_sat_pix_frac` | - | highest input level / saturation; highest fraction of SAT pixels in an input |
| `master_median` | - | median of the normalized master (info) |
| `pix_rms_frac` | - | clipped sigma of the master / 32-px smooth model - 1 (PRNU + noise) |
| `master_poisson_frac`, `pix_rms_poisson_ratio` | - | 1/sqrt(sum of input electrons), and pix_rms / that (info: PRNU dominates) |
| `input_noise_ratio` | - | sigma(frame / master) / Poisson expectation sqrt(S g + rn^2)/(S g) x sqrt(1 - 1/N); ~1 for the right gain, brighter-fatter pulls it below |
| `low_qe_frac` | - | fraction of pixels below 0.9 x the local level |
| `n_dust_spots` | spots | compact depressions (Gaussian-smoothed high-pass < -5 sigma and < -0.5%, area 12-40000 px, elongation <= 3). Includes DECam's 6 tape bumps per CCD, which are stable |
| `treering_proxy_frac` | - | rms of the 3-px-smoothed high-pass; dominated by tree rings on DECam, but no ring centre is fitted |
| `edge_feature_frac` | - | signed largest (outer 10 px / 20-60 px band) - 1 over the amp's CCD-border sides |
| `amp_ratio`, `amp_ratio_vs_gain` (`A/B`) | - | median A / B in strips either side of the amp boundary, and that / (gain_B / gain_A) |
| `gradient_ptp_frac` (CCD) | - | peak-to-peak of an order-2 Legendre fit to the amp-stitched, 64-binned CCD / median |
| `ref_amp_ratio` | - | median(master / reference) of the amp / of the CCD |
| `ref_ratio_rms`, `ref_ratio_frac_out` | - | clipped rms of the amp-normalized ratio, fraction of pixels off by > 1% |
| `n_new_dust_spots` | spots | dust-like depressions in the ratio image (dust that appeared since the reference) |

The photometric closure test (zero-point scatter vs focal-plane position) needs source
catalogs and is not implemented: `flat_metrics.photometric_closure` raises
`NotImplementedError`.

## Fringe metrics

Inputs per template: the template (fringe / sky per pixel), the metadata of its
`cpFringeMeasuredIsrExp` inputs, a reference template (previous period, or
`--reference-collection`), and `n_science` (12) exposures to fit. These are **never the
template's own inputs**: one input carries ~5% of a DECam template, so fitting it
inflates the scale several-fold, and a 1/N leave-one-out correction overshoots. The
default is the inputs of the nearest other period (disjoint exposures); with
`--science-collection`, held-out exposures in the validity range are used instead.
ISR'd frames that are not yet sky-normalized are normalized here.

DECam z fringes are weak: about 1e-4 of sky rms, against a template noise of 4-8e-4 per
pixel. So images are high-passed per amp (minus a 256-px interpolated median), mean-binned
8x8, and the white noise expected from neighbouring-pixel differences and each bin's
good-pixel count is subtracted in quadrature.

| metric | unit | definition |
|--------|------|------------|
| `fringe_rms_frac`, `fringe_ptp_frac` | - | binned template rms with the noise removed; p99 - p1 of the binned template |
| `template_noise_frac`, `template_snr` | - | per-pixel white noise of the template; fringe rms / binned noise |
| `n_template_inputs` | exposures | inputs listed in the template's provenance |
| `amp_step_frac` (`A/B`) | - | template median in a strip of A minus B at their boundary |
| `ref_template_corr`, `ref_template_coherence` | - | correlation with the reference template (32x32 bins), and that / the value expected for an identical pattern given both templates' noise (~1 = stable; unstable when the signal is near zero) |
| `n_science` | exposures | exposures fitted |
| `scale_median`, `scale_snr_median` | - | fitted scale of the template per exposure (least squares on binned images) |
| `residual_rms_frac`, `residual_excess_frac` | - | binned rms after subtraction, and its part above white noise |
| `fringe_removed_frac` | - | 1 - sum(post var - noise) / sum(pre var - noise) over the fitted exposures |
| `scale_sky_corr` | - | Pearson r of scale vs sky level (`BGMEAN`); strongly negative on DECam |
| `n_scale_outliers`, `scale_time_jump_sigma` | -, sigma | residuals from a scale = a + b / sky fit in units of sqrt(fit error^2 + (0.1 x model)^2) (the 10% is night-to-night OH variability, `scale_intrinsic_frac`): count beyond 5, and median jump between consecutive exposures (~0.95 = noise) |

The fitted scale is attenuated by s^2 / (s^2 + n^2) for a template with signal s and
noise n per bin (`template_snr`).

## Configuration

`calqa_config.yaml` holds the parameters and thresholds. **Its numbers are placeholders**
(`placeholder: true`); every run prints a note until it is given a `--build-baseline`
config. Per metric, `warn`/`fail` are `[lo, hi]` intervals (`null` = unbounded) read as
`mode: absolute` (default), `offset` (reference + lo/hi) or `fraction` (reference x
(1 + lo/hi)). Single amps are overridden under `<type>.per_amp."<detector>".<amp>`.
`known_bad` lists amps to report as `skip`.

`--build-baseline OUT.yaml` treats `--collection` as known good. It writes a copy of the
config with `per_amp` filled in: reference = the median over that collection's calibs,
and `warn`/`fail` = reference +- max(k x MAD-sigma, min_frac x |reference|,
`baseline_min_abs`), with k and min_frac taken from the `baseline:` section. Count-like
metrics (`baseline_side: upper`) get only an upper bound.

## Where geometry and gain come from

Nothing DECam-specific is hard-coded. With the Butler, the amp boxes, gain, read noise and
saturation come from each exposure's `Detector` (the obs_decam camera). Focal-plane
positions and detector types come from `camera` in `DECam/calib/curated/unbounded`, and
the bad pixel mask is the union of the `DECam/calib/curated` `defects` valid during the
calib's validity range. This deliberately queries the CALIBRATION collection directly:
`DECam/calib` chains the curated RUNs, so it ignores their validity ranges. With
`--fits-dir`, the geometry comes from `DATASEC<amp>`/`BIASSEC<amp>`/`GAIN<amp>`/
`RDNOISE<amp>`/`SATURAT<amp>` headers (raw DECam), or else from `amp_geometry.yaml`:

```python
from common import calio
calio.dump_amp_geometry(os.environ["REPO"], "<fits_dir>/amp_geometry.yaml")
```

FITS directory layout: `<fits_dir>/<type>/*.fits` (masters), optionally
`<fits_dir>/<type>_inputs/*.fits` (the single frames) and `<fits_dir>/defects/*.fits`.
Detectors are identified by `CCDNUM` (or `LSST BUTLER DATAID DETECTOR`). Stack-written
Exposure FITS (`IMAGE`/`MASK` HDUs with `MP_<plane>` keys) and plain images both work.

## From a notebook

```python
from common import calio, qa
import bias_metrics, run_bias_qa

source = calio.ButlerSource(os.environ["REPO"], "DECam/calib/bias")
info = source.list_calibs("bias", detectors=[10])[0]
rows, trend, _ = run_bias_qa.measure_bias(source, info, run_bias_qa.bias_params(qa.load_config()))
# or call single functions, e.g. bias_metrics.pickup(frames, master, master.amp("A"))
```

## Tests

```bash
cd ~/decam_template_tools/verify/calibs && python -m pytest tests
```

These use synthetic two-amp frames with injected gain, read noise, overscan offsets,
pickup, common-mode row noise, gradients, column structure, hot pixels, bad columns and
an outlier input (bias); flats with PRNU, Poisson noise, gain steps, illumination
gradients, dust donuts, low-QE patches, bright edges and tree rings; and fringe templates
and science frames with known fringe amplitude, template noise, amp steps, scales vs sky
and an outlier exposure. They check recovery within tolerance, every flag path, the FITS
source, baselines, and serial vs parallel runs. No Butler is needed.

## Notes

- Each worker opens its own Butler (spawned processes, one Postgres connection each) and
  uses about 0.5 GB. One master takes about a minute, mostly reading ~200 input headers.
  A flat also reads its reference master and two input frames.
- Inputs (`cp<Type>IsrExp`, `cpFringeMeasuredIsrExp`) are read from the master's own RUN.
  A master made by a RUN resumed with `--skip-existing-in` has none there; its inputs are
  then found from its provenance header (`LSST BUTLER INPUT <n> DATASETTYPE/RUN/ID`).
- Edge columns/rows with steep bias ramps count as column structure, not hot pixels.
  The curated defects normally cover them; check `n_new_defect_pix` if they don't.
