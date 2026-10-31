"""Write the per-job BPS override configs for the flat/fringe rebuild into ./rebuild/."""
import pathlib

import pandas as pd

QA = "/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/calibQa/flat/certified"
BASE = "/global/u1/d/descdm/seanmacb/decam_template_tools/endurance/submit"
SAT = 0.8 * 46155    # 80% of the lowest DECam amp saturation (ADU)
DEAD = [2, 61]       # S30, N30: ~0 ADU in every frame, so they can't vote

df = pd.read_parquet(f"{QA}/flat_qa.parquet")
cal = df.drop_duplicates("calib_id").set_index("calib_id")[["filter", "validity_begin"]]
t = pd.read_parquet(f"{QA}/flat_qa_trend.parquet").join(cal, on="calib_id")
t["year"] = t["validity_begin"].str[:4]
t = t[~t["detector"].isin(DEAD)]
out = pathlib.Path("rebuild")
out.mkdir(exist_ok=True)


BAD_FRINGE_RAWS = {"2016": [(34, 563122), (16, 563896)], "2017": [(4, 649905)]}   # (detector, exposure)
S30_DEAD = " AND NOT (detector=2 AND day_obs<20161229)"    # S30 back on 2016-12-29; same cut as the chunk queries


def day_range(y):
    return f"AND (day_obs > {y}0000) AND (day_obs < {int(y) + 1}0000)"


flats = [("2013", "r")] + [(y, b) for y in ("2014", "2016", "2017", "2018") for b in "griz"]
for y, b in flats:
    g = t[(t["year"] == y) & (t["filter"] == b)]
    bad = (g["level_adu"] < 5000) | (g["level_adu"] > SAT)
    frac = bad.groupby(g["exposure"]).mean()          # fraction of live CCDs flagging it
    excl = sorted(int(e) for e in frac[frac > 0.5].index)
    ex = f" AND exposure NOT IN ({','.join(map(str, excl))})" if excl else ""
    print(f"flat_{y}_{b}: {frac.size} inputs, excluding {len(excl)}, keeping {frac.size - len(excl)}")
    resume = ""
    if (y, b) == ("2018", "i"):
        # 2018-09-01 set (767733-767743): detector 41 amp B reads ~1.9k ADU instead of ~19k, as in
        # the certified 2018 i flat, which used only the 2018-11-01 set (789775-789785)
        ex += " AND exposure NOT IN (" + ",".join(str(e) for e in range(767733, 767744)) + ")"
    if (y, b) == ("2016", "g"):
        # truncated raw; the first build kept 2727 good cpFlatIsrExp, so skip those quanta
        ex += " AND NOT (detector=39 AND exposure=588492)"
        resume = ('extraQgraphOptions: "--dataset-query-constraint off --skip-existing-in '
                  'u/descdm/DECam/calib/ELAIS_pilot/flat_v2/2016/g_band/20261027T102646Z"\n')
    (out / f"flat_{y}_{b}.yaml").write_text(f"""includeConfigs:
  - {BASE}/bps_build_flat.yaml
year: "{y}"
submitPath: "/pscratch/sd/d/descdm/butlersubmits/{{outputRun}}"  # repo symlink ./submit points at seanmacb scratch
band: "{b}"
range: "{day_range(y)}"
{resume}payload:
  payloadName: "DECam/calib/ELAIS_pilot/flat_v2/{{year}}/{{band}}_band"
  dataQuery: "instrument='DECam' {{range}} AND band='{{band}}' AND exposure.observation_type='dome flat' AND NOT (detector=51 AND exposure=598458){ex}"
""")
for y in ("2013", "2014", "2016", "2017", "2018"):
    extra = S30_DEAD if y < "2017" else ""
    if y == "2016":
        extra += " AND NOT (detector=59 AND exposure=591839)"  # truncated raw
    resume = ""
    if y in BAD_FRINGE_RAWS:
        # truncated raws (cfitsio: missing ZZERO/ZSCALE) that blocked those detectors' masters in the
        # first build (run 20261028T225158Z); skip the ISR quanta that already succeeded
        extra += "".join(f" AND NOT (detector={d} AND exposure={e})" for d, e in BAD_FRINGE_RAWS[y])
        resume = ('extraQgraphOptions: "--dataset-query-constraint off --skip-existing-in '
                  f'u/descdm/DECam/calib/ELAIS_pilot/fringe_v2/{y}/20261028T225158Z"\n')
    (out / f"fringe_{y}.yaml").write_text(f"""includeConfigs:
  - {BASE}/bps_build_fringe.yaml
year: "{y}"
submitPath: "/pscratch/sd/d/descdm/butlersubmits/{{outputRun}}"  # repo symlink ./submit points at seanmacb scratch
band: "z"
range: "{day_range(y)}"
{resume}payload:
  payloadName: "DECam/calib/ELAIS_pilot/fringe_v2/{{year}}"
  dataQuery: "instrument='DECam' {{range}} AND band='{{band}}' AND exposure.observation_type='science'{extra}"
""")
    print(f"fringe_{y}: science exposures only{extra.replace(' AND NOT', ', minus')}")
