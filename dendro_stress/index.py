"""Water-stress index and traffic light.

Two ways to say what MDS a non-stressed tree would show today:

* reference trees - the median MDS of well-watered trees on the same day (an over-irrigated reference row);
* VPD baseline    - expected MDS = a + b * VPD_max, fitted on days known to be well watered (calibration window).

Signal intensity SI = MDS / expected MDS, smoothed with a 3-day rolling median. A tree under moderate water
deficit shrinks more than expected (SI > 1). In severe drought MDS can drop again because the stem no longer
refills, so the predawn tree water deficit (TWD) is checked as well, scaled by the tree's own calibration MDS.

Traffic light:
  green - SI < si_amber and relative predawn TWD < twd_amber
  amber - SI >= si_amber or relative predawn TWD >= twd_amber        -> irrigate soon / increase the dose
  red   - SI >= si_red   or relative predawn TWD >= twd_red          -> irrigate now
  grey  - not enough valid readings that day
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = {"grey": -1, "green": 0, "amber": 1, "red": 2}
ADVICE = {
    "green": "no extra irrigation needed",
    "amber": "water deficit building: irrigate soon or raise the dose",
    "red": "strong water stress: irrigate now",
    "grey": "not enough sensor data today",
}


@dataclass
class IndexConfig:
    si_amber: float = 1.3
    si_red: float = 1.6
    twd_amber: float = 1.5  # (predawn TWD - calibration level) / calibration median MDS
    twd_red: float = 3.0
    smooth_days: int = 3
    min_vpd_kpa: float = 0.3  # days with lower VPD max give no MDS signal (rain, fog)
    twd_baseline: bool = True  # measure predawn TWD above its calibration-window level


def fit_vpd_baseline(mds: pd.Series, vpd: pd.Series, n_iter: int = 3) -> tuple[float, float]:
    """Least-squares MDS = a + b * VPD on the given days, with 2.5-sigma trimming. Returns (a, b), b >= 0."""
    df = pd.concat([mds.rename("m"), vpd.rename("v")], axis=1).dropna()
    if len(df) < 3:
        raise ValueError("need at least 3 days with MDS and VPD to fit the VPD baseline")
    keep = np.ones(len(df), dtype=bool)
    a, b = float(df["m"].median()), 0.0
    for _ in range(n_iter):
        x, y = df["v"].to_numpy()[keep], df["m"].to_numpy()[keep]
        if len(x) < 3 or np.ptp(x) < 1e-6:
            break
        b, a = np.polyfit(x, y, 1)
        b = max(float(b), 0.0)
        a = float(np.mean(y - b * x))
        res = df["m"].to_numpy() - (a + b * df["v"].to_numpy())
        s = np.std(res[keep])
        if s == 0:
            break
        keep = np.abs(res) <= 2.5 * s
    return float(a), float(b)


def expected_from_vpd(vpd: pd.Series, a: float, b: float) -> pd.Series:
    return (a + b * vpd).clip(lower=1e-6)


def signal_intensity(mds: pd.Series, expected: pd.Series, smooth_days: int = 3) -> pd.Series:
    si = mds / expected.reindex(mds.index)
    si = si.replace([np.inf, -np.inf], np.nan)
    if smooth_days > 1:
        si = si.rolling(smooth_days, min_periods=1).median()
    return si


def traffic_light(si: pd.Series, twd_rel: pd.Series, valid: pd.Series | None = None,
                  cfg: IndexConfig | None = None) -> pd.Series:
    """Daily colour from smoothed SI and relative predawn TWD."""
    cfg = cfg or IndexConfig()
    idx = si.index.union(twd_rel.index)
    si, tw = si.reindex(idx), twd_rel.reindex(idx)
    color = pd.Series("green", index=idx, dtype=object)
    amber = (si >= cfg.si_amber) | (tw >= cfg.twd_amber)
    red = (si >= cfg.si_red) | (tw >= cfg.twd_red)
    color[amber.fillna(False)] = "amber"
    color[red.fillna(False)] = "red"
    nodata = si.isna() & tw.isna()
    if valid is not None:
        nodata |= ~valid.reindex(idx).fillna(False).astype(bool)
    color[nodata] = "grey"
    return color


def stress_table(daily: pd.DataFrame, expected_mds: pd.Series, calib_mds: float, vpd_max: pd.Series | None = None,
                 cfg: IndexConfig | None = None, twd_calib: float = 0.0) -> pd.DataFrame:
    """Add si, twd_rel, light and advice columns to a daily metrics table."""
    cfg = cfg or IndexConfig()
    if not np.isfinite(calib_mds) or calib_mds <= 0:
        raise ValueError("calibration MDS must be a positive number")
    out = daily.copy()
    mds = out["mds"].copy()
    if vpd_max is not None:
        low = vpd_max.reindex(out.index) < cfg.min_vpd_kpa
        mds[low.fillna(False)] = np.nan
    out["mds_expected"] = expected_mds.reindex(out.index)
    out["si"] = signal_intensity(mds, out["mds_expected"], cfg.smooth_days)
    out["twd_rel"] = (out["twd_predawn"] - (twd_calib if cfg.twd_baseline else 0.0)) / calib_mds
    out["light"] = traffic_light(out["si"], out["twd_rel"], out["valid"], cfg)
    out["advice"] = out["light"].map(ADVICE)
    return out


def analyse_tree(daily: pd.DataFrame, vpd_max: pd.Series, calib_days: pd.Index, reference_mds: pd.Series | None = None,
                 cfg: IndexConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Stress table for one tree. Uses reference trees if given, else the VPD baseline fitted on `calib_days`.

    With reference trees the expected MDS is reference_mds * k, where k = median(MDS_tree / MDS_ref) on the
    calibration days (corrects for tree size and sensor position).
    """
    cfg = cfg or IndexConfig()
    cal = daily.loc[daily.index.intersection(calib_days)]
    cal = cal[cal["valid"] & cal["mds"].notna()]
    if len(cal) < 3:
        raise ValueError("fewer than 3 valid calibration days")
    calib_mds = float(cal["mds"].median())
    twd_calib = float(cal["twd_predawn"].median()) if cal["twd_predawn"].notna().any() else 0.0
    info: dict = {"calib_days": int(len(cal)), "calib_mds_um": calib_mds, "calib_twd_predawn_um": twd_calib}
    if reference_mds is not None:
        ratio = (cal["mds"] / reference_mds.reindex(cal.index)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < 3:
            raise ValueError("reference MDS missing on calibration days")
        k = float(ratio.median())
        expected = reference_mds * k
        info.update(method="reference", scale=k)
    else:
        v = vpd_max.reindex(cal.index)
        use = v >= cfg.min_vpd_kpa
        a, b = fit_vpd_baseline(cal["mds"][use], v[use])
        expected = expected_from_vpd(vpd_max, a, b)
        info.update(method="vpd", a=a, b=b)
    return stress_table(daily, expected, calib_mds, vpd_max, cfg, twd_calib), info
