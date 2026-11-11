"""Daily dendrometer metrics: maximum daily shrinkage (MDS), growth and tree water deficit (TWD).

Definitions (radius r in micrometres, one value per 10-15 min):

* MDS  - morning maximum (00:00-13:00) minus the lowest value after it, up to 21:00. The daily shrink caused by
         transpiration drawing water out of elastic stem (or fruit) tissue.
* GRO  - zero-growth line: the running maximum of r. Growth happens only when the stem exceeds its previous
         maximum; any time r is below that line the tree is shrunken, not shrinking.
* TWD  - tree water deficit = GRO - r (>= 0). Predawn TWD (04:00-06:00 mean) is the deficit that did not refill
         overnight and works like a predawn water potential.
* daily growth - increase of GRO over the day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zero_growth_line(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (GRO, TWD) for a radius series. NaN readings keep the line flat; TWD is NaN where r is NaN."""
    v = r.to_numpy(dtype=float)
    filled = np.where(np.isfinite(v), v, -np.inf)
    gro = np.maximum.accumulate(filled)
    gro[np.isneginf(gro)] = np.nan
    twd = gro - v
    return pd.Series(gro, index=r.index, name="gro"), pd.Series(twd, index=r.index, name="twd")


def _mds_one_day(t_hours: np.ndarray, v: np.ndarray, max_end: float = 13.0, min_end: float = 21.0) -> float:
    ok = np.isfinite(v)
    morning = ok & (t_hours < max_end)
    if not morning.any():
        return np.nan
    i_max = np.flatnonzero(morning)[np.argmax(v[morning])]
    after = ok & (np.arange(len(v)) >= i_max) & (t_hours < min_end)
    if not after.any():
        return np.nan
    return float(v[i_max] - np.min(v[after]))


def daily_metrics(r: pd.Series, min_coverage: float = 0.8, step: str | None = None) -> pd.DataFrame:
    """Daily table from a cleaned, regular radius series (micrometres).

    Columns: coverage, valid, mxsd, mnsd, mds, growth, twd_mean, twd_max, twd_predawn.
    """
    if r.empty:
        return pd.DataFrame(columns=["coverage", "valid", "mxsd", "mnsd", "mds", "growth", "twd_mean", "twd_max",
                                     "twd_predawn"])
    if step is None:
        diffs = r.index.to_series().diff().dropna()
        step_td = diffs.median() if len(diffs) else pd.Timedelta("10min")
    else:
        step_td = pd.Timedelta(step)
    per_day = int(round(pd.Timedelta("1D") / step_td))
    gro, twd = zero_growth_line(r)
    day = r.index.normalize()
    hours = (r.index - day).total_seconds().to_numpy() / 3600.0
    rows = []
    v_all, gro_all, twd_all = r.to_numpy(float), gro.to_numpy(float), twd.to_numpy(float)
    codes, uniques = pd.factorize(day)
    prev_gro_end = np.nan
    for k, d in enumerate(uniques):
        sel = codes == k
        v, h, g, w = v_all[sel], hours[sel], gro_all[sel], twd_all[sel]
        cov = float(np.isfinite(v).sum()) / per_day
        pre = (h >= 4) & (h < 6) & np.isfinite(w)
        g_end = g[np.isfinite(g)][-1] if np.isfinite(g).any() else np.nan
        growth = g_end - prev_gro_end if np.isfinite(prev_gro_end) and np.isfinite(g_end) else np.nan
        rows.append(dict(
            date=d, coverage=min(cov, 1.0), valid=cov >= min_coverage,
            mxsd=np.nanmax(v) if np.isfinite(v).any() else np.nan,
            mnsd=np.nanmin(v) if np.isfinite(v).any() else np.nan,
            mds=_mds_one_day(h, v), growth=growth,
            twd_mean=np.nanmean(w) if np.isfinite(w).any() else np.nan,
            twd_max=np.nanmax(w) if np.isfinite(w).any() else np.nan,
            twd_predawn=float(np.mean(w[pre])) if pre.any() else np.nan,
        ))
        if np.isfinite(g_end):
            prev_gro_end = g_end
    out = pd.DataFrame(rows).set_index("date")
    for col in ("mds", "growth", "twd_mean", "twd_max", "twd_predawn"):
        out.loc[~out["valid"], col] = np.nan
    return out
