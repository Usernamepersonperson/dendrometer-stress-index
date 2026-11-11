"""Read dendrometer CSVs (long or wide) and run the whole chain for each sensor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .clean import CleanConfig, clean_series
from .index import ADVICE, IndexConfig, analyse_tree, stress_table
from .metrics import daily_metrics

TIME_COLS = ("time", "timestamp", "datetime", "date_time")
TREE_COLS = ("tree", "sensor", "sensor_id", "tree_id", "series")
VALUE_COLS = ("radius_um", "value", "radius", "diameter_um", "src")


def read_dendro_csv(path: Path | str) -> dict[str, pd.Series]:
    """Return {sensor name: radius series (um) with DatetimeIndex}.

    Long format: a time column, a tree/sensor column and a value column (radius_um, value, ...).
    Wide format: a time column and one numeric column per sensor.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("input CSV has no rows")
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in TIME_COLS if c in cols), None)
    if tcol is None:
        raise ValueError(f"input CSV needs a time column (one of {', '.join(TIME_COLS)})")
    times = pd.to_datetime(df[tcol], errors="coerce")
    if times.isna().all():
        raise ValueError("could not parse any timestamps")
    if times.dt.tz is not None:
        times = times.dt.tz_localize(None)
    tree_col = next((cols[c] for c in TREE_COLS if c in cols), None)
    val_col = next((cols[c] for c in VALUE_COLS if c in cols), None)
    out: dict[str, pd.Series] = {}
    if tree_col is not None and val_col is not None:
        for name, g in df.assign(_t=times).dropna(subset=["_t"]).groupby(tree_col, sort=True):
            s = pd.Series(pd.to_numeric(g[val_col], errors="coerce").to_numpy(), index=pd.DatetimeIndex(g["_t"]))
            out[str(name)] = s.sort_index()
    else:
        for c in df.columns:
            if c == tcol:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(times)).loc[lambda x: x.index.notna()]
            out[str(c)] = s.sort_index()
    if not out:
        raise ValueError("no numeric dendrometer columns found")
    return out


@dataclass
class TreeReport:
    name: str
    table: pd.DataFrame
    info: dict
    n_jumps: int
    n_spikes: int
    n_gaps: int


def run_all(series: dict[str, pd.Series], daily_weather: pd.DataFrame | None = None,
            reference: list[str] | None = None, calib_days: int = 21, calib_start: str | None = None,
            step: str | None = None, clean_cfg: CleanConfig | None = None,
            index_cfg: IndexConfig | None = None) -> list[TreeReport]:
    """Clean every sensor, compute daily metrics and the stress table.

    Expected MDS: median of `reference` sensors if given, else the VPD baseline (needs `daily_weather`),
    else the tree's own calibration median (no weather: SI only reacts to MDS changes, weaker).
    """
    reference = reference or []
    missing = [r for r in reference if r not in series]
    if missing:
        raise ValueError(f"reference sensors not in input: {missing}")
    clean_cfg = clean_cfg or CleanConfig()
    if step:
        clean_cfg.step = step
    else:
        any_s = next(iter(series.values()))
        med = any_s.index.to_series().diff().median()
        clean_cfg.step = f"{max(1, round(med.total_seconds() / 60))}min" if pd.notna(med) else "10min"
    cleaned, dailies = {}, {}
    for name, s in series.items():
        res = clean_series(s, clean_cfg)
        cleaned[name] = res
        dailies[name] = daily_metrics(res.series, step=clean_cfg.step)
    all_days = sorted(set().union(*[d.index for d in dailies.values()]))
    if not all_days:
        raise ValueError("no data")
    c0 = pd.Timestamp(calib_start) if calib_start else all_days[0]
    calib = pd.date_range(c0, periods=calib_days)
    vpd = daily_weather["vpd_max"] if daily_weather is not None and "vpd_max" in daily_weather else None
    ref_mds = None
    if reference:
        ref_mds = pd.concat([dailies[r]["mds"] for r in reference], axis=1).median(axis=1)
    reports = []
    for name, dm in dailies.items():
        res = cleaned[name]
        own_ref = ref_mds
        if reference and name in reference and len(reference) > 1:
            own_ref = pd.concat([dailies[r]["mds"] for r in reference if r != name], axis=1).median(axis=1)
        if own_ref is not None or vpd is not None:
            table, info = analyse_tree(dm, vpd if vpd is not None else pd.Series(np.nan, index=dm.index), calib,
                                       reference_mds=own_ref, cfg=index_cfg)
        else:
            cal = dm.loc[dm.index.intersection(calib)]
            cm = float(cal["mds"].median())
            table = stress_table(dm, pd.Series(cm, index=dm.index), cm, None, index_cfg)
            info = {"method": "own-median", "calib_mds_um": cm}
        reports.append(TreeReport(name, table, info, res.count("jump") + res.count("reset"), res.count("spike"),
                                  res.count("gap") + res.count("gap_filled")))
    return reports


def reports_to_frame(reports: list[TreeReport]) -> pd.DataFrame:
    cols = ["coverage", "mds", "growth", "twd_predawn", "mds_expected", "si", "twd_rel", "light", "advice"]
    frames = []
    for r in reports:
        t = r.table[cols].copy()
        t.insert(0, "tree", r.name)
        frames.append(t)
    out = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def latest_status(reports: list[TreeReport]) -> list[dict]:
    """One line per tree: the last day with data, its colour and advice."""
    out = []
    for r in reports:
        t = r.table[r.table["light"] != "grey"]
        if t.empty:
            out.append(dict(tree=r.name, date=None, light="grey", advice=ADVICE["grey"]))
            continue
        last = t.iloc[-1]
        out.append(dict(tree=r.name, date=str(t.index[-1].date()), light=last["light"], advice=last["advice"],
                        mds_um=None if pd.isna(last["mds"]) else round(float(last["mds"]), 1),
                        si=None if pd.isna(last["si"]) else round(float(last["si"]), 2),
                        twd_predawn_um=None if pd.isna(last["twd_predawn"]) else round(float(last["twd_predawn"]), 1),
                        method=r.info.get("method"), jumps_fixed=r.n_jumps, spikes_removed=r.n_spikes))
    return out
