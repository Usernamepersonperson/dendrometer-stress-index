"""Experiments behind the README numbers.

simulated : seeded seasons driven by real Visp weather, known drought windows -> detection rate, delay,
            false stress days; glitch injection with exact truth.
real      : Pfynwald 2024-2025 (irrigated / control / roof-covered Scots pines), Visp weather -> relation of MDS
            and TWD to VPD and dry spells, traffic-light share per treatment, TWD agreement with the published
            series, glitch handling on real series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .clean import CleanConfig, clean_series
from .index import IndexConfig, analyse_tree
from .metrics import daily_metrics, zero_growth_line
from .simulate import Schedule, inject_glitches, jitter_params, random_droughts, simulate_tree, to_10min
from .weather import daily_weather

YEARS = (2024, 2025)
SEASON = ("04-15", "09-30")
STRESS_REW = 0.4  # truth: soil water limits transpiration below this relative extractable water
WET_REW = 0.6  # truth: well watered above this


def _non_green(light: pd.Series) -> pd.Series:
    return light.isin(["amber", "red"])


# --------------------------------------------------------------------------------------------- simulated
def simulate_season(hourly: pd.DataFrame, year: int, seed: int, cfg: IndexConfig | None = None) -> dict:
    start, end = pd.Timestamp(f"{year}-{SEASON[0]}"), pd.Timestamp(f"{year}-{SEASON[1]}")
    w10 = to_10min(hourly, str(start.date()), f"{end.date()} 23:00")
    dw = daily_weather(hourly.loc[start:end + pd.Timedelta("23h")])
    rng = np.random.default_rng(seed)
    droughts = random_droughts(start, end, rng)
    tree = simulate_tree(w10, Schedule(droughts=droughts), jitter_params(rng), seed=seed * 2 + 1)
    ref = simulate_tree(w10, Schedule(), jitter_params(rng), seed=seed * 2 + 2)
    dm, dref = daily_metrics(tree.radius), daily_metrics(ref.radius)
    calib = dm.index[:30]
    st_ref, _ = analyse_tree(dm, dw["vpd_max"], calib, reference_mds=dref["mds"], cfg=cfg)
    st_vpd, _ = analyse_tree(dm, dw["vpd_max"], calib, cfg=cfg)
    return dict(year=year, seed=seed, droughts=droughts, truth=tree.daily_truth, daily=dm, ref=st_ref, vpd=st_vpd,
                tree=tree, weather=dw)


def score_season(season: dict, method: str) -> tuple[list[dict], dict]:
    """Per-drought detection records and well-watered false-alarm counts for one method ('ref' or 'vpd')."""
    truth, st = season["truth"], season[method]
    flag = _non_green(st["light"]).reindex(truth.index).fillna(False)
    recs = []
    for a, b in season["droughts"]:
        win = truth.loc[a:b - pd.Timedelta("1D")]
        stressed = win.index[win["rew_min"] < STRESS_REW]
        if len(stressed) == 0:
            recs.append(dict(start=a, end=b, reached_stress=False))
            continue
        onset = stressed[0]
        look = flag.loc[onset - pd.Timedelta("5D"):b + pd.Timedelta("2D")]
        hits = look.index[look.to_numpy()]
        rec = dict(start=a, end=b, reached_stress=True, onset=onset, detected=len(hits) > 0)
        if len(hits):
            rec["delay_days"] = int((hits[0] - onset).days)
        # recovery: first green day after the drought ends
        after = st["light"].loc[b:]
        green = after.index[after.to_numpy() == "green"]
        if len(green):
            rec["recovery_days"] = int((green[0] - b).days)
        recs.append(rec)
    near = pd.Series(False, index=truth.index)
    for a, b in season["droughts"]:
        near |= (truth.index >= a) & (truth.index < b + pd.Timedelta("3D"))
    wet = (truth["rew_min"] > WET_REW) & ~near & st["light"].reindex(truth.index).ne("grey")
    fa = dict(wet_days=int(wet.sum()), false_days=int((flag & wet).sum()))
    return recs, fa


def run_simulated(hourly: pd.DataFrame, n_seeds: int = 10, cfg: IndexConfig | None = None) -> tuple[dict, list]:
    seasons = [simulate_season(hourly, y, s, cfg) for y in YEARS for s in range(n_seeds)]
    out: dict = {"seasons": len(seasons), "weather_years": list(YEARS), "seeds_per_year": n_seeds}
    for method in ("ref", "vpd"):
        recs, wet, false = [], 0, 0
        for se in seasons:
            r, fa = score_season(se, method)
            recs += r
            wet += fa["wet_days"]
            false += fa["false_days"]
        rs = [r for r in recs if r["reached_stress"]]
        det = [r for r in rs if r["detected"]]
        delays = np.array([r["delay_days"] for r in det])
        rec_days = np.array([r["recovery_days"] for r in rs if "recovery_days" in r])
        out[method] = dict(
            droughts=len(recs), reached_stress=len(rs), detected=len(det),
            detection_rate=len(det) / len(rs) if rs else float("nan"),
            delay_median_days=float(np.median(delays)) if len(delays) else float("nan"),
            delay_p25_days=float(np.percentile(delays, 25)) if len(delays) else float("nan"),
            delay_p75_days=float(np.percentile(delays, 75)) if len(delays) else float("nan"),
            delay_max_days=int(delays.max()) if len(delays) else None,
            detected_within_3d=float(np.mean(delays <= 3)) if len(delays) else float("nan"),
            wet_days=wet, false_stress_days=false, false_stress_rate=false / wet if wet else float("nan"),
            recovery_median_days=float(np.median(rec_days)) if len(rec_days) else float("nan"),
        )
    return out, seasons


def glitch_experiment(series_list: list[pd.Series], seed: int = 7, tol_steps: int = 2,
                      cfg: CleanConfig | None = None) -> dict:
    """Inject known faults into clean series; measure detection and the daily-metric error with and without cleaning."""
    rng = np.random.default_rng(seed)
    tot = {k: [0, 0] for k in ("jump", "reset", "spike")}  # [injected, found]
    false_events, days = 0, 0
    mae_raw, mae_clean, twd_raw, twd_clean = [], [], [], []
    for s in series_list:
        s = s.astype(float)
        faulty, truth = inject_glitches(s, rng)
        res = clean_series(faulty, cfg)
        ev = res.events
        found_idx = set()
        for _, t in truth.iterrows():
            if t["kind"] == "gap":
                continue
            kinds = ("jump", "reset") if t["kind"] in ("jump", "reset") else ("spike",)
            near = ev[ev["kind"].isin(kinds) & (abs(ev["time"] - t["time"]) <= pd.Timedelta(minutes=10 * tol_steps))]
            tot[t["kind"]][0] += 1
            if len(near):
                tot[t["kind"]][1] += 1
                found_idx.update(near.index)
        det = ev[ev["kind"].isin(["jump", "reset", "spike"])]
        false_events += len(set(det.index) - found_idx)
        d_true = daily_metrics(s)
        d_raw = daily_metrics(faulty)
        d_cln = daily_metrics(res.series.reindex(s.index))
        days += int(d_true["valid"].sum())
        both = d_true["valid"] & d_raw["valid"] & d_cln["valid"]
        mae_raw.append(np.abs(d_raw["mds"] - d_true["mds"])[both].to_numpy())
        mae_clean.append(np.abs(d_cln["mds"] - d_true["mds"])[both].to_numpy())
        twd_raw.append(np.abs(d_raw["twd_predawn"] - d_true["twd_predawn"])[both].to_numpy())
        twd_clean.append(np.abs(d_cln["twd_predawn"] - d_true["twd_predawn"])[both].to_numpy())

    def cat(x):
        x = np.concatenate(x)
        return x[np.isfinite(x)]

    out = {f"{k}_recall": (v[1] / v[0] if v[0] else float("nan")) for k, v in tot.items()}
    out.update({f"{k}_injected": v[0] for k, v in tot.items()})
    out.update(series=len(series_list), series_days=days, false_detections=false_events,
               false_detections_per_100_days=100 * false_events / days if days else float("nan"),
               mds_mae_raw_um=float(np.mean(cat(mae_raw))), mds_mae_cleaned_um=float(np.mean(cat(mae_clean))),
               twd_predawn_mae_raw_um=float(np.mean(cat(twd_raw))),
               twd_predawn_mae_cleaned_um=float(np.mean(cat(twd_clean))))
    return out


# ------------------------------------------------------------------------------------------------- real
def real_daily(cache: Path, year: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Daily metrics for every tree of a year (stacked), the year arrays and the tree table."""
    from .fetch import load_year

    d = load_year(cache, year)
    rows = []
    for c in range(d["radius"].shape[1]):
        r = pd.Series(d["radius"][:, c], index=d["time"]).astype(float)
        dm = daily_metrics(r)
        dm["col"] = c
        rows.append(dm)
    return pd.concat(rows), d, d["trees"]


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 5:
        return float("nan"), len(df)
    return float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1]), len(df)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 5:
        return float("nan"), len(df)
    return _pearson(df.iloc[:, 0].rank(), df.iloc[:, 1].rank())


def longest_dry_spell(dw: pd.DataFrame, start: str, end: str, wet_mm: float = 5.0) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Longest run of days with rain < wet_mm in [start, end]."""
    w = dw.loc[start:end, "rain"].fillna(0.0) < wet_mm
    best, cur_start, best_rng = 0, None, (None, None)
    for d, dry in w.items():
        if dry:
            cur_start = cur_start or d
            n = (d - cur_start).days + 1
            if n > best:
                best, best_rng = n, (cur_start, d)
        else:
            cur_start = None
    return best_rng


MAIN_GROUPS = ("irrigation", "control", "roof")


def _wide(daily: pd.DataFrame, col: str, days: pd.DatetimeIndex, cols: list[int]) -> pd.DataFrame:
    return daily.pivot_table(index=daily.index, columns="col", values=col, dropna=False).reindex(index=days, columns=cols)


def run_real(cache: Path, hourly: pd.DataFrame, cfg: IndexConfig | None = None) -> tuple[dict, dict]:
    dw_all = daily_weather(hourly)
    out: dict = {}
    keep: dict = {}
    for year in YEARS:
        daily, arrs, trees = real_daily(cache, year)
        dw = dw_all.loc[f"{year}-04-01":f"{year}-10-31"]
        s0, s1 = f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}"
        calib = pd.date_range(f"{year}-04-20", f"{year}-05-20")
        summer = pd.date_range(f"{year}-06-01", f"{year}-09-15")
        trees = trees.assign(group=trees["treatment"])
        cols_by = {g: trees.index[trees["group"] == g].tolist() for g in trees["group"].unique()}
        allcols = list(range(arrs["radius"].shape[1]))
        days = pd.date_range(s0, s1)

        mds_wide, valid_wide, twd_wide = (_wide(daily, c, days, allcols) for c in ("mds", "valid", "twd_predawn"))
        valid_wide = valid_wide.fillna(0)
        res_year: dict = {"trees": {g: len(c) for g, c in cols_by.items()}}
        # TWD agreement with the published series (zero-growth line, 10-min, whole year)
        r_list, mae_list = [], []
        for c in range(arrs["radius"].shape[1]):
            _, twd = zero_growth_line(pd.Series(arrs["radius"][:, c], index=arrs["time"]).astype(float))
            pub = arrs["twd"][:, c].astype(float)
            ok = np.isfinite(pub) & np.isfinite(twd.to_numpy())
            if ok.sum() > 1000:
                r_list.append(np.corrcoef(pub[ok], twd.to_numpy()[ok])[0, 1])
                mae_list.append(np.mean(np.abs(pub[ok] - twd.to_numpy()[ok])))
        res_year["twd_vs_published"] = dict(trees=len(r_list), r_median=float(np.median(r_list)),
                                            r_min=float(np.min(r_list)), mae_median_um=float(np.median(mae_list)))
        # per-tree stress tables
        lights: dict[str, dict[str, list]] = {}
        light_tables: dict[int, pd.DataFrame] = {}
        rel_mds = {}
        for g, cols in cols_by.items():
            for c in cols:
                dm = daily[daily["col"] == c].drop(columns="col").loc[s0:s1]
                ref_cols = [k for k in cols_by.get("irrigation", []) if k != c]
                ref = mds_wide[ref_cols].where(valid_wide[ref_cols].astype(bool)).median(axis=1)
                for method, refser in (("ref", ref), ("vpd", None)):
                    try:
                        st, info = analyse_tree(dm, dw["vpd_max"], calib, reference_mds=refser, cfg=cfg)
                    except ValueError:
                        continue
                    sl = st["light"].reindex(summer)
                    sl = sl[sl.notna() & (sl != "grey")]
                    lights.setdefault(method, {}).setdefault(g, []).append(
                        (len(sl), int(_non_green(sl).sum()), int((sl == "red").sum())))
                    if method == "ref":
                        light_tables[c] = st
                        rel_mds[c] = dm["mds"] / info["calib_mds_um"]
        res_year["summer_window"] = [str(summer[0].date()), str(summer[-1].date())]
        res_year["light_share"] = {
            m: {g: dict(tree_days=sum(x[0] for x in v), non_green=sum(x[1] for x in v) / max(1, sum(x[0] for x in v)),
                        red=sum(x[2] for x in v) / max(1, sum(x[0] for x in v)), trees=len(v))
                for g, v in d.items()} for m, d in lights.items()}
        # relations with weather (treatment medians)
        rel = {}
        rain_free = dw["rain"].fillna(0) < 1.0
        for g in MAIN_GROUPS:
            cols = [c for c in cols_by.get(g, []) if c in rel_mds]
            if not cols:
                continue
            m = pd.concat([rel_mds[c] for c in cols], axis=1).median(axis=1)
            tw = twd_wide[cols].median(axis=1)
            vpd = dw["vpd_max"].reindex(m.index)
            r_mds, n_mds = _pearson(m[rain_free.reindex(m.index).fillna(False)], vpd)
            r_twd, n_twd = _spearman(tw.loc[f"{year}-06-01":s1], dw["days_since_rain"].reindex(tw.index))
            rel[g] = dict(mds_vs_vpdmax_r=r_mds, mds_days=n_mds, twd_predawn_vs_days_since_rain_rho=r_twd,
                          twd_days=n_twd, twd_predawn_summer_median_um=float(tw.reindex(summer).median()))
        res_year["weather_relation"] = rel
        a, b = longest_dry_spell(dw, f"{year}-06-01", f"{year}-09-15")
        spell = {"start": str(a.date()), "end": str(b.date()), "days": int((b - a).days + 1)}
        for g in MAIN_GROUPS:
            cols = cols_by.get(g, [])
            tw = twd_wide[cols].median(axis=1)
            spell[f"{g}_twd_change_um"] = float(tw.loc[b - pd.Timedelta("2D"):b].mean()
                                                - tw.loc[a:a + pd.Timedelta("2D")].mean())
            fl = [_non_green(light_tables[c]["light"].loc[a:b]).mean() for c in cols if c in light_tables]
            spell[f"{g}_non_green_share"] = float(np.mean(fl)) if fl else float("nan")
        res_year["longest_dry_spell"] = spell
        out[str(year)] = res_year
        keep[year] = dict(daily=daily, trees=trees, cols_by=cols_by, twd_wide=twd_wide, weather=dw,
                          lights=light_tables, arrs=arrs)
    return out, keep


def real_glitch_series(keep: dict, n_series: int = 30, seed: int = 11) -> tuple[list[pd.Series], dict]:
    """Season slices (Apr 15 - Sep 30) of real series with >= 97 % coverage; also cleans them unmodified."""
    rng = np.random.default_rng(seed)
    cands = []
    for year, k in keep.items():
        arrs = k["arrs"]
        sel = (arrs["time"] >= pd.Timestamp(f"{year}-{SEASON[0]}")) & (arrs["time"] < pd.Timestamp(f"{year}-10-01"))
        for c in range(arrs["radius"].shape[1]):
            v = arrs["radius"][sel, c].astype(float)
            if np.isfinite(v).mean() >= 0.97:
                cands.append(pd.Series(v, index=arrs["time"][sel]).interpolate(limit_area="inside"))
    # unmodified real series: anything flagged is a false alarm or a real fast change
    false_jump, false_spike, days = 0, 0, 0
    for s in cands:
        res = clean_series(s)
        false_jump += res.count("jump") + res.count("reset")
        false_spike += res.count("spike")
        days += len(s) / 144
    pick = rng.choice(len(cands), size=min(n_series, len(cands)), replace=False)
    stats = dict(clean_series=len(cands), series_days=round(days),
                 jumps_flagged_per_100_days=100 * false_jump / days, spikes_flagged_per_100_days=100 * false_spike / days)
    return [cands[i].dropna() for i in sorted(pick)], stats
