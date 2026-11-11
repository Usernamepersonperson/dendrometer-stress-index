"""Command line: run on a CSV, reproduce the README numbers, simulate test data, fetch open data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .weather import DATA_DIR, daily_weather, load_weather

SAMPLE = DATA_DIR / "sample_pfynwald_2025_jja.csv.gz"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def cmd_run(args) -> int:
    from .io import latest_status, read_dendro_csv, reports_to_frame, run_all
    from .plots import run_figure

    series = read_dendro_csv(args.input)
    dw = daily_weather(load_weather(args.weather)) if args.weather else None
    reports = run_all(series, dw, reference=args.reference, calib_days=args.calib_days, calib_start=args.calib_start)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = reports_to_frame(reports)
    table.to_csv(out / "daily_stress.csv", index=False, float_format="%.3f")
    status = latest_status(reports)
    (out / "status.json").write_text(json.dumps(status, indent=2, default=_json_default), encoding="utf-8")
    focus = next((r for r in reports if r.name not in (args.reference or [])), reports[0])
    run_figure(out / "stress_figure.png", focus.table, focus.name)
    print(json.dumps(status, indent=2, default=_json_default))
    return 0


def cmd_simulate(args) -> int:
    from .simulate import Schedule, jitter_params, random_droughts, simulate_tree, to_10min

    hw = load_weather()
    start, end = pd.Timestamp(f"{args.year}-04-15"), pd.Timestamp(f"{args.year}-09-30")
    w10 = to_10min(hw, str(start.date()), f"{end.date()} 23:00")
    rng = np.random.default_rng(args.seed)
    droughts = random_droughts(start, end, rng)
    tree = simulate_tree(w10, Schedule(droughts=droughts), jitter_params(rng), seed=args.seed * 2 + 1)
    ref = simulate_tree(w10, Schedule(), jitter_params(rng), seed=args.seed * 2 + 2)
    df = pd.DataFrame({"time": w10.index, "orchard_tree": tree.radius.round(2).to_numpy(),
                       "reference_tree": ref.radius.round(2).to_numpy()})
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: simulated, drought windows (irrigation off): "
          + ", ".join(f"{a.date()}..{(b - pd.Timedelta('1D')).date()}" for a, b in droughts))
    return 0


def cmd_fetch(args) -> int:
    from .fetch import build_sample, build_weather_fixture, fetch_all

    cache = Path(args.cache)
    fetch_all(cache)
    build_weather_fixture(cache)
    build_sample(cache)
    print(f"cache ready in {cache}")
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    return run_demo(Path(args.out), Path(args.cache), seeds=args.seeds, offline=args.offline)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dendro-stress", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="daily stress table + traffic light + figure for a dendrometer CSV")
    r.add_argument("input", help="CSV: long (time, tree, radius_um) or wide (time, one column per sensor)")
    r.add_argument("--weather", help="hourly weather CSV (time, t_air, rh[, rain]) for the VPD baseline")
    r.add_argument("--reference", nargs="*", default=None, help="names of well-watered reference sensors")
    r.add_argument("--calib-days", type=int, default=21, help="days at the start known to be well watered")
    r.add_argument("--calib-start", default=None, help="first calibration day (YYYY-MM-DD)")
    r.add_argument("--out", default="out")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("simulate", help="write a simulated season CSV (orchard tree + reference tree)")
    s.add_argument("--year", type=int, default=2025, choices=[2024, 2025])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="simulated_dendrometer.csv")
    s.set_defaults(func=cmd_simulate)
    f = sub.add_parser("fetch", help="download the open real data into cache/ (about 140 MB)")
    f.add_argument("--cache", default="cache")
    f.set_defaults(func=cmd_fetch)
    d = sub.add_parser("demo", help="reproduce the README numbers and figures")
    d.add_argument("--out", default=".")
    d.add_argument("--cache", default="cache")
    d.add_argument("--seeds", type=int, default=10)
    d.add_argument("--offline", action="store_true", help="skip the real-data part if the cache is missing")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
