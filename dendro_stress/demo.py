"""`python -m dendro_stress demo`: all experiments, results/metrics.json, docs figures, example run."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .cli import SAMPLE, _json_default
from .evaluate import glitch_experiment, real_glitch_series, run_real, run_simulated
from .weather import daily_weather, load_weather

FIG_SEASON = (2025, 3)  # (weather year, seed) shown in the figures


def run_demo(out: Path, cache: Path, seeds: int = 10, offline: bool = False) -> int:
    t0 = time.time()
    hw = load_weather()
    results: dict = {"labels": {"simulated": "seeded simulator driven by real MeteoSwiss Visp weather, Apr 15-Sep 30",
                                "real": "open data: Pfynwald (Valais) Scots pines, EnviDat CC BY 4.0, Apr-Oct "
                                        "2024 and 2025; weather MeteoSwiss Visp"}}
    print("simulated seasons ...", flush=True)
    sim, seasons = run_simulated(hw, n_seeds=seeds)
    results["simulated"] = sim
    print(json.dumps(sim, indent=1, default=_json_default), flush=True)
    results["simulated_glitches"] = glitch_experiment([s["tree"].radius for s in seasons[:10]], seed=5)
    season = next(s for s in seasons if (s["year"], s["seed"]) == FIG_SEASON)
    have_cache = all((cache / f"pfynwald_{y}.npz").exists() for y in (2024, 2025))
    if not have_cache and not offline:
        from .fetch import fetch_all

        print("real data missing, fetching (about 140 MB) ...", flush=True)
        fetch_all(cache)
        have_cache = True
    keep = None
    if have_cache:
        print("real data ...", flush=True)
        real, keep = run_real(cache, hw)
        results["real"] = real
        series, clean_stats = real_glitch_series(keep)
        results["real_glitches"] = dict(unmodified=clean_stats, injected=glitch_experiment(series, seed=13))
    else:
        print("offline and no cache: real-data part skipped", flush=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default),
                                                  encoding="utf-8")
    # example run on the committed real sample (VPD baseline from Visp weather)
    from .io import latest_status, read_dendro_csv, reports_to_frame, run_all

    sample = read_dendro_csv(SAMPLE)
    reports = run_all(sample, daily_weather(hw), calib_days=21, calib_start="2025-06-01")
    reports_to_frame(reports).to_csv(out / "results" / "example_sample_daily.csv", index=False, float_format="%.3f")
    (out / "results" / "example_sample_status.json").write_text(
        json.dumps(latest_status(reports), indent=2, default=_json_default), encoding="utf-8")
    if keep is not None:
        from .plots import main_figure, post_figure

        (out / "docs").mkdir(parents=True, exist_ok=True)
        gl = results["real_glitches"]["injected"]
        main_figure(out / "docs" / "figure.png", results["real"], keep, sim, season, gl)
        post_figure(out / "docs" / "post-figure.png", keep, results["real"])
    print(f"done in {time.time() - t0:.0f} s -> {out / 'results' / 'metrics.json'}", flush=True)
    return 0


