import json

import numpy as np
import pandas as pd
import pytest
from conftest import diurnal

from dendro_stress.cli import main
from dendro_stress.evaluate import glitch_experiment, longest_dry_spell, score_season, simulate_season
from dendro_stress.io import latest_status, read_dendro_csv, reports_to_frame, run_all
from dendro_stress.simulate import Schedule, TreeParams, inject_glitches, random_droughts, simulate_tree, to_10min
from dendro_stress.weather import daily_weather, load_weather


@pytest.fixture(scope="module")
def weather():
    return load_weather()


@pytest.fixture(scope="module")
def w10(weather):
    return to_10min(weather, "2025-06-01", "2025-07-31 23:00")


# ---------------------------------------------------------------------------------------------- weather data


def test_weather_fixture_covers_two_seasons(weather):
    assert weather.index.min() <= pd.Timestamp("2024-04-15")
    assert weather.index.max() >= pd.Timestamp("2025-09-30")
    assert weather["vpd"].between(0, 8).all()


def test_to_10min_conserves_rain(weather):
    w = to_10min(weather, "2025-06-01", "2025-06-30 23:00")
    hourly = weather.loc["2025-06-01":"2025-06-30 23:00", "rain"].sum()
    assert w["rain"].sum() == pytest.approx(hourly, rel=1e-6)
    assert len(w) == 30 * 144


# ------------------------------------------------------------------------------------------------ simulator


def test_simulation_is_deterministic(w10):
    a = simulate_tree(w10, Schedule(), seed=4)
    b = simulate_tree(w10, Schedule(), seed=4)
    np.testing.assert_array_equal(a.radius.to_numpy(), b.radius.to_numpy())


def test_irrigated_tree_stays_wet(w10):
    res = simulate_tree(w10, Schedule(), TreeParams(), seed=1)
    assert res.daily_truth["rew_min"].min() > 0.6


def test_drought_window_dries_soil_and_raises_twd(w10):
    dr = [(pd.Timestamp("2025-06-20"), pd.Timestamp("2025-07-15"))]
    res = simulate_tree(w10, Schedule(droughts=dr), TreeParams(), seed=1)
    t = res.daily_truth
    assert t.loc["2025-07-10", "rew_min"] < 0.4
    assert t.loc["2025-06-15", "rew_min"] > 0.6
    assert t.loc["2025-07-01":"2025-07-14", "in_drought"].all()
    # irrigation resumes after the window
    assert t.loc["2025-07-15":"2025-07-17", "irrigation_mm"].sum() > 0


def test_random_droughts_do_not_overlap():
    rng = np.random.default_rng(0)
    dr = random_droughts(pd.Timestamp("2025-04-15"), pd.Timestamp("2025-09-30"), rng, n=2)
    assert len(dr) == 2
    (a1, b1), (a2, b2) = dr
    assert b1 + pd.Timedelta(days=20) <= a2
    assert a1 >= pd.Timestamp("2025-05-15")


def test_inject_glitches_truth_table():
    s = diurnal(days=20, noise=0.5)
    faulty, truth = inject_glitches(s, np.random.default_rng(1), n_jumps=2, n_resets=1, n_spikes=4, n_gaps=2)
    assert truth["kind"].value_counts().to_dict() == {"spike": 4, "jump": 2, "gap": 2, "reset": 1}
    assert faulty.isna().sum() >= 4
    reset_i = int(truth.loc[truth["kind"] == "reset", "i"].iloc[0])
    assert faulty.iloc[reset_i] == pytest.approx(0.0, abs=1e-9)


def test_glitch_experiment_recovers_faults():
    series = [diurnal(days=25, noise=0.5, seed=k) for k in range(3)]
    res = glitch_experiment(series, seed=2)
    assert res["jump_recall"] >= 0.8 and res["spike_recall"] >= 0.8
    assert res["mds_mae_cleaned_um"] < res["mds_mae_raw_um"]


def test_simulated_season_detects_drought(weather):
    season = simulate_season(weather, 2025, seed=3)
    recs, fa = score_season(season, "ref")
    reached = [r for r in recs if r["reached_stress"]]
    assert reached and all(r["detected"] for r in reached)
    assert fa["wet_days"] > 50


def test_longest_dry_spell():
    d = pd.DataFrame({"rain": [10, 0, 0, 0, 7, 0, 0, 1, 0, 9.0]}, index=pd.date_range("2025-07-01", periods=10))
    a, b = longest_dry_spell(d, "2025-07-01", "2025-07-10")
    assert (a, b) == (pd.Timestamp("2025-07-06"), pd.Timestamp("2025-07-09"))


# ------------------------------------------------------------------------------------------------------- io


def _long_csv(tmp_path, n_days=30):
    rows = []
    for name, amp, k in (("ref_a", 60, 0), ("ref_b", 64, 1), ("tree_x", 80, 2)):
        s = diurnal(days=n_days, amp=amp, noise=0.3, seed=k)
        if name == "tree_x":
            s.iloc[-8 * 144:] *= 1.8  # shrinks much more in the last 8 days
        rows.append(pd.DataFrame({"time": s.index, "tree": name, "radius_um": s.to_numpy()}))
    p = tmp_path / "long.csv"
    pd.concat(rows).to_csv(p, index=False)
    return p


def test_read_long_and_wide(tmp_path):
    p = _long_csv(tmp_path, 3)
    long = read_dendro_csv(p)
    assert sorted(long) == ["ref_a", "ref_b", "tree_x"]
    wide = pd.read_csv(p).pivot(index="time", columns="tree", values="radius_um").reset_index()
    pw = tmp_path / "wide.csv"
    wide.to_csv(pw, index=False)
    w = read_dendro_csv(pw)
    assert sorted(w) == sorted(long)
    assert len(w["ref_a"]) == len(long["ref_a"])


def test_read_rejects_missing_time(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        read_dendro_csv(p)


def test_read_rejects_non_numeric(tmp_path):
    p = tmp_path / "bad2.csv"
    pd.DataFrame({"time": ["2025-01-01 00:00", "2025-01-01 00:10"], "s1": ["x", "y"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        read_dendro_csv(p)


def test_run_all_with_reference_flags_stressed_tree(tmp_path):
    series = read_dendro_csv(_long_csv(tmp_path))
    reports = run_all(series, reference=["ref_a", "ref_b"], calib_days=14)
    status = {s["tree"]: s for s in latest_status(reports)}
    assert status["tree_x"]["light"] == "red"
    assert status["ref_a"]["light"] == "green"
    frame = reports_to_frame(reports)
    assert {"tree", "date", "mds", "si", "light", "advice"} <= set(frame.columns)


def test_run_all_unknown_reference(tmp_path):
    series = read_dendro_csv(_long_csv(tmp_path, 5))
    with pytest.raises(ValueError):
        run_all(series, reference=["nope"])


def test_run_all_without_weather_or_reference(tmp_path):
    series = read_dendro_csv(_long_csv(tmp_path))
    reports = run_all(series, calib_days=14)
    assert reports[0].info["method"] == "own-median"


def test_cli_simulate_then_run(tmp_path):
    csv = tmp_path / "sim.csv"
    assert main(["simulate", "--year", "2025", "--seed", "1", "--out", str(csv)]) == 0
    out = tmp_path / "out"
    assert main(["run", str(csv), "--reference", "reference_tree", "--out", str(out)]) == 0
    status = json.loads((out / "status.json").read_text())
    assert {s["tree"] for s in status} == {"orchard_tree", "reference_tree"}
    assert (out / "stress_figure.png").stat().st_size > 10_000
    daily = pd.read_csv(out / "daily_stress.csv")
    assert set(daily["light"]) <= {"green", "amber", "red", "grey"}


def test_cli_run_on_real_sample_with_weather(tmp_path):
    from dendro_stress.cli import SAMPLE
    from dendro_stress.weather import WEATHER_FIXTURE

    out = tmp_path / "o"
    assert main(["run", str(SAMPLE), "--weather", str(WEATHER_FIXTURE), "--calib-start", "2025-06-01",
                 "--out", str(out)]) == 0
    daily = pd.read_csv(out / "daily_stress.csv")
    assert daily["tree"].nunique() == 6
    assert daily["mds"].median() > 10


def test_daily_weather_on_fixture(weather):
    d = daily_weather(weather)
    assert d.loc["2025-07-01":"2025-07-31", "vpd_max"].mean() > 1.0
