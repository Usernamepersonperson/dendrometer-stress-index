import numpy as np
import pandas as pd
import pytest
from conftest import diurnal

from dendro_stress.clean import CleanConfig, clean_series, regularize
from dendro_stress.index import (
    IndexConfig,
    analyse_tree,
    fit_vpd_baseline,
    signal_intensity,
    stress_table,
    traffic_light,
)
from dendro_stress.metrics import daily_metrics, zero_growth_line
from dendro_stress.weather import daily_weather, days_since, parse_meteoswiss_hourly, saturation_vp, vpd_kpa

# ----------------------------------------------------------------------------------------------- weather


def test_saturation_vp_known_values():
    # FAO-56 table: es(20 C) = 2.338 kPa, es(0 C) = 0.6108 kPa
    assert saturation_vp(20.0) == pytest.approx(2.338, abs=1e-3)
    assert saturation_vp(0.0) == pytest.approx(0.6108, abs=1e-4)


def test_vpd_zero_at_saturation_and_clipped():
    assert vpd_kpa(25.0, 100.0) == pytest.approx(0.0)
    assert vpd_kpa(25.0, 120.0) == pytest.approx(0.0)  # RH > 100 clipped
    assert vpd_kpa(25.0, -5.0) == pytest.approx(saturation_vp(25.0))
    assert vpd_kpa(30.0, 40.0) == pytest.approx(0.6 * saturation_vp(30.0))


def test_parse_meteoswiss_shifts_to_interval_start_cet():
    text = ("station_abbr;reference_timestamp;tre200h0;ure200h0;rre150h0\n"
            "VIS;01.07.2025 12:00;25.0;40;0\nVIS;01.07.2025 13:00;26.0;35;1.2\n")
    df = parse_meteoswiss_hourly(text)
    # 12:00 UTC end -> 11:00 UTC start -> 12:00 CET
    assert df.index[0] == pd.Timestamp("2025-07-01 12:00")
    assert df["rain"].iloc[1] == pytest.approx(1.2)


def test_parse_meteoswiss_rejects_other_files():
    with pytest.raises(ValueError):
        parse_meteoswiss_hourly("a;b\n1;2\n")


def test_days_since_and_daily_weather():
    ev = pd.Series([True, False, False, True, False], index=pd.date_range("2025-07-01", periods=5))
    assert days_since(ev).tolist() == [0, 1, 2, 0, 1]
    idx = pd.date_range("2025-07-01", periods=48, freq="h")
    h = pd.DataFrame({"t_air": 20.0, "rh": 50.0, "rain": 0.0, "et0": 0.1}, index=idx)
    h.loc["2025-07-02 03:00", "rain"] = 6.0
    d = daily_weather(h)
    assert len(d) == 2
    assert d["rain"].tolist() == [0.0, 6.0]
    assert d["days_since_rain"].tolist() == [1, 0]
    assert d["vpd_max"].iloc[0] == pytest.approx(0.5 * saturation_vp(20.0))


# ------------------------------------------------------------------------------------------------- clean


def test_regularize_inserts_gaps_and_merges_duplicates():
    idx = pd.to_datetime(["2025-06-01 00:00", "2025-06-01 00:10", "2025-06-01 00:10", "2025-06-01 00:40"])
    s = regularize(pd.Series([1.0, 2.0, 4.0, 5.0], index=idx))
    assert len(s) == 5
    assert s.iloc[1] == pytest.approx(3.0)
    assert s.isna().sum() == 2


def test_regularize_needs_datetime_index():
    with pytest.raises(TypeError):
        regularize(pd.Series([1.0, 2.0]))


def test_clean_series_no_events_on_smooth_signal(series):
    res = clean_series(series)
    assert len(res.events) == 0
    np.testing.assert_allclose(res.series.to_numpy(), series.to_numpy())


def test_jump_is_detected_and_removed(series):
    s = series.copy()
    s.iloc[500:] += 150.0
    res = clean_series(s)
    assert res.count("jump") == 1
    np.testing.assert_allclose(res.series.to_numpy(), series.to_numpy(), atol=0.5)


def test_negative_jump_detected(series):
    s = series.copy()
    s.iloc[700:] -= 80.0
    res = clean_series(s)
    assert res.count("jump") == 1
    assert res.events.loc[res.events["kind"] == "jump", "size_um"].iloc[0] == pytest.approx(-80.0, abs=6)


def test_reset_to_zero_is_called_reset(series):
    s = series.copy() + 500.0
    s.iloc[400:] -= s.iloc[400]  # logger restarts at 0
    res = clean_series(s)
    assert res.count("reset") == 1
    np.testing.assert_allclose(res.series.to_numpy(), (series + 500.0).to_numpy(), atol=0.5)


def test_spike_removed_and_filled(series):
    s = series.copy()
    s.iloc[300] += 200.0
    res = clean_series(s)
    assert res.count("spike") == 1
    assert res.count("jump") == 0
    assert res.series.iloc[300] == pytest.approx(series.iloc[300], abs=1.0)


def test_gradual_rehydration_not_called_a_jump():
    # stem swells 300 um over ~3 hours after irrigation: fast, but not a one-interval step
    idx = pd.date_range("2025-07-01", periods=600, freq="10min")
    t = np.arange(600)
    v = np.where(t < 300, 0.0, 300.0 * (1 - np.exp(-(t - 300) / 9.0)))
    res = clean_series(pd.Series(v, index=idx))
    assert res.count("jump") == 0


def test_short_gap_filled_long_gap_kept(series):
    s = series.copy()
    s.iloc[100:104] = np.nan  # 40 min
    s.iloc[600:700] = np.nan  # ~17 h
    res = clean_series(s)
    assert res.series.iloc[100:104].notna().all()
    assert res.series.iloc[600:700].isna().all()
    assert res.count("gap_filled") == 1 and res.count("gap") == 1


def test_jump_across_short_gap_corrected(series):
    s = series.copy()
    s.iloc[400:403] = np.nan
    s.iloc[403:] += 120.0
    res = clean_series(s, CleanConfig())
    assert res.count("jump") == 1
    assert abs(res.series.iloc[-1] - series.iloc[-1]) < 0.5


def test_clean_empty_series():
    res = clean_series(pd.Series([], index=pd.DatetimeIndex([]), dtype=float))
    assert res.series.empty


# ----------------------------------------------------------------------------------------------- metrics


def test_zero_growth_line_known_values():
    idx = pd.date_range("2025-01-01", periods=6, freq="10min")
    r = pd.Series([0.0, 2.0, 1.0, np.nan, 3.0, 2.5], index=idx)
    gro, twd = zero_growth_line(r)
    assert gro.tolist() == [0.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    assert twd.iloc[2] == pytest.approx(1.0)
    assert np.isnan(twd.iloc[3])
    assert twd.iloc[5] == pytest.approx(0.5)


def test_mds_matches_known_amplitude(series):
    d = daily_metrics(series)
    # interior days: the amplitude is exactly 50 um
    assert d["mds"].iloc[1:-1].to_numpy() == pytest.approx(50.0, abs=0.5)
    assert d["valid"].all()


def test_growth_and_twd_with_steady_growth():
    s = diurnal(days=8, amp=40.0, growth_per_day=10.0)
    d = daily_metrics(s)
    assert d["growth"].iloc[2:-1].to_numpy() == pytest.approx(10.0, abs=0.5)
    assert (d["twd_predawn"].dropna() >= 0).all()


def test_drought_raises_predawn_twd():
    s = diurnal(days=10, amp=50.0)
    s.iloc[5 * 144:] -= np.linspace(0, 200, len(s) - 5 * 144)  # stem fails to refill: slow decline
    d = daily_metrics(s)
    assert d["twd_predawn"].iloc[-1] > 150
    assert d["twd_predawn"].iloc[2] < 5


def test_low_coverage_day_masked(series):
    s = series.copy()
    s.iloc[144 * 3: 144 * 3 + 60] = np.nan  # 10 h missing on day 4
    d = daily_metrics(s)
    assert not d["valid"].iloc[3]
    assert np.isnan(d["mds"].iloc[3])
    assert d["valid"].iloc[4]


def test_daily_metrics_empty():
    d = daily_metrics(pd.Series([], index=pd.DatetimeIndex([]), dtype=float))
    assert d.empty and "mds" in d.columns


# ------------------------------------------------------------------------------------------------- index


def test_fit_vpd_baseline_recovers_line_and_ignores_outlier():
    v = pd.Series(np.linspace(0.5, 3.0, 20))
    m = 40.0 + 30.0 * v
    m.iloc[5] = 400.0
    a, b = fit_vpd_baseline(m, v)
    assert a == pytest.approx(40.0, abs=1.0)
    assert b == pytest.approx(30.0, abs=0.5)


def test_fit_vpd_baseline_needs_data():
    with pytest.raises(ValueError):
        fit_vpd_baseline(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))


def test_fit_vpd_baseline_slope_not_negative():
    v = pd.Series(np.linspace(0.5, 3.0, 10))
    a, b = fit_vpd_baseline(100 - 10 * v, v)
    assert b == 0.0


def test_signal_intensity_smoothing():
    idx = pd.date_range("2025-07-01", periods=5)
    si = signal_intensity(pd.Series([100, 100, 300, 100, 100.0], index=idx), pd.Series(100.0, index=idx), 3)
    assert si.max() == pytest.approx(1.0)  # single-day outlier removed by the 3-day median
    raw = signal_intensity(pd.Series([100, 100, 300, 100, 100.0], index=idx), pd.Series(100.0, index=idx), 1)
    assert raw.iloc[2] == pytest.approx(3.0)


def test_traffic_light_thresholds():
    idx = pd.date_range("2025-07-01", periods=5)
    si = pd.Series([1.0, 1.35, 1.7, 1.0, np.nan], index=idx)
    tw = pd.Series([0.1, 0.1, 0.1, 3.5, np.nan], index=idx)
    light = traffic_light(si, tw, cfg=IndexConfig())
    assert light.tolist() == ["green", "amber", "red", "red", "grey"]


def test_traffic_light_invalid_day_is_grey():
    idx = pd.date_range("2025-07-01", periods=2)
    light = traffic_light(pd.Series([1.0, 1.0], index=idx), pd.Series([0.0, 0.0], index=idx),
                          valid=pd.Series([True, False], index=idx))
    assert light.tolist() == ["green", "grey"]


def test_stress_table_rejects_bad_calibration(series):
    d = daily_metrics(series)
    with pytest.raises(ValueError):
        stress_table(d, d["mds"], 0.0)


def test_analyse_tree_reference_scale_and_alarm():
    days = pd.date_range("2025-06-01", periods=40)
    ref = pd.Series(100.0, index=days)
    mds = pd.Series(150.0, index=days)  # bigger tree: 1.5 x the reference when well watered
    mds.iloc[30:] = 300.0  # then shrinks twice as much as expected
    daily = pd.DataFrame({"mds": mds, "valid": True, "twd_predawn": 5.0}, index=days)
    st, info = analyse_tree(daily, pd.Series(2.0, index=days), days[:20], reference_mds=ref)
    assert info["method"] == "reference" and info["scale"] == pytest.approx(1.5)
    assert (st["light"].iloc[:28] == "green").all()
    assert (st["light"].iloc[32:] == "red").all()


def test_analyse_tree_vpd_method():
    days = pd.date_range("2025-06-01", periods=30)
    vpd = pd.Series(np.tile([1.0, 2.0, 3.0], 10), index=days)
    mds = 20.0 + 40.0 * vpd
    daily = pd.DataFrame({"mds": mds, "valid": True, "twd_predawn": 1.0}, index=days)
    st, info = analyse_tree(daily, vpd, days[:15])
    assert info["method"] == "vpd" and info["b"] == pytest.approx(40.0, abs=1e-6)
    assert st["si"].dropna().to_numpy() == pytest.approx(1.0)
    assert (st["light"] == "green").all()


def test_analyse_tree_too_few_calibration_days():
    days = pd.date_range("2025-06-01", periods=10)
    daily = pd.DataFrame({"mds": 50.0, "valid": False, "twd_predawn": 1.0}, index=days)
    with pytest.raises(ValueError):
        analyse_tree(daily, pd.Series(2.0, index=days), days[:5])


def test_low_vpd_days_give_no_si():
    days = pd.date_range("2025-06-01", periods=5)
    daily = pd.DataFrame({"mds": 80.0, "valid": True, "twd_predawn": 1.0}, index=days)
    vpd = pd.Series([2.0, 0.1, 0.1, 0.1, 2.0], index=days)
    st = stress_table(daily, pd.Series(80.0, index=days), 80.0, vpd, IndexConfig(smooth_days=1))
    assert st["si"].isna().sum() == 3
    assert (st["light"] == "green").all()  # TWD still known, so not grey
