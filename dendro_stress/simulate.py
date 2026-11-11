"""Seeded dendrometer simulator driven by real hourly weather (simplified soil-plant water model).

Per 10-min step:
  soil bucket W (mm): + rain + irrigation - transpiration, capped at capacity; REW = W / capacity
  transpiration T = Kc * ET0 * min(1, REW / 0.4)   (stomata close below 40 % relative extractable water)
  soil water potential psi_s = -0.03 * REW**-2 MPa (floored at -3)
  stem water potential target psi* = psi_s - R * T / T_ref, with soil-root resistance R rising as soil dries
  elastic radius e relaxes to k_el * psi* with time constant tau (stem capacitance)
  growth g increases only while the stem is turgid (psi_stem above -0.8 MPa) and T_air > 5 C
  reading r = g + e + thermal artefact + sensor noise

Drought periods (irrigation off, rain excluded) and the irrigation schedule are injected, so the true water
status (REW) is known every day. This is a test bench, not a calibrated crop model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TreeParams:
    capacity_mm: float = 100.0
    kc: float = 0.9
    t_ref_mm_h: float = 0.6
    r0_mpa: float = 0.8  # midday drop of stem water potential at T = T_ref, wet soil
    k_el_um_per_mpa: float = 190.0
    tau_h: float = 1.5
    growth_um_day: float = 18.0
    psi_growth_mpa: float = -0.8
    noise_um: float = 0.7
    thermal_um_per_k: float = 0.3
    daily_var: float = 0.15  # day-to-day biological variability of the shrink amplitude (log-normal sigma)
    bark_um_per_mm: float = 2.0  # bark swelling after rain (wetting of the outer bark), decays over ~1.5 days
    bark_max_um: float = 30.0


@dataclass
class Schedule:
    irrigate: bool = True
    refill_below: float = 0.7  # irrigate at night when REW falls below this
    refill_to: float = 0.85
    droughts: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)  # [start, end) irrigation off,
    # rain excluded


@dataclass
class SimResult:
    radius: pd.Series  # micrometres, 10-min
    rew: pd.Series  # 10-min relative extractable water
    daily_truth: pd.DataFrame  # rew_min, rew_mean, in_drought, irrigation_mm
    params: TreeParams


def to_10min(hourly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Hourly weather (interval start, CET) -> 10-min grid. Rain and ET0 are spread evenly over the hour."""
    h = hourly.loc[start:end].copy()
    if h.empty:
        raise ValueError(f"no weather between {start} and {end}")
    grid = pd.date_range(h.index.min(), h.index.max() + pd.Timedelta("50min"), freq="10min")
    w = pd.DataFrame(index=grid)
    for col in ("t_air", "rh", "ghi", "vpd"):
        w[col] = h[col].reindex(grid).interpolate(limit_direction="both")
    for col in ("rain", "et0"):
        w[col] = h[col].reindex(grid, method="ffill") / 6.0
    if w["et0"].isna().all():
        # simple fallback: radiation and VPD driven (mm per 10 min)
        w["et0"] = (0.0011 * w["ghi"].clip(lower=0) * (0.5 + 0.5 * w["vpd"].clip(0, 4))) / 6.0
    w["et0"] = w["et0"].fillna(0.0).clip(lower=0)
    w["rain"] = w["rain"].fillna(0.0).clip(lower=0)
    return w


def simulate_tree(w10: pd.DataFrame, schedule: Schedule, params: TreeParams | None = None, seed: int = 0,
                  w0_rew: float = 0.9) -> SimResult:
    p = params or TreeParams()
    rng = np.random.default_rng(seed)
    n = len(w10)
    dt_h = 1.0 / 6.0
    idx = w10.index
    in_dr = np.zeros(n, dtype=bool)
    for a, b in schedule.droughts:
        in_dr |= (idx >= a) & (idx < b)
    rain = w10["rain"].to_numpy()
    et0 = w10["et0"].to_numpy()
    tair = w10["t_air"].to_numpy()
    is_irr_time = (idx.hour == 22) & (idx.minute == 0)
    W = w0_rew * p.capacity_mm
    e = -p.k_el_um_per_mpa * 0.03 / w0_rew**2
    g = 0.0
    r = np.empty(n)
    rew_arr = np.empty(n)
    irr = np.zeros(n)
    noise = rng.normal(0.0, p.noise_um, n)
    # slow random-walk drift of the sensor (mounting creep), small
    drift = np.cumsum(rng.normal(0.0, 0.02, n))
    # day-to-day variability of the shrink amplitude, AR(1) over days
    day_codes, day_uniq = pd.factorize(idx.normalize())
    z = np.zeros(len(day_uniq))
    for k in range(len(z)):
        z[k] = (0.5 * z[k - 1] if k else 0.0) + rng.normal(0.0, p.daily_var * np.sqrt(0.75))
    day_factor = np.exp(z)[day_codes]
    bark = 0.0
    for i in range(n):
        if not in_dr[i]:
            W += rain[i]
            if schedule.irrigate and is_irr_time[i] and W / p.capacity_mm < schedule.refill_below:
                add = (schedule.refill_to * p.capacity_mm) - W
                irr[i] = add
                W += add
        rew = min(max(W / p.capacity_mm, 1e-3), 1.0)
        fs = min(1.0, rew / 0.4)
        T = p.kc * et0[i] * fs  # mm per 10 min
        W = min(max(W - T, 0.0), p.capacity_mm)
        rew = min(max(W / p.capacity_mm, 1e-3), 1.0)
        psi_s = max(-0.03 / rew**2, -3.0)
        dry = min(max((0.6 - rew) / 0.6, 0.0), 1.0)
        R = p.r0_mpa * (1.0 + 3.0 * dry**2) * day_factor[i]
        k_el = p.k_el_um_per_mpa * (1.0 + 0.4 * dry)
        psi_target = psi_s - R * (T / dt_h) / p.t_ref_mm_h
        e += (k_el * psi_target - e) * min(1.0, dt_h / p.tau_h)
        psi_stem = e / k_el
        if tair[i] > 5.0:
            turgor = min(max((psi_stem - p.psi_growth_mpa) / (-p.psi_growth_mpa), 0.0), 1.0)
            g += p.growth_um_day * dt_h / 24.0 * turgor * 2.0  # x2: growth happens on roughly half the hours
        if not in_dr[i]:
            bark = min(bark + p.bark_um_per_mm * rain[i], p.bark_max_um)
        bark *= np.exp(-dt_h / 36.0)
        r[i] = g + e + bark + p.thermal_um_per_k * (tair[i] - 15.0) + noise[i] + drift[i]
        rew_arr[i] = rew
    rew_s = pd.Series(rew_arr, index=idx, name="rew")
    day = idx.normalize()
    truth = pd.DataFrame({
        "rew_min": rew_s.groupby(day).min(),
        "rew_mean": rew_s.groupby(day).mean(),
        "in_drought": pd.Series(in_dr, index=idx).groupby(day).any(),
        "irrigation_mm": pd.Series(irr, index=idx).groupby(day).sum(),
        "rain_mm": pd.Series(np.where(in_dr, 0.0, rain), index=idx).groupby(day).sum(),
    })
    truth.index.name = "date"
    return SimResult(pd.Series(r, index=idx, name="radius_um"), rew_s, truth, p)


def random_droughts(start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator, n: int = 2,
                    calib_days: int = 30, min_len: int = 14, max_len: int = 28, min_sep: int = 20) -> list:
    """Place `n` non-overlapping drought windows after the calibration period."""
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    lo = start + pd.Timedelta(days=calib_days)
    for _ in range(200):
        if len(out) == n:
            break
        length = int(rng.integers(min_len, max_len + 1))
        span = (end - lo).days - length
        if span <= 0:
            break
        a = lo + pd.Timedelta(days=int(rng.integers(0, span)))
        b = a + pd.Timedelta(days=length)
        if all(b + pd.Timedelta(days=min_sep) <= x or a >= y + pd.Timedelta(days=min_sep) for x, y in out):
            out.append((a, b))
    return sorted(out)


def jitter_params(rng: np.random.Generator) -> TreeParams:
    """Tree-to-tree variation (size, capacitance, growth)."""
    return TreeParams(
        capacity_mm=float(rng.uniform(80, 120)),
        k_el_um_per_mpa=float(rng.uniform(150, 230)),
        r0_mpa=float(rng.uniform(0.65, 0.95)),
        growth_um_day=float(rng.uniform(12, 24)),
        tau_h=float(rng.uniform(1.0, 2.0)),
    )


def inject_glitches(r: pd.Series, rng: np.random.Generator, n_jumps: int = 3, n_resets: int = 1, n_spikes: int = 10,
                    n_gaps: int = 4) -> tuple[pd.Series, pd.DataFrame]:
    """Add known sensor faults to a clean series. Returns (faulty series, truth table of injected events)."""
    v = r.to_numpy(dtype=float).copy()
    n = len(v)
    events = []
    busy: list[tuple[int, int]] = []  # occupied index ranges, kept apart so faults do not overlap

    def pick(length: int = 1, margin: int = 200) -> int:
        for _ in range(2000):
            i = int(rng.integers(margin, n - margin - length))
            if all(i + length + 30 < a or i > b + 30 for a, b in busy):
                busy.append((i, i + length))
                return i
        raise RuntimeError("series too short for the requested glitches")

    for _ in range(n_gaps):
        length = int(rng.integers(2, 145))  # 20 min .. 1 day
        i = pick(length)
        v[i:i + length] = np.nan
        events.append(dict(i=i, kind="gap", size_um=float(length)))
    for _ in range(n_jumps):
        i = pick()
        size = float(rng.choice([-1, 1]) * rng.uniform(40, 200))
        v[i:] += size
        events.append(dict(i=i, kind="jump", size_um=size))
    for _ in range(n_resets):
        i = pick()
        size = -float(v[i])
        v[i:] += size
        events.append(dict(i=i, kind="reset", size_um=size))
    for _ in range(n_spikes):
        i = pick()
        size = float(rng.choice([-1, 1]) * rng.uniform(40, 300))
        v[i] += size
        events.append(dict(i=i, kind="spike", size_um=size))
    truth = pd.DataFrame(events)
    truth["time"] = r.index[truth["i"].to_numpy()]
    return pd.Series(v, index=r.index, name=r.name), truth
