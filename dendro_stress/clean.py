"""Sensor-glitch handling for point-dendrometer series: spikes, jumps, resets and gaps.

A point dendrometer reads stem (or fruit) radius in micrometres every 10-15 min. Typical glitches:

* spike  - one or two wild readings that come straight back (electrical noise, a bump on the sensor);
* jump   - a lasting step in the level (sensor slipped, bark flaked, re-mounted, logger swapped);
* reset  - the logger restarts at (about) zero; handled as a jump that lands near zero;
* gap    - missing readings (radio loss, battery). Short gaps are interpolated, long ones stay missing.

Jumps and resets are corrected by removing the step from all later readings, so growth and tree water deficit
are not polluted by a fake level change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleanConfig:
    step: str = "10min"
    spike_window: int = 5  # samples in the centred rolling median
    spike_k: float = 8.0  # robust sigmas from the rolling median
    spike_min_um: float = 8.0
    jump_k: float = 12.0  # robust sigmas of the 1-step difference
    jump_min_um: float = 20.0
    jump_isolation: float = 5.0  # step must exceed this many times the neighbouring changes
    reset_tol_um: float = 5.0  # a jump landing within this of zero is called a reset
    max_gap: str = "1h"  # gaps up to this length are linearly interpolated


@dataclass
class CleanResult:
    series: pd.Series
    events: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["time", "kind", "size_um"]))

    def count(self, kind: str) -> int:
        return int((self.events["kind"] == kind).sum()) if len(self.events) else 0


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def regularize(s: pd.Series, step: str = "10min") -> pd.Series:
    """Snap timestamps to the step grid (mean of duplicates) and insert NaN rows for missing samples."""
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series needs a DatetimeIndex")
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s[s.index.notna()]
    if s.empty:
        return s
    s = s.groupby(s.index.round(step)).mean()
    grid = pd.date_range(s.index.min(), s.index.max(), freq=step)
    return s.reindex(grid)


def remove_spikes(s: pd.Series, cfg: CleanConfig) -> tuple[pd.Series, list[dict]]:
    med = s.rolling(cfg.spike_window, center=True, min_periods=3).median()
    resid = (s - med).to_numpy()
    thr = max(cfg.spike_min_um, cfg.spike_k * _robust_sigma(resid))
    bad = np.abs(resid) > thr
    out = s.copy()
    out[bad] = np.nan
    events = [dict(time=t, kind="spike", size_um=float(r)) for t, r in zip(s.index[bad], resid[bad], strict=True)]
    return out, events


def correct_jumps(s: pd.Series, cfg: CleanConfig, max_gap_steps: int) -> tuple[pd.Series, list[dict]]:
    """Remove lasting level steps between neighbouring valid samples (at most `max_gap_steps` apart)."""
    v = s.to_numpy(dtype=float).copy()
    idx = np.flatnonzero(np.isfinite(v))
    if len(idx) < 3:
        return s.copy(), []
    d = np.diff(v[idx])
    gaps = np.diff(idx)
    close = gaps <= max_gap_steps
    sigma = _robust_sigma(d[close & (gaps == 1)]) if np.any(close & (gaps == 1)) else _robust_sigma(d)
    thr = max(cfg.jump_min_um, cfg.jump_k * sigma)
    events = []
    step_thr = thr * np.sqrt(np.maximum(gaps, 1))
    cand = np.flatnonzero(close & (np.abs(d) > step_thr))
    step_off = np.zeros(len(d))
    for k in cand:
        # a fault is a step inside one interval; natural swelling after rain or irrigation is spread over
        # several intervals, so the neighbouring changes are of similar size
        lo, hi = max(0, k - 3), k + 4
        nb_d = np.concatenate([d[lo:k], d[k + 1:hi]])
        nb_g = np.concatenate([gaps[lo:k], gaps[k + 1:hi]])
        if len(nb_d) and abs(d[k]) <= cfg.jump_isolation * max(float(np.median(np.abs(nb_d))), sigma):
            continue
        i_next = idx[k + 1]
        after = v[i_next]
        kind = "reset" if abs(after) <= cfg.reset_tol_um and abs(v[idx[k]]) > cfg.reset_tol_um else "jump"
        # keep the natural change expected over the interval (mean of neighbouring per-step changes)
        natural = float(np.mean(nb_d / nb_g)) * gaps[k] if len(nb_d) else 0.0
        step_off[k] = d[k] - natural
        events.append(dict(time=s.index[i_next], kind=kind, size_um=float(d[k] - natural)))
    corrected = v.copy()
    if events:
        cum = np.concatenate([[0.0], np.cumsum(step_off)])  # offset to remove at each valid sample
        corrected[idx] = v[idx] - cum
    return pd.Series(corrected, index=s.index, name=s.name), events


def fill_gaps(s: pd.Series, max_gap_steps: int) -> tuple[pd.Series, list[dict]]:
    isna = s.isna().to_numpy()
    events = []
    if not isna.any():
        return s.copy(), events
    # run lengths of NaN blocks
    edges = np.diff(np.concatenate([[0], isna.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = s.copy()
    interp = s.interpolate(method="time", limit_area="inside")
    for a, b in zip(starts, ends, strict=True):
        n = b - a
        inside = a > 0 and b < len(s)
        kind = "gap_filled" if (n <= max_gap_steps and inside) else "gap"
        if kind == "gap_filled":
            out.iloc[a:b] = interp.iloc[a:b]
        events.append(dict(time=s.index[a], kind=kind, size_um=float(n)))
    return out, events


def clean_series(s: pd.Series, cfg: CleanConfig | None = None) -> CleanResult:
    """Full cleaning: regularize -> drop spikes -> correct jumps/resets -> interpolate short gaps."""
    cfg = cfg or CleanConfig()
    s = regularize(s, cfg.step)
    if s.empty:
        return CleanResult(s)
    max_gap_steps = max(1, int(pd.Timedelta(cfg.max_gap) / pd.Timedelta(cfg.step)))
    s1, ev1 = remove_spikes(s, cfg)
    s2, ev2 = correct_jumps(s1, cfg, max_gap_steps + 1)
    s3, ev3 = fill_gaps(s2, max_gap_steps)
    events = pd.DataFrame(ev1 + ev2 + ev3, columns=["time", "kind", "size_um"])
    events = events.sort_values("time").reset_index(drop=True)
    return CleanResult(s3, events)
