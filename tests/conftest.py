import numpy as np
import pandas as pd
import pytest


def diurnal(days: int = 10, amp: float = 50.0, growth_per_day: float = 0.0, step: str = "10min",
            start: str = "2025-06-01", noise: float = 0.0, seed: int = 0) -> pd.Series:
    """Synthetic stem radius: max at 06:00, min at 16:00 (cosine), optional linear growth and noise."""
    idx = pd.date_range(start, periods=int(days * pd.Timedelta("1D") / pd.Timedelta(step)), freq=step)
    h = (idx - idx.normalize()).total_seconds().to_numpy() / 3600.0
    # piecewise cosine: shrink 06->16 h, refill 16->30 h
    phase = np.where((h >= 6) & (h < 16), (h - 6) / 10.0, np.where(h >= 16, 1 - (h - 16) / 14.0, 1 - (h + 8) / 14.0))
    shrink = amp * (1 - np.cos(np.pi * phase)) / 2.0
    t_days = np.arange(len(idx)) * pd.Timedelta(step) / pd.Timedelta("1D")
    v = growth_per_day * t_days - shrink
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, len(v))
    return pd.Series(v, index=idx, name="radius_um")


@pytest.fixture
def series():
    return diurnal()
