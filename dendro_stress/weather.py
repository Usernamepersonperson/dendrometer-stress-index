"""Weather inputs: vapour pressure deficit (VPD), daily summaries, MeteoSwiss hourly parser.

MeteoSwiss timestamps are UTC and mark the END of the hourly interval. Dendrometer data here are in CET
(UTC+1, no daylight saving), so weather is shifted to the interval start and then by +1 h.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
WEATHER_FIXTURE = DATA_DIR / "weather_visp_hourly.csv.gz"

# MeteoSwiss hourly parameter id -> column
PARAMS = {
    "tre200h0": "t_air",  # air temperature 2 m (deg C)
    "ure200h0": "rh",  # relative humidity 2 m (%)
    "rre150h0": "rain",  # precipitation (mm per hour)
    "gre000h0": "ghi",  # global radiation (W/m2)
    "erefaoh0": "et0",  # FAO reference evapotranspiration (mm per hour)
}


def saturation_vp(t_c) -> np.ndarray:
    """Saturation vapour pressure over water in kPa (Tetens / FAO-56 eq. 11)."""
    t = np.asarray(t_c, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def vpd_kpa(t_c, rh_pct) -> np.ndarray:
    """Vapour pressure deficit in kPa from air temperature (deg C) and relative humidity (%), RH clipped 0-100."""
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    return saturation_vp(t_c) * (1.0 - rh / 100.0)


def parse_meteoswiss_hourly(text: str, tz_offset_h: int = 1) -> pd.DataFrame:
    """Parse one MeteoSwiss SwissMetNet hourly CSV; returns hourly rows indexed by interval START in local CET."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    need = {"reference_timestamp", "tre200h0", "ure200h0"}
    if not need <= set(df.columns):
        raise ValueError(f"not a MeteoSwiss hourly file, missing {sorted(need - set(df.columns))}")
    end_utc = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")
    out = pd.DataFrame(index=end_utc - pd.Timedelta(hours=1) + pd.Timedelta(hours=tz_offset_h))
    out.index.name = "time"
    for pid, col in PARAMS.items():
        out[col] = pd.to_numeric(df[pid], errors="coerce").to_numpy() if pid in df.columns else np.nan
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_weather(path: Path | str | None = None) -> pd.DataFrame:
    """Load an hourly weather table (columns time, t_air, rh, rain, ghi, et0); adds a vpd column."""
    df = pd.read_csv(path or WEATHER_FIXTURE, parse_dates=["time"]).set_index("time").sort_index()
    for col in ("t_air", "rh"):
        if col not in df.columns:
            raise ValueError(f"weather table needs column {col!r}")
    for col in ("rain", "ghi", "et0"):
        if col not in df.columns:
            df[col] = np.nan
    df["vpd"] = vpd_kpa(df["t_air"], df["rh"])
    return df


def daily_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily summaries: max and daytime (08-18 h) mean VPD, rain sum, mean temperature, ET0 sum, days since rain."""
    h = hourly.copy()
    if "vpd" not in h.columns:
        h["vpd"] = vpd_kpa(h["t_air"], h["rh"])
    day = h.index.normalize()
    hour = h.index.hour
    g = h.groupby(day)
    out = pd.DataFrame({
        "vpd_max": g["vpd"].max(),
        "vpd_day": h[(hour >= 8) & (hour < 18)].groupby(day[(hour >= 8) & (hour < 18)])["vpd"].mean(),
        "t_mean": g["t_air"].mean(),
        "rain": g["rain"].sum(min_count=1),
        "et0": g["et0"].sum(min_count=1),
        "n_hours": g["t_air"].count(),
    })
    out.index.name = "date"
    out["days_since_rain"] = days_since(out["rain"].fillna(0.0) >= 5.0)
    return out


def days_since(event: pd.Series) -> pd.Series:
    """Days since the last True in a daily boolean series (0 on the event day; counts from series start before)."""
    n, out = 0, []
    for e in event.to_numpy():
        n = 0 if e else n + 1
        out.append(n)
    return pd.Series(out, index=event.index, dtype=int)
