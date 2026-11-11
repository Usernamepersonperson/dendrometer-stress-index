"""Rebuild the real-data cache from the public sources (needs network, about 140 MB of downloads).

    python -m dendro_stress fetch

* Point-dendrometer series, Pfynwald (Valais) irrigation / roof / VPD experiment, 10-min, 2024 and 2025.
  EnviDat, DOI 10.16904/envidat.751, licence CC BY 4.0.
* Hourly weather, MeteoSwiss SwissMetNet station Visp (VIS), open data (source must be cited).

The 10-min text files are 0.67 GB each unzipped, so they are streamed out of the zip and stored as compact
float32 arrays in ``cache/pfynwald_<year>.npz`` (one column per tree, 10-min CET grid).
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ENVIDAT = "https://www.envidat.ch/dataset/571cf7ad-edda-4e1f-9e9d-45014c39a9e2/resource"
DENDRO_ZIPS = {
    2024: f"{ENVIDAT}/425594e2-7210-49a8-82b7-1bdfc4cbaf33/download/pfynwald_vpdrought_dendrometer_2024.zip",
    2025: f"{ENVIDAT}/cea43ffe-6437-4b0e-8d58-398026cbde41/download/pfynwald_vpdrought_dendrometer_2025.zip",
}
SMN = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
WEATHER_FILES = ("historical_2020-2029", "recent")
STEP = pd.Timedelta("10min")


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "dendrometer-stress-index/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def _member(z: zipfile.ZipFile, key: str) -> str:
    names = [n for n in z.namelist() if key in n and not n.startswith("__MACOSX")]
    if not names:
        raise ValueError(f"no file containing {key!r} in zip")
    return names[0]


def zip_to_arrays(zpath: Path, year: int) -> dict:
    """Stream the 10-min file out of the zip into (time x tree) float32 arrays: radius, published TWD, flags."""
    z = zipfile.ZipFile(zpath)
    with z.open(_member(z, "metadata")) as f:
        meta = pd.read_csv(f)
    with z.open(_member(z, "_1Day_")) as f:
        daily = pd.read_csv(f)
    meta["series"] = meta["imp_name"].str.split("_").str[-1].astype(int)
    trees = meta[meta["imp_name"].str.startswith("value_")].sort_values("series").reset_index(drop=True)
    twd_ids = meta[meta["imp_name"].str.startswith("twd_")].set_index("series")["messvar_id"]
    val_col = {mid: i for i, mid in enumerate(trees["messvar_id"])}
    twd_col = {int(twd_ids[s]): i for i, s in enumerate(trees["series"]) if s in twd_ids.index}
    t0 = pd.Timestamp(f"{year}-01-01")
    n = int((pd.Timestamp(f"{year + 1}-01-01") - t0) / STEP)
    radius = np.full((n, len(trees)), np.nan, dtype="float32")
    twd = np.full((n, len(trees)), np.nan, dtype="float32")
    flagged = np.zeros((n, len(trees)), dtype=bool)
    with z.open(_member(z, "10Minutes")) as f:
        for ch in pd.read_csv(f, chunksize=1_000_000, usecols=["messvar_id", "variable", "messtime", "messval",
                                                                "quality_flag"]):
            ch = ch[ch["variable"].isin(["value", "twd"])]
            idx = ((pd.to_datetime(ch["messtime"]) - t0) / STEP).round().astype(int).to_numpy()
            ok = (idx >= 0) & (idx < n)
            ch, idx = ch[ok], idx[ok]
            isval = (ch["variable"] == "value").to_numpy()
            mids = ch["messvar_id"].to_numpy()
            vals = ch["messval"].to_numpy(dtype="float64")
            flags = (ch["quality_flag"].fillna("") != "clean").to_numpy()
            cols = np.array([val_col.get(m, -1) for m in mids[isval]], dtype=int)
            good = cols >= 0
            radius[idx[isval][good], cols[good]] = vals[isval][good]
            flagged[idx[isval][good], cols[good]] = flags[isval][good]
            cols = np.array([twd_col.get(m, -1) for m in mids[~isval]], dtype=int)
            good = cols >= 0
            twd[idx[~isval][good], cols[good]] = vals[~isval][good]
    keep = ["series", "installation_name", "treatment", "tree_id", "species", "dbh", "height"]
    return dict(radius=radius, twd=twd, flagged=flagged, trees=trees[keep], daily=daily, t0=t0)


def save_year(arrs: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, radius=arrs["radius"], twd=arrs["twd"], flagged=arrs["flagged"],
                        t0=str(arrs["t0"]))
    arrs["trees"].to_csv(out.with_suffix(".trees.csv"), index=False)


def fetch_all(cache: Path) -> None:
    raw = cache / "raw"
    for year, url in DENDRO_ZIPS.items():
        out = cache / f"pfynwald_{year}.npz"
        if out.exists():
            continue
        zpath = download(url, raw / f"pfynwald_{year}.zip")
        print(f"converting {zpath.name} (streams 0.67 GB, a few minutes) ...", flush=True)
        save_year(zip_to_arrays(zpath, year), out)
    for period in WEATHER_FILES:
        download(f"{SMN}/vis/ogd-smn_vis_h_{period}.csv", raw / f"ogd-smn_vis_h_{period}.csv")


def build_weather_fixture(cache: Path, out: Path | None = None) -> pd.DataFrame:
    """Visp hourly weather, April-October 2024 and 2025, CET interval start -> data/weather_visp_hourly.csv.gz."""
    from .weather import WEATHER_FIXTURE, parse_meteoswiss_hourly

    frames = [parse_meteoswiss_hourly((cache / "raw" / f"ogd-smn_vis_h_{p}.csv").read_text(encoding="utf-8"))
              for p in WEATHER_FILES]
    h = pd.concat(frames).sort_index()
    h = h[~h.index.duplicated(keep="last")]
    h = h[(h.index.year.isin([2024, 2025])) & (h.index.month.isin(range(4, 11)))]
    out = out or WEATHER_FIXTURE
    out.parent.mkdir(parents=True, exist_ok=True)
    h.reset_index().to_csv(out, index=False, float_format="%.4g", compression="gzip")
    return h


def load_year(cache: Path, year: int) -> dict:
    """Load one cached year: radius / twd / flagged arrays (time x tree), the CET time index and tree table."""
    path = cache / f"pfynwald_{year}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m dendro_stress fetch` first")
    z = np.load(path)
    trees = pd.read_csv(path.with_suffix(".trees.csv"))
    t0 = pd.Timestamp(str(z["t0"]))
    idx = pd.date_range(t0, periods=z["radius"].shape[0], freq=STEP)
    return dict(radius=z["radius"], twd=z["twd"], flagged=z["flagged"], time=idx, trees=trees)


SAMPLE_TREATMENTS = ("irrigation", "control", "roof")


def build_sample(cache: Path, out: Path | None = None, year: int = 2025, start: str = "2025-06-01",
                 end: str = "2025-08-31") -> pd.DataFrame:
    """Small committed sample (CC BY 4.0, see README): 2 trees per treatment, 10-min, June-August 2025."""
    from .weather import DATA_DIR

    d = load_year(cache, year)
    trees = d["trees"]
    sel = (d["time"] >= pd.Timestamp(start)) & (d["time"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    rows = []
    for tr in SAMPLE_TREATMENTS:
        cols = trees.index[trees["treatment"] == tr].tolist()
        cover = [np.isfinite(d["radius"][sel, c]).mean() for c in cols]
        best = [c for _, c in sorted(zip(cover, cols, strict=True), reverse=True)[:2]]
        for c in best:
            rows.append(pd.DataFrame({
                "time": d["time"][sel], "tree": f"{tr}_{int(trees.loc[c, 'tree_id'])}",
                "radius_um": d["radius"][sel, c],
            }))
    df = pd.concat(rows, ignore_index=True).dropna(subset=["radius_um"])
    out = out or (DATA_DIR / "sample_pfynwald_2025_jja.csv.gz")
    df.to_csv(out, index=False, float_format="%.2f", compression="gzip")
    return df
