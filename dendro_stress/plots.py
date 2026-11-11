"""Figures: README figure (1600x900), phone-sized post figure (960x700) and a per-run figure. Dark palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
LIGHT_COLORS = {"green": GREEN, "amber": AMBER, "red": RED, "grey": GREY}
GROUP_COLORS = {"irrigation": GREEN, "control": AMBER, "roof": RED}
GROUP_LABELS = {"irrigation": "irrigated", "control": "rain only", "roof": "roof, half the rain"}
REAL_SRC = "Source: Zweifel et al. 2026, EnviDat, CC BY 4.0 (dendrometers); MeteoSwiss (weather, Visp)"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": TEXT, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.facecolor": BG,
    "figure.facecolor": BG, "savefig.facecolor": BG,
})


def _style(ax, fs: float = 11) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(labelsize=fs)
    for s in ax.spines.values():
        s.set_color(GRID)


def _months(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def real_twd_panel(ax, twd_wide: pd.DataFrame, cols_by: dict, weather: pd.DataFrame, fs: float = 11,
                   spell: tuple | None = None, legend_fs: float = 10.5) -> None:
    _style(ax, fs)
    days = twd_wide.index
    if spell:
        ax.axvspan(spell[0], spell[1] + pd.Timedelta("1D"), color=GREY, alpha=0.18, lw=0)
    for g in ("roof", "control", "irrigation"):
        cols = cols_by.get(g, [])
        if not cols:
            continue
        m = twd_wide[cols].median(axis=1).rolling(3, min_periods=1, center=True).median()
        ax.plot(days, m, color=GROUP_COLORS[g], lw=2.6, label=f"{GROUP_LABELS[g]} ({len(cols)} trees)")
    ax.set_ylabel("predawn tree water deficit (um)", fontsize=fs)
    ax2 = ax.twinx()
    rain = weather["rain"].reindex(days).fillna(0)
    ax2.bar(days, rain, color=ACCENT, alpha=0.45, width=0.9)
    ax2.set_ylim(0, max(60, rain.max() * 3))
    ax2.set_ylabel("rain (mm/day)", fontsize=fs, color=ACCENT)
    ax2.tick_params(labelsize=fs, colors=ACCENT)
    for s in ax2.spines.values():
        s.set_color(GRID)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylim(bottom=0)
    _months(ax)
    ax.legend(fontsize=legend_fs, loc="upper left", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, framealpha=0.9)


def light_strip(ax, light: pd.Series, y: float, h: float) -> None:
    for d, c in light.items():
        ax.add_patch(plt.Rectangle((mdates.date2num(d), y), 1.0, h, color=LIGHT_COLORS.get(c, GREY), lw=0))


def sim_panel(ax, season: dict, method: str = "ref", fs: float = 11, legend_fs: float = 10.0) -> None:
    _style(ax, fs)
    truth, st = season["truth"], season[method]
    for a, b in season["droughts"]:
        ax.axvspan(a, b, color=RED, alpha=0.12, lw=0)
    ax.plot(truth.index, truth["rew_min"], color=VIOLET, lw=2.6, label="true soil water (REW)")
    ax.axhline(0.4, color=VIOLET, lw=1, ls="--", alpha=0.8)
    ax.set_ylabel("soil water REW (0-1)", fontsize=fs, color=VIOLET)
    light_strip(ax, st["light"].reindex(truth.index).fillna("grey"), -0.22, 0.12)
    ax.set_ylim(-0.25, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.text(truth.index[0] + pd.Timedelta("1D"), -0.085, "traffic light", fontsize=fs - 1, color=MUTED, va="bottom")
    ax.set_xlim(truth.index[0], truth.index[-1] + pd.Timedelta("1D"))
    ax2 = ax.twinx()
    si = st["si"].reindex(truth.index)
    ax2.plot(si.index, si, color=ACCENT, lw=1.8, label="stress index SI (MDS / expected)")
    ax2.set_ylim(-0.75, 3.25)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_ylabel("SI", fontsize=fs, color=ACCENT)
    ax2.tick_params(labelsize=fs, colors=ACCENT)
    for sp in ax2.spines.values():
        sp.set_color(GRID)
    _months(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=legend_fs, loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
               framealpha=0.9, ncol=2)


def share_panel(ax, real: dict, method: str = "ref", fs: float = 11) -> None:
    _style(ax, fs)
    years = sorted(real)
    groups = ["irrigation", "control", "roof"]
    w = 0.36
    for k, y in enumerate(years):
        vals = [real[y]["light_share"][method].get(g, {}).get("non_green", np.nan) * 100 for g in groups]
        x = np.arange(len(groups)) + (k - 0.5) * w
        bars = ax.bar(x, vals, w * 0.92, color=[GROUP_COLORS[g] for g in groups], alpha=0.55 + 0.4 * k)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=fs - 1, color=TEXT)
            ax.text(b.get_x() + b.get_width() / 2, 2, y, ha="center", fontsize=fs - 2.5, color=BG, rotation=90,
                    va="bottom")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=fs)
    ax.set_ylabel("summer days amber or red (%)", fontsize=fs)
    ax.set_ylim(0, 100)


def _title(fig, title: str, subtitle: str, fs_title: float = 19, fs_sub: float = 12.5) -> None:
    fig.text(0.03, 0.965, title, fontsize=fs_title, fontweight="bold", color=TEXT, va="top", ha="left")
    fig.text(0.03, 0.925, subtitle, fontsize=fs_sub, color=MUTED, va="top", ha="left")


def main_figure(path: Path, real: dict, keep: dict, sim: dict, season: dict, glitch: dict, year: int = 2025) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    _title(fig, "dendrometer-stress-index: daily water-stress traffic light from stem shrinkage",
           "Top: real open data, Scots pines in an irrigation / roof experiment (Valais). Bottom: simulated orchard "
           "tree driven by real weather, with injected droughts.", fs_title=21, fs_sub=13)
    k = keep[year]
    ry = real[str(year)]
    spell = ry["longest_dry_spell"]
    ax = fig.add_axes([0.05, 0.53, 0.56, 0.33])
    real_twd_panel(ax, k["twd_wide"], k["cols_by"], k["weather"], fs=11.5,
                   spell=(pd.Timestamp(spell["start"]), pd.Timestamp(spell["end"])))
    ax.set_title(f"A  Real {year}: predawn tree water deficit by treatment (grey = longest dry spell)", fontsize=13.5,
                 color=TEXT, loc="left")
    ax = fig.add_axes([0.70, 0.53, 0.27, 0.33])
    share_panel(ax, real, "ref", fs=11.5)
    ax.set_title("B  Real: summer days flagged, 1 Jun-15 Sep", fontsize=13.5, color=TEXT, loc="left")
    ax = fig.add_axes([0.05, 0.08, 0.56, 0.33])
    sim_panel(ax, season, "ref", fs=11.5)
    ax.set_title(f"C  Simulated season ({season['year']} weather, seed {season['seed']}): shaded = irrigation off",
                 fontsize=13.5, color=TEXT, loc="left")
    axk = fig.add_axes([0.70, 0.06, 0.28, 0.36])
    axk.axis("off")
    s = sim["ref"]
    inj = int(glitch["jump_injected"] + glitch["reset_injected"])
    found = int(round(glitch["jump_recall"] * glitch["jump_injected"] + glitch["reset_recall"] * glitch["reset_injected"]))
    lines = [
        (f"{s['detected']}/{s['reached_stress']}", "simulated droughts detected", GREEN),
        (f"{s['delay_median_days']:+.1f} d", f"median delay vs true onset (IQR {s['delay_p25_days']:+.0f} to "
                                              f"{s['delay_p75_days']:+.0f} d)", ACCENT),
        (f"{s['false_stress_rate'] * 100:.1f}%", f"false stress days ({s['false_stress_days']}/{s['wet_days']} wet)",
         AMBER),
        (f"{found}/{inj}", "injected jumps + resets found (real series)", VIOLET),
    ]
    axk.text(0, 1.0, "D  Measured (simulated unless noted)", fontsize=13.5, color=TEXT, va="top")
    for i, (val, lab, col) in enumerate(lines):
        y = 0.80 - i * 0.215
        axk.text(0.0, y, val, fontsize=23, fontweight="bold", color=col, va="center")
        axk.text(0.36, y, lab, fontsize=11.5, color=MUTED, va="center", wrap=True)
    fig.text(0.03, 0.012, REAL_SRC + ". TWD thresholds set on 2024 + simulation; 2025 held out. Not calibrated per "
             "species.", fontsize=10, color=MUTED)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def post_figure(path: Path, keep: dict, real: dict, year: int = 2025) -> None:
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    ry = real[str(year)]
    rel = ry["weather_relation"]
    _title(fig, "Stem dendrometers separate dry trees from watered ones",
           f"Open data: Scots pines, irrigation / roof experiment, Valais, Apr-Sep {year}", fs_title=19, fs_sub=14)
    k = keep[year]
    spell = ry["longest_dry_spell"]
    ax = fig.add_axes([0.10, 0.37, 0.79, 0.43])
    real_twd_panel(ax, k["twd_wide"], k["cols_by"], k["weather"], fs=14,
                   spell=(pd.Timestamp(spell["start"]), pd.Timestamp(spell["end"])), legend_fs=13)
    ax.set_ylabel("predawn deficit (um)", fontsize=14)
    ax.set_title("Tree water deficit at dawn (treatment median)", fontsize=15, color=TEXT, loc="left")
    share = ry["light_share"]["ref"]
    ratio = rel["control"]["twd_predawn_summer_median_um"] / rel["irrigation"]["twd_predawn_summer_median_um"]
    stats = [
        (f"{ratio:.1f}x", "dawn deficit, rain only", "vs irrigated (summer)", AMBER),
        (f"{share['control']['non_green'] * 100:.0f}%", "summer days flagged,", "rain only", RED),
        (f"{share['irrigation']['non_green'] * 100:.0f}%", "summer days flagged,", "irrigated", GREEN),
    ]
    for i, (val, lab1, lab2, col) in enumerate(stats):
        x = 0.06 + i * 0.32
        fig.text(x, 0.215, val, fontsize=30, fontweight="bold", color=col, va="center")
        fig.text(x, 0.135, lab1, fontsize=14, color=MUTED, va="center")
        fig.text(x, 0.095, lab2, fontsize=14, color=MUTED, va="center")
    fig.text(0.03, 0.035, "Source: EnviDat (CC BY 4.0), MeteoSwiss. Forest trees, not orchards; thresholds uncalibrated.",
             fontsize=12, color=MUTED)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def run_figure(path: Path, table: pd.DataFrame, name: str) -> None:
    """Figure for one sensor of a `run`: MDS vs expected, predawn TWD, traffic light."""
    fig = plt.figure(figsize=(12, 6.5), dpi=100, facecolor=BG)
    _title(fig, f"Water-stress index: {name}", "MDS = max daily shrinkage; SI = MDS / expected; TWD = tree water "
           "deficit", fs_title=17, fs_sub=11.5)
    ax = fig.add_axes([0.07, 0.42, 0.88, 0.42])
    _style(ax)
    ax.plot(table.index, table["mds"], color=ACCENT, lw=1.8, label="MDS (um)")
    ax.plot(table.index, table["mds_expected"], color=GREY, lw=1.6, ls="--", label="expected MDS")
    ax.legend(fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    ax.set_ylabel("um")
    ax2 = fig.add_axes([0.07, 0.08, 0.88, 0.27])
    _style(ax2)
    ax2.plot(table.index, table["twd_predawn"], color=VIOLET, lw=1.8)
    ax2.set_ylabel("predawn TWD (um)")
    lo = 0 if table["twd_predawn"].isna().all() else float(np.nanmin(table["twd_predawn"]))
    hi = 1 if table["twd_predawn"].isna().all() else float(np.nanmax(table["twd_predawn"]))
    span = max(hi - lo, 1.0)
    ax2.set_ylim(lo - 0.35 * span, hi + 0.1 * span)
    light_strip(ax2, table["light"], lo - 0.3 * span, 0.2 * span)
    for a in (ax, ax2):
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
