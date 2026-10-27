"""Data checks on the interim tables, written as a Markdown report with figures.

Run by ``make data`` (and ``wpforecast check``). Each check yields a row in the summary
table with a status: ``ok`` or ``warn``. Nothing here changes the data.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wpforecast.config import Config
from wpforecast.data import build, eds, openmeteo

log = logging.getLogger(__name__)

REPORT = "data_check.md"
FIGURES = "data_check"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name, bool(ok), detail))


def _table(df: pd.DataFrame) -> str:
    """Minimal Markdown table (avoids a tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def coverage(series_index: pd.DatetimeIndex, freq: str) -> dict:
    """First/last stamp, expected vs present rows, duplicates and the longest gap."""
    idx = series_index.sort_values()
    if len(idx) == 0:
        return {
            "first": "-",
            "last": "-",
            "rows": 0,
            "missing": "-",
            "dupes": 0,
            "longest_gap": "-",
        }
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    missing = pd.date_range(idx[0], idx[-1], freq=freq).difference(idx)
    runs = (missing.to_series().diff() != step).cumsum()  # consecutive missing stamps
    longest = int(runs.value_counts().max()) if len(missing) else 0
    return {
        "first": f"{idx[0]:%Y-%m-%d %H:%M}",
        "last": f"{idx[-1]:%Y-%m-%d %H:%M}",
        "rows": len(idx),
        "missing": len(missing),
        "dupes": int(idx.duplicated().sum()),
        "longest_gap": f"{step * longest / pd.Timedelta(hours=1):g} h",
    }


def check_coverage(cfg: Config, report: Report) -> None:
    rows = []
    for name, freq in (
        ("production", "h"),
        ("capacity", "h"),
        ("energinet_forecast", "h"),
        ("da_price", "h"),
        ("imbalance", "15min"),
    ):
        df = build.read(cfg, name)
        values = df.dropna(how="all")
        cov = coverage(pd.DatetimeIndex(values.index), freq)
        rows.append({"table": name, "freq": freq, **cov})
        report.add(f"{name}: no duplicate stamps", cov["dupes"] == 0, f"{cov['dupes']} duplicates")
        if name in ("production", "capacity", "da_price"):
            report.add(
                f"{name}: complete between first and last stamp",
                cov["missing"] in (0, "-") and cov["rows"] > 0,
                f"{cov['missing']} missing, longest gap {cov['longest_gap']}",
            )
    report.sections.append(
        "## Coverage\n\nUTC stamps, hour-beginning.\n\n" + _table(pd.DataFrame(rows))
    )


def check_nwp(cfg: Config, report: Report) -> pd.DataFrame:
    nwp = build.read(cfg, "nwp_hres")
    missing = json.loads((cfg.paths.interim / "nwp_missing_runs.json").read_text())
    expected_runs = len(openmeteo.run_times(cfg))
    per_run = nwp.groupby("run_utc", observed=True).size()
    full = len(cfg.points) * (cfg.nwp.max_lead_h + 1)
    incomplete = per_run[per_run != full]
    after_init = nwp[nwp["lead_h"] > 0]  # hourly max/accumulated fields have no lead 0
    nans = {var: int(after_init[var].isna().sum()) for var in cfg.nwp.variables}
    speed, direction = nwp["wind_speed_100m"], nwp["wind_direction_100m"]

    report.add(
        "NWP: every run present",
        not missing,
        f"{expected_runs - len(missing)}/{expected_runs} runs; missing: {missing[:10]}",
    )
    report.add(
        "NWP: every run complete", incomplete.empty, f"{len(incomplete)} runs without {full} rows"
    )
    report.add("NWP: no missing values after lead 0", sum(nans.values()) == 0, str(nans))
    report.add(
        "NWP: wind speed in [0, 60] m/s",
        speed.between(0, 60).all(),
        f"min {speed.min():.2f}, max {speed.max():.2f}",
    )
    report.add(
        "NWP: direction in [0, 360] deg",
        direction.between(0, 360).all(),
        f"min {direction.min():.0f}, max {direction.max():.0f}",
    )

    points = build.read(cfg, "points")
    land_ok = (points.loc[points["region"] == "onshore", "grid_elevation_m"] > 0).all()
    sea_ok = (points.loc[points["region"] == "offshore", "grid_elevation_m"] == 0).all()
    report.add(
        "Points: onshore on land, offshore at sea", land_ok and sea_ok, "grid-cell elevation"
    )
    report.add(
        "Points: grid cell within 10 km",
        (points["distance_km"] <= 10).all(),
        f"max {points['distance_km'].max():.1f} km",
    )
    section = [
        "## NWP: ECMWF IFS HRES 00 UTC runs",
        f"{expected_runs - len(missing)} of {expected_runs} runs, leads 0–{cfg.nwp.max_lead_h} h, "
        f"{len(cfg.points)} points, {len(nwp):,} rows.",
        f"Missing upstream: {', '.join(missing) if missing else 'none'}.",
        "### Grid cells used\n\nOnshore points use land cells, offshore points sea cells "
        "(Open-Meteo `cell_selection`).",
        _table(points.round(3)),
    ]
    report.sections.append("\n\n".join(section))
    return nwp


def check_time(cfg: Config, report: Report) -> None:
    """Energinet's local stamps agree with UTC + Europe/Copenhagen; DST days have 23/25 h."""
    raw = eds.load(cfg, "production")
    utc = pd.to_datetime(raw["HourUTC"], utc=True)
    local = pd.to_datetime(raw["HourDK"])
    expected = utc.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)
    mismatch = int((expected != local).sum())
    report.add(
        "Time: HourDK = HourUTC in Europe/Copenhagen", mismatch == 0, f"{mismatch} rows differ"
    )

    per_day = utc.dt.tz_convert("Europe/Copenhagen").dt.date.value_counts().sort_index()
    inner = per_day.iloc[1:-1]  # first/last local day are cut by the UTC download window
    days = pd.Series(inner.index)
    expected_len = days.map(
        lambda d: int(
            (
                pd.Timestamp(d + dt.timedelta(days=1), tz="Europe/Copenhagen")
                - pd.Timestamp(d, tz="Europe/Copenhagen")
            )
            / pd.Timedelta(hours=1)
        )
    )
    bad = int((inner.to_numpy() != expected_len.to_numpy()).sum())
    dst = sorted(int(n) for n in set(expected_len) - {24})
    report.add(
        "Time: hours per local day match DST",
        bad == 0,
        f"{bad} days differ; DST day lengths seen: {dst}",
    )


def check_ranges(cfg: Config, report: Report) -> None:
    prod, cap = build.read(cfg, "production"), build.read(cfg, "capacity")
    ds = build.read(cfg, "dataset")
    ratio = (prod["wind_mw"] / cap["total_mw"].reindex(prod.index)).dropna()
    report.add(
        "Production: non-negative",
        (prod.drop(columns="wind_mw") >= 0).all().all(),
        "all four columns",
    )
    report.add(
        "Production: below installed capacity",
        (ratio <= 1.0).mean() > 0.999,
        f"max P/C {ratio.max():.3f}; {(ratio > 1).sum()} hours above 1",
    )
    report.add(
        "Capacity: positive, no gaps",
        (cap["total_mw"] > 0).all(),
        f"{cap['total_mw'].min():.0f}–{cap['total_mw'].max():.0f} MW",
    )

    fc = build.read(cfg, "energinet_forecast")
    if not fc.empty:
        fc_ratio = (fc["total_da_mw"] / cap["total_mw"].reindex(fc.index)).dropna()
        report.add(
            "Energinet forecast: within capacity",
            (fc_ratio <= 1.0).all(),
            f"max {fc_ratio.max():.3f}",
        )

    price = build.read(cfg, "da_price")["price_eur_mwh"]
    negative = ds.index.isin(price[price <= cfg.curtailment.da_price_max].index)
    report.sections.append(
        "## Ranges and units\n\n"
        + _table(
            pd.DataFrame(
                [
                    {
                        "quantity": "wind production (MW)",
                        "min": f"{prod['wind_mw'].min():.0f}",
                        "median": f"{prod['wind_mw'].median():.0f}",
                        "max": f"{prod['wind_mw'].max():.0f}",
                    },
                    {
                        "quantity": "capacity (MW)",
                        "min": f"{cap['total_mw'].min():.0f}",
                        "median": f"{cap['total_mw'].median():.0f}",
                        "max": f"{cap['total_mw'].max():.0f}",
                    },
                    {
                        "quantity": "normalised power y = P / C",
                        "min": f"{ratio.min():.3f}",
                        "median": f"{ratio.median():.3f}",
                        "max": f"{ratio.max():.3f}",
                    },
                    {
                        "quantity": "day-ahead price (EUR/MWh)",
                        "min": f"{price.min():.1f}",
                        "median": f"{price.median():.1f}",
                        "max": f"{price.max():.1f}",
                    },
                ]
            )
        )
        + f"\n\nHours with day-ahead price ≤ {cfg.curtailment.da_price_max:g} EUR/MWh "
        f"(curtailment flag, Step 2): {int(negative.sum())} of {len(ds)} ({negative.mean():.1%})."
    )


def figures(cfg: Config, nwp: pd.DataFrame, report: Report) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = cfg.paths.reports / FIGURES
    out.mkdir(parents=True, exist_ok=True)

    cap = build.read(cfg, "capacity")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.stackplot(
        cap.index, cap["onshore_mw"], cap["offshore_mw"], labels=["onshore", "offshore"], alpha=0.8
    )
    ax.set(ylabel="MW", title=f"{cfg.data.price_area} installed wind capacity (Energinet, monthly)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "capacity.png", dpi=130)
    plt.close(fig)

    # Normalised power vs mean 100 m wind speed, at day-ahead leads (13-36 h).
    ds = build.read(cfg, "dataset")
    da = nwp[nwp["lead_h"].between(13, 36)]
    mean_speed = da.groupby("valid_utc")["wind_speed_100m"].mean()
    joined = pd.DataFrame(
        {"speed": mean_speed, "y": ds["power_norm"].reindex(mean_speed.index)}
    ).dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if not joined.empty:
        ax.hexbin(joined["speed"], joined["y"], gridsize=40, mincnt=1, bins="log", cmap="Blues")
        bins = np.arange(0, joined["speed"].max() + 1, 1.0)
        med = joined.groupby(pd.cut(joined["speed"], bins), observed=True)["y"].median()
        ax.plot(
            [b.mid for b in med.index],
            med.to_numpy(),
            color="black",
            lw=1.5,
            label="median per 1 m/s",
        )
        ax.legend(loc="upper left", frameon=False)
    ax.set(
        xlabel="mean 100 m wind speed, 19 points, lead 13–36 h (m/s)",
        ylabel="y = P / C",
        ylim=(0, 1),
    )
    ax.set_title(f"Normalised power vs NWP wind speed ({len(joined):,} hours)")
    fig.tight_layout()
    fig.savefig(out / "power_curve.png", dpi=130)
    plt.close(fig)

    report.sections.append(
        "## Figures\n\n"
        f"![Installed capacity]({FIGURES}/capacity.png)\n\n"
        f"![Normalised power vs wind speed]({FIGURES}/power_curve.png)"
    )


def provenance(cfg: Config) -> str:
    manifest_path = cfg.paths.raw / "eds" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dates = sorted(v["downloaded_at"][:10] for v in manifest.values())
    span = f"{dates[0]} to {dates[-1]}" if dates else "unknown"
    return (
        "## Provenance\n\n"
        f"- Download window: {cfg.data.start} to {cfg.data.end} (UTC days).\n"
        f"- Energi Data Service chunks downloaded {span}; settlement data is revised for up "
        "to two years, so these dates define the snapshot.\n"
        f"- NWP: `{cfg.nwp.model}` 00 UTC runs from the Open-Meteo Single Runs API. Runs "
        f"are assumed usable {cfg.nwp.availability_lag_h} h after initialisation "
        "(observed 6.1–6.5 h on 2026-10-26/27), before the 09:00 UTC summer issue time.\n"
        "- Sources: Energinet (www.energidataservice.dk), CC BY 4.0; "
        "weather data by Open-Meteo.com (ECMWF), CC BY 4.0."
    )


def run(cfg: Config) -> Report:
    report = Report()
    check_coverage(cfg, report)
    nwp = check_nwp(cfg, report)
    check_time(cfg, report)
    check_ranges(cfg, report)
    figures(cfg, nwp, report)

    summary = _table(
        pd.DataFrame(
            [
                {"check": c.name, "status": "ok" if c.ok else "**warn**", "detail": c.detail}
                for c in report.checks
            ]
        )
    )
    n_warn = sum(not c.ok for c in report.checks)
    text = "\n\n".join(
        [
            "# Data check",
            f"Generated by `make data` for {cfg.data.price_area}, "
            f"{cfg.data.start} to {cfg.data.end}. {len(report.checks) - n_warn} checks ok, "
            f"{n_warn} warnings.",
            "## Summary",
            summary,
            *report.sections,
            provenance(cfg),
        ]
    )
    path = cfg.paths.reports / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    log.info("check: %d ok, %d warnings -> %s", len(report.checks) - n_warn, n_warn, path)
    return report
