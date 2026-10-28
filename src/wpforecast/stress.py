"""Regime change (Step 9): how the adaptivity regimes ride out Thor and ECMWF's new cycle.

From 2026-03-01 the capacity register counts Thor (1,058 MW) in full, but its output only shows
from mid-May 2026; ECMWF's IFS 50r1 feeds the 00 UTC runs from 2026-05-13 (``[stress]
events``). With C(t) ahead of production, y = P / C is biased low, so everything here is in
MW: forecast quantiles times the capacity known at issue time, against settled production.

Models: the frozen and walk-forward curves of the backtest; the RLS curve rerun for every
lambda of ``[model.rls] forgetting_grid``; and ``rls_reset``, the tuned lambda with its
covariance reset at each event (covariance resetting, a system-identification remedy for
known breaks). The report compares them over the test year (stable) and the stress period
(changing), with rolling bias and CRPS and the curve's level over time, in MW.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast import backtest, scoring
from wpforecast.config import Config
from wpforecast.data import build
from wpforecast.models.power_curve import PowerCurve

log = logging.getLogger(__name__)

REPORT = "stress.md"
FIGURES = "stress"
REGIMES = ("curve_frozen", "curve_walk_forward")  # taken from the backtest
RESET = "rls_reset"
PERIODS = ("test", "stress")
LEVEL_SPEED = 10.0  # onshore 100 m wind (m/s) at which the curve's level is followed


def event_times(cfg: Config) -> list[pd.Timestamp]:
    """Local midnight of each event date, in UTC."""
    return [pd.Timestamp(e.date, tz=cfg.issue.tz).tz_convert("UTC") for e in cfg.stress.events]


def rls_name(forgetting: float) -> str:
    return f"rls_{forgetting:g}"


def variants(cfg: Config, tuned: float) -> list[PowerCurve]:
    """The RLS curve for every lambda of the grid, and the tuned lambda with resets."""

    def curve(name: str, forgetting: float, resets=()) -> PowerCurve:
        levels = scoring.crps_levels(cfg)
        n = cfg.model.baselines.min_error_samples
        return PowerCurve(name, cfg.model.rls, forgetting, levels, n, resets=tuple(resets))

    grid = [curve(rls_name(lam), lam) for lam in cfg.model.rls.forgetting_grid]
    return [*grid, curve(RESET, tuned, event_times(cfg))]


def in_mw(cfg: Config, fc: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Product hours with an observation, in MW: quantiles and point times the capacity known
    at issue time; bias = point - observed; CRPS of the MW quantiles."""
    keys = ["issue_time_utc", "time_utc"]
    fc = fc[fc["in_product"] & fc["period"].isin(PERIODS)].dropna(subset=["power_mw"])
    fc = fc.merge(table[[*keys, "capacity_issue_mw"]], on=keys, how="left")
    grid = np.asarray(scoring.crps_levels(cfg))
    cap = fc["capacity_issue_mw"].to_numpy()
    q = fc[[scoring.qcol(x) for x in grid]].to_numpy() * cap[:, None]
    observed = fc["power_mw"].to_numpy()
    point = fc["point"].to_numpy() * cap
    return fc[["model", "period", "target_day", *keys]].assign(
        observed_mw=observed,
        bias_mw=point - observed,
        abs_mw=np.abs(point - observed),
        crps_mw=scoring.crps(observed, q, grid),
    )


def common(rows: pd.DataFrame) -> pd.DataFrame:
    """Per period, the hours every model forecast."""
    n = rows.groupby("period")["model"].transform("nunique")
    per_hour = rows.groupby(["period", "issue_time_utc", "time_utc"])["model"].transform("nunique")
    return rows[per_hour == n]


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model and period: hours, bias, MAE, RMSE and CRPS in MW."""
    g = rows.groupby(["model", "period"], sort=False)
    out = g.agg(
        hours=("bias_mw", "size"),
        bias_mw=("bias_mw", "mean"),
        mae_mw=("abs_mw", "mean"),
        crps_mw=("crps_mw", "mean"),
    )
    out["rmse_mw"] = np.sqrt(g["bias_mw"].apply(lambda b: (b**2).mean()))
    return out.unstack("period")  # noqa: PD010  (a reshape of unique keys)


def first_output(cfg: Config) -> pd.Timestamp | None:
    """First hour of the stress period with offshore production above its maximum in the
    year before, plus ``first_output_margin``: when the new fleet first shows (descriptive,
    from data in hindsight; never used by a model)."""
    production = build.read(cfg, "production")
    offshore = production[[c for c in production.columns if c.startswith("offshore")]].sum(axis=1)
    start = pd.Timestamp(cfg.dates.stress_start, tz=cfg.issue.tz).tz_convert("UTC")
    before = offshore[(offshore.index < start) & (offshore.index >= start - pd.Timedelta(days=365))]
    threshold = before.max() * (1 + cfg.stress.first_output_margin)
    above = offshore[(offshore.index >= start) & (offshore > threshold)]
    return above.index[0] if len(above) else None


def rolling(cfg: Config, rows: pd.DataFrame) -> pd.DataFrame:
    """Trailing ``rolling_days`` means of the daily bias and CRPS (MW), from ``plot_start``."""
    cols = ["bias_mw", "crps_mw"]
    rows = rows[pd.to_datetime(rows["target_day"]) >= pd.Timestamp(cfg.stress.plot_start)]
    out = []
    for name, g in rows.groupby("model", sort=False):
        daily = g.groupby("target_day")[cols].mean()
        smooth = daily.rolling(cfg.stress.rolling_days, min_periods=1).mean()
        out.append(smooth.assign(model=name).reset_index())
    return pd.concat(out, ignore_index=True)


def level(curve: PowerCurve, capacity: pd.Series, ratio: float) -> pd.DataFrame:
    """The curve at LEVEL_SPEED onshore (offshore ratio x), mean over directions, after every
    update, in MW with the registered capacity of the day."""
    phi = np.vstack(
        [curve.basis.along(np.array([LEVEL_SPEED]), ratio, d) for d in range(0, 360, 45)]
    )
    times, betas = zip(*curve.history, strict=True)
    times = pd.DatetimeIndex(times)
    y = (phi @ np.vstack(betas).T).mean(axis=0)
    cap = capacity.reindex(times, method="ffill").to_numpy()
    return pd.DataFrame({"time_utc": times, "level_mw": y * cap, "model": curve.name})


def _figures(cfg: Config, roll: pd.DataFrame, levels: pd.DataFrame, names, first, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mark(ax) -> None:
        for e, t in zip(cfg.stress.events, event_times(cfg), strict=True):
            ax.axvline(t, color="0.3", ls="--", lw=0.8)
            ax.annotate(
                e.label,
                (t, 1),
                xycoords=("data", "axes fraction"),
                fontsize=7,
                rotation=90,
                va="top",
                ha="right",
            )
        if first is not None:
            ax.axvline(first, color="C3", ls=":", lw=1)
            ax.annotate(
                "new offshore output first shows",
                (first, 0.02),
                fontsize=7,
                rotation=90,
                xycoords=("data", "axes fraction"),
                va="bottom",
                ha="left",
                color="C3",
            )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for name in names:
        r = roll[roll["model"] == name]
        when = pd.to_datetime(r["target_day"]).dt.tz_localize("UTC")
        for ax, col in zip(axes, ("bias_mw", "crps_mw"), strict=True):
            ax.plot(when, r[col], lw=1.2, label=name)
    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[0].set_ylabel("bias, point - observed (MW)")
    axes[1].set_ylabel("CRPS (MW)")
    for ax in axes:
        mark(ax)
    axes[0].set_title(f"Trailing {cfg.stress.rolling_days}-day means, day-ahead product hours")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "rolling.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, g in levels.groupby("model", sort=False):
        ax.plot(g["time_utc"], g["level_mw"], lw=1.2, label=name)
    mark(ax)
    ax.set_xlim(pd.Timestamp(cfg.stress.plot_start, tz="UTC"), levels["time_utc"].max())
    ax.set(
        ylabel="DK1 power (MW)",
        title=f"RLS curve at {LEVEL_SPEED:g} m/s onshore, x registered capacity",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out / "curve_level.png", dpi=150)
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    cols = [f"{m} {p}" for m, p in df.columns]
    lines = ["| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, row in df.iterrows():
        cells = [
            f"{v:,.0f}" if "hours" in c else f"{v:+.0f}" if "bias" in c else f"{v:.0f}"
            for c, v in zip(cols, row, strict=True)
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(cfg: Config, fc: pd.DataFrame, table: pd.DataFrame, tuned: float, ratio: float) -> Path:
    """Rerun the RLS variants, score every regime in MW, write the report and figures."""
    last = pd.to_datetime(table["target_day"]).dt.date.max()
    frames, levels = [fc[fc["model"].isin(REGIMES)]], []
    capacity = build.read(cfg, "capacity")["total_mw"]
    curves = variants(cfg, tuned)
    for curve in curves:
        frames.append(backtest.run(cfg, curve, table, "recursive", cfg.dates.train_start, last))
        levels.append(level(curve, capacity, ratio))
    rows = common(in_mw(cfg, pd.concat(frames, ignore_index=True), table))
    if rows.empty or "stress" not in set(rows["period"]):
        raise ValueError("no stress-period forecasts with observations")

    table_mw = summary(rows)
    names = [*REGIMES, *(c.name for c in curves)]
    table_mw = table_mw.reindex(names)
    shown = [*REGIMES, rls_name(tuned), rls_name(min(cfg.model.rls.forgetting_grid)), RESET]
    first = first_output(cfg)
    out = cfg.paths.reports
    figs = out / FIGURES
    figs.mkdir(parents=True, exist_ok=True)
    level_rows = pd.concat(levels, ignore_index=True)
    _figures(
        cfg, rolling(cfg, rows), level_rows[level_rows["model"].isin(shown)], shown, first, figs
    )

    crps = table_mw["crps_mw"]
    grid = [rls_name(lam) for lam in cfg.model.rls.forgetting_grid]
    bias = table_mw["bias_mw"]
    best = {p: crps.loc[grid, p].idxmin() for p in PERIODS}
    slow = rls_name(max(cfg.model.rls.forgetting_grid))
    fast = rls_name(min(cfg.model.rls.forgetting_grid))
    n = rows.groupby("period")["time_utc"].nunique()
    events = "; ".join(f"{e.date}: {e.label}" for e in cfg.stress.events)
    text = [
        "# Regime change: Thor and IFS 50r1",
        "",
        f"Generated by `make stress`. Events: {events}."
        + (
            f" New offshore output first shows on {first:%Y-%m-%d %H:%M} UTC (offshore "
            "production above its maximum of the year before)."
            if first is not None
            else ""
        ),
        "",
        "All scores in MW on the day-ahead product hours every model forecast: test year "
        f"({n.get('test', 0):,} hours, stable fleet) and stress period ({n.get('stress', 0):,} "
        "hours). Forecasts in MW are the quantiles times the capacity known at issue time; the "
        "register counts Thor in full from its first event, months before Thor produces. "
        "`rls_<lambda>`: the RLS curve with forgetting lambda (memory ~ 1 / (1 - lambda) rows, "
        f"~37 a day); `{RESET}`: lambda = {tuned:g} with the covariance reset at every event.",
        "",
        "## Scores (MW)",
        "",
        _md(table_mw[["hours", "bias_mw", "mae_mw", "rmse_mw", "crps_mw"]]),
        "",
        "## Forgetting: tracking against variance",
        "",
        f"Shorter memory tracks the change: the stress-period bias falls from "
        f"{bias.loc[slow, 'stress']:+.0f} MW ({slow}) to {bias.loc[fast, 'stress']:+.0f} MW "
        f"({fast}), while the test-year CRPS rises from {crps.loc[best['test'], 'test']:.0f} MW "
        f"({best['test']}) to {crps.loc[fast, 'test']:.0f} MW ({fast}). The best lambda by CRPS "
        f"is {best['test']} in the stable test year and {best['stress']} under change: no single "
        "memory suits both, the trade-off of exponential forgetting.",
        "",
        f"`{RESET}` removes the bias ({bias.loc[RESET, 'stress']:+.0f} MW) but has the worst "
        f"stress CRPS ({crps.loc[RESET, 'stress']:.0f} MW). Resetting P to delta I forgets the "
        "precision of all 64 parameters, not only those the event changed, so the curve swings "
        "until the data pin it down again (curve-level figure). A partial reset (P + alpha I) "
        "or one limited to the offshore block would be the next step; it is not tuned here, "
        "since the stress period is the evaluation.",
        "",
        "## Figures",
        "",
        f"![Rolling bias and CRPS]({FIGURES}/rolling.png)",
        "",
        f"![Curve level over time]({FIGURES}/curve_level.png)",
        "",
    ]
    path = out / REPORT
    path.write_text("\n".join(text))
    log.info("stress: %d models, report -> %s", len(names), path)
    return path
