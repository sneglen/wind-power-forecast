"""DK1 imbalance-market backtest (Step 7): what the forecast's uncertainty is worth in euros.

A price-taking portfolio produces E_t = C_p y_t MWh in hour t (C_p = ``portfolio_mw``), sells
its day-ahead bid b_t at the day-ahead price and settles the imbalance E_t - b_t at the single
imbalance price (one price for long and short positions since 2025-03-19):

    R_t = lambda_DA b_t + lambda_imb (E_t - b_t).

The imbalance cost is the revenue lost against a perfect bid (b_t = E_t),

    C_t = (lambda_DA - lambda_imb) (E_t - b_t),

reported in EUR per MWh produced (sum C / sum E) over the market window. Hourly bids are
repeated over the four 15-min MTUs and settled against the hourly mean of the 15-min prices,
which is exact if production is flat within the hour (settlement data is hourly).

Bids b = C_p q_tau, from each model's issued quantiles:

- ``mean``: the mean of the predictive distribution (the mean of the 99 quantiles);
- ``median``: tau = 0.5;
- ``tau*``: the level with the lowest cost over the trailing days (walk-forward);
- ``newsvendor``: tau = c_down / (c_up + c_down) from the trailing days, where c_up is the
  mean premium lambda_imb - lambda_DA paid per MWh short (E < b) and c_down the mean discount
  lambda_DA - lambda_imb per MWh long (E > b), positions taken from the median bid.

Energinet's day-ahead forecast is bid as published and recalibrated (Step 6). Trailing data
for target day D ends with local day D-2, the last day complete at the issue time on D-1, and
never reaches back before the market window (one imbalance-price method).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast import evaluation, features, scoring
from wpforecast.config import Config
from wpforecast.data import build
from wpforecast.scoring import qcol

log = logging.getLogger(__name__)

REPORT = "market.md"
FIGURES = "market"
RESULTS = "market.json"
QUARTERS = 4  # 15-min imbalance settlement periods per hour
MEDIAN = 0.5
STRATEGIES = ("mean", "median", "tau*", "newsvendor")
ENERGINET = {"energinet": "energinet, as published", "energinet_recal": "energinet, recalibrated"}
DAY = dt.timedelta(days=1)


# --- inputs ---------------------------------------------------------------------------------


def hourly_prices(cfg: Config) -> pd.DataFrame:
    """Per UTC hour with all four quarters: the day-ahead price ``da``, the mean imbalance
    price ``imb`` (EUR/MWh) and the mean system imbalance ``system_mw`` (satisfied demand,
    > 0 when the system is short and regulated up)."""
    quarters = build.read(cfg, "imbalance").resample("1h")
    hourly = pd.DataFrame(
        {
            "imb": quarters["imbalance_price_eur_mwh"].mean(),
            "system_mw": quarters["satisfied_demand_mw"].mean(),
            "n": quarters["imbalance_price_eur_mwh"].count(),
        }
    )
    hourly = hourly[hourly["n"] == QUARTERS].drop(columns="n")
    return hourly.join(build.read(cfg, "da_price")["price_eur_mwh"].rename("da"), how="inner")


def market_rows(cfg: Config, fc: pd.DataFrame) -> pd.DataFrame:
    """Product hours of the market window with prices and Energinet's bid, forecast by every
    model; Energinet's forecast is a fraction of capacity, clipped to [0, 1]."""
    mk = cfg.market
    days = pd.to_datetime(fc["target_day"]).dt.date
    fc = fc[fc["in_product"] & (days >= mk.start) & (days <= mk.end)].dropna(subset=["y"])
    fc = fc.merge(hourly_prices(cfg), left_on="time_utc", right_index=True, how="inner")
    found = evaluation.energinet_forecast(cfg)
    if found is not None:
        energinet, a, b = found
        mw = energinet.reindex(fc["time_utc"]).to_numpy()
        cap = fc["capacity_mw"].to_numpy()
        fc = fc.assign(
            energinet=np.clip(mw / cap, 0, 1), energinet_recal=np.clip((a * mw + b) / cap, 0, 1)
        ).dropna(subset=["energinet"])
    per_hour = fc.groupby("time_utc")["model"].transform("nunique")
    return fc[per_hour == fc["model"].nunique()].sort_values("time_utc").copy()


# --- costs and bid levels -------------------------------------------------------------------


def cost(cfg: Config, rows: pd.DataFrame, bid: np.ndarray) -> np.ndarray:
    """Imbalance cost (EUR) per hour of bidding ``bid`` (fraction of capacity; a matrix of
    bids gives one column per bid)."""
    spread = (rows["da"] - rows["imb"]).to_numpy()
    y = rows["y"].to_numpy()
    bid = np.asarray(bid, dtype=float)
    if bid.ndim == 2:
        spread, y = spread[:, None], y[:, None]
    return cfg.market.portfolio_mw * spread * (y - bid)


def positions(cfg: Config, rows: pd.DataFrame, bid: np.ndarray) -> pd.DataFrame:
    """Per hour: MWh short and long against ``bid`` and their cost in EUR (up + down = cost)."""
    cp = cfg.market.portfolio_mw
    energy, bid = cp * rows["y"].to_numpy(), cp * np.asarray(bid, dtype=float)
    premium = (rows["imb"] - rows["da"]).to_numpy()  # paid per MWh short, above day-ahead
    short, long = np.maximum(bid - energy, 0), np.maximum(energy - bid, 0)
    return pd.DataFrame(
        {
            "short_mwh": short,
            "long_mwh": long,
            "up_eur": premium * short,
            "down_eur": -premium * long,
        },
        index=rows.index,
    )


def newsvendor(totals: pd.Series) -> tuple[float, float, float]:
    """Effective penalties c_up, c_down (EUR/MWh) and tau = c_down / (c_up + c_down).

    tau falls back to the median when the penalties do not define a level (sum <= 0)."""
    c_up = totals["up_eur"] / totals["short_mwh"] if totals["short_mwh"] > 0 else np.nan
    c_down = totals["down_eur"] / totals["long_mwh"] if totals["long_mwh"] > 0 else np.nan
    total = c_up + c_down
    tau = c_down / total if total > 0 else MEDIAN
    return float(c_up), float(c_down), float(tau)


def nearest(levels: np.ndarray, tau: float) -> float:
    return float(levels[np.abs(levels - tau).argmin()])


def trailing_window(cfg: Config, day: dt.date) -> tuple[dt.date, dt.date]:
    """First and last target day known in full at the issue time of ``day``."""
    issue_day = features.issue_time_utc(cfg, day).tz_convert(cfg.issue.tz).date()
    last = issue_day - DAY  # the issue day itself is not complete at the issue time
    first = max(cfg.market.start, last - (cfg.market.tau_lookback_days - 1) * DAY)
    return first, last


def walk_forward(cfg: Config, by_tau: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    """Per target day, the bid levels chosen from the trailing days only.

    ``by_tau``: daily cost (EUR) per grid level; ``parts``: daily ``positions`` of the median
    bid. Both are indexed by target day."""
    levels = by_tau.columns.to_numpy(dtype=float)
    days = np.array(by_tau.index)
    out = []
    for day in days:
        first, last = trailing_window(cfg, day)
        sel = (days >= first) & (days <= last)
        if sel.sum() < cfg.market.tau_min_days:
            out.append((day, MEDIAN, MEDIAN))
            continue
        star = float(levels[by_tau[sel].sum().to_numpy().argmin()])
        out.append((day, star, nearest(levels, newsvendor(parts[sel].sum())[2])))
    return pd.DataFrame(out, columns=["target_day", "tau*", "newsvendor"]).set_index("target_day")


def calm(cfg: Config, rows: pd.DataFrame) -> np.ndarray:
    """Hours whose spread |lambda_DA - lambda_imb| stays within ``spike_eur_mwh``."""
    return np.abs((rows["da"] - rows["imb"]).to_numpy()) <= cfg.market.spike_eur_mwh


def analyse(cfg: Config, g: pd.DataFrame) -> dict:
    """One model: hourly cost of every strategy, daily cost per grid level (all hours and
    calm hours only), and the daily choices of the walk-forward strategies."""
    levels = np.asarray(scoring.crps_levels(cfg))
    q = g[[qcol(x) for x in levels]].to_numpy()
    day = g["target_day"].to_numpy()
    median = q[:, list(levels).index(MEDIAN)]
    by_level = cost(cfg, g, q)
    by_tau = pd.DataFrame(by_level, columns=levels).groupby(day).sum()
    by_tau_calm = pd.DataFrame(by_level * calm(cfg, g)[:, None], columns=levels).groupby(day).sum()
    parts = positions(cfg, g, median).groupby(day).sum()
    choice = walk_forward(cfg, by_tau, parts)
    column = {x: i for i, x in enumerate(levels)}
    bids = {"mean": q.mean(axis=1), "median": median}
    for name in ("tau*", "newsvendor"):
        idx = pd.Series(day).map(choice[name]).map(column).to_numpy()
        bids[name] = q[np.arange(len(q)), idx]
    hourly = pd.DataFrame({k: cost(cfg, g, b) for k, b in bids.items()}, index=g.index)
    return {
        "hourly": hourly,
        "by_tau": by_tau,
        "by_tau_calm": by_tau_calm,
        "choice": choice,
        "median": median,
    }


def decompose(cfg: Config, g: pd.DataFrame, bid: np.ndarray) -> dict[str, float]:
    """Cost per MWh = covariance part + mean-spread part.

    With s = lambda_DA - lambda_imb and e = E - b over n hours, sum s e = sum (s - s_bar)(e -
    e_bar) + n s_bar e_bar: the first part is how the own error moves with the price spread,
    the second the bias of the bid times the mean spread (a bet, not a forecast property)."""
    cp = cfg.market.portfolio_mw
    s = (g["da"] - g["imb"]).to_numpy()
    e = cp * (g["y"].to_numpy() - bid)
    energy = cp * g["y"].sum()
    covariance = ((s - s.mean()) * (e - e.mean())).sum() / energy
    return {
        "cost": float((s * e).sum() / energy),
        "covariance": float(covariance),
        "mean_spread": float(len(s) * s.mean() * e.mean() / energy),
        "corr": float(np.corrcoef(s, e)[0, 1]),
        "mae_mw": float(np.abs(e).mean()),
    }


# --- uncertainty ----------------------------------------------------------------------------


def _weights(cfg: Config, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """How often each day enters each day-block bootstrap sample, shape (draws, days)."""
    ev = cfg.evaluation
    idx = evaluation.block_bootstrap(n_days, ev.bootstrap_block_days, ev.n_bootstrap, rng)
    w = np.zeros((ev.n_bootstrap, n_days))
    np.add.at(w, (np.repeat(np.arange(ev.n_bootstrap), idx.shape[1]), idx.ravel()), 1.0)
    return w


def _band(cfg: Config, w: np.ndarray, daily_cost: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Bootstrap band of sum(cost) / sum(energy); a matrix of costs gives one band per column."""
    energy = w @ energy
    ratio = (w @ daily_cost) / (energy[:, None] if daily_cost.ndim == 2 else energy)
    band = cfg.evaluation.band
    return np.quantile(ratio, [(1 - band) / 2, (1 + band) / 2], axis=0)


# --- report ---------------------------------------------------------------------------------


def _figures(cfg: Config, res: dict, energy: pd.DataFrame, energinet, w, names, out: Path):
    """Cost against the bid level (all hours, calm hours) and the walk-forward levels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (
        ("all hours", "by_tau", "all"),
        (
            f"excl. hours with |spread| > {cfg.market.spike_eur_mwh:g} EUR/MWh",
            "by_tau_calm",
            "calm",
        ),
    )
    for ax, (title, key, which) in zip(axes, panels, strict=True):
        e = energy[which].to_numpy()
        for i, name in enumerate(n for n in names if n in res):
            by_tau = res[name][key]
            lo, hi = _band(cfg, w, by_tau.to_numpy(), e)
            levels = by_tau.columns.to_numpy(dtype=float)
            ax.fill_between(levels, lo, hi, color=f"C{i}", alpha=0.15, lw=0)
            ax.plot(levels, by_tau.sum().to_numpy() / e.sum(), color=f"C{i}", lw=1.4, label=name)
        for (label, daily), style in zip(energinet[which].items(), ("--", ":"), strict=False):
            ax.axhline(daily.sum() / e.sum(), color="0.3", ls=style, lw=1, label=label)
        ax.axvline(MEDIAN, color="0.7", lw=0.8)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set(xlabel="bid quantile level tau", title=title)
    axes[0].set_ylabel("imbalance cost (EUR per MWh produced)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8, frameon=False
    )
    fig.suptitle(f"DK1 single price, {cfg.market.start} to {cfg.market.end}")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "cost_vs_tau.png", dpi=150)
    plt.close(fig)

    ref = cfg.market.reference_model
    if ref in res:
        choice = res[ref]["choice"]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        when = pd.to_datetime(pd.Series(choice.index))
        for name in ("tau*", "newsvendor"):
            ax.step(when, choice[name], where="post", lw=1.2, label=name)
        ax.axhline(MEDIAN, color="0.7", lw=0.8)
        ax.set(ylim=(0, 1), ylabel="bid level tau", title=f"{ref}: walk-forward bid levels")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "tau_over_time.png", dpi=150)
        plt.close(fig)


def _md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)])


def run(
    cfg: Config, fc: pd.DataFrame, figure_models: tuple[str, ...], baselines: tuple[str, ...]
) -> Path:
    """Write the market report, its figures and ``market.json``.

    ``figure_models`` are drawn in the cost-vs-tau figure; the median bids of ``baselines``
    are compared with the reference model's median bid."""
    rows = market_rows(cfg, fc)
    if rows.empty:
        raise ValueError("no forecasts with prices and observations in the market window")
    mk, cp = cfg.market, cfg.market.portfolio_mw
    rng = np.random.default_rng(cfg.evaluation.seed)
    res = {name: analyse(cfg, g) for name, g in rows.groupby("model", sort=False)}
    ref = mk.reference_model if mk.reference_model in res else next(iter(res))
    g = rows[rows["model"] == ref]  # every model has the same hours, in time order
    day = g["target_day"].to_numpy()
    quiet = calm(cfg, g)
    energy = pd.DataFrame({"all": cp * g["y"], "calm": cp * g["y"] * quiet}).groupby(day).sum()
    total_energy = energy["all"].sum()
    w = _weights(cfg, len(energy), rng)

    # Every model: strategies, the median's band, the best fixed level in hindsight, and the
    # median's cost split into its covariance and mean-spread parts.
    table, parts, results = [], [], {}
    for name, r in res.items():
        daily = r["hourly"].groupby(day).sum()
        total = daily.sum() / total_energy
        lo, hi = _band(cfg, w, daily["median"].to_numpy(), energy["all"].to_numpy())
        best = r["by_tau"].sum() / total_energy
        best_calm = r["by_tau_calm"].sum() / energy["calm"].sum()
        split = decompose(cfg, rows[rows["model"] == name], r["median"])
        results[name] = {
            **total.to_dict(),
            "median_band": [float(lo), float(hi)],
            "best_tau": float(best.idxmin()),
            "best_tau_calm_hours": float(best_calm.idxmin()),
            "median_split": split,
        }
        table.append(
            [
                name,
                *(f"{total[s]:.2f}" for s in STRATEGIES),
                f"{lo:.2f} to {hi:.2f}",
                f"{best.idxmin():.2f} ({best.min():.2f})",
                f"{best_calm.idxmin():.2f} ({best_calm.min():.2f})",
            ]
        )
        parts.append(
            [
                name,
                f"{split['mae_mw']:.1f}",
                f"{split['cost']:.2f}",
                f"{split['covariance']:+.2f}",
                f"{split['mean_spread']:+.2f}",
                f"{split['corr']:+.3f}",
            ]
        )

    # The reference model's strategies and Energinet, with bands and savings vs the median.
    hourly = res[ref]["hourly"].rename(columns=lambda c: f"{ref}, {c}")
    median = f"{ref}, median"
    for name in baselines:
        if name in res:
            hourly[f"{name}, median"] = res[name]["hourly"]["median"].to_numpy()
    for col, label in ENERGINET.items():
        if col in g:
            hourly[label] = cost(cfg, g, g[col].to_numpy())
    daily = hourly.groupby(day).sum()
    energinet = {
        "all": {c: daily[c] for c in ENERGINET.values() if c in daily},
        "calm": {
            c: hourly.loc[quiet, c].groupby(day[quiet]).sum()
            for c in ENERGINET.values()
            if c in daily
        },
    }
    curtailed = g["curtailed"].to_numpy()
    clean = hourly[~curtailed].sum() / (cp * g.loc[~curtailed, "y"]).sum()
    bands = _band(cfg, w, daily.to_numpy(), energy["all"].to_numpy())
    gain = daily[median].to_numpy()[:, None] - daily.to_numpy()
    savings = _band(cfg, w, gain, energy["all"].to_numpy())
    detail = [
        [
            col,
            f"{daily[col].sum() / total_energy:.2f}",
            f"{bands[0, j]:.2f} to {bands[1, j]:.2f}",
            f"{(daily[median].sum() - daily[col].sum()) / total_energy:+.2f}",
            f"{savings[0, j]:+.2f} to {savings[1, j]:+.2f}",
            f"{clean[col]:.2f}",
        ]
        for j, col in enumerate(daily.columns)
    ]

    # Why the single price still prices the forecast: own position against the system.
    pos = positions(cfg, g, res[ref]["median"])
    system_short = g["system_mw"].to_numpy() > 0
    c_up, c_down, tau_eff = newsvendor(pos.sum())
    error = cp * (g["y"].to_numpy() - res[ref]["median"])
    corr = float(np.corrcoef(error, g["system_mw"].fillna(0).to_numpy())[0, 1])
    pos_rows = []
    for label, mwh, penalty in (
        ("short (E < b): pays lambda_imb - lambda_DA", "short_mwh", c_up),
        ("long (E > b): forgoes lambda_DA - lambda_imb", "long_mwh", c_down),
    ):
        sel = (pos[mwh] > 0).to_numpy()
        pos_rows.append(
            [
                label,
                f"{sel.sum()}",
                f"{pos[mwh].sum():.0f}",
                f"{system_short[sel].mean():.0%}",
                f"{penalty:+.1f}",
            ]
        )
    choice = res[ref]["choice"]
    spread = (g["da"] - g["imb"]).to_numpy()

    out = cfg.paths.reports
    figs = out / FIGURES
    figs.mkdir(parents=True, exist_ok=True)
    _figures(cfg, res, energy, energinet, w, figure_models, figs)
    band = f"{cfg.evaluation.band:.0%}"
    spike = f"{mk.spike_eur_mwh:g}"
    text = [
        "# DK1 imbalance-market backtest",
        "",
        f"Generated by `make market`. Window {mk.start} to {mk.end} (single imbalance price): "
        f"{len(energy)} days, {len(g)} hours bid by every model and by Energinet. A "
        f"price-taking {cp:g} MW portfolio produces y x {cp:g} MW and bids one quantile of its "
        "forecast day-ahead; hourly bids are settled per 15 minutes against the hourly mean "
        "price. Cost = revenue lost against a perfect bid, sum (lambda_DA - lambda_imb)(E - b) "
        "/ sum E, in EUR per MWh produced (lower is better; negative = earned). Bands are "
        f"{band} day-block bootstrap bands.",
        "",
        "## Cost by bid strategy (EUR/MWh)",
        "",
        "`tau*` and `newsvendor` choose a level each day from the trailing "
        f"{mk.tau_lookback_days} complete days (median until {mk.tau_min_days} days exist). "
        "The last two columns are the best fixed level in hindsight (and its cost), over all "
        f"hours and over the hours with |lambda_DA - lambda_imb| <= {spike} EUR/MWh; they are "
        "not strategies.",
        "",
        _md(
            [
                "model",
                *STRATEGIES,
                f"median, {band} band",
                "best tau, all hours",
                f"best tau, spread <= {spike}",
            ],
            table,
        ),
        "",
        f"## {ref} against its alternatives",
        "",
        f"Saving = cost of {ref}'s median bid minus cost of the bid (positive = cheaper), "
        "with a paired band over the same days. Energinet's forecast is issued at 18:00, "
        "after gate closure: an information edge.",
        "",
        _md(
            ["bid", "cost", f"{band} band", "saving vs median", f"{band} band", "excl. curtailed"],
            detail,
        ),
        "",
        f"Levels chosen for {ref} (mean, 10-90 % range): "
        + ", ".join(
            f"{s} {choice[s].mean():.2f} ({choice[s].quantile(0.1):.2f} to "
            f"{choice[s].quantile(0.9):.2f})"
            for s in ("tau*", "newsvendor")
        )
        + ".",
        "",
        "## What drives the cost of the median bid",
        "",
        "With s = lambda_DA - lambda_imb and e = E - b, sum s e = sum (s - s_bar)(e - e_bar) "
        "+ n s_bar e_bar. The covariance part prices how the own error moves with the spread; "
        "the mean-spread part is the bid's bias times the mean spread, a bet rather than a "
        f"property of the forecast. {(~quiet).sum()} hours ({(~quiet).mean():.1%}) have "
        f"|spread| > {spike} EUR/MWh (largest {np.abs(spread).max():.0f}); the mean spread "
        f"is {spread.mean():+.1f} EUR/MWh over all hours and {spread[quiet].mean():+.1f} over "
        "the rest.",
        "",
        _md(
            ["model", "MAE (MW)", "cost", "covariance part", "mean-spread part", "corr(s, e)"],
            parts,
        ),
        "",
        "## Position against the system",
        "",
        f"Hours by the sign of {ref}'s median-bid imbalance, the share in which the system was "
        "short, and the mean effective penalty per MWh (EUR/MWh; positive = a cost).",
        "",
        _md(["own position", "hours", "MWh", "system short", "penalty"], pos_rows),
        "",
        f"Correlation of the own imbalance E - b (MW) with the system imbalance: {corr:+.2f}. "
        f"Whole-window newsvendor level c_down / (c_up + c_down) = {tau_eff:.2f} (hindsight).",
        "",
        "## Figures",
        "",
        "![cost vs tau](market/cost_vs_tau.png)",
        "",
        "![tau over time](market/tau_over_time.png)",
        "",
    ]
    path = out / REPORT
    path.write_text("\n".join(text))
    summary = {
        "window": [mk.start.isoformat(), mk.end.isoformat()],
        "days": len(energy),
        "hours": len(g),
        "spike_hours": int((~quiet).sum()),
        "eur_per_mwh": results,
        "energinet": {c: float(v.sum() / total_energy) for c, v in energinet["all"].items()},
        "reference": {
            "model": ref,
            "c_up": c_up,
            "c_down": c_down,
            "tau_newsvendor_hindsight": tau_eff,
            "correlation_with_system": corr,
        },
    }
    out_json = cfg.paths.outputs / RESULTS
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("market: %d hours, report -> %s", len(g), path)
    return path
