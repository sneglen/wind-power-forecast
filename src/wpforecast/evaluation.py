"""Test-year verification (Step 6): one honest score of every model, written as a report.

Protocol after Madsen et al. (2005) and Pinson et al. (2007): every model is scored on the
same hours (the day-ahead product hours of the test period that every model forecast), with
and without curtailed hours, on

- CRPS and its skill over climatology, overall (with a day-block bootstrap band) and by k;
- MAE and RMSE of the point forecast in % of capacity, against Energinet's day-ahead
  forecast on the hours it covers (it is issued at 18:00 with newer weather: an edge);
- reliability: the observed frequency below each quantile, with bootstrap bands;
- sharpness: the coverage and mean width of central prediction intervals;
- PIT histograms (randomised ranks of the observation among the 99 quantiles);
- Diebold-Mariano tests on daily mean CRPS differences, with a Newey-West variance.
- with ENS models (Step 8): the CRPS change from adding the ENS spread to each layer, and
  spread-skill: the RMSE of the point forecast against the predicted spread, in bins.

Days are the resampling unit because errors within a day are strongly dependent.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from wpforecast import scoring
from wpforecast.config import Config
from wpforecast.data import build

log = logging.getLogger(__name__)

REPORT = "evaluation.md"
FIGURES = "evaluation"
REFERENCE = "climatology"
RAW_ENSEMBLE, HRES_CURVE, WITH_SPREAD = "ens_raw", "curve_rls", "_ens"


def scored_rows(fc: pd.DataFrame) -> pd.DataFrame:
    """Test-period product hours with an observation, forecast by every model."""
    fc = fc[(fc["period"] == "test") & fc["in_product"]].dropna(subset=["y"])
    n_models = fc["model"].nunique()
    per_hour = fc.groupby(["issue_time_utc", "time_utc"])["model"].transform("nunique")
    return fc[per_hour == n_models].copy()


def add_row_scores(cfg: Config, fc: pd.DataFrame) -> pd.DataFrame:
    """CRPS, point errors and central-interval hits and widths per row."""
    grid = np.asarray(scoring.crps_levels(cfg))
    y = fc["y"].to_numpy()
    q = fc[[scoring.qcol(x) for x in grid]].to_numpy()
    out = fc.assign(
        crps=scoring.crps(y, q, grid),
        abs_error=np.abs(y - fc["point"].to_numpy()),
        sq_error=(y - fc["point"].to_numpy()) ** 2,
    )
    for width in cfg.evaluation.intervals:
        lo = fc[scoring.qcol((1 - width) / 2)].to_numpy()
        hi = fc[scoring.qcol((1 + width) / 2)].to_numpy()
        out[f"in_{width:g}"] = (y >= lo) & (y <= hi)
        out[f"width_{width:g}"] = hi - lo
    return out


# --- resampling and tests ------------------------------------------------------------------


def block_bootstrap(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``draws`` moving-block bootstrap samples of a length-n series."""
    block = min(block, n)
    per_sample = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(draws, per_sample))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)
    return idx[:, :n]


def diebold_mariano(d: np.ndarray, max_lag: int) -> tuple[float, float]:
    """DM statistic and two-sided p-value for mean(d) = 0 (Newey-West long-run variance)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    e = d - d.mean()
    lrv = e @ e / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        lrv += 2 * (1 - lag / (max_lag + 1)) * (e[lag:] @ e[:-lag]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(lrv / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def pit(cfg: Config, fc: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Randomised rank of y among the 99 quantiles, scaled to (0, 1): uniform if calibrated."""
    grid = scoring.crps_levels(cfg)
    q = fc[[scoring.qcol(x) for x in grid]].to_numpy()
    y = fc["y"].to_numpy()[:, None]
    below, at = (q < y).sum(axis=1), (q <= y).sum(axis=1)
    rank = below + rng.random(len(y)) * (at - below + 1)  # ties at the bounds share the mass
    return rank / (len(grid) + 1)


# --- tables ---------------------------------------------------------------------------------


def summary(cfg: Config, rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per model: CRPS, skill with a bootstrap band, MAE, RMSE, interval coverage and width."""
    ev = cfg.evaluation
    daily = rows.pivot_table(index="target_day", columns="model", values="crps", aggfunc="sum")
    idx = block_bootstrap(len(daily), ev.bootstrap_block_days, ev.n_bootstrap, rng)
    sums = daily.to_numpy()[idx].sum(axis=1)  # (draws, models)
    ref = list(daily.columns).index(REFERENCE) if REFERENCE in daily.columns else None
    lo_q, hi_q = (1 - ev.band) / 2, (1 + ev.band) / 2

    out = []
    for name, g in rows.groupby("model", sort=False):
        r = {
            "model": name,
            "hours": len(g),
            "crps": g["crps"].mean(),
            "mae_pct": 100 * g["abs_error"].mean(),
            "rmse_pct": 100 * math.sqrt(g["sq_error"].mean()),
        }
        if ref is not None:
            j = list(daily.columns).index(name)
            boot = 1 - sums[:, j] / sums[:, ref]
            r["skill"] = 1 - daily[name].sum() / daily[REFERENCE].sum()
            r["skill_lo"], r["skill_hi"] = np.quantile(boot, [lo_q, hi_q])
        for width in ev.intervals:
            r[f"cover_{width:g}"] = g[f"in_{width:g}"].mean()
            r[f"width_{width:g}"] = g[f"width_{width:g}"].mean()
        out.append(r)
    return pd.DataFrame(out).set_index("model")


def reliability(cfg: Config, rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Observed frequency below each reporting quantile, with day-block bootstrap bands."""
    ev = cfg.evaluation
    levels = cfg.quantiles.levels
    out = []
    for name, g in rows.groupby("model", sort=False):
        below = g[[scoring.qcol(x) for x in levels]].to_numpy() >= g["y"].to_numpy()[:, None]
        per_day = pd.DataFrame(below, index=g["target_day"].to_numpy()).groupby(level=0)
        hits, hours = per_day.sum().to_numpy(), per_day.size().to_numpy()
        idx = block_bootstrap(len(hours), ev.bootstrap_block_days, ev.n_bootstrap, rng)
        freq = hits[idx].sum(axis=1) / hours[idx].sum(axis=1)[:, None]
        lo, hi = np.quantile(freq, [(1 - ev.band) / 2, (1 + ev.band) / 2], axis=0)
        for i, level in enumerate(levels):
            out.append(
                {
                    "model": name,
                    "level": level,
                    "observed": hits[:, i].sum() / hours.sum(),
                    "lo": lo[i],
                    "hi": hi[i],
                }
            )
    return pd.DataFrame(out)


def dm_matrix(cfg: Config, rows: pd.DataFrame) -> pd.DataFrame:
    """DM statistic of row model minus column model (negative: the row model is better)."""
    daily = rows.pivot_table(index="target_day", columns="model", values="crps", aggfunc="mean")
    names = list(dict.fromkeys(rows["model"]))
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a != b:
                table.loc[a, b] = diebold_mariano(
                    (daily[a] - daily[b]).to_numpy(), cfg.evaluation.dm_max_lag
                )[0]
    return table


def energinet_forecast(cfg: Config) -> tuple[pd.Series, float, float] | None:
    """Energinet's day-ahead DK1 forecast (MW) and its recalibration P = a F + b.

    Energinet's forecast does not cover exactly the settled production (it runs about 16 %
    low in 2024-25), so it is also used after a linear recalibration fitted on the training
    period only.
    """
    path = cfg.paths.interim / "energinet_forecast.parquet"
    if not path.exists():
        return None
    energinet = build.read(cfg, "energinet_forecast")["total_da_mw"]
    dataset = build.read(cfg, "dataset")
    train = dataset.loc[str(cfg.dates.train_start) : str(cfg.dates.train_end), "power_mw"]
    pairs = pd.DataFrame({"p": train, "f": energinet.reindex(train.index)}).dropna()
    a, b = np.polyfit(pairs["f"], pairs["p"], 1) if len(pairs) > 1 else (1.0, 0.0)
    return energinet, float(a), float(b)


def versus_energinet(cfg: Config, rows: pd.DataFrame) -> pd.DataFrame | None:
    """Point MAE and RMSE (% of capacity) on the hours Energinet's day-ahead forecast covers,
    as published and recalibrated (``energinet_forecast``)."""
    found = energinet_forecast(cfg)
    if found is None:
        return None
    energinet, a, b = found

    first = rows.drop_duplicates("time_utc").set_index("time_utc")
    published = energinet.reindex(first.index).dropna()
    hours = published.index
    keep = rows[rows["time_utc"].isin(hours)]
    table = keep.groupby("model", sort=False).agg(
        mae_pct=("abs_error", "mean"), rmse_pct=("sq_error", "mean")
    )
    table["mae_pct"] *= 100
    table["rmse_pct"] = 100 * np.sqrt(table["rmse_pct"])
    y, cap = first.loc[hours, "y"], first.loc[hours, "capacity_mw"]
    for label, forecast in (
        ("energinet, as published", published),
        (f"energinet, recalibrated ({a:.2f} F {b:+.0f} MW)", a * published + b),
    ):
        err = y - forecast / cap
        table.loc[label] = [100 * err.abs().mean(), 100 * math.sqrt((err**2).mean())]
    table.attrs["hours"] = len(hours)
    return table


# --- ensemble (Step 8) -----------------------------------------------------------------------


def ensemble_pairs(cfg: Config, rows: pd.DataFrame) -> pd.DataFrame | None:
    """Each layer with the ENS spread against itself without; the raw ensemble against the
    HRES curve. DM < 0: the ENS model is better."""
    names = list(dict.fromkeys(rows["model"]))
    pairs = [(n.removesuffix(WITH_SPREAD), n) for n in names if n.endswith(WITH_SPREAD)]
    if RAW_ENSEMBLE in names and HRES_CURVE in names:
        pairs.insert(0, (HRES_CURVE, RAW_ENSEMBLE))
    pairs = [(a, b) for a, b in pairs if a in names]
    if not pairs:
        return None
    daily = rows.pivot_table(index="target_day", columns="model", values="crps", aggfunc="mean")
    crps = rows.groupby("model")["crps"].mean()
    out = []
    for base, ens in pairs:
        stat, p = diebold_mariano((daily[ens] - daily[base]).to_numpy(), cfg.evaluation.dm_max_lag)
        out.append(
            {
                "model": ens,
                "against": base,
                "crps_without": crps[base],
                "crps_with": crps[ens],
                "change": crps[ens] / crps[base] - 1,
                "dm": stat,
                "p": p,
            }
        )
    return pd.DataFrame(out).set_index("model")


def predicted_spread(cfg: Config, rows: pd.DataFrame) -> np.ndarray:
    """Standard deviation of the predictive distribution, from its 99 quantiles."""
    return rows[[scoring.qcol(x) for x in scoring.crps_levels(cfg)]].to_numpy().std(axis=1)


def _binned(x: pd.Series, sq_error: pd.Series, bins: int) -> pd.DataFrame:
    """Mean x and RMSE per bin of equal size by x."""
    b = pd.qcut(x.rank(method="first"), bins, labels=False)
    return (
        pd.DataFrame({"x": x, "se": sq_error})
        .groupby(b)
        .agg(x=("x", "mean"), rmse=("se", lambda e: math.sqrt(e.mean())))
    )


def _spread_skill_figure(cfg: Config, rows: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = cfg.evaluation.spread_bins
    ens = rows[rows["model"] == RAW_ENSEMBLE].set_index(["issue_time_utc", "time_utc"])
    hres = rows[rows["model"] == HRES_CURVE].set_index(["issue_time_utc", "time_utc"])
    spread = pd.Series(predicted_spread(cfg, ens.reset_index()), ens.index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for label, err in (("HRES curve (curve_rls)", hres["sq_error"]), ("ENS mean", ens["sq_error"])):
        b = _binned(spread, err.reindex(spread.index), bins)
        ax.plot(b["x"], b["rmse"], marker="o", ms=3, label=label)
    ax.set(
        xlabel="raw ENS spread (std of member powers)",
        ylabel="RMSE of the point forecast",
        title="Does the ENS spread predict the error?",
    )
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    with_spread = [n for n in dict.fromkeys(rows["model"]) if n.endswith(WITH_SPREAD)]
    bases = [RAW_ENSEMBLE, *(n.removesuffix(WITH_SPREAD) for n in with_spread)]
    for name in [RAW_ENSEMBLE, *(x for n in with_spread for x in (n.removesuffix(WITH_SPREAD), n))]:
        g = rows[rows["model"] == name]
        b = _binned(pd.Series(predicted_spread(cfg, g), g.index), g["sq_error"], bins)
        color = f"C{bases.index(name.removesuffix(WITH_SPREAD))}"
        style = "--" if name.endswith(WITH_SPREAD) else "-"
        ax.plot(b["x"], b["rmse"], style, marker="o", ms=3, color=color, label=name)
    top = ax.get_xlim()[1]
    for a in axes:
        a.plot([0, top], [0, top], color="0.6", lw=0.8)
    ax.set(
        xlabel="predicted spread (std of the 99 quantiles)",
        title="Spread-skill per model (diagonal = consistent)",
    )
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "spread_skill.png", dpi=150)
    plt.close(fig)


# --- figures --------------------------------------------------------------------------------


def _by_horizon(cfg: Config, rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean of ``column`` per horizon and model, for horizons with enough hours."""
    first = rows["model"].iloc[0]
    count = rows[rows["model"] == first].groupby("horizon_h").size()
    keep = count[count >= cfg.evaluation.min_hours_per_horizon].index
    kept = rows[rows["horizon_h"].isin(keep)]
    return kept.pivot_table(index="horizon_h", columns="model", values=column, aggfunc="mean")


def _figures(cfg: Config, rows: pd.DataFrame, rel: pd.DataFrame, pits: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(rows["model"]))
    others = [n for n in names if n != REFERENCE]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (title, sel) in zip(
        axes,
        (("all hours", rows), ("excluding curtailed hours", rows[~rows["curtailed"]])),
        strict=True,
    ):
        by_k = _by_horizon(cfg, sel, "crps")
        ax.axhline(0, color="0.6", lw=1)
        if REFERENCE in by_k:
            for n in others:
                ax.plot(by_k.index, 1 - by_k[n] / by_k[REFERENCE], lw=1.2, label=n)
        ax.set(title=title, xlabel="horizon k (h)")
    axes[0].set_ylabel(f"CRPS skill vs {REFERENCE}")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "skill_by_horizon.png", dpi=150)
    plt.close(fig)

    cols = 3
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(10, 2.6 * nrows), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, names, strict=False):
        r = rel[rel["model"] == n]
        ax.fill_between(r["level"], r["lo"] - r["level"], r["hi"] - r["level"], alpha=0.3)
        ax.plot(r["level"], r["observed"] - r["level"], marker="o", ms=3)
        ax.axhline(0, color="0.6", lw=1)
        ax.set_title(n, fontsize=9)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.supxlabel("nominal level")
    fig.supylabel("observed minus nominal frequency")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)

    bins = np.linspace(0, 1, cfg.evaluation.pit_bins + 1)
    fig, axes = plt.subplots(nrows, cols, figsize=(10, 2.4 * nrows), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, names, strict=False):
        ax.hist(pits[n], bins=bins, density=True, alpha=0.8)
        ax.axhline(1, color="0.3", lw=1, ls="--")
        ax.set_title(n, fontsize=9)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.supxlabel("PIT")
    fig.tight_layout()
    fig.savefig(out / "pit.png", dpi=150)
    plt.close(fig)

    widest = max(cfg.evaluation.intervals)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    by_k = _by_horizon(cfg, rows, f"width_{widest:g}")
    for n in [n for n in names if n in by_k]:
        ax.plot(by_k.index, by_k[n], lw=1.2, label=n)
    ax.set(xlabel="horizon k (h)", ylabel=f"mean width of the {widest:.0%} interval")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "sharpness.png", dpi=150)
    plt.close(fig)


# --- report ---------------------------------------------------------------------------------


def _md(df: pd.DataFrame, fmt: dict[str, str]) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join([df.index.name or "model", *cols]) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for name, row in df.iterrows():
        cells = ["\u2013" if pd.isna(row[c]) else fmt.get(c, "{:.4f}").format(row[c]) for c in cols]
        lines.append("| " + " | ".join([str(name), *cells]) + " |")
    return "\n".join(lines)


def run(cfg: Config, fc: pd.DataFrame) -> Path:
    """Write the evaluation report and its figures under ``paths.reports``."""
    rng = np.random.default_rng(cfg.evaluation.seed)
    rows = add_row_scores(cfg, scored_rows(fc))
    if rows.empty:
        raise ValueError("no out-of-sample forecasts with observations to evaluate")
    out = cfg.paths.reports
    figs = out / FIGURES
    figs.mkdir(parents=True, exist_ok=True)

    widths = cfg.evaluation.intervals
    pct = "{:.1%}"
    fmt = {"hours": "{:.0f}", "mae_pct": "{:.2f}", "rmse_pct": "{:.2f}", "skill": "{:+.1%}"}
    fmt |= {"skill_lo": "{:+.1%}", "skill_hi": "{:+.1%}"}
    fmt |= {f"cover_{w:g}": pct for w in widths} | {f"width_{w:g}": "{:.3f}" for w in widths}
    all_hours = summary(cfg, rows, rng)
    clean = summary(cfg, rows[~rows["curtailed"]], rng)
    rel = reliability(cfg, rows, rng)
    pits = {n: pit(cfg, g, rng) for n, g in rows.groupby("model", sort=False)}
    dm = dm_matrix(cfg, rows)
    energinet = versus_energinet(cfg, rows)
    pairs = ensemble_pairs(cfg, rows)
    _figures(cfg, rows, rel, pits, figs)
    if pairs is not None and RAW_ENSEMBLE in pairs.index:
        _spread_skill_figure(cfg, rows, figs)

    d = cfg.dates
    band = f"{cfg.evaluation.band:.0%}"
    text = [
        "# Test-year evaluation",
        "",
        f"Generated by `make evaluate`. Test period {d.test_start} to {d.test_end}: "
        f"{rows['target_day'].nunique()} days, {len(rows) // rows['model'].nunique()} "
        f"day-ahead product hours forecast by every model "
        f"({int(rows.loc[rows['model'] == rows['model'].iloc[0], 'curtailed'].sum())} "
        "curtailed). Scores in units of capacity; MAE and RMSE in % of capacity.",
        "",
        "## All hours",
        "",
        f"Skill = 1 − CRPS / CRPS({REFERENCE}); [lo, hi] = {band} day-block bootstrap band. "
        "`cover_x` / `width_x` = coverage and mean width of the central x interval.",
        "",
        _md(all_hours, fmt),
        "",
        "## Excluding curtailed hours (day-ahead price ≤ 0)",
        "",
        _md(clean[["hours", "crps", "skill", "mae_pct", "rmse_pct"]], fmt),
        "",
    ]
    if energinet is not None:
        text += [
            "## Point forecasts against Energinet's day-ahead forecast",
            "",
            f"On the {energinet.attrs['hours']} hours Energinet's forecast covers. It is "
            "issued at 18:00 on D−1 with newer weather data, seven hours after this "
            "project's issue time, and it excludes Thor: a benchmark with an edge. As "
            "published it runs low against settled production (it does not cover the same "
            "plants), so it is also shown after a linear recalibration fitted on the "
            "training period only.",
            "",
            _md(energinet, fmt),
            "",
        ]
    if pairs is not None:
        text += [
            "## Ensemble NWP (Step 8)",
            "",
            "Each `_ens` layer adds the spread of the ECMWF ENS members' power (51 members "
            "through the same RLS curve) as a covariate; `ens_raw` is the members' quantiles "
            "without post-processing, against the HRES curve with empirical errors. DM < 0: "
            "the ENS model is better.",
            "",
            _md(
                pairs,
                {
                    "against": "{}",
                    "crps_without": "{:.4f}",
                    "crps_with": "{:.4f}",
                    "change": "{:+.1%}",
                    "dm": "{:+.2f}",
                    "p": "{:.3f}",
                },
            ),
            "",
        ]
        if RAW_ENSEMBLE in pairs.index:
            text += [f"![Spread-skill]({FIGURES}/spread_skill.png)", ""]
    text += [
        "## Diebold–Mariano tests on daily mean CRPS",
        "",
        "Statistic of the row model minus the column model; negative means the row model "
        f"is better. |DM| > 1.96 is significant at 5 % (Newey–West variance, "
        f"{cfg.evaluation.dm_max_lag} lags).",
        "",
        _md(dm, {c: "{:+.1f}" for c in dm.columns}),
        "",
        "## Figures",
        "",
        f"![CRPS skill by horizon]({FIGURES}/skill_by_horizon.png)",
        "",
        f"![Reliability with {band} bootstrap bands]({FIGURES}/reliability.png)",
        "",
        f"![PIT histograms]({FIGURES}/pit.png)",
        "",
        f"![Sharpness]({FIGURES}/sharpness.png)",
        "",
    ]
    path = out / REPORT
    path.write_text("\n".join(text))
    log.info("evaluate: report -> %s", path)
    return path
