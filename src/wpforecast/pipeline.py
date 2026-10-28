"""Pipeline stages: data -> features -> backtest -> evaluate -> market -> figures.

``data`` downloads, builds and checks the inputs (``wpforecast.data``). ``features`` builds
one row per issue time x target hour from the information set at issue time
(``wpforecast.features``). ``backtest`` runs every model through the event loop of
``wpforecast.backtest``; ``evaluate`` scores them on the same hours (``wpforecast.scoring``);
``market`` prices their bids in the imbalance market (``wpforecast.market``).

Conventions: timestamps are UTC, hour-beginning. Power is capacity-normalised to [0, 1].
Horizon ``k`` = hours from the issue time to the END of the target hour; the day-ahead
product (local day D) is k = 14..37. Scores cover the product hours only.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast import backtest, data, evaluation, features, market, scoring
from wpforecast.backtest import Model
from wpforecast.config import Config
from wpforecast.data import checks, ens
from wpforecast.models.baselines import Climatology, NewReference, Persistence
from wpforecast.models.power_curve import REGIONS, PowerCurve
from wpforecast.models.probabilistic import GradientBoosting, LogitNormal, QuantileRegression

log = logging.getLogger(__name__)

FORECASTS = "forecasts.parquet"
METRICS = "metrics.json"
SCORES = "scores.md"
TUNING = "tuning.json"
CURVE_PARAMS = "power_curve_params.parquet"
CURVE_BY_SECTOR = "power_curve_by_sector.png"
CURVE_TRAJECTORY = "power_curve_trajectory.png"
FAN_CHART = "fan_chart_stormy_week.png"
PROBABILISTIC = ("quantreg", "logitnormal", "lightgbm")
BASELINES = ("climatology", "persistence", "new_reference")
REFERENCE = "climatology"


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path


def run_data(cfg: Config) -> Path:
    """Download what is missing, build the interim tables and write the data-check report."""
    data.acquire(cfg)
    return cfg.paths.interim


def run_check(cfg: Config) -> Path:
    """Re-run the data checks on the existing interim tables."""
    checks.run(cfg)
    return cfg.paths.reports / checks.REPORT


def run_ens(cfg: Config) -> Path:
    """Download ECMWF ENS point values (Extension A); separate from `data`, not in `all`."""
    ens.download(cfg)
    return cfg.paths.raw / "ens"


def run_features(cfg: Config) -> Path:
    """One row per issue time x target hour, inputs restricted to the issue-time information."""
    return features.run(cfg)


def power_curve(cfg: Config, name: str, forgetting: float) -> PowerCurve:
    return PowerCurve(
        name,
        cfg.model.rls,
        forgetting,
        scoring.crps_levels(cfg),
        cfg.model.baselines.min_error_samples,
    )


def tune_forgetting(cfg: Config, table: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """RLS forgetting factor with the lowest point RMSE on the validation product hours."""
    rmse = {}
    for lam in cfg.model.rls.forgetting_grid:
        fc = backtest.run(
            cfg,
            power_curve(cfg, "tuning", lam),
            table,
            "recursive",
            cfg.dates.train_start,
            cfg.dates.train_end,
        )
        val = fc[(fc["period"] == "validation") & fc["in_product"]].dropna(subset=["y"])
        rmse[f"{lam:g}"] = float(np.sqrt(((val["y"] - val["point"]) ** 2).mean()))
    best = min(rmse, key=rmse.get)
    log.info("tune: RLS forgetting %s (validation RMSE %s)", best, rmse)
    return float(best), rmse


def models(cfg: Config, forgetting: float) -> list[tuple[Model, str, dt.date]]:
    """Every model in the backtest, with its adaptivity regime and first target day."""
    levels = scoring.crps_levels(cfg)
    m = cfg.model
    n = m.baselines.min_error_samples
    first = cfg.dates.train_start
    layer = (scoring.fit_levels(cfg), levels, m.probabilistic.min_train_rows)
    k_max = cfg.issue.max_horizon_h
    return [
        (Climatology(levels), "walk_forward", first),
        (Persistence(levels, n), "walk_forward", first),
        (NewReference(levels, n, cfg.issue.power_lag_h), "walk_forward", first),
        (power_curve(cfg, "curve_frozen", 1.0), "frozen", cfg.dates.test_start),
        (power_curve(cfg, "curve_walk_forward", 1.0), "walk_forward", first),
        (power_curve(cfg, "curve_rls", forgetting), "recursive", first),
        # Step 5: distributions on the RLS curve; each layer keeps its own refit schedule.
        (
            QuantileRegression(
                "quantreg", power_curve(cfg, "quantreg", forgetting), *layer, m.quantreg, k_max
            ),
            "recursive",
            first,
        ),
        (
            LogitNormal(
                "logitnormal",
                power_curve(cfg, "logitnormal", forgetting),
                *layer,
                m.logitnormal,
                k_max,
            ),
            "recursive",
            first,
        ),
        (
            GradientBoosting("lightgbm", power_curve(cfg, "lightgbm", forgetting), *layer, m.gbm),
            "recursive",
            first,
        ),
    ]


def run_backtest(cfg: Config) -> Path:
    """Tune the RLS forgetting factor, then forecast every target day with every model."""
    table = features.read(cfg)
    last = pd.to_datetime(table["target_day"]).dt.date.max()
    forgetting, rmse = tune_forgetting(cfg, table)
    out_dir = cfg.paths.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / TUNING).write_text(
        json.dumps({"rls_forgetting": forgetting, "validation_rmse": rmse}, indent=2) + "\n"
    )

    frames, params = [], []
    for model, regime, first in models(cfg, forgetting):
        frames.append(backtest.run(cfg, model, table, regime, first, last))
        if isinstance(model, PowerCurve):
            params.append(_trajectory(model))
    fc = pd.concat(frames, ignore_index=True)
    fc.to_parquet(out_dir / FORECASTS, index=False)
    pd.concat(params, ignore_index=True).to_parquet(out_dir / CURVE_PARAMS, index=False)
    log.info(
        "backtest: %d models, %d target days, %d rows -> %s",
        fc["model"].nunique(),
        fc["target_day"].nunique(),
        len(fc),
        out_dir / FORECASTS,
    )
    return out_dir / FORECASTS


def _trajectory(model: PowerCurve) -> pd.DataFrame:
    """beta after every update, one row per update."""
    times, betas = zip(*model.history, strict=True)
    frame = pd.DataFrame(np.vstack(betas), columns=[f"b{i}" for i in range(len(betas[0]))])
    return frame.assign(model=model.name, time_utc=list(times))


def run_evaluate(cfg: Config) -> Path:
    """Score the test year for every model: metrics, a score table and the evaluation report."""
    fc = pd.read_parquet(_require(cfg.paths.outputs / FORECASTS, "Run `make backtest` first."))
    fc = evaluation.scored_rows(fc)
    if fc.empty:
        raise ValueError("no out-of-sample forecasts with observations to evaluate")

    results = {}
    for name, group in fc.groupby("model", sort=False):
        result = scoring.scores(cfg, group)
        clean = scoring.scores(cfg, group[~group["curtailed"]])
        result["excl_curtailed"] = {k: clean[k] for k in ("n_hours", "crps", "mean_pinball")}
        by_k = scoring.crps_by(cfg, group, "horizon_h")
        result["crps_by_horizon"] = {str(k): float(v) for k, v in by_k.items()}
        results[name] = result
    if REFERENCE in results:
        ref = results[REFERENCE]
        for result in results.values():
            result["crps_skill"] = scoring.skill(result["crps"], ref["crps"])
            result["crps_skill_by_horizon"] = {
                k: scoring.skill(v, ref["crps_by_horizon"][k])
                for k, v in result["crps_by_horizon"].items()
            }

    metrics = {
        "test_period": [cfg.dates.test_start.isoformat(), cfg.dates.test_end.isoformat()],
        "n_hours": int(fc.groupby("model").size().iloc[0]),
        "n_curtailed": int(fc.loc[fc["model"] == fc["model"].iloc[0], "curtailed"].sum()),
        "reference": REFERENCE,
        "models": results,
    }
    out = cfg.paths.outputs / METRICS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    (cfg.paths.outputs / SCORES).write_text(score_table(metrics))
    for name, result in results.items():
        log.info(
            "evaluate: %-14s CRPS %.4f over %d hours", name, result["crps"], len(fc) // len(results)
        )
    evaluation.run(cfg, fc)
    return out


def run_market(cfg: Config) -> Path:
    """Imbalance cost of day-ahead bids under DK1 single pricing (Step 7)."""
    fc = pd.read_parquet(_require(cfg.paths.outputs / FORECASTS, "Run `make backtest` first."))
    return market.run(cfg, fc, ("curve_rls", *PROBABILISTIC), BASELINES)


def score_table(metrics: dict) -> str:
    """Markdown summary of ``metrics.json``."""
    lines = [
        f"Test period {metrics['test_period'][0]} .. {metrics['test_period'][1]}, "
        f"{metrics['n_hours']} product hours ({metrics['n_curtailed']} curtailed). "
        "y = P / C; scores in units of capacity.",
        "",
        "| model | CRPS | CRPS skill vs climatology | mean pinball | RMSE (point) "
        "| MAE (point) | CRPS, excl. curtailed |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in metrics["models"].items():
        skill = r.get("crps_skill", float("nan"))
        lines.append(
            f"| {name} | {r['crps']:.4f} | {skill:+.1%} | {r['mean_pinball']:.4f} "
            f"| {r['rmse_point']:.4f} | {r['mae_point']:.4f} "
            f"| {r['excl_curtailed']['crps']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_figures(cfg: Config) -> Path:
    """Model figures: the power curve by direction and over time, and a fan chart.

    The verification figures (skill, reliability, PIT, sharpness) come with the evaluation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _require(cfg.paths.outputs / FORECASTS, "Run `make backtest` first.")
    out_dir = cfg.paths.figures
    out_dir.mkdir(parents=True, exist_ok=True)

    params = cfg.paths.outputs / CURVE_PARAMS
    if params.exists():
        _curve_figures(cfg, pd.read_parquet(params), out_dir, plt)
    _fan_chart(cfg, pd.read_parquet(cfg.paths.outputs / FORECASTS), out_dir, plt)
    log.info("figures: -> %s", out_dir)
    return out_dir


def stormiest_week(fc: pd.DataFrame) -> list:
    """The seven consecutive test days with the highest mean observed y (product hours)."""
    test = fc[(fc["period"] == "test") & fc["in_product"]].drop_duplicates("time_utc")
    daily = test.groupby("target_day")["y"].mean()
    if len(daily) < 7:
        return list(daily.index)
    end = daily.rolling(7).mean().idxmax()
    position = list(daily.index).index(end)
    return list(daily.index[position - 6 : position + 1])


def _fan_chart(cfg: Config, fc: pd.DataFrame, out_dir: Path, plt) -> None:
    names = [n for n in PROBABILISTIC if n in set(fc["model"])]
    if not names:
        return
    week = stormiest_week(fc)
    rows = fc[fc["target_day"].isin(week) & fc["in_product"]]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.4 * len(names)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        f = rows[rows["model"] == name].sort_values("time_utc")
        t = f["time_utc"]
        for lo, hi, alpha in ((0.05, 0.95, 0.25), (0.25, 0.75, 0.45)):
            ax.fill_between(
                t,
                f[scoring.qcol(lo)],
                f[scoring.qcol(hi)],
                color="C0",
                alpha=alpha,
                lw=0,
                label=f"{lo:.0%}–{hi:.0%}",
            )
        ax.plot(t, f[scoring.qcol(0.5)], color="C0", lw=1, label="median")
        ax.plot(t, f["y"], color="k", lw=0.8, label="observed")
        ax.set(ylim=(0, 1), ylabel="y = P / C", title=name)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7, ncol=4, frameon=False)
    fig.suptitle(f"Day-ahead forecasts, stormiest test week from {week[0]}")
    fig.tight_layout()
    fig.savefig(out_dir / FAN_CHART, dpi=150)
    plt.close(fig)


SECTORS = np.arange(0, 360, 45)


def _curve(
    cfg: Config, beta: np.ndarray, onshore: np.ndarray, ratio: float, direction: float
) -> np.ndarray:
    """The curve along onshore speed v, offshore speed ratio * v, diurnal terms at mean 0."""
    basis = power_curve(cfg, "figure", 1.0).basis
    n = len(onshore)
    phi = basis.matrix(
        {"onshore": onshore, "offshore": ratio * onshore},
        {r: np.full(n, direction) for r in REGIONS},
        np.zeros(n),
    )
    phi[:, basis.size - 2 * basis.harmonics :] = 0.0
    return phi @ beta


def _speed_path(cfg: Config) -> tuple[np.ndarray, float]:
    """Onshore speeds up to the 99.9 % quantile seen, and the median offshore/onshore ratio."""
    on, off = (f"wind_speed_100m@{r}" for r in REGIONS)
    table = pd.read_parquet(cfg.paths.interim / features.FEATURES, columns=[on, off]).dropna()
    table = table[table[on] > 1.0]
    ratio = float((table[off] / table[on]).median())
    return np.linspace(0, table[on].quantile(0.999), 101), ratio


def _curve_figures(cfg: Config, params: pd.DataFrame, out_dir: Path, plt) -> None:
    beta_cols = [c for c in params.columns if c.startswith("b")]
    speed, ratio = _speed_path(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, name in zip(axes, ("curve_frozen", "curve_rls"), strict=True):
        rows = params[params["model"] == name]
        if rows.empty:
            continue
        beta = rows[beta_cols].iloc[-1].to_numpy()
        for sector in SECTORS:
            ax.plot(speed, _curve(cfg, beta, speed, ratio, sector), lw=1.2, label=f"{sector}°")
        when = pd.Timestamp(rows["time_utc"].iloc[-1]).date()
        ax.set(
            xlabel=f"onshore 100 m speed (m/s), offshore = {ratio:.2f} x", title=f"{name}, {when}"
        )
    axes[0].set_ylabel("y = P / C")
    axes[1].legend(title="from", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / CURVE_BY_SECTOR, dpi=150)
    plt.close(fig)

    rows = params[params["model"] == "curve_rls"]
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for v in (6.0, 9.0, 12.0):
        level = [
            np.mean([_curve(cfg, b, np.array([v]), ratio, s)[0] for s in SECTORS])
            for b in rows[beta_cols].to_numpy()
        ]
        ax.plot(pd.to_datetime(rows["time_utc"]), level, lw=1, label=f"{v:g} m/s")
    ax.set(ylabel="curve, mean over directions", title="RLS power curve over time")
    ax.legend(title="wind speed", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / CURVE_TRAJECTORY, dpi=150)
    plt.close(fig)


STAGES = {
    "data": run_data,
    "check": run_check,
    "ens": run_ens,
    "features": run_features,
    "backtest": run_backtest,
    "evaluate": run_evaluate,
    "market": run_market,
    "figures": run_figures,
}


def run_all(cfg: Config) -> None:
    for name in ("data", "features", "backtest", "evaluate", "market", "figures"):  # data checks
        STAGES[name](cfg)
