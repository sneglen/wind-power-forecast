"""Pipeline stages: data -> features -> backtest -> evaluate -> figures.

``data`` downloads, builds and checks the inputs (``wpforecast.data``). ``features`` builds
one row per issue time x target hour from the information set at issue time
(``wpforecast.features``). ``backtest`` runs a model through the event loop of
``wpforecast.backtest``; it is still the climatology benchmark, and later steps add models
behind the same interfaces.

Conventions: timestamps are UTC, hour-beginning. Power is capacity-normalised to [0, 1].
Horizon ``k`` = hours from the issue time to the END of the target hour; the day-ahead
product (local day D) is k = 14..37. Scores cover the product hours only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast import backtest, data, features
from wpforecast.backtest import qcol
from wpforecast.config import Config
from wpforecast.data import checks
from wpforecast.models.baselines import Climatology

log = logging.getLogger(__name__)

FORECASTS = "forecasts.parquet"
METRICS = "metrics.json"
RELIABILITY = "reliability.png"


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


def run_features(cfg: Config) -> Path:
    """One row per issue time x target hour, inputs restricted to the issue-time information."""
    return features.run(cfg)


def run_backtest(cfg: Config) -> Path:
    """Climatology benchmark, updated daily, for every target day from ``train_start``."""
    table = features.read(cfg)
    last = pd.to_datetime(table["target_day"]).dt.date.max()
    model = Climatology(cfg.quantiles.levels, cfg.model.climatology.window_days)
    fc = backtest.run(cfg, model, table, "recursive", cfg.dates.train_start, last)

    out = cfg.paths.outputs / FORECASTS
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out, index=False)
    log.info("backtest: %d target days, %d rows -> %s", fc["target_day"].nunique(), len(fc), out)
    return out


def pinball(y: np.ndarray, f: np.ndarray, level: float) -> float:
    """Mean pinball (quantile) loss of forecast ``f`` for quantile ``level``."""
    diff = y - f
    return float(np.mean(np.maximum(level * diff, (level - 1) * diff)))


def _scores(cfg: Config, fc: pd.DataFrame) -> list[dict]:
    y = fc["y"].to_numpy()
    return [
        {
            "level": level,
            "pinball": pinball(y, fc[qcol(level)].to_numpy(), level),
            "coverage": float(np.mean(y <= fc[qcol(level)].to_numpy())),
        }
        for level in cfg.quantiles.levels
    ]


def run_evaluate(cfg: Config) -> Path:
    """Score the day-ahead product hours of the test period, with and without curtailment."""
    fc = pd.read_parquet(_require(cfg.paths.outputs / FORECASTS, "Run `make backtest` first."))
    fc = fc[(fc["period"] == "test") & fc["in_product"]].dropna(subset=["y"])
    if fc.empty:
        raise ValueError("no out-of-sample forecasts with observations to evaluate")

    per_level = _scores(cfg, fc)
    clean = _scores(cfg, fc[~fc["curtailed"]])
    metrics = {
        "model": str(fc["model"].iloc[0]),
        "regime": str(fc["regime"].iloc[0]),
        "test_period": [cfg.dates.test_start.isoformat(), cfg.dates.test_end.isoformat()],
        "n_hours": len(fc),
        "n_curtailed": int(fc["curtailed"].sum()),
        "mean_pinball": float(np.mean([m["pinball"] for m in per_level])),
        "mean_pinball_excl_curtailed": float(np.mean([m["pinball"] for m in clean])),
        "quantiles": per_level,
    }

    out = cfg.paths.outputs / METRICS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    log.info(
        "evaluate: mean pinball %.4f over %d hours -> %s", metrics["mean_pinball"], len(fc), out
    )
    return out


def run_figures(cfg: Config) -> Path:
    """Reliability diagram: nominal vs empirical quantile coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads(
        _require(cfg.paths.outputs / METRICS, "Run `make evaluate` first.").read_text()
    )
    nominal = [m["level"] for m in metrics["quantiles"]]
    empirical = [m["coverage"] for m in metrics["quantiles"]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="ideal")
    ax.plot(nominal, empirical, marker="o", ms=4, label=metrics["model"])
    ax.set(xlabel="nominal level", ylabel="empirical coverage", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"Reliability, {metrics['n_hours']} h out-of-sample")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out = cfg.paths.figures / RELIABILITY
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("figures: -> %s", out)
    return out


STAGES = {
    "data": run_data,
    "check": run_check,
    "features": run_features,
    "backtest": run_backtest,
    "evaluate": run_evaluate,
    "figures": run_figures,
}


def run_all(cfg: Config) -> None:
    for name in ("data", "features", "backtest", "evaluate", "figures"):  # data runs the check
        STAGES[name](cfg)
