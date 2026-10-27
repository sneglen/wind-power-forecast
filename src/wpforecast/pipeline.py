"""Pipeline stages: data -> features -> backtest -> evaluate -> figures.

``data`` downloads, builds and checks the inputs (``wpforecast.data``). ``features`` builds
one row per issue time x target hour from the information set at issue time
(``wpforecast.features``). ``backtest`` runs every model through the event loop of
``wpforecast.backtest``; ``evaluate`` scores them on the same hours (``wpforecast.scoring``).

Conventions: timestamps are UTC, hour-beginning. Power is capacity-normalised to [0, 1].
Horizon ``k`` = hours from the issue time to the END of the target hour; the day-ahead
product (local day D) is k = 14..37. Scores cover the product hours only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from wpforecast import backtest, data, features, scoring
from wpforecast.backtest import Model
from wpforecast.config import Config
from wpforecast.data import checks, ens
from wpforecast.models.baselines import Climatology, NewReference, Persistence

log = logging.getLogger(__name__)

FORECASTS = "forecasts.parquet"
METRICS = "metrics.json"
SCORES = "scores.md"
RELIABILITY = "reliability.png"
SKILL_BY_HORIZON = "crps_skill_by_horizon.png"
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


def models(cfg: Config) -> list[tuple[Model, str]]:
    """Every model in the backtest, with its adaptivity regime."""
    levels = scoring.crps_levels(cfg)
    n = cfg.model.baselines.min_error_samples
    return [
        (Climatology(levels), "walk_forward"),
        (Persistence(levels, n), "walk_forward"),
        (NewReference(levels, n, cfg.issue.power_lag_h), "walk_forward"),
    ]


def run_backtest(cfg: Config) -> Path:
    """Forecast every target day from ``train_start`` with every model."""
    table = features.read(cfg)
    last = pd.to_datetime(table["target_day"]).dt.date.max()
    fc = pd.concat(
        [
            backtest.run(cfg, model, table, regime, cfg.dates.train_start, last)
            for model, regime in models(cfg)
        ],
        ignore_index=True,
    )
    out = cfg.paths.outputs / FORECASTS
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out, index=False)
    log.info(
        "backtest: %d models, %d target days, %d rows -> %s",
        fc["model"].nunique(),
        fc["target_day"].nunique(),
        len(fc),
        out,
    )
    return out


def _scored_rows(cfg: Config, fc: pd.DataFrame) -> pd.DataFrame:
    """Test-period product hours with an observation, forecast by every model."""
    fc = fc[(fc["period"] == "test") & fc["in_product"]].dropna(subset=["y"])
    n_models = fc["model"].nunique()
    per_hour = fc.groupby(["issue_time_utc", "time_utc"])["model"].transform("nunique")
    return fc[per_hour == n_models]


def run_evaluate(cfg: Config) -> Path:
    """Score the test period's product hours for every model, with and without curtailment."""
    fc = pd.read_parquet(_require(cfg.paths.outputs / FORECASTS, "Run `make backtest` first."))
    fc = _scored_rows(cfg, fc)
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
    return out


def score_table(metrics: dict) -> str:
    """Markdown summary of ``metrics.json``."""
    lines = [
        f"Test period {metrics['test_period'][0]} .. {metrics['test_period'][1]}, "
        f"{metrics['n_hours']} product hours ({metrics['n_curtailed']} curtailed). "
        "y = P / C; scores in units of capacity.",
        "",
        "| model | CRPS | CRPS skill vs climatology | mean pinball | MAE (median) "
        "| CRPS, excl. curtailed |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in metrics["models"].items():
        skill = r.get("crps_skill", float("nan"))
        lines.append(
            f"| {name} | {r['crps']:.4f} | {skill:+.1%} | {r['mean_pinball']:.4f} "
            f"| {r['mae_median']:.4f} | {r['excl_curtailed']['crps']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_figures(cfg: Config) -> Path:
    """Reliability diagram and CRPS skill by horizon, one line per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads(
        _require(cfg.paths.outputs / METRICS, "Run `make evaluate` first.").read_text()
    )
    out_dir = cfg.paths.figures
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="ideal")
    for name, r in metrics["models"].items():
        nominal = [float(x) for x in r["coverage"]]
        ax.plot(nominal, list(r["coverage"].values()), marker="o", ms=3, label=name)
    ax.set(xlabel="nominal level", ylabel="observed frequency below", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"Reliability, {metrics['n_hours']} h out-of-sample")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / RELIABILITY, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0, color="0.6", lw=1)
    for name, r in metrics["models"].items():
        if name == metrics["reference"] or "crps_skill_by_horizon" not in r:
            continue
        k = [int(x) for x in r["crps_skill_by_horizon"]]
        ax.plot(k, list(r["crps_skill_by_horizon"].values()), marker="o", ms=3, label=name)
    ax.set(xlabel="horizon k (h)", ylabel=f"CRPS skill vs {metrics['reference']}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / SKILL_BY_HORIZON, dpi=150)
    plt.close(fig)
    log.info("figures: -> %s", out_dir)
    return out_dir


STAGES = {
    "data": run_data,
    "check": run_check,
    "ens": run_ens,
    "features": run_features,
    "backtest": run_backtest,
    "evaluate": run_evaluate,
    "figures": run_figures,
}


def run_all(cfg: Config) -> None:
    for name in ("data", "features", "backtest", "evaluate", "figures"):  # data runs the check
        STAGES[name](cfg)
