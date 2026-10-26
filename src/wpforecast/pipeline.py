"""Pipeline stages: data -> backtest -> evaluate -> figures.

Step 0 wires the stages end to end with a climatological quantile benchmark. Later steps
replace the internals (real DK1 data, adaptive models, market backtest) behind the same
stage interfaces.

Conventions: timestamps are UTC, hour-beginning. Power is capacity-normalised to [0, 1].
An observation stamped ``t`` covers ``[t, t + 1h)`` and becomes usable at
``t + 1h + power_lag_h``; it may enter a forecast only if that instant is <= the issue time.
Horizon ``k`` = hours from the issue time to the END of the target hour, so k = 1 is the
first hour after issue and the day-ahead product (local day D) is k = 14..37.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast.config import Config

log = logging.getLogger(__name__)

DATASET = "dataset.parquet"
FORECASTS = "forecasts.parquet"
METRICS = "metrics.json"
RELIABILITY = "reliability.png"


def qcol(level: float) -> str:
    """Column name holding the forecast of quantile ``level``."""
    return f"q{level:g}"


def issue_time_utc(cfg: Config, target_day: dt.date) -> pd.Timestamp:
    """Instant at which the forecast for local ``target_day`` is issued (day before, local)."""
    local = dt.datetime.combine(target_day - dt.timedelta(days=1), cfg.issue.time_local)
    return pd.Timestamp(local, tz=cfg.issue.tz).tz_convert("UTC")


def target_hours_utc(cfg: Config, target_day: dt.date) -> pd.DatetimeIndex:
    """Hour-beginning UTC stamps of local ``target_day`` (23, 24 or 25 hours with DST)."""
    start = pd.Timestamp(target_day, tz=cfg.issue.tz)
    end = pd.Timestamp(target_day + dt.timedelta(days=1), tz=cfg.issue.tz)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def horizons(issue: pd.Timestamp, targets: pd.DatetimeIndex) -> np.ndarray:
    """Horizon k (hours, to the end of each hour-beginning target stamp)."""
    return ((targets - issue) / pd.Timedelta(hours=1)).to_numpy() + 1


def period(cfg: Config, day: dt.date) -> str:
    """Which evaluation period a target day belongs to."""
    d = cfg.dates
    if day < d.validation_start:
        return "train"
    if day <= d.train_end:
        return "validation"
    if day <= d.test_end:
        return "test"
    return "stress"


def _last_day(cfg: Config, index: pd.DatetimeIndex) -> dt.date:
    """Last target day: stress_end if set, else the last local day fully covered by data."""
    if cfg.dates.stress_end is not None:
        return cfg.dates.stress_end
    day = index[-1].tz_convert(cfg.issue.tz).date()
    return day if target_hours_utc(cfg, day)[-1] <= index[-1] else day - dt.timedelta(days=1)


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path


def run_data(cfg: Config) -> Path:
    """Load raw hourly data, validate its schema and write the interim dataset."""
    raw = _require(
        cfg.paths.raw,
        "Real DK1 data ingestion arrives in Step 1; use `make smoke` for the synthetic sample.",
    )
    df = pd.read_csv(raw)

    required = ["time_utc", "power_mw", "capacity_mw"] + [f"wind_speed_{p}" for p in cfg.points]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{raw} is missing columns {missing}")

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.set_index("time_utc").sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{raw} has duplicate timestamps")
    if len(df) > 1 and not (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError(f"{raw} is not a regular hourly series")
    if (df["capacity_mw"] <= 0).any():
        raise ValueError(f"{raw} has non-positive capacity")

    df["power_norm"] = (df["power_mw"] / df["capacity_mw"]).clip(0.0, 1.0)

    lookback = dt.timedelta(days=cfg.model.climatology.window_days + 2)
    lo = issue_time_utc(cfg, cfg.dates.train_start) - lookback
    hi = target_hours_utc(cfg, cfg.dates.stress_end)[-1] if cfg.dates.stress_end else None
    df = df.loc[lo:hi, required[1:] + ["power_norm"]]

    out = cfg.paths.interim / DATASET
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    log.info("data: %d rows %s .. %s -> %s", len(df), df.index[0], df.index[-1], out)
    return out


def run_backtest(cfg: Config) -> Path:
    """Issue one day-ahead forecast per target day using only information at issue time.

    Step 0 benchmark: empirical quantiles of the last ``window_days`` of observed
    normalised power (climatology), identical for every hour of the target day.
    """
    df = pd.read_parquet(_require(cfg.paths.interim / DATASET, "Run `make data` first."))
    power = df["power_norm"]
    usable_at = power.index + pd.Timedelta(hours=1 + cfg.issue.power_lag_h)
    window = pd.Timedelta(days=cfg.model.climatology.window_days)
    levels = np.asarray(cfg.quantiles.levels)

    frames = []
    last = _last_day(cfg, power.index)
    for day in pd.date_range(cfg.dates.train_start, last, freq="D").date:
        issue = issue_time_utc(cfg, day)
        targets = target_hours_utc(cfg, day)
        k = horizons(issue, targets)
        if k.max() > cfg.issue.max_horizon_h:
            raise ValueError(f"target day {day} needs k = {k.max():g} > max_horizon_h")
        history = power[(usable_at <= issue) & (power.index >= issue - window)].dropna()
        if len(history) < 24:
            log.warning("backtest: skipping %s, only %d usable hours", day, len(history))
            continue
        if not history.index[-1] < issue < targets[0]:
            raise RuntimeError(f"issue-time discipline violated for target day {day}")

        q = np.quantile(history.to_numpy(), levels)
        frame = pd.DataFrame(
            {
                "target_day": day,
                "period": period(cfg, day),
                "issue_time_utc": issue,
                "time_utc": targets,
                "horizon_h": k,
                "observed": power.reindex(targets).to_numpy(),
            }
        )
        for level, value in zip(levels, q, strict=True):
            frame[qcol(level)] = value
        frames.append(frame)

    if not frames:
        raise ValueError("backtest produced no forecasts; check [dates] against the data range")
    fc = pd.concat(frames, ignore_index=True)

    out = cfg.paths.outputs / FORECASTS
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out)
    log.info("backtest: %d target days, %d rows -> %s", len(frames), len(fc), out)
    return out


def pinball(y: np.ndarray, f: np.ndarray, level: float) -> float:
    """Mean pinball (quantile) loss of forecast ``f`` for quantile ``level``."""
    diff = y - f
    return float(np.mean(np.maximum(level * diff, (level - 1) * diff)))


def run_evaluate(cfg: Config) -> Path:
    """Score the test period: pinball loss and empirical coverage per quantile."""
    fc = pd.read_parquet(_require(cfg.paths.outputs / FORECASTS, "Run `make backtest` first."))
    fc = fc[fc["period"] == "test"]
    fc = fc.dropna(subset=["observed"])
    if fc.empty:
        raise ValueError("no out-of-sample forecasts with observations to evaluate")

    y = fc["observed"].to_numpy()
    per_level = [
        {
            "level": level,
            "pinball": pinball(y, fc[qcol(level)].to_numpy(), level),
            "coverage": float(np.mean(y <= fc[qcol(level)].to_numpy())),
        }
        for level in cfg.quantiles.levels
    ]
    metrics = {
        "model": "climatology",
        "test_period": [cfg.dates.test_start.isoformat(), cfg.dates.test_end.isoformat()],
        "n_hours": len(fc),
        "mean_pinball": float(np.mean([m["pinball"] for m in per_level])),
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
    "backtest": run_backtest,
    "evaluate": run_evaluate,
    "figures": run_figures,
}


def run_all(cfg: Config) -> None:
    for stage in STAGES.values():
        stage(cfg)
