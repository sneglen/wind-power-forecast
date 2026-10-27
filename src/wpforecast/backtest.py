"""Backtest harness: an event loop of predict -> observe -> update over issue times.

At each issue time t0, in order:

1. **observe**: rows whose target hour is known at t0 (stamp + 1 h + power_lag_h <= t0) are
   released, except curtailed hours and missing values, which never reach a model;
2. **fit or update**, by adaptivity regime:
   - ``frozen``: ``fit`` once, at the first issue time, on everything known then;
   - ``walk_forward``: ``fit`` again at the first issue time of every month (local target
     day), on everything known then; the model chooses its own window;
   - ``recursive``: ``fit`` at the first issue time, then ``update`` with the rows released
     since the previous issue time (prequential: predict, then update);
3. **predict**: the model sees the input columns of t0's rows only, never the targets.

Models predict the quantiles of the CRPS grid (``wpforecast.scoring.crps_levels``), which
contains the reporting levels. A model with a ``point`` method also gives its point
forecast (column ``point``); otherwise the median is used. Quantile crossing is fixed by
sorting. A model returns NaN where it cannot forecast; those rows are dropped with a warning.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Protocol

import numpy as np
import pandas as pd

from wpforecast.config import REGIMES, Config
from wpforecast.features import HOUR, KEYS, TARGETS, input_columns
from wpforecast.scoring import crps_levels, qcol

log = logging.getLogger(__name__)


class Model(Protocol):
    name: str

    def fit(self, known: pd.DataFrame) -> None:
        """(Re-)estimate from every row released so far (inputs and targets)."""

    def update(self, new: pd.DataFrame) -> None:
        """Absorb rows released since the previous issue time."""

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Quantiles on the CRPS grid, shape (rows, levels), for the rows of one issue time."""


def trainable(cfg: Config, features: pd.DataFrame) -> pd.DataFrame:
    """Rows a model may learn from, sorted by the time their target becomes known."""
    rows = features[features["y"].notna() & ~features["curtailed"]]
    known_at = rows["time_utc"] + HOUR * (1 + cfg.issue.power_lag_h)
    return rows.assign(known_at=known_at).sort_values("known_at", kind="stable")


def run(
    cfg: Config,
    model: Model,
    features: pd.DataFrame,
    regime: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Forecast every target day in [start, end]; returns keys, targets and quantiles."""
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} not in {REGIMES}")
    days = pd.to_datetime(features["target_day"]).dt.date
    issues = features.loc[(days >= start) & (days <= end), "issue_time_utc"]
    issues = issues.drop_duplicates().sort_values()
    if len(issues) == 0:
        raise ValueError(f"no target days in [{start}, {end}]")

    learn = trainable(cfg, features)
    known_at = learn["known_at"].dt.tz_convert(None).to_numpy()  # naive UTC for searchsorted
    by_issue = features.groupby("issue_time_utc", sort=False)
    inputs = input_columns(features)
    levels = crps_levels(cfg)  # contains the reporting levels; see wpforecast.scoring
    columns = [qcol(x) for x in levels]
    median = qcol(0.5)

    frames, released, month = [], 0, None
    for n, issue in enumerate(issues):
        upto = int(np.searchsorted(known_at, issue.tz_convert(None).to_datetime64(), "right"))
        rows = by_issue.get_group(issue)
        new_month = rows["target_day"].iloc[0].replace(day=1)
        if n == 0 or (regime == "walk_forward" and new_month != month):
            model.fit(learn.iloc[:upto].drop(columns="known_at"))
        elif regime == "recursive" and upto > released:
            model.update(learn.iloc[released:upto].drop(columns="known_at"))
        released, month = upto, new_month

        q = np.asarray(model.predict(rows[inputs]), dtype=float)
        if q.shape != (len(rows), len(levels)):
            raise ValueError(f"{model.name}: predict returned {q.shape} for {len(rows)} rows")
        quantiles = pd.DataFrame(np.sort(q, axis=1), index=rows.index, columns=columns)
        point = model.point(rows[inputs]) if hasattr(model, "point") else quantiles[median]
        out = pd.concat([rows[KEYS + TARGETS], quantiles], axis=1)
        frames.append(out.assign(point=np.asarray(point, dtype=float)))

    fc = pd.concat(frames, ignore_index=True)
    missing = fc[columns].isna().any(axis=1)
    if missing.any():
        log.warning("backtest: %s has no forecast for %d rows", model.name, int(missing.sum()))
        fc = fc[~missing]
    return fc.assign(model=model.name, regime=regime)
