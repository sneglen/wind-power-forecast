"""Reference forecasts that every later model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Climatology:
    """Empirical quantiles of the last ``window_days`` of observed y, the same for every hour."""

    name = "climatology"

    def __init__(self, levels: tuple[float, ...], window_days: int) -> None:
        self.levels = np.asarray(levels)
        self.window = pd.Timedelta(days=window_days)
        self.observed = pd.Series(dtype=float)

    def fit(self, known: pd.DataFrame) -> None:
        self.observed = pd.Series(dtype=float)
        self.update(known)

    def update(self, new: pd.DataFrame) -> None:
        # A target hour appears once per issue time that covers it; keep one value per hour.
        latest = new.drop_duplicates("time_utc", keep="last").set_index("time_utc")["y"]
        both = pd.concat([self.observed, latest])
        self.observed = both[~both.index.duplicated(keep="last")].sort_index()

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        issue = inputs["issue_time_utc"].iloc[0]
        history = self.observed[self.observed.index >= issue - self.window]
        if len(history) < 24:
            return np.full((len(inputs), len(self.levels)), np.nan)
        return np.tile(np.quantile(history.to_numpy(), self.levels), (len(inputs), 1))
