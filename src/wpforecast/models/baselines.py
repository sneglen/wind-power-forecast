"""Reference forecasts that every later model must beat.

- ``Climatology``: empirical quantiles of every y known so far, the same for every hour.
- ``Persistence``: the last known y, plus empirical quantiles of its past errors at the
  same horizon.
- ``NewReference`` (Nielsen et al., 1998): the horizon-wise optimal blend
  y(t+k|t) = a_k y(t) + (1 - a_k) mean(y), with a_k the lag-k autocorrelation of y, plus
  empirical error quantiles by horizon.

All are fitted on the rows released by the backtest harness (curtailed hours excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hourly(known: pd.DataFrame) -> pd.Series:
    """One observed y per target hour (a target hour appears once per issue that covers it)."""
    y = known.drop_duplicates("time_utc", keep="last").set_index("time_utc")["y"].sort_index()
    return y.asfreq("h") if len(y) else y


class Climatology:
    """Empirical quantiles of every y known at the last fit."""

    name = "climatology"

    def __init__(self, levels: tuple[float, ...]) -> None:
        self.levels = np.asarray(levels)
        self.q = np.full(len(levels), np.nan)
        self.known = pd.DataFrame()

    def fit(self, known: pd.DataFrame) -> None:
        self.known = known
        y = _hourly(known).dropna().to_numpy()
        self.q = np.quantile(y, self.levels) if len(y) else np.full(len(self.levels), np.nan)

    def update(self, new: pd.DataFrame) -> None:
        self.fit(pd.concat([self.known, new]))

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        return np.tile(self.q, (len(inputs), 1))


class _PointWithErrors:
    """A point forecast plus empirical quantiles of its past errors at the same horizon.

    Horizons with fewer than ``min_samples`` past errors borrow the nearest horizon that
    has enough (e.g. k = 38, which only occurs on the autumn DST day).
    """

    name = "point_with_errors"

    def __init__(self, levels: tuple[float, ...], min_samples: int) -> None:
        self.levels = np.asarray(levels)
        self.min_samples = min_samples
        self.errors = pd.DataFrame()
        self.known = pd.DataFrame()

    def _fit_point(self, known: pd.DataFrame) -> None:
        """Estimate the point forecast's parameters (none by default)."""

    def point(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit(self, known: pd.DataFrame) -> None:
        self.known = known
        self._fit_point(known)
        error = pd.Series(known["y"].to_numpy() - self.point(known), index=known.index)
        error = error.dropna()
        horizon = known.loc[error.index, "horizon_h"]
        enough = horizon.map(horizon.value_counts()) >= self.min_samples
        error, horizon = error[enough], horizon[enough]
        if error.empty:
            self.errors = pd.DataFrame(columns=range(len(self.levels)), dtype=float)
            return
        by_k = {k: np.quantile(e.to_numpy(), self.levels) for k, e in error.groupby(horizon)}
        self.errors = pd.DataFrame.from_dict(by_k, orient="index").sort_index()

    def update(self, new: pd.DataFrame) -> None:
        self.fit(pd.concat([self.known, new]))

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        if self.errors.empty:
            return np.full((len(inputs), len(self.levels)), np.nan)
        k = inputs["horizon_h"].to_numpy()
        nearest = self.errors.index.to_numpy()[
            np.abs(self.errors.index.to_numpy()[None, :] - k[:, None]).argmin(axis=1)
        ]
        q = self.point(inputs)[:, None] + self.errors.loc[nearest].to_numpy()
        return np.clip(q, 0.0, 1.0)


class Persistence(_PointWithErrors):
    """The value at issue time, y(t + k | t) = y(t)."""

    name = "persistence"

    def point(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["y_issue"].to_numpy(dtype=float)


class NewReference(_PointWithErrors):
    """Nielsen et al. (1998): a_k y(t) + (1 - a_k) mean(y), a_k the lag-k autocorrelation.

    y(t) is the last known hour, stamped ``power_lag_h`` + 1 h before the issue time, so the
    lag to the target hour at horizon k is k + power_lag_h hours.
    """

    name = "new_reference"

    def __init__(self, levels: tuple[float, ...], min_samples: int, power_lag_h: int) -> None:
        super().__init__(levels, min_samples)
        self.power_lag_h = power_lag_h
        self.mean = np.nan
        self.a = pd.Series(dtype=float)

    def _fit_point(self, known: pd.DataFrame) -> None:
        y = _hourly(known)
        self.mean = float(y.mean())
        lags = np.unique(known["horizon_h"].to_numpy()) + self.power_lag_h
        self.a = pd.Series({lag: y.autocorr(lag=int(lag)) for lag in lags}).fillna(0.0)

    def point(self, rows: pd.DataFrame) -> np.ndarray:
        lag = rows["horizon_h"].to_numpy() + self.power_lag_h
        a = self.a.reindex(lag).fillna(0.0).to_numpy()  # unseen lag: fall back to the mean
        return a * rows["y_issue"].to_numpy(dtype=float) + (1 - a) * self.mean
