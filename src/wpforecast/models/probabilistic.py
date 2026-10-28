"""Predictive distributions on top of the RLS power curve's point forecast p (Step 5).

Each layer owns a power curve, updated daily, and learns the distribution of y given p, the
horizon k and (for the gradient-boosted model) the NWP inputs:

- ``QuantileRegression`` (Nielsen et al., 2006): q_tau = [B(p), B(p) k/K]' gamma_tau, linear
  quantile regression on a B-spline basis in p; refit weekly on a rolling window. A
  pragmatic stand-in for time-adaptive quantile regression (Moller et al., 2008).
- ``LogitNormal`` (Pinson, 2012): the generalised logit z = log(y^nu / (1 - y^nu)) is normal,
  mu and log sigma linear in z(p), z(p)^2 and k, fitted by maximum likelihood with the
  bounds as censored observations; refit monthly on all known data.
- ``GradientBoosting``: LightGBM quantile regression, one model per level, on p, k, calendar
  terms and NWP features; refit monthly. (LightGBM rejects monotone constraints with the
  quantile objective, so the plan's monotonicity in wind speed is not imposed.)

The training pairs use p as it was issued (or, before the first forecast, the curve's
prequential point), never a p fitted with hindsight. The quantile models are fitted at the
reporting and tail levels and interpolated in tau onto the CRPS grid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import BSpline

from wpforecast.config import GBM, QuantReg
from wpforecast.config import LogitNormal as LogitNormalConfig
from wpforecast.models.power_curve import PowerCurve
from wpforecast.scoring import to_grid

DAY = pd.Timedelta(days=1)


class _Layer:
    """Shared plumbing: the curve, the training pairs, the refit schedule, the grid."""

    name = "layer"
    columns: tuple[str, ...] = ()  # input columns kept for training besides p and k

    def __init__(
        self,
        name: str,
        curve: PowerCurve,
        fit_levels: tuple[float, ...],
        grid: tuple[float, ...],
        min_train_rows: int,
    ) -> None:
        self.name = name
        self.curve = curve
        self.fit_levels = np.asarray(fit_levels)
        self.grid = np.asarray(grid)
        self.min_train_rows = min_train_rows
        self.train = pd.DataFrame()
        self.issued = pd.Series(dtype=float)
        self.fitted_at: pd.Timestamp | None = None

    # --- schedule and data ----------------------------------------------------------------

    def _due(self, now: pd.Timestamp) -> bool:
        raise NotImplementedError

    def _window(self, train: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return train

    def _pairs(self, rows: pd.DataFrame) -> pd.DataFrame:
        p = self.issued.reindex(rows.index).fillna(self.curve.prequential.reindex(rows.index))
        keep = rows[["time_utc", "horizon_h", "y", *self.columns]].assign(p_hat=p)
        return keep.dropna(subset=["p_hat", "y"])

    def _refit(self, now: pd.Timestamp) -> None:
        data = self._window(self.train, now)
        if len(data) >= self.min_train_rows:
            self._fit(data)
            self.fitted_at = now

    # --- the backtest Model protocol ------------------------------------------------------

    def fit(self, known: pd.DataFrame) -> None:
        self.curve.fit(known)
        self.train = self._pairs(known)
        self.fitted_at = None
        if len(known):
            self._refit(known["time_utc"].max())

    def update(self, new: pd.DataFrame) -> None:
        self.curve.update(new)
        self.train = pd.concat([self.train, self._pairs(new)])
        now = new["time_utc"].max()
        if self.fitted_at is None or self._due(now):
            self._refit(now)

    def point(self, inputs: pd.DataFrame) -> np.ndarray:
        return self.curve.point(inputs)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        p = self.curve.point(inputs)
        self.issued = pd.concat([self.issued, pd.Series(p, index=inputs.index)])
        if self.fitted_at is None:
            return np.full((len(inputs), len(self.grid)), np.nan)
        q = self._quantiles(inputs.assign(p_hat=p))
        return np.clip(to_grid(q, self.fit_levels, self.grid), 0.0, 1.0)

    # --- model-specific -------------------------------------------------------------------

    def _fit(self, data: pd.DataFrame) -> None:
        raise NotImplementedError

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        """Quantiles at ``fit_levels``, shape (rows, levels)."""
        raise NotImplementedError


def _monthly(fitted_at: pd.Timestamp | None, now: pd.Timestamp) -> bool:
    return fitted_at is None or (now.year, now.month) != (fitted_at.year, fitted_at.month)


class QuantileRegression(_Layer):
    def __init__(self, name, curve, fit_levels, grid, min_train_rows, params: QuantReg, max_k):
        super().__init__(name, curve, fit_levels, grid, min_train_rows)
        self.params, self.max_k = params, max_k
        inner = np.linspace(0, 1, params.spline_knots + 2)[1:-1]
        ends = np.ones(params.spline_degree + 1)
        self.t = np.r_[0 * ends, inner, ends]
        self.gamma = np.empty(0)

    def _due(self, now: pd.Timestamp) -> bool:
        return now - self.fitted_at >= self.params.refit_days * DAY

    def _window(self, train: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return train[train["time_utc"] > now - self.params.window_days * DAY]

    def design(self, p: np.ndarray, k: np.ndarray) -> np.ndarray:
        b = BSpline.design_matrix(np.clip(p, 0, 1 - 1e-9), self.t, self.params.spline_degree)
        b = b.toarray()
        return np.hstack([b, b * (np.asarray(k, dtype=float) / self.max_k)[:, None]])

    def _fit(self, data: pd.DataFrame) -> None:
        import statsmodels.api as sm

        X = self.design(data["p_hat"].to_numpy(), data["horizon_h"].to_numpy())
        y = data["y"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # iteration-limit warnings on flat tails
            self.gamma = np.column_stack(
                [sm.QuantReg(y, X).fit(q=tau, max_iter=2000).params for tau in self.fit_levels]
            )

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        return self.design(rows["p_hat"].to_numpy(), rows["horizon_h"].to_numpy()) @ self.gamma


class LogitNormal(_Layer):
    def __init__(
        self, name, curve, fit_levels, grid, min_train_rows, params: LogitNormalConfig, max_k
    ):
        super().__init__(name, curve, fit_levels, grid, min_train_rows)
        self.nu, self.eps, self.max_k = params.nu, params.epsilon, max_k
        self.theta = np.zeros(7)

    def _due(self, now: pd.Timestamp) -> bool:
        return _monthly(self.fitted_at, now)

    def logit(self, y: np.ndarray) -> np.ndarray:
        y = np.clip(y, self.eps, 1 - self.eps) ** self.nu
        return np.log(y / (1 - y))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-z))) ** (1 / self.nu)

    def _moments(self, theta: np.ndarray, p: np.ndarray, k: np.ndarray):
        zp, kk = self.logit(p), np.asarray(k, dtype=float) / self.max_k
        mu = theta[0] + theta[1] * zp + theta[2] * kk
        log_sigma = theta[3] + theta[4] * zp + theta[5] * zp**2 + theta[6] * kk
        return mu, np.exp(np.clip(log_sigma, -10, 5))

    def _nll(self, theta: np.ndarray, p, k, y) -> float:
        mu, sigma = self._moments(theta, p, k)
        low, high = y <= self.eps, y >= 1 - self.eps
        mid = ~(low | high)
        ll = stats.norm.logpdf(self.logit(y[mid]), mu[mid], sigma[mid]).sum()
        ll += stats.norm.logcdf((self.logit(self.eps) - mu[low]) / sigma[low]).sum()
        ll += stats.norm.logsf((self.logit(1 - self.eps) - mu[high]) / sigma[high]).sum()
        return -ll / len(y)

    def _fit(self, data: pd.DataFrame) -> None:
        p, k = data["p_hat"].to_numpy(dtype=float), data["horizon_h"].to_numpy()
        y = data["y"].to_numpy(dtype=float)
        start = self.theta
        if not start.any():  # first fit: mu = z(p), constant sigma from the residuals
            z = self.logit(y) - self.logit(p)
            start = np.array([z.mean(), 1.0, 0.0, np.log(z.std() + 1e-6), 0.0, 0.0, 0.0])
        result = optimize.minimize(self._nll, start, args=(p, k, y), method="L-BFGS-B")
        self.theta = result.x

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        mu, sigma = self._moments(
            self.theta, rows["p_hat"].to_numpy(dtype=float), rows["horizon_h"].to_numpy()
        )
        z = mu[:, None] + sigma[:, None] * stats.norm.ppf(self.fit_levels)[None, :]
        q = self.inverse(z)
        q[q <= self.eps] = 0.0  # censored mass at the bounds
        q[q >= 1 - self.eps] = 1.0
        return q


def calendar(time_utc: pd.Series) -> pd.DataFrame:
    hour = (time_utc.dt.hour + 0.5).to_numpy()
    doy = time_utc.dt.day_of_year.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=time_utc.index,
    )


class GradientBoosting(_Layer):
    def __init__(self, name, curve, fit_levels, grid, min_train_rows, params: GBM):
        super().__init__(name, curve, fit_levels, grid, min_train_rows)
        self.params = params
        self.columns = params.features
        self.models: list = []

    def _due(self, now: pd.Timestamp) -> bool:
        return _monthly(self.fitted_at, now)

    def _x(self, rows: pd.DataFrame) -> pd.DataFrame:
        base = rows[["p_hat", "horizon_h", *self.params.features]].astype(float)
        return pd.concat([base, calendar(rows["time_utc"])], axis=1)

    def _fit(self, data: pd.DataFrame) -> None:
        import lightgbm as lgb

        X, y = self._x(data), data["y"].to_numpy(dtype=float)
        dataset = lgb.Dataset(X, y, free_raw_data=False)
        self.models = [
            lgb.train(
                {
                    "objective": "quantile",
                    "alpha": float(tau),
                    "learning_rate": self.params.learning_rate,
                    "num_leaves": self.params.num_leaves,
                    "min_data_in_leaf": self.params.min_child_samples,
                    "num_threads": self.params.num_threads,
                    "verbosity": -1,
                },
                dataset,
                num_boost_round=self.params.n_estimators,
            )
            for tau in self.fit_levels
        ]

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        X = self._x(rows)
        return np.column_stack([m.predict(X) for m in self.models])
