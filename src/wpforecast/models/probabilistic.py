"""Predictive distributions on top of the RLS power curve's point forecast p (Step 5).

Each layer owns a power curve, updated daily, and learns the distribution of y given p, the
horizon k and (for the gradient-boosted model) the NWP inputs:

- ``QuantileRegression`` (Nielsen et al., 2006): q_tau = [B(p), B(p) k/K]' gamma_tau, linear
  quantile regression on a B-spline basis in p, knots at quantiles of the training p and
  constant beyond its maximum; refit weekly on a rolling window. A
  pragmatic stand-in for time-adaptive quantile regression (Moller et al., 2008). Solved by
  statsmodels' IRLS or, exactly, as a linear program by HiGHS (``solver``).
- ``LogitNormal`` (Pinson, 2012): the generalised logit z = log(y^nu / (1 - y^nu)) is normal,
  mu and log sigma linear in z(p), z(p)^2 and k, fitted by maximum likelihood with the
  bounds as censored observations; refit monthly on all known data.
- ``GradientBoosting``: LightGBM quantile regression, one model per level, on p, k, calendar
  terms and NWP features; refit monthly. (LightGBM rejects monotone constraints with the
  quantile objective, so the plan's monotonicity in wind speed is not imposed.)

The training pairs use p as it was issued (or, before the first forecast, the curve's
prequential point), never a p fitted with hindsight. The quantile models are fitted at the
reporting and tail levels and interpolated in tau onto the CRPS grid.

With ``ensemble=True`` (Step 8) each layer also uses s, the spread (standard deviation) of
the ENS members' power through the same curve, issued like p: quantile regression adds
B(p) s to its basis, the logit-normal adds s to log sigma, LightGBM takes s as a feature.
``RawEnsemble`` is the benchmark without post-processing: the members' quantiles.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats
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
        ensemble: bool = False,
    ) -> None:
        self.name = name
        self.ensemble = ensemble
        self.curve = curve
        self.fit_levels = np.asarray(fit_levels)
        self.grid = np.asarray(grid)
        self.min_train_rows = min_train_rows
        self.train = pd.DataFrame()
        self.issued = pd.Series(dtype=float)
        self.issued_spread = pd.Series(dtype=float)
        self.fitted_at: pd.Timestamp | None = None

    # --- schedule and data ----------------------------------------------------------------

    def _due(self, now: pd.Timestamp) -> bool:
        raise NotImplementedError

    def _window(self, train: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return train

    def _pairs(self, rows: pd.DataFrame) -> pd.DataFrame:
        def issued(mine: pd.Series, curve: pd.Series) -> pd.Series:
            return mine.reindex(rows.index).fillna(curve.reindex(rows.index))

        keep = rows[["time_utc", "horizon_h", "y", *self.columns]].assign(
            p_hat=issued(self.issued, self.curve.prequential)
        )
        if not self.ensemble:
            return keep.dropna(subset=["p_hat", "y"])
        keep["ens_spread"] = issued(self.issued_spread, self.curve.prequential_spread)
        return keep.dropna(subset=["p_hat", "y", "ens_spread"])

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
        rows = inputs.assign(p_hat=p)
        ok = np.isfinite(p)  # no point forecast (no usable weather run): no distribution
        if self.ensemble:
            spread = self.curve.spread(inputs)
            self.issued_spread = pd.concat([self.issued_spread, pd.Series(spread, inputs.index)])
            rows["ens_spread"] = spread
            ok &= np.isfinite(spread)
        out = np.full((len(inputs), len(self.grid)), np.nan)
        if self.fitted_at is None or not ok.any():
            return out
        q = self._quantiles(rows[ok])
        out[ok] = np.clip(to_grid(q, self.fit_levels, self.grid), 0.0, 1.0)
        return out

    # --- model-specific -------------------------------------------------------------------

    def _fit(self, data: pd.DataFrame) -> None:
        raise NotImplementedError

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        """Quantiles at ``fit_levels``, shape (rows, levels)."""
        raise NotImplementedError


def quantile_lp(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Exact linear quantile regression (Koenker & Bassett, 1978), solved by HiGHS.

    The primal LP, min tau 1'u + (1 - tau) 1'v s.t. X b + u - v = y, u, v >= 0, has 2n + p
    variables. Its dual, max y'a s.t. X'a = (1 - tau) X'1, 0 <= a <= 1, has n bounded variables
    and p constraints, and solves ~15x faster; b is the dual's constraint multipliers. With
    a rank-deficient X (small samples) the dual's redundant constraints can be declared
    infeasible numerically; the primal, always feasible, is then solved instead."""
    dual = optimize.linprog(
        -y, A_eq=X.T, b_eq=(1 - tau) * X.sum(axis=0), bounds=(0, 1), method="highs"
    )
    if dual.success:
        return -dual.eqlin.marginals
    n, p = X.shape
    eye = sparse.identity(n, format="csr")
    primal = optimize.linprog(
        np.r_[np.zeros(p), np.full(n, tau), np.full(n, 1 - tau)],
        A_eq=sparse.hstack([sparse.csr_matrix(X), eye, -eye], format="csr"),
        b_eq=y,
        bounds=np.r_[np.tile([-np.inf, np.inf], (p, 1)), np.tile([0.0, np.inf], (2 * n, 1))],
        method="highs",
    )
    if not primal.success:
        raise RuntimeError(f"quantile regression LP failed at tau = {tau}: {primal.message}")
    return primal.x[:p]


def _monthly(fitted_at: pd.Timestamp | None, now: pd.Timestamp) -> bool:
    return fitted_at is None or (now.year, now.month) != (fitted_at.year, fitted_at.month)


class QuantileRegression(_Layer):
    def __init__(
        self, name, curve, fit_levels, grid, min_train_rows, params: QuantReg, max_k, ensemble=False
    ):
        super().__init__(name, curve, fit_levels, grid, min_train_rows, ensemble)
        self.params, self.max_k = params, max_k
        self.t = self.knots(np.linspace(0, 1, 101))  # replaced at every fit
        self.gamma = np.empty(0)

    def knots(self, p: np.ndarray) -> np.ndarray:
        """Knot vector over [0, max p], interior knots at equally spaced quantiles of p.

        Every basis function then rests on about n / (knots + 1) training rows. Uniform
        knots over [0, 1] left the top functions almost without data, since p rarely
        exceeds 0.9, and their coefficients unidentified (D-042)."""
        degree, n_inner = self.params.spline_degree, self.params.spline_knots
        top = float(np.max(p))
        inner = np.quantile(p, np.linspace(0, 1, n_inner + 2)[1:-1])
        if not (np.diff(np.r_[0.0, inner, top]) > 0).all():  # ties, e.g. many p = 0
            inner = np.linspace(0, top, n_inner + 2)[1:-1]
        return np.r_[np.zeros(degree + 1), inner, np.full(degree + 1, top)]

    def _due(self, now: pd.Timestamp) -> bool:
        return now - self.fitted_at >= self.params.refit_days * DAY

    def _window(self, train: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return train[train["time_utc"] > now - self.params.window_days * DAY]

    def design(self, p: np.ndarray, k: np.ndarray, spread: np.ndarray | None = None) -> np.ndarray:
        """[B(p), B(p) k / K] and, with a spread, B(p) s; rows with a missing p come out NaN."""
        p = np.asarray(p, dtype=float)
        ok = np.isfinite(p)
        b = np.full((len(p), len(self.t) - self.params.spline_degree - 1), np.nan)
        if ok.any():
            top = self.t[-1] * (1 - 1e-9)  # beyond the data: constant in p
            b[ok] = BSpline.design_matrix(
                np.clip(p[ok], 0, top), self.t, self.params.spline_degree
            ).toarray()
        parts = [b, b * (np.asarray(k, dtype=float) / self.max_k)[:, None]]
        if spread is not None:
            parts.append(b * np.asarray(spread, dtype=float)[:, None])
        return np.hstack(parts)

    def _design(self, rows: pd.DataFrame) -> np.ndarray:
        spread = rows["ens_spread"].to_numpy() if self.ensemble else None
        return self.design(rows["p_hat"].to_numpy(), rows["horizon_h"].to_numpy(), spread)

    def _fit(self, data: pd.DataFrame) -> None:
        self.t = self.knots(data["p_hat"].to_numpy(dtype=float))
        X = self._design(data)
        y = data["y"].to_numpy(dtype=float)
        if self.params.solver == "highs":
            self.gamma = np.column_stack([quantile_lp(X, y, tau) for tau in self.fit_levels])
            return

        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # iteration-limit warnings on flat tails
            self.gamma = np.column_stack(
                [sm.QuantReg(y, X).fit(q=tau, max_iter=2000).params for tau in self.fit_levels]
            )

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        return self._design(rows) @ self.gamma


class LogitNormal(_Layer):
    def __init__(
        self,
        name,
        curve,
        fit_levels,
        grid,
        min_train_rows,
        params: LogitNormalConfig,
        max_k,
        ensemble=False,
    ):
        super().__init__(name, curve, fit_levels, grid, min_train_rows, ensemble)
        self.nu, self.eps, self.max_k = params.nu, params.epsilon, max_k
        self.theta = np.zeros(8 if ensemble else 7)

    def _due(self, now: pd.Timestamp) -> bool:
        return _monthly(self.fitted_at, now)

    def logit(self, y: np.ndarray) -> np.ndarray:
        y = np.clip(y, self.eps, 1 - self.eps) ** self.nu
        return np.log(y / (1 - y))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-z))) ** (1 / self.nu)

    def _moments(self, theta: np.ndarray, rows: pd.DataFrame):
        zp = self.logit(rows["p_hat"].to_numpy(dtype=float))
        kk = rows["horizon_h"].to_numpy(dtype=float) / self.max_k
        mu = theta[0] + theta[1] * zp + theta[2] * kk
        log_sigma = theta[3] + theta[4] * zp + theta[5] * zp**2 + theta[6] * kk
        if self.ensemble:
            log_sigma = log_sigma + theta[7] * rows["ens_spread"].to_numpy(dtype=float)
        return mu, np.exp(np.clip(log_sigma, -10, 5))

    def _nll(self, theta: np.ndarray, rows: pd.DataFrame, y: np.ndarray) -> float:
        mu, sigma = self._moments(theta, rows)
        low, high = y <= self.eps, y >= 1 - self.eps
        mid = ~(low | high)
        ll = stats.norm.logpdf(self.logit(y[mid]), mu[mid], sigma[mid]).sum()
        ll += stats.norm.logcdf((self.logit(self.eps) - mu[low]) / sigma[low]).sum()
        ll += stats.norm.logsf((self.logit(1 - self.eps) - mu[high]) / sigma[high]).sum()
        return -ll / len(y)

    def _fit(self, data: pd.DataFrame) -> None:
        y = data["y"].to_numpy(dtype=float)
        start = self.theta
        if not start.any():  # first fit: mu = z(p), constant sigma from the residuals
            z = self.logit(y) - self.logit(data["p_hat"].to_numpy(dtype=float))
            start = np.zeros_like(self.theta)
            start[:4] = [z.mean(), 1.0, 0.0, np.log(z.std() + 1e-6)]
        result = optimize.minimize(self._nll, start, args=(data, y), method="L-BFGS-B")
        self.theta = result.x

    def _quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        mu, sigma = self._moments(self.theta, rows)
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
    def __init__(self, name, curve, fit_levels, grid, min_train_rows, params: GBM, ensemble=False):
        super().__init__(name, curve, fit_levels, grid, min_train_rows, ensemble)
        self.params = params
        self.columns = params.features
        self.models: list = []

    def _due(self, now: pd.Timestamp) -> bool:
        return _monthly(self.fitted_at, now)

    def _x(self, rows: pd.DataFrame) -> pd.DataFrame:
        spread = ["ens_spread"] if self.ensemble else []
        base = rows[["p_hat", "horizon_h", *spread, *self.params.features]].astype(float)
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


class RawEnsemble:
    """Benchmark: every ENS member through the RLS curve; the members' quantiles, no
    post-processing. Its point forecast is the member mean."""

    def __init__(self, name: str, curve: PowerCurve, grid: tuple[float, ...]) -> None:
        self.name, self.curve, self.grid = name, curve, np.asarray(grid)

    def fit(self, known: pd.DataFrame) -> None:
        self.curve.fit(known)

    def update(self, new: pd.DataFrame) -> None:
        self.curve.update(new)

    def point(self, inputs: pd.DataFrame) -> np.ndarray:
        p = self.curve.members(inputs)
        return p.mean(axis=1) if p.shape[1] else np.full(len(inputs), np.nan)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        p = self.curve.members(inputs)
        if p.shape[1] == 0:
            return np.full((len(inputs), len(self.grid)), np.nan)
        return np.quantile(p, self.grid, axis=1).T  # NaN where a member is missing
