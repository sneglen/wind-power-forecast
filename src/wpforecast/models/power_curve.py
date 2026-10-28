"""Adaptive power curve in the WPPT spirit: recursive least squares with forgetting.

Point forecast for horizon k, issued at t with the last known hour p_t:

    p(t+k | t) = phi(t+k)' beta_t + a_k (p_t - phi(t)' beta_t)

- ``phi`` = cubic B-splines of the regional 100 m wind speed (onshore, offshore), each times
  [1, cos theta, sin theta] of the regional direction, plus diurnal harmonics of the UTC
  hour. One curve is pooled over all horizons.
- ``beta`` is estimated by recursive least squares with exponential forgetting factor
  lambda (memory ~ 1 / (1 - lambda) rows), a regularised start P0 = delta I and a covariance
  whose trace is bounded by that of P0. With lambda = 1 it equals ridge regression with
  penalty 1 / delta, i.e. OLS for a large delta: that is the frozen and walk-forward twin.
- ``a_k`` corrects the curve's error at the last known hour; it is estimated per horizon
  block from prequential residuals, with the same forgetting, so it fades with k.

The predictive distribution is the point plus empirical quantiles of the model's own
prequential errors at the same horizon over the last ``error_window_days`` (Step 5 replaces
this with proper probabilistic models).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from wpforecast.config import RLS as RLSConfig

REGIONS = ("onshore", "offshore")
SPEED, DIRECTION = "wind_speed_100m", "wind_direction_100m"
HALF_HOUR = pd.Timedelta(minutes=30)


@dataclass(frozen=True)
class Basis:
    """phi = [B_r(v_r) (x) (1, cos theta_r, sin theta_r) for r in regions] + diurnal harmonics.

    B_r: B-splines with ``knots`` interior knots spread uniformly over [0, speed_max[r]].
    """

    knots: int
    degree: int
    speed_max: dict[str, float]
    harmonics: int
    t: dict[str, np.ndarray] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ends = np.ones(self.degree + 1)
        t = {}
        for region in REGIONS:
            top = self.speed_max[region]
            inner = np.linspace(0.0, top, self.knots + 2)[1:-1]
            t[region] = np.r_[0.0 * ends, inner, top * ends]
        object.__setattr__(self, "t", t)

    @property
    def n_speed(self) -> int:
        return self.knots + self.degree + 1

    @property
    def size(self) -> int:
        return len(REGIONS) * 3 * self.n_speed + 2 * self.harmonics

    def speed(self, v: np.ndarray, region: str) -> np.ndarray:
        top = self.speed_max[region]
        v = np.clip(np.asarray(v, dtype=float), 0.0, top * (1 - 1e-9))
        if len(v) == 0:  # e.g. an issue time without a usable weather run
            return np.empty((0, self.n_speed))
        return BSpline.design_matrix(v, self.t[region], self.degree).toarray()

    def matrix(
        self, speeds: dict[str, np.ndarray], directions: dict[str, np.ndarray], hour: np.ndarray
    ) -> np.ndarray:
        """Rows with any missing input come out as NaN."""
        parts = []
        for region in REGIONS:
            v, theta = speeds[region], np.deg2rad(directions[region])
            ok = np.isfinite(v) & np.isfinite(theta)
            b = np.full((len(v), self.n_speed), np.nan)
            b[ok] = self.speed(v[ok], region)
            parts += [b, b * np.cos(theta)[:, None], b * np.sin(theta)[:, None]]
        for m in range(1, self.harmonics + 1):
            angle = 2 * np.pi * m * np.asarray(hour, dtype=float) / 24
            parts += [np.sin(angle)[:, None], np.cos(angle)[:, None]]
        return np.hstack(parts)

    def rows(self, rows: pd.DataFrame, at_issue: bool = False) -> np.ndarray:
        """phi for the target hours, or for the last known hour (``at_issue``)."""
        suffix = "@issue" if at_issue else ""
        stamp = rows["last_known_utc" if at_issue else "time_utc"] + HALF_HOUR
        hour = (stamp.dt.hour + stamp.dt.minute / 60).to_numpy()
        return self.matrix(
            {r: rows[f"{SPEED}@{r}{suffix}"].to_numpy(dtype=float) for r in REGIONS},
            {r: rows[f"{DIRECTION}@{r}{suffix}"].to_numpy(dtype=float) for r in REGIONS},
            hour,
        )


class RLS:
    """Recursive least squares with exponential forgetting and a bounded covariance."""

    def __init__(self, size: int, forgetting: float, delta: float) -> None:
        self.forgetting, self.delta = forgetting, delta
        self.beta = np.zeros(size)
        self.P = delta * np.eye(size)
        self.max_trace = delta * size

    def update(self, x: np.ndarray, y: float) -> None:
        px = self.P @ x
        gain = px / (self.forgetting + x @ px)
        self.beta += gain * (y - x @ self.beta)
        self.P = (self.P - np.outer(gain, px)) / self.forgetting
        self.P = (self.P + self.P.T) / 2
        trace = np.trace(self.P)
        if trace > self.max_trace:  # no excitation: stop the covariance from winding up
            self.P *= self.max_trace / trace


def ridge(X: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
    """Batch solution that RLS with forgetting 1 and P0 = delta I converges to exactly."""
    return np.linalg.solve(X.T @ X + np.eye(X.shape[1]) / delta, X.T @ y)


class PowerCurve:
    """The model above, for the backtest harness; regime and forgetting set the variant."""

    def __init__(
        self,
        name: str,
        params: RLSConfig,
        forgetting: float,
        levels: tuple[float, ...],
        min_error_samples: int,
    ) -> None:
        self.name = name
        self.params = params
        self.forgetting = forgetting
        self.levels = np.asarray(levels)
        self.min_error_samples = min_error_samples
        self.basis = Basis(
            params.spline_knots, params.spline_degree, params.speed_max_ms, params.diurnal_harmonics
        )
        self.blocks = np.asarray(params.horizon_blocks)
        self.reset()

    def reset(self) -> None:
        self.rls = RLS(self.basis.size, self.forgetting, self.params.delta)
        self.sxx = np.zeros(len(self.blocks))
        self.sxy = np.zeros(len(self.blocks))
        self.errors = pd.DataFrame(columns=["time_utc", "horizon_h", "error"])
        self.prequential = pd.Series(dtype=float)  # point forecast per released row (by index)
        self.history: list[tuple[pd.Timestamp, np.ndarray]] = []

    # --- forecast -------------------------------------------------------------------------

    def _block(self, horizon: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.blocks, horizon, side="right") - 1

    @property
    def a(self) -> np.ndarray:
        """Short-horizon correction per block, a_b = S_xy / S_xx."""
        return np.where(self.sxx > 0, self.sxy / np.maximum(self.sxx, 1e-12), 0.0)

    def _parts(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Curve at the target hour, its error at the last known hour, and phi."""
        phi = self.basis.rows(rows)
        curve = phi @ self.rls.beta
        issue_error = (
            rows["y_issue"].to_numpy(dtype=float) - self.basis.rows(rows, True) @ self.rls.beta
        )
        return curve, np.nan_to_num(issue_error), phi

    def point(self, rows: pd.DataFrame) -> np.ndarray:
        curve, issue_error, _ = self._parts(rows)
        a = self.a[self._block(rows["horizon_h"].to_numpy())]
        return np.clip(curve + a * issue_error, 0.0, 1.0)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        point = self.point(inputs)
        table = self._error_quantiles()
        if table.empty:
            return np.full((len(inputs), len(self.levels)), np.nan)
        k = inputs["horizon_h"].to_numpy()
        index = table.index.to_numpy()
        nearest = index[np.abs(index[None, :] - k[:, None]).argmin(axis=1)]
        return np.clip(point[:, None] + table.loc[nearest].to_numpy(), 0.0, 1.0)

    def _error_quantiles(self) -> pd.DataFrame:
        errors = self.errors
        if errors.empty:
            return pd.DataFrame()
        recent = errors["time_utc"] > errors["time_utc"].max() - pd.Timedelta(
            days=self.params.error_window_days
        )
        errors = errors[recent]
        counts = errors["horizon_h"].map(errors["horizon_h"].value_counts())
        errors = errors[counts >= self.min_error_samples]
        by_k = {
            k: np.quantile(e.to_numpy(dtype=float), self.levels)
            for k, e in errors.groupby("horizon_h")["error"]
        }
        return pd.DataFrame.from_dict(by_k, orient="index").sort_index()

    # --- learn ----------------------------------------------------------------------------

    def fit(self, known: pd.DataFrame) -> None:
        """Start afresh and learn from ``known`` day by day, exactly as ``update`` would."""
        self.reset()
        days = known["time_utc"].dt.floor("D")
        for _, batch in known.groupby(days, sort=True):
            self.update(batch)

    def update(self, new: pd.DataFrame) -> None:
        new = new.sort_values("time_utc", kind="stable")
        curve, issue_error, phi = self._parts(new)
        y = new["y"].to_numpy(dtype=float)
        ok = np.isfinite(phi).all(axis=1) & np.isfinite(y)
        if not ok.any():
            return

        # Prequential residuals: with the parameters as they were before seeing these rows.
        horizon = new["horizon_h"].to_numpy()
        block = self._block(horizon)
        point = np.clip(curve + self.a[block] * issue_error, 0.0, 1.0)
        self.errors = pd.concat(
            [
                self.errors,
                pd.DataFrame(
                    {"time_utc": new["time_utc"], "horizon_h": horizon, "error": y - point}
                )[ok],
            ],
            ignore_index=True,
        )
        self.prequential = pd.concat([self.prequential, pd.Series(point[ok], new.index[ok])])
        residual = y - curve
        for b in np.unique(block[ok]):
            sel = ok & (block == b)
            decay = self.forgetting ** sel.sum()
            self.sxx[b] = decay * self.sxx[b] + np.sum(issue_error[sel] ** 2)
            self.sxy[b] = decay * self.sxy[b] + np.sum(issue_error[sel] * residual[sel])

        for x, target in zip(phi[ok], y[ok], strict=True):
            self.rls.update(x, target)
        self.history.append((new["time_utc"].max(), self.rls.beta.copy()))
