"""Scores shared by every model: pinball loss, CRPS from quantiles, coverage, MAE, skill.

Forecasts are quantiles on the CRPS grid tau = 1/n .. (n-1)/n (``[quantiles] crps_levels``),
which contains the 19 reporting levels. CRPS is approximated from all of them (scoringrules
``crps_quantile`` = 2 x mean pinball loss); pinball loss and coverage use the reporting
levels; MAE is that of the median and of the point forecast (the median for models without
one), RMSE that of the point forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scoringrules as sr

from wpforecast.config import Config


def crps_levels(cfg: Config) -> tuple[float, ...]:
    n = cfg.quantiles.crps_levels + 1
    return tuple(i / n for i in range(1, n))


def qcol(level: float) -> str:
    """Column name holding the forecast of quantile ``level``."""
    return f"q{round(level, 6):g}"


def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Pinball loss per row and level; ``q`` has shape (rows, levels)."""
    diff = y[:, None] - q
    return np.maximum(levels * diff, (levels - 1) * diff)


def crps(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """CRPS per row, approximated from the quantiles ``q`` at ``levels``."""
    return np.asarray(sr.crps_quantile(y, q, levels, backend="numpy"))


def scores(cfg: Config, fc: pd.DataFrame) -> dict:
    """Mean scores of one model's forecasts (rows with an observation)."""
    grid = np.asarray(crps_levels(cfg))
    report = np.asarray(cfg.quantiles.levels)
    y = fc["y"].to_numpy()
    q_grid = fc[[qcol(x) for x in grid]].to_numpy()
    q_report = fc[[qcol(x) for x in report]].to_numpy()
    return {
        "n_hours": len(fc),
        "crps": float(crps(y, q_grid, grid).mean()),
        "mean_pinball": float(pinball(y, q_report, report).mean()),
        "mae_median": float(np.abs(y - fc[qcol(0.5)].to_numpy()).mean()),
        "mae_point": float(np.abs(y - fc["point"].to_numpy()).mean()),
        "rmse_point": float(np.sqrt(((y - fc["point"].to_numpy()) ** 2).mean())),
        "coverage": dict(
            zip(map(str, report), (y[:, None] <= q_report).mean(axis=0).tolist(), strict=True)
        ),
    }


def crps_by(cfg: Config, fc: pd.DataFrame, key: str) -> pd.Series:
    """Mean CRPS per value of ``key`` (e.g. horizon)."""
    grid = np.asarray(crps_levels(cfg))
    per_row = crps(fc["y"].to_numpy(), fc[[qcol(x) for x in grid]].to_numpy(), grid)
    return pd.Series(per_row, index=fc.index).groupby(fc[key]).mean()


def skill(score: float, reference: float) -> float:
    """Skill score: 1 - score / reference (positive = better than the reference)."""
    return 1.0 - score / reference
