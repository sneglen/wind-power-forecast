import dataclasses

import numpy as np
import pandas as pd
import pytest

from conftest import curve_table, smoke_config
from wpforecast import backtest, scoring
from wpforecast.models.power_curve import PowerCurve
from wpforecast.models.probabilistic import GradientBoosting, LogitNormal, QuantileRegression

DAYS = list(pd.date_range("2024-04-01", "2024-06-30", freq="D").date)
LEVELS = np.array([0.01, 0.1, 0.5, 0.9, 0.99])


def test_to_grid_interpolates_linearly_in_tau():
    q = np.array([[0.0, 0.2, 1.0]])
    levels = np.array([0.01, 0.5, 0.99])
    grid = np.array([0.01, 0.255, 0.5, 0.99])
    assert np.allclose(scoring.to_grid(q, levels, grid), [[0.0, 0.1, 0.2, 1.0]])


def _pairs(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """p uniform, y = p + noise whose spread peaks mid-range, clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = np.clip(p + rng.normal(0, 0.02 + 0.15 * p * (1 - p), n), 0, 1)
    times = pd.date_range("2024-04-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"time_utc": times, "horizon_h": rng.integers(1, 38, n), "y": y, "p_hat": p}
    )


def _coverage(model, data: pd.DataFrame) -> np.ndarray:
    q = model._quantiles(data)
    return (data["y"].to_numpy()[:, None] <= q).mean(axis=0)


@pytest.fixture
def layers(cfg):
    m = cfg.model
    common = ("t", None, tuple(LEVELS), tuple(LEVELS), 1)
    return {
        "quantreg": QuantileRegression(*common, m.quantreg, cfg.issue.max_horizon_h),
        "logitnormal": LogitNormal(*common, m.logitnormal, cfg.issue.max_horizon_h),
    }


@pytest.mark.parametrize("name", ["quantreg", "logitnormal"])
def test_layers_are_calibrated_on_fresh_data(layers, name):
    model = layers[name]
    model._fit(_pairs(seed=0))
    coverage = _coverage(model, _pairs(seed=1))
    assert np.allclose(coverage, LEVELS, atol=0.04), coverage


def test_logit_normal_puts_mass_on_the_bounds(layers):
    model = layers["logitnormal"]
    data = _pairs()
    data["y"] = np.where(data["p_hat"] < 0.05, 0.0, data["y"])  # calm hours produce nothing
    model._fit(data)
    calm = data[data["p_hat"] < 0.02].head(5)
    assert (model._quantiles(calm)[:, 0] == 0.0).all()


def test_gradient_boosting_is_calibrated(cfg):
    params = dataclasses.replace(cfg.model.gbm, features=("wind_speed_100m@onshore",))
    model = GradientBoosting("gbm", None, tuple(LEVELS), tuple(LEVELS), 1, params)
    data = _pairs().assign(**{"wind_speed_100m@onshore": lambda d: 3 + 12 * d["p_hat"]})
    model._fit(data)
    fresh = _pairs(seed=1).assign(**{"wind_speed_100m@onshore": lambda d: 3 + 12 * d["p_hat"]})
    assert np.allclose(_coverage(model, fresh), LEVELS, atol=0.05)


def test_layers_in_the_harness(tmp_path):
    cfg = smoke_config(tmp_path)
    table = curve_table(cfg, DAYS)
    grid = scoring.crps_levels(cfg)
    fit = scoring.fit_levels(cfg)

    def curve() -> PowerCurve:
        return PowerCurve("c", cfg.model.rls, 1.0, grid, 10)

    model = QuantileRegression("qr", curve(), fit, grid, 200, cfg.model.quantreg, 38)
    fc = backtest.run(cfg, model, table, "recursive", DAYS[45], DAYS[-1])
    q = fc[[scoring.qcol(x) for x in grid]].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all()
    inside = (fc["y"] >= fc[scoring.qcol(0.05)]) & (fc["y"] <= fc[scoring.qcol(0.95)])
    assert 0.8 <= inside.mean() <= 0.97  # a 90 % interval, out of sample
    # Training pairs carry the p that was issued, never one refitted with hindsight.
    issued = model.issued.reindex(model.train.index).dropna()
    assert len(issued) and np.allclose(model.train.loc[issued.index, "p_hat"], issued)
