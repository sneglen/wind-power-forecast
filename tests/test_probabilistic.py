import dataclasses

import numpy as np
import pandas as pd
import pytest

from conftest import curve_table, smoke_config
from wpforecast import backtest, scoring
from wpforecast.models.power_curve import PowerCurve
from wpforecast.models.probabilistic import (
    GradientBoosting,
    LogitNormal,
    QuantileRegression,
    RawEnsemble,
)

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


@pytest.mark.parametrize("kind", ["curve", "quantreg", "logitnormal", "lightgbm"])
def test_a_day_without_weather_is_skipped(tmp_path, kind):
    """An issue time with no usable run (all NWP missing) must not break any model."""
    cfg = smoke_config(tmp_path)
    days = DAYS[:40]
    table = curve_table(cfg, days)
    gap = days[35]
    nwp = [c for c in table.columns if c.startswith("wind_")]
    table.loc[table["target_day"] == gap, nwp] = np.nan
    grid, fit = scoring.crps_levels(cfg), scoring.fit_levels(cfg)
    curve = PowerCurve("c", cfg.model.rls, 1.0, grid, 10)
    gbm = dataclasses.replace(cfg.model.gbm, features=("wind_speed_100m@onshore",))
    model = {
        "curve": lambda: curve,
        "quantreg": lambda: QuantileRegression("qr", curve, fit, grid, 200, cfg.model.quantreg, 38),
        "logitnormal": lambda: LogitNormal("ln", curve, fit, grid, 200, cfg.model.logitnormal, 38),
        "lightgbm": lambda: GradientBoosting("gbm", curve, fit, grid, 200, gbm),
    }[kind]()
    fc = backtest.run(cfg, model, table, "recursive", days[20], days[-1])
    assert gap not in set(fc["target_day"]) and days[-1] in set(fc["target_day"])


def _pairs_with_spread(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """y = p + noise whose standard deviation is the ensemble spread s."""
    rng = np.random.default_rng(seed)
    data = _pairs(n, seed).assign(
        p_hat=rng.uniform(0.3, 0.7, n), ens_spread=rng.uniform(0.01, 0.15, n)
    )
    data["y"] = np.clip(data["p_hat"] + rng.normal(0, data["ens_spread"]), 0, 1)
    return data


@pytest.mark.parametrize("kind", ["quantreg", "logitnormal"])
def test_layers_widen_with_the_ensemble_spread(cfg, kind):
    common = ("t", None, tuple(LEVELS), tuple(LEVELS), 1)
    k = cfg.issue.max_horizon_h
    model = {
        "quantreg": QuantileRegression(*common, cfg.model.quantreg, k, ensemble=True),
        "logitnormal": LogitNormal(*common, cfg.model.logitnormal, k, ensemble=True),
    }[kind]
    model._fit(_pairs_with_spread(seed=0))
    fresh = _pairs_with_spread(seed=1)
    q = model._quantiles(fresh)
    width = q[:, 3] - q[:, 1]  # the 10-90 % interval
    s = fresh["ens_spread"].to_numpy()
    assert width[s > 0.12].mean() > 3 * width[s < 0.04].mean()
    assert np.allclose(_coverage(model, fresh), LEVELS, atol=0.04)


def test_ensemble_models_in_the_harness(tmp_path):
    cfg = smoke_config(tmp_path)
    table = curve_table(cfg, DAYS)
    rng = np.random.default_rng(0)
    for m in range(5):
        for r in ("onshore", "offshore"):
            table[f"ens{m:02d}_wind_speed_100m@{r}"] = table[f"wind_speed_100m@{r}"] + rng.normal(
                0, 1, len(table)
            )
            table[f"ens{m:02d}_wind_direction_100m@{r}"] = 270.0
    grid, fit = scoring.crps_levels(cfg), scoring.fit_levels(cfg)

    def curve() -> PowerCurve:
        return PowerCurve("c", cfg.model.rls, 1.0, grid, 10, ensemble=True)

    gbm = dataclasses.replace(cfg.model.gbm, features=("wind_speed_100m@onshore",))
    for model in (
        RawEnsemble("raw", curve(), grid),
        QuantileRegression("qr", curve(), fit, grid, 200, cfg.model.quantreg, 38, ensemble=True),
        LogitNormal("ln", curve(), fit, grid, 200, cfg.model.logitnormal, 38, ensemble=True),
        GradientBoosting("gbm", curve(), fit, grid, 200, gbm, ensemble=True),
    ):
        fc = backtest.run(cfg, model, table, "recursive", DAYS[45], DAYS[-1])
        q = fc[[scoring.qcol(x) for x in grid]].to_numpy()
        assert len(fc) and np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all(), model.name
    # Training pairs carry the spread that was issued, like p.
    issued = model.issued_spread.reindex(model.train.index).dropna()
    assert len(issued) and np.allclose(model.train.loc[issued.index, "ens_spread"], issued)
