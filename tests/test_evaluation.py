import numpy as np
import pandas as pd
import pytest

from wpforecast import evaluation, scoring


def test_block_bootstrap_draws_whole_blocks():
    rng = np.random.default_rng(0)
    idx = evaluation.block_bootstrap(10, 3, 50, rng)
    assert idx.shape == (50, 10) and idx.min() >= 0 and idx.max() <= 9
    steps = np.diff(idx[:, :3], axis=1)
    assert (steps == 1).all()  # the first block is three consecutive days


def test_diebold_mariano_separates_noise_from_a_real_difference():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 365)
    stat, p = evaluation.diebold_mariano(noise, max_lag=7)
    assert abs(stat) < 3 and p > 0.003
    stat, p = evaluation.diebold_mariano(noise - 0.5, max_lag=7)
    assert stat < -1.96 and p < 0.05


def test_pit_is_uniform_for_a_calibrated_forecast(cfg):
    rng = np.random.default_rng(2)
    grid = np.asarray(scoring.crps_levels(cfg))
    n = 20000
    y = rng.random(n)  # y ~ U(0, 1) and the forecast quantiles are exactly those of U(0, 1)
    fc = pd.DataFrame(np.tile(grid, (n, 1)), columns=[scoring.qcol(x) for x in grid]).assign(y=y)
    counts, _ = np.histogram(evaluation.pit(cfg, fc, rng), bins=10, range=(0, 1))
    assert np.allclose(counts / n, 0.1, atol=0.01)


def test_pit_spreads_ties_at_a_bound(cfg):
    rng = np.random.default_rng(3)
    grid = np.asarray(scoring.crps_levels(cfg))
    q = np.where(grid < 0.3, 0.0, grid)  # 30 % of the mass sits at 0
    n = 20000
    fc = pd.DataFrame(np.tile(q, (n, 1)), columns=[scoring.qcol(x) for x in grid]).assign(y=0.0)
    p = evaluation.pit(cfg, fc, rng)
    assert p.max() <= 0.31 and p.mean() == pytest.approx(0.15, abs=0.01)
