import numpy as np
import pytest

from wpforecast import scoring


def test_crps_grid_contains_the_reporting_levels(cfg):
    grid = scoring.crps_levels(cfg)
    assert len(grid) == 99 and grid[0] == pytest.approx(0.01)
    names = {scoring.qcol(x) for x in grid}
    assert {scoring.qcol(x) for x in cfg.quantiles.levels} <= names
    assert scoring.qcol(0.07) == "q0.07" and scoring.qcol(7 / 100) == "q0.07"


def test_pinball_by_hand():
    y = np.array([0.3])
    q = np.array([[0.5, 0.5]])
    levels = np.array([0.1, 0.9])
    # under-forecast pays tau, over-forecast pays 1 - tau
    assert np.allclose(scoring.pinball(y, q, levels), [[0.9 * 0.2, 0.1 * 0.2]])


def test_crps_of_a_point_forecast_is_the_absolute_error(cfg):
    grid = np.asarray(scoring.crps_levels(cfg))
    y = np.array([0.3, 0.9])
    q = np.full((2, len(grid)), 0.5)
    assert np.allclose(scoring.crps(y, q, grid), [0.2, 0.4])


def test_crps_is_twice_the_mean_pinball(cfg):
    rng = np.random.default_rng(0)
    grid = np.asarray(scoring.crps_levels(cfg))
    y = rng.random(50)
    q = np.sort(rng.random((50, len(grid))), axis=1)
    assert np.allclose(scoring.crps(y, q, grid), 2 * scoring.pinball(y, q, grid).mean(axis=1))


def test_skill():
    assert scoring.skill(0.08, 0.10) == pytest.approx(0.2)
    assert scoring.skill(0.10, 0.10) == 0
