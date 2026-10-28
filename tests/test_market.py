import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wpforecast import features, market


def _rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": rng.random(n), "da": rng.normal(60, 20, n), "imb": rng.normal(60, 80, n)}
    )


def test_cost_by_hand(cfg):
    rows = pd.DataFrame({"y": [0.3], "da": [50.0], "imb": [80.0]})
    # 100 MW x y = 30 MWh against a 50 MWh bid: 20 MWh short, bought back 30 EUR above DA.
    assert market.cost(cfg, rows, np.array([0.5])) == pytest.approx([600.0])
    # Long by 20 MWh while the imbalance price is above DA: sold dearer, a gain.
    assert market.cost(cfg, rows, np.array([0.1])) == pytest.approx([-600.0])


def test_positions_split_the_cost(cfg):
    rows = _rows(500)
    bid = np.random.default_rng(1).random(500)
    pos = market.positions(cfg, rows, bid)
    assert np.allclose(pos["up_eur"] + pos["down_eur"], market.cost(cfg, rows, bid))
    assert ((pos["short_mwh"] == 0) | (pos["long_mwh"] == 0)).all()


def test_cost_splits_into_covariance_and_mean_spread(cfg):
    rows = _rows(500)
    split = market.decompose(cfg, rows, np.full(500, 0.4))
    assert split["cost"] == pytest.approx(split["covariance"] + split["mean_spread"])


def test_single_price_cost_is_linear_in_the_bid(cfg):
    """With one price for both directions the cost of a bid shift is the mean spread: no
    interior optimum, the textbook quantile rule has nothing to balance."""
    rows = _rows(1000)
    base, shifted = (market.cost(cfg, rows, np.full(1000, b)).sum() for b in (0.3, 0.5))
    spread = (rows["da"] - rows["imb"]).sum()
    assert shifted - base == pytest.approx(-cfg.market.portfolio_mw * 0.2 * spread)


def test_newsvendor_level_from_effective_penalties():
    totals = pd.Series({"short_mwh": 10.0, "up_eur": 300.0, "long_mwh": 20.0, "down_eur": 200.0})
    c_up, c_down, tau = market.newsvendor(totals)
    assert (c_up, c_down) == (30.0, 10.0)
    assert tau == pytest.approx(0.25)  # dear shortfalls: bid low
    rewarded = pd.concat([totals.drop("up_eur"), pd.Series({"up_eur": -300.0})])  # sum < 0
    assert market.newsvendor(rewarded)[2] == market.MEDIAN


def test_trailing_window_ends_before_the_issue(cfg):
    day = dt.date(2025, 3, 26)
    first, last = market.trailing_window(cfg, day)
    assert last == day - dt.timedelta(days=2)
    end_of_last = pd.Timestamp(last + dt.timedelta(days=1), tz=cfg.issue.tz)
    assert end_of_last <= features.issue_time_utc(cfg, day)
    assert first == cfg.market.start  # never before the market window
    later = market.trailing_window(cfg, day + dt.timedelta(days=cfg.market.tau_lookback_days))
    assert (later[1] - later[0]).days == cfg.market.tau_lookback_days - 1


def test_walk_forward_ignores_days_not_complete_at_issue(cfg):
    """Prices and production from the issue day on cannot move the chosen level."""
    mk = dataclasses.replace(cfg.market, tau_min_days=2, tau_lookback_days=5)
    cfg = dataclasses.replace(cfg, market=mk)
    days = [mk.start + dt.timedelta(days=i) for i in range(8)]
    levels = np.array([0.1, 0.5, 0.9])
    by_tau = pd.DataFrame([[1.0, 2.0, 3.0]] * 8, index=days, columns=levels)  # low tau wins
    parts = pd.DataFrame(
        {"short_mwh": 1.0, "long_mwh": 1.0, "up_eur": 30.0, "down_eur": 10.0}, index=days
    )
    clean = market.walk_forward(cfg, by_tau, parts)
    assert (clean["tau*"].iloc[:2] == market.MEDIAN).all()  # too few complete days yet
    assert clean["tau*"].iloc[3] == 0.1 and clean["newsvendor"].iloc[3] == 0.1

    target = days[5]
    future = by_tau.index >= target - dt.timedelta(days=1)  # the issue day and later
    corrupted = by_tau.copy()
    corrupted.loc[future] = [[1e6, 2.0, 0.0]] * future.sum()
    upset = parts.copy()
    upset.loc[future, "up_eur"] = -1e6
    moved = market.walk_forward(cfg, corrupted, upset)
    upto = clean.index <= target
    pd.testing.assert_frame_equal(clean[upto], moved[upto])
    assert not clean[~upto].equals(moved[~upto])
