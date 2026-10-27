import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wpforecast import backtest, features
from wpforecast.config import Config

DAYS = list(pd.date_range("2025-01-20", "2025-02-10", freq="D").date)


def _features(cfg: Config) -> pd.DataFrame:
    """A synthetic feature table spanning a month boundary; every 7th hour is curtailed."""
    rows = features.skeleton(cfg, DAYS)
    rng = np.random.default_rng(0)
    rows["x"] = rng.random(len(rows))
    rows["y"] = rng.random(len(rows))
    rows["power_mw"] = rows["capacity_mw"] = 1.0
    rows["curtailed"] = rows["time_utc"].dt.hour % 7 == 0
    return rows


class Spy:
    """Records what it is shown and fails if anything is known too early."""

    name = "spy"

    def __init__(self, cfg: Config) -> None:
        self.lag = features.HOUR * (1 + cfg.issue.power_lag_h)
        self.levels = len(cfg.quantiles.levels)
        self.seen = pd.DataFrame()
        self.fits, self.updates = 0, 0

    def fit(self, known: pd.DataFrame) -> None:
        self.fits += 1
        self.seen = known

    def update(self, new: pd.DataFrame) -> None:
        self.updates += 1
        assert not new["time_utc"].isin(self.seen["time_utc"]).any(), "hour released twice"
        self.seen = pd.concat([self.seen, new])

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        issue = inputs["issue_time_utc"].iloc[0]
        assert not set(features.TARGETS) & set(inputs.columns), "targets leaked into inputs"
        if len(self.seen):
            assert (self.seen["time_utc"] + self.lag <= issue).all(), "future target seen"
            assert not self.seen["curtailed"].any() and self.seen["y"].notna().all()
        return np.tile(np.linspace(1, 0, self.levels), (len(inputs), 1))  # crossing on purpose


@pytest.mark.parametrize(
    ("regime", "fits", "updates"),
    # 19 issue times from 2025-01-23 to 2025-02-10: one fit, then 18 daily updates.
    [("frozen", 1, 0), ("walk_forward", 2, 0), ("recursive", 1, 18)],
)
def test_event_loop(cfg, regime, fits, updates):
    spy = Spy(cfg)
    start = dt.date(2025, 1, 23)
    fc = backtest.run(cfg, spy, _features(cfg), regime, start, DAYS[-1])
    assert (spy.fits, spy.updates) == (fits, updates)
    assert fc["target_day"].min() == start and fc["target_day"].max() == DAYS[-1]
    q = fc[[backtest.qcol(x) for x in cfg.quantiles.levels]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all(), "quantile crossing must be fixed by sorting"


def test_recursive_sees_every_released_hour_once(cfg):
    spy = Spy(cfg)
    table = _features(cfg)
    backtest.run(cfg, spy, table, "recursive", DAYS[3], DAYS[-1])
    last_issue = features.issue_time_utc(cfg, DAYS[-1])
    learnable = backtest.trainable(cfg, table)
    expected = learnable[learnable["known_at"] <= last_issue]
    assert len(spy.seen) == len(expected)


def test_wrong_shape_is_rejected(cfg):
    class Bad(Spy):
        def predict(self, inputs):
            return np.zeros((len(inputs), 2))

    with pytest.raises(ValueError, match="predict returned"):
        backtest.run(cfg, Bad(cfg), _features(cfg), "frozen", DAYS[3], DAYS[5])
