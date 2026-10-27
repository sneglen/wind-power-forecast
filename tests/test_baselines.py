import numpy as np
import pandas as pd
import pytest

from conftest import smoke_config
from wpforecast import backtest, features, scoring
from wpforecast.config import Config
from wpforecast.models.baselines import Climatology, NewReference, Persistence

PHI, MEAN = 0.95, 0.4
DAYS = list(pd.date_range("2024-04-01", "2025-03-31", freq="D").date)  # contains a k = 38 day


def _ar1(index: pd.DatetimeIndex) -> pd.Series:
    """Hourly AR(1) around MEAN, small enough never to leave [0, 1]."""
    rng = np.random.default_rng(1)
    x = np.zeros(len(index))
    for i in range(1, len(index)):
        x[i] = PHI * x[i - 1] + rng.normal(0, 0.02)
    return pd.Series(MEAN + x, index=index)


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> pd.DataFrame:
    cfg = smoke_config(tmp_path_factory.mktemp("baselines"))
    rows = features.skeleton(cfg, DAYS)
    hours = pd.date_range(
        rows["time_utc"].min() - pd.Timedelta(days=2), rows["time_utc"].max(), freq="h"
    )
    y = _ar1(hours)
    rows["y"] = y.reindex(rows["time_utc"]).to_numpy()
    rows["y_issue"] = y.reindex(features.last_known_stamp(cfg, rows["issue_time_utc"])).to_numpy()
    rows["curtailed"] = False
    return rows


def test_climatology_is_the_empirical_distribution(cfg, table):
    model = Climatology((0.1, 0.5, 0.9))
    model.fit(table)
    hourly = table.drop_duplicates("time_utc")["y"]
    assert np.allclose(model.predict(table.head(3))[0], np.quantile(hourly, [0.1, 0.5, 0.9]))


def test_new_reference_learns_the_autocorrelation(table):
    model = NewReference((0.5,), min_samples=30, power_lag_h=0)
    model.fit(table)
    for k in (1, 14, 24, 37):
        assert model.a[k] == pytest.approx(PHI**k, abs=0.15)
    assert model.mean == pytest.approx(MEAN, abs=0.02)
    # Far horizons shrink to the mean, short ones stay close to persistence.
    row = table[table["horizon_h"].isin([1, 37])].head(37)
    point = model.point(row)
    assert abs(point[-1] - MEAN) < abs(row["y_issue"].iloc[-1] - MEAN) + 1e-12


def test_persistence_adds_error_quantiles_by_horizon(table):
    model = Persistence((0.1, 0.5, 0.9), min_samples=30)
    model.fit(table)
    at_k = table[table["horizon_h"] == 20]
    expected = np.quantile(at_k["y"] - at_k["y_issue"], [0.1, 0.5, 0.9])
    row = at_k.head(1)
    assert np.allclose(model.predict(row)[0], np.clip(row["y_issue"].iloc[0] + expected, 0, 1))


def test_rare_horizon_borrows_the_nearest(table):
    model = Persistence((0.1, 0.9), min_samples=30)
    model.fit(table)
    assert (table["horizon_h"] == 38).sum() == 1 and 38 not in model.errors.index
    row = table[table["horizon_h"] == 38]
    as_37 = row.assign(horizon_h=37)
    assert np.allclose(model.predict(row), model.predict(as_37))


@pytest.mark.parametrize("make", [Climatology, Persistence, NewReference])
def test_baselines_run_in_the_harness(cfg: Config, table, make):
    levels = scoring.crps_levels(cfg)
    model = {
        Climatology: lambda: Climatology(levels),
        Persistence: lambda: Persistence(levels, 30),
        NewReference: lambda: NewReference(levels, 30, cfg.issue.power_lag_h),
    }[make]()
    table = table.assign(power_mw=1.0, capacity_mw=1.0)
    fc = backtest.run(cfg, model, table, "walk_forward", DAYS[200], DAYS[230])
    assert fc["target_day"].nunique() == 31
    q = fc[[scoring.qcol(x) for x in levels]].to_numpy()
    assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
