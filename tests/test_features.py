import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import smoke_config
from wpforecast import features
from wpforecast.config import Config
from wpforecast.data import build, ens

DAY = dt.date(2025, 3, 22)  # an ordinary CET day inside the sample


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> tuple[Config, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = smoke_config(tmp_path_factory.mktemp("features"))
    build.build(cfg)
    return (
        cfg,
        build.read(cfg, "dataset"),
        build.read(cfg, "nwp_hres"),
        build.read(cfg, "da_price"),
    )


@pytest.fixture(scope="module")
def table(tables) -> pd.DataFrame:
    return features.build_features(*tables)


@pytest.fixture(scope="module")
def ens_table(tables) -> pd.DataFrame:
    return pd.read_parquet(ens.build(tables[0]))


def _horizons(cfg: Config, day: dt.date):
    k = features.horizons(features.issue_time_utc(cfg, day), features.target_hours_utc(cfg, day))
    return int(k.min()), int(k.max()), len(k)


def test_issue_time_horizons_and_dst(cfg):
    # 11:00 local on D-1: 10:00 UTC in winter (CET), 09:00 UTC in summer (CEST).
    assert features.issue_time_utc(cfg, dt.date(2025, 1, 10)) == pd.Timestamp(
        "2025-01-09 10:00", tz="UTC"
    )
    assert features.issue_time_utc(cfg, dt.date(2025, 7, 10)) == pd.Timestamp(
        "2025-07-09 09:00", tz="UTC"
    )
    # Day-ahead product: k = 14..37; 23-hour spring day ends at 36, 25-hour autumn day at 38.
    assert _horizons(cfg, dt.date(2025, 1, 10)) == (14, 37, 24)
    assert _horizons(cfg, dt.date(2025, 3, 30)) == (14, 36, 23)
    assert _horizons(cfg, dt.date(2025, 10, 26)) == (14, 38, 25)
    assert cfg.issue.max_horizon_h >= 38


def test_periods(cfg):
    d = cfg.dates
    assert features.period(cfg, d.train_start - dt.timedelta(days=1)) == "warmup"
    assert features.period(cfg, d.train_start) == "train"
    assert features.period(cfg, d.validation_start) == "validation"
    assert features.period(cfg, d.train_end) == "validation"
    assert features.period(cfg, d.test_start) == "test"
    assert features.period(cfg, d.test_end) == "test"
    assert features.period(cfg, d.stress_start) == "stress"


def test_last_day_defaults_to_data_end(cfg):
    open_ended = dataclasses.replace(cfg, dates=dataclasses.replace(cfg.dates, stress_end=None))
    index = pd.date_range("2025-01-01", "2025-01-21 23:00", freq="h", tz="UTC")
    # Local day 2025-01-21 ends at 22:00 UTC (hour-beginning stamp): fully covered.
    assert features.last_day(open_ended, index) == dt.date(2025, 1, 21)
    assert features.last_day(open_ended, index[:-2]) == dt.date(2025, 1, 20)


def test_rows_per_issue(table):
    day = table[table["target_day"] == DAY]
    assert day["horizon_h"].tolist() == list(range(1, 38))
    product = day[day["in_product"]]
    assert len(product) == 24 and product["horizon_h"].min() == 14
    dst = table[table["target_day"] == dt.date(2025, 3, 30)]
    assert len(dst) == 36 and dst.loc[dst["in_product"], "horizon_h"].max() == 36


def test_nwp_run_lead_and_hour_mean(tables, table):
    cfg, _, nwp, _ = tables
    row = table[(table["target_day"] == DAY) & (table["horizon_h"] == 20)].iloc[0]
    run = pd.Timestamp("2025-03-21 00:00", tz="UTC")  # 00 UTC run of D-1
    assert row["run_utc"] == run
    assert row["lead_h"] == (row["time_utc"] + features.HOUR - run) / features.HOUR

    raw = nwp[(nwp["run_utc"] == run) & (nwp["point"] == "horns_rev_1")].set_index("valid_utc")
    t = row["time_utc"]
    speed = raw["wind_speed_100m"].astype(float)
    assert row["wind_speed_100m@horns_rev_1"] == pytest.approx(
        (speed[t] + speed[t + features.HOUR]) / 2
    )
    gusts = raw["wind_gusts_10m"].astype(float)
    assert row["wind_gusts_10m@horns_rev_1"] == pytest.approx(gusts[t + features.HOUR])


def test_direction_round_trip():
    speed = pd.Series([5.0, 5.0, 5.0, 5.0])
    direction = pd.Series([0.0, 90.0, 180.0, 270.0])
    u, v = features._uv(speed, direction)
    assert np.allclose(u, [0, -5, 0, 5]) and np.allclose(v, [-5, 0, 5, 0])  # westerly: u > 0
    assert np.allclose(features._direction(u, v), direction)


def test_falls_back_to_the_previous_run(tables):
    cfg, dataset, nwp, price = tables
    d1 = pd.Timestamp("2025-03-21 00:00", tz="UTC")
    fallback = features.build_features(cfg, dataset, nwp[nwp["run_utc"] != d1], price)
    rows = fallback[fallback["target_day"] == DAY]
    assert (rows["run_utc"] == d1 - pd.Timedelta(days=1)).all()
    assert rows["lead_h"].max() == 71  # 47 h + 24 h, inside max_lead_h = 72


def _issue_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[table["target_day"] == DAY]
    return rows[features.input_columns(table)].reset_index(drop=True)


def test_information_set_at_issue_time(tables, table):
    """Nothing stamped at or after the issue time can change the inputs of that issue."""
    cfg, dataset, nwp, price = tables
    issue = features.issue_time_utc(cfg, DAY)
    last_known = features.last_known_stamp(cfg, issue)
    lag = pd.Timedelta(hours=cfg.nwp.availability_lag_h)

    future = dataset.copy()
    future.loc[future.index > last_known] = 0.999  # power, capacity and y from the future
    late_nwp = nwp.copy()
    late = late_nwp["run_utc"] + lag > issue
    late_nwp.loc[late, cfg.nwp.variables] = 99.0
    prices = price.assign(price_eur_mwh=-1.0)  # published after issue: may only flag targets
    corrupted = features.build_features(cfg, future, late_nwp, prices)
    pd.testing.assert_frame_equal(_issue_rows(table), _issue_rows(corrupted))

    # The boundary is tight: the last known hour and the current run do reach the forecast.
    edge = dataset.copy()
    edge.loc[last_known] = 0.999
    moved = _issue_rows(features.build_features(cfg, edge, nwp, price))
    assert (moved["y_issue"] == 0.999).all() and (moved["capacity_issue_mw"] == 0.999).all()
    current = nwp.copy()
    current.loc[current["run_utc"] == issue.floor("D"), "wind_speed_100m"] = 99.0
    moved = _issue_rows(features.build_features(cfg, dataset, current, price))
    assert (moved["wind_speed_100m@offshore"] > 50).all()


def test_prices_are_never_inputs(table):
    assert not [c for c in features.input_columns(table) if "price" in c]
    assert set(features.TARGETS) <= set(table.columns)
    assert table["curtailed"].dtype == bool


def test_ensemble_information_set_at_issue_time(tables, ens_table):
    """ENS runs published after the issue time cannot reach it; the current one does."""
    cfg, dataset, nwp, price = tables
    issue = features.issue_time_utc(cfg, DAY)
    lag = pd.Timedelta(hours=cfg.nwp.ensemble.availability_lag_h)
    values = [c for c in ens_table.columns if "@" in c]
    clean = _issue_rows(features.build_features(cfg, dataset, nwp, price, ens_table))
    assert len(features.ensemble_members(clean.columns)) == 51
    assert (clean["ens_run_utc"] == issue.floor("D")).all()

    late = ens_table.copy()
    late.loc[late["run_utc"] + lag > issue, values] = 99.0
    corrupted = _issue_rows(features.build_features(cfg, dataset, nwp, price, late))
    pd.testing.assert_frame_equal(clean, corrupted)

    current = ens_table.copy()
    current.loc[current["run_utc"] == issue.floor("D"), values] = 99.0
    moved = _issue_rows(features.build_features(cfg, dataset, nwp, price, current))
    assert (moved["ens_mean_wind_speed_100m@onshore"] > 50).all()
    assert (moved["ens_std_wind_speed_100m@onshore"] < 1e-3).all()

    slow = dataclasses.replace(cfg.nwp.ensemble, availability_lag_h=11)  # 01:00 UTC too late
    later = dataclasses.replace(cfg, nwp=dataclasses.replace(cfg.nwp, ensemble=slow))
    rows = _issue_rows(features.build_features(later, dataset, nwp, price, ens_table))
    assert (rows["ens_run_utc"] == issue.floor("D") - pd.Timedelta(days=1)).all()


def test_an_ens_run_used_too_early_is_refused(cfg, table):
    rows = table.assign(ens_run_utc=table["issue_time_utc"])
    with pytest.raises(RuntimeError, match="ENS run"):
        features._assert_information_set(cfg, rows)
