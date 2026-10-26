import dataclasses
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from wpforecast import pipeline
from wpforecast.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "sample" / "smoke.toml"


@pytest.fixture
def cfg(tmp_path) -> Config:
    base = load_config(SMOKE)
    paths = dataclasses.replace(
        base.paths,
        raw=ROOT / base.paths.raw,
        interim=tmp_path / "interim",
        outputs=tmp_path / "outputs",
        figures=tmp_path / "figures",
    )
    return dataclasses.replace(base, paths=paths)


def _horizons(cfg: Config, day: dt.date):
    k = pipeline.horizons(pipeline.issue_time_utc(cfg, day), pipeline.target_hours_utc(cfg, day))
    return int(k.min()), int(k.max()), len(k)


def test_issue_time_horizons_and_dst(cfg):
    # 11:00 local on D-1: 10:00 UTC in winter (CET), 09:00 UTC in summer (CEST).
    assert pipeline.issue_time_utc(cfg, dt.date(2025, 1, 10)) == pd.Timestamp(
        "2025-01-09 10:00", tz="UTC"
    )
    assert pipeline.issue_time_utc(cfg, dt.date(2025, 7, 10)) == pd.Timestamp(
        "2025-07-09 09:00", tz="UTC"
    )
    # Day-ahead product: k = 14..37; 23-hour spring day ends at 36, 25-hour autumn day at 38.
    assert _horizons(cfg, dt.date(2025, 1, 10)) == (14, 37, 24)
    assert _horizons(cfg, dt.date(2025, 3, 30)) == (14, 36, 23)
    assert _horizons(cfg, dt.date(2025, 10, 26)) == (14, 38, 25)
    assert cfg.issue.max_horizon_h >= 38


def test_periods(cfg):
    d = cfg.dates
    assert pipeline.period(cfg, d.train_start) == "train"
    assert pipeline.period(cfg, d.validation_start) == "validation"
    assert pipeline.period(cfg, d.train_end) == "validation"
    assert pipeline.period(cfg, d.test_start) == "test"
    assert pipeline.period(cfg, d.test_end) == "test"
    assert pipeline.period(cfg, d.stress_start) == "stress"


def test_last_day_defaults_to_data_end(cfg):
    open_ended = dataclasses.replace(cfg, dates=dataclasses.replace(cfg.dates, stress_end=None))
    index = pd.date_range("2025-01-01", "2025-01-21 23:00", freq="h", tz="UTC")
    # Local day 2025-01-21 ends at 22:00 UTC (hour-beginning stamp): fully covered.
    assert pipeline._last_day(open_ended, index) == dt.date(2025, 1, 21)
    assert pipeline._last_day(open_ended, index[:-2]) == dt.date(2025, 1, 20)


def test_committed_sample_matches_generator(cfg):
    from scripts.make_synthetic_sample import make_sample

    committed = pd.read_csv(cfg.paths.raw)
    regenerated = make_sample(list(cfg.points))
    pd.testing.assert_frame_equal(committed, regenerated, check_exact=False, atol=0.01)


@pytest.mark.smoke
def test_pipeline_end_to_end(cfg):
    pipeline.run_all(cfg)

    fc = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)
    n_days = (cfg.dates.stress_end - cfg.dates.train_start).days + 1
    assert fc["target_day"].nunique() == n_days
    assert set(fc["period"]) == {"train", "validation", "test", "stress"}
    assert (fc["issue_time_utc"] < fc["time_utc"]).all()
    assert fc["horizon_h"].between(14, 37).all()
    q = fc[[pipeline.qcol(x) for x in cfg.quantiles.levels]].to_numpy()
    assert (q[:, 1:] >= q[:, :-1]).all(), "quantiles must not cross"

    metrics = json.loads((cfg.paths.outputs / pipeline.METRICS).read_text())
    n_test_days = (cfg.dates.test_end - cfg.dates.test_start).days + 1
    assert metrics["n_hours"] == 24 * n_test_days
    assert metrics["mean_pinball"] > 0
    assert (cfg.paths.figures / pipeline.RELIABILITY).stat().st_size > 0


def _forecast_after_corrupting(cfg: Config, since: pd.Timestamp) -> pd.DataFrame:
    path = cfg.paths.interim / pipeline.DATASET
    data = pd.read_parquet(path)
    data.loc[data.index >= since, "power_norm"] = 1.0
    data.to_parquet(path)
    pipeline.run_backtest(cfg)
    fc = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)
    fc["target_day"] = pd.to_datetime(fc["target_day"]).dt.date
    pipeline.run_data(cfg)  # restore the clean dataset
    return fc


def test_information_set_at_issue_time(cfg):
    """Data stamped from the issue time on is invisible; the hour just before it is used."""
    pipeline.run_data(cfg)
    pipeline.run_backtest(cfg)
    clean = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)
    clean["target_day"] = pd.to_datetime(clean["target_day"]).dt.date

    day = dt.date(2025, 1, 15)
    issue = pipeline.issue_time_utc(cfg, day)
    cols = [pipeline.qcol(x) for x in cfg.quantiles.levels]
    on_day = clean["target_day"] == day

    future = _forecast_after_corrupting(cfg, issue)
    pd.testing.assert_frame_equal(
        clean.loc[clean["target_day"] <= day, cols], future.loc[future["target_day"] <= day, cols]
    )
    assert not clean.loc[clean["target_day"] > day, cols].equals(
        future.loc[future["target_day"] > day, cols]
    )

    # With power_lag_h = 0 the hour [issue - 1h, issue) is known at issue time.
    last_hour = _forecast_after_corrupting(cfg, issue - pd.Timedelta(hours=1))
    assert not clean.loc[on_day, cols].equals(last_hour.loc[last_hour["target_day"] == day, cols])
