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


def test_issue_time_and_targets_respect_dst(cfg):
    # 2025-03-30 is the CET -> CEST switch: 23 local hours.
    assert len(pipeline.target_hours_utc(cfg, dt.date(2025, 3, 30))) == 23
    assert len(pipeline.target_hours_utc(cfg, dt.date(2025, 10, 26))) == 25
    issue = pipeline.issue_time_utc(cfg, dt.date(2025, 1, 10))
    assert issue == pd.Timestamp("2025-01-09 09:00", tz="UTC")  # 10:00 CET


def test_committed_sample_matches_generator(cfg):
    from scripts.make_synthetic_sample import make_sample

    committed = pd.read_csv(cfg.paths.raw)
    regenerated = make_sample(list(cfg.points))
    pd.testing.assert_frame_equal(committed, regenerated, check_exact=False, atol=0.01)


@pytest.mark.smoke
def test_pipeline_end_to_end(cfg):
    pipeline.run_all(cfg)

    fc = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)
    n_days = (cfg.dates.end - cfg.dates.start).days + 1
    assert fc["target_day"].nunique() == n_days
    assert (fc["issue_time_utc"] < fc["time_utc"]).all()
    assert fc["lead_h"].between(14, 39).all()
    q = fc[[pipeline.qcol(x) for x in cfg.quantiles]].to_numpy()
    assert (q[:, 1:] >= q[:, :-1]).all(), "quantiles must not cross"

    metrics = json.loads((cfg.paths.outputs / pipeline.METRICS).read_text())
    assert metrics["n_hours"] > 0 and metrics["mean_pinball"] > 0
    assert (cfg.paths.figures / pipeline.RELIABILITY).stat().st_size > 0


def test_no_lookahead(cfg):
    """Corrupting data after a day's issue time must not change that day's forecast."""
    pipeline.run_data(cfg)
    pipeline.run_backtest(cfg)
    before = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)

    day = dt.date(2025, 1, 15)
    issue = pipeline.issue_time_utc(cfg, day)
    path = cfg.paths.interim / pipeline.DATASET
    data = pd.read_parquet(path)
    data.loc[data.index >= issue - pd.Timedelta(hours=1), "power_norm"] = 1.0
    data.to_parquet(path)
    pipeline.run_backtest(cfg)
    after = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)

    cols = [pipeline.qcol(x) for x in cfg.quantiles]
    for frame in (before, after):
        frame["target_day"] = pd.to_datetime(frame["target_day"]).dt.date
    pd.testing.assert_frame_equal(
        before.loc[before["target_day"] <= day, cols],
        after.loc[after["target_day"] <= day, cols],
    )
    assert not before.loc[before["target_day"] > day, cols].equals(
        after.loc[after["target_day"] > day, cols]
    )
