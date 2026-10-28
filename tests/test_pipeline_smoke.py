import datetime as dt
import json

import pandas as pd
import pytest

from wpforecast import evaluation, features, market, pipeline
from wpforecast.config import Config
from wpforecast.scoring import qcol


@pytest.mark.smoke
def test_pipeline_end_to_end(cfg):
    pipeline.run_all(cfg)

    fc = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)
    n_days = (cfg.dates.stress_end - cfg.dates.train_start).days + 1
    assert fc["target_day"].nunique() == n_days
    assert set(fc["period"]) == {"train", "validation", "test", "stress"}
    assert (fc["time_utc"] >= fc["issue_time_utc"]).all()
    assert fc["horizon_h"].between(1, 37).all()
    product = fc[fc["in_product"]]
    assert product["horizon_h"].between(14, 37).all()
    dst_day = product[
        (product["target_day"].astype(str) == "2025-03-30") & (product["model"] == "climatology")
    ]
    assert len(dst_day) == 23 and dst_day["horizon_h"].max() == 36
    q = fc[[qcol(x) for x in cfg.quantiles.levels]].to_numpy()
    assert (q[:, 1:] >= q[:, :-1]).all(), "quantiles must not cross"

    assert set(fc["model"]) == {
        "climatology",
        "persistence",
        "new_reference",
        "curve_frozen",
        "curve_walk_forward",
        "curve_rls",
        *pipeline.PROBABILISTIC,
    }
    assert fc.loc[fc["model"] == "curve_frozen", "period"].isin(["test", "stress"]).all()
    tuning = json.loads((cfg.paths.outputs / pipeline.TUNING).read_text())
    assert tuning["rls_forgetting"] in cfg.model.rls.forgetting_grid
    metrics = json.loads((cfg.paths.outputs / pipeline.METRICS).read_text())
    n_test_days = (cfg.dates.test_end - cfg.dates.test_start).days + 1
    assert metrics["n_hours"] == 24 * n_test_days
    for result in metrics["models"].values():
        assert result["n_hours"] == metrics["n_hours"], "models are scored on the same hours"
        assert 0 < result["crps"] <= result["mae_median"] + 1e-12  # CRPS <= MAE of the median
    assert metrics["models"]["climatology"]["crps_skill"] == 0
    assert "| persistence |" in (cfg.paths.outputs / pipeline.SCORES).read_text()
    report = (cfg.paths.reports / evaluation.REPORT).read_text()
    assert "## Diebold" in report and "energinet" in report
    for figure in ("skill_by_horizon", "reliability", "pit", "sharpness"):
        assert (cfg.paths.reports / evaluation.FIGURES / f"{figure}.png").stat().st_size > 0
    market_report = (cfg.paths.reports / market.REPORT).read_text()
    assert "## Position against the system" in market_report and "energinet" in market_report
    for figure in ("cost_vs_tau", "tau_over_time"):
        assert (cfg.paths.reports / market.FIGURES / f"{figure}.png").stat().st_size > 0
    costs = json.loads((cfg.paths.outputs / market.RESULTS).read_text())
    assert costs["hours"] == 24 * ((cfg.market.end - cfg.market.start).days + 1)
    assert set(costs["eur_per_mwh"]["logitnormal"]) >= set(market.STRATEGIES)
    figures = (
        pipeline.CURVE_BY_SECTOR,
        pipeline.CURVE_TRAJECTORY,
        pipeline.FAN_CHART,
    )
    for figure in figures:
        assert (cfg.paths.figures / figure).stat().st_size > 0


def _forecast_after_corrupting(cfg: Config, since: pd.Timestamp) -> pd.DataFrame:
    path = cfg.paths.interim / "dataset.parquet"
    clean = pd.read_parquet(path)
    corrupted = clean.copy()
    corrupted.loc[corrupted.index >= since, "power_norm"] = 1.0
    corrupted.to_parquet(path)
    pipeline.run_features(cfg)
    pipeline.run_backtest(cfg)
    clean.to_parquet(path)
    return pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)


def test_forecasts_use_only_the_information_set(cfg):
    """Through the whole chain: production from the issue time on cannot move a forecast."""
    pipeline.run_data(cfg)
    pipeline.run_features(cfg)
    pipeline.run_backtest(cfg)
    clean = pd.read_parquet(cfg.paths.outputs / pipeline.FORECASTS)

    day = dt.date(2025, 3, 22)
    issue = features.issue_time_utc(cfg, day)
    cols = [qcol(x) for x in cfg.quantiles.levels]

    future = _forecast_after_corrupting(cfg, issue)
    upto = clean["target_day"] <= day
    pd.testing.assert_frame_equal(clean.loc[upto, cols], future.loc[upto, cols])
    assert not clean.loc[~upto, cols].equals(future.loc[~upto, cols])

    # With power_lag_h = 0 the hour [issue - 1h, issue) is known at issue time.
    last_hour = _forecast_after_corrupting(cfg, issue - pd.Timedelta(hours=1))
    on_day = clean["target_day"] == day
    assert not clean.loc[on_day, cols].equals(last_hour.loc[on_day, cols])
