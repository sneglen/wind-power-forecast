import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wpforecast import features
from wpforecast.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "sample" / "smoke.toml"


def smoke_config(tmp: Path) -> Config:
    """The smoke config on the committed sample, writing everything below ``tmp``."""
    base = load_config(SMOKE)
    paths = dataclasses.replace(
        base.paths,
        raw=ROOT / base.paths.raw,
        interim=tmp / "interim",
        outputs=tmp / "outputs",
        figures=tmp / "figures",
        reports=tmp / "reports",
    )
    return dataclasses.replace(base, paths=paths)


@pytest.fixture
def cfg(tmp_path) -> Config:
    return smoke_config(tmp_path)


def truth(speed: np.ndarray) -> np.ndarray:
    """The synthetic power curve."""
    return 1 / (1 + np.exp(-(speed - 9.0)))


def curve_table(cfg: Config, days: list[dt.date], seed: int = 2) -> pd.DataFrame:
    """Feature rows with perfect NWP: y = truth(speed) + an hourly AR(1) error (persists)."""
    rows = features.skeleton(cfg, days)
    hours = pd.date_range(
        rows["time_utc"].min() - pd.Timedelta(days=2), rows["time_utc"].max(), freq="h"
    )
    rng = np.random.default_rng(seed)
    speed = pd.Series(
        np.clip(9 + 4 * np.sin(np.arange(len(hours)) / 17) + rng.normal(0, 1, len(hours)), 0, 25),
        hours,
    )
    error = np.zeros(len(hours))
    for i in range(1, len(hours)):
        error[i] = 0.97 * error[i - 1] + rng.normal(0, 0.01)
    y = pd.Series(np.clip(truth(speed.to_numpy()) + error, 0, 1), hours)
    rows["last_known_utc"] = features.last_known_stamp(cfg, rows["issue_time_utc"])
    for suffix, stamp in (("", "time_utc"), ("@issue", "last_known_utc")):
        for region in ("onshore", "offshore"):
            rows[f"wind_speed_100m@{region}{suffix}"] = speed.reindex(rows[stamp]).to_numpy()
            rows[f"wind_direction_100m@{region}{suffix}"] = 270.0
    rows["y"] = y.reindex(rows["time_utc"]).to_numpy()
    rows["y_issue"] = y.reindex(rows["last_known_utc"]).to_numpy()
    rows["curtailed"] = False
    return rows.assign(power_mw=1.0, capacity_mw=1.0)
