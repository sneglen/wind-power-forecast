import datetime as dt
from pathlib import Path

import pytest

from wpforecast.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.toml"
SMOKE = ROOT / "data" / "sample" / "smoke.toml"


def test_main_config_matches_plan():
    cfg = load_config(CONFIG)
    q = cfg.quantiles.levels
    assert (q[0], q[-1], len(q), cfg.quantiles.crps_levels) == (0.05, 0.95, 19, 99)
    assert cfg.issue.time_local == dt.time(11, 0) and cfg.issue.power_lag_h == 0
    assert cfg.dates.train_start == dt.date(2024, 4, 1)
    assert (cfg.dates.test_start, cfg.dates.test_end) == (dt.date(2025, 3, 1), dt.date(2026, 2, 28))
    assert cfg.dates.stress_end is None
    regions = [p.region for p in cfg.points.values()]
    assert regions.count("onshore") == 12 and regions.count("offshore") == 7
    assert cfg.points["thor"].region == "offshore"
    assert cfg.market.portfolio_mw == 100.0
    assert cfg.data.start == dt.date(2024, 3, 14) and not cfg.data.offline
    assert cfg.nwp.model == "ecmwf_ifs" and cfg.nwp.variables[:2] == (
        "wind_speed_100m",
        "wind_direction_100m",
    )


def test_smoke_config_inherits_and_overrides():
    base, smoke = load_config(CONFIG), load_config(SMOKE)
    assert smoke.paths.raw == Path("data/sample/raw")
    assert smoke.data.offline
    assert smoke.model.climatology.window_days == 5
    assert smoke.quantiles == base.quantiles
    assert smoke.points == base.points
    assert smoke.model.rls == base.model.rls


def _write(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "cfg.toml"
    path.write_text(f'base = "{CONFIG}"\n{extra}')
    return path


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("[quantiles]\nlevels = [0.5, 0.1]", "increasing"),
        ("[quantiles]\nlevels = [0.0, 0.5]", "strictly inside"),
        ("[dates]\ntest_start = 2023-01-01", "train_start <= validation_start"),
        ("[dates]\nstress_end = 2026-01-01", "stress_start <= stress_end"),
        ('[issue]\ntime_local = "13:00"', "precede gate closure"),
        ('[issue]\ntime_local = "08:00"', "available by the issue time"),
        ("[nwp]\nmax_lead_h = 40", "max_lead_h"),
        ("[market]\nend = 2026-06-30", "inside the test period"),
        ('[adaptivity]\nregimes = ["frozen", "magic"]', "regimes"),
        ("[points.offshore]\non_5500_0900 = [55.0, 8.0]", "duplicate"),
        ('[issue]\ntimezone = "Mars/Olympus"', "unknown timezone"),
        ("[model.rls]\nforgetting = 1.5", "forgetting"),
        ("[points.onshore]\nbad = [95.0, 8.0]", "valid lat/lon"),
        ('[market]\npricing = "dual"', "single pricing"),
        ("[data]\nstart = 2024-05-01", "must precede"),
        ("[data.openmeteo]\ncalls_per_hour = 20000", "budgets"),
        ("[model.rls]\ntypo = 1", "unexpected or missing keys"),
    ],
)
def test_invalid_config_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, override))
