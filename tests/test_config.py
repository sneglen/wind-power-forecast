from pathlib import Path

import pytest

from wpforecast.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.toml"
SMOKE = ROOT / "data" / "sample" / "smoke.toml"


def test_main_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.quantiles[0] == 0.05 and cfg.quantiles[-1] == 0.95
    assert len(cfg.quantiles) == 19
    assert "thor" in cfg.points
    assert cfg.dates.start.year == 2024


def test_smoke_config_inherits_and_overrides():
    base, smoke = load_config(CONFIG), load_config(SMOKE)
    assert smoke.paths.raw == Path("data/sample/dk1_synthetic.csv")
    assert smoke.model.climatology.window_days == 5
    assert smoke.quantiles == base.quantiles
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
        ("[dates]\ntest_start = 2023-01-01", "start <= train_end"),
        ('[issue]\ntime_local = "13:00"', "gate_closure"),
        ('[issue]\ntimezone = "Mars/Olympus"', "unknown timezone"),
        ("[model.rls]\nforgetting = 1.5", "forgetting"),
        ("[points]\nbad = [95.0, 8.0]", "valid lat/lon"),
        ('[market]\npricing = "dual"', "single pricing"),
        ("[model.rls]\ntypo = 1", "unexpected or missing keys"),
    ],
)
def test_invalid_config_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, override))
