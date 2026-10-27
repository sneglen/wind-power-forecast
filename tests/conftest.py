import dataclasses
from pathlib import Path

import pytest

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
