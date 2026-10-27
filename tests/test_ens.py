import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wpforecast.config import Config
from wpforecast.data import ens


def _config(cfg: Config, tmp_path) -> Config:
    """Three runs from 2025-03-08; the archive below lacks the middle one."""
    return dataclasses.replace(
        cfg,
        paths=dataclasses.replace(cfg.paths, raw=tmp_path / "raw"),
        data=dataclasses.replace(cfg.data, end=dt.date(2025, 3, 10), offline=False),
        nwp=dataclasses.replace(
            cfg.nwp, ensemble=dataclasses.replace(cfg.nwp.ensemble, start=dt.date(2025, 3, 8))
        ),
    )


def _archive() -> xr.Dataset:
    """A dynamical.org-shaped dataset over Denmark: value = member + lead/1000."""
    inits = pd.to_datetime(["2025-03-08", "2025-03-10"])
    leads = pd.to_timedelta(np.arange(0, 145, 3), unit="h")
    members = np.arange(4, dtype=np.int16)
    lat = np.arange(58.0, 54.0, -0.25)
    lon = np.arange(7.0, 12.0, 0.25)
    shape = (len(inits), len(leads), len(members), len(lat), len(lon))
    value = (
        members[None, None, :, None, None]
        + (leads / pd.Timedelta(hours=1)).to_numpy()[None, :, None, None, None] / 1000
    )
    data = np.broadcast_to(value, shape).astype(np.float32)
    coords = {
        "init_time": inits,
        "lead_time": leads,
        "ensemble_member": members,
        "latitude": lat,
        "longitude": lon,
    }
    dims = list(coords)
    return xr.Dataset({"wind_u_100m": (dims, data), "wind_v_100m": (dims, -data)}, coords)


def test_download_extract_and_resume(cfg, tmp_path):
    cfg = _config(cfg, tmp_path)
    opened = []

    def opener(_: Config) -> xr.Dataset:
        opened.append(1)
        return _archive()

    ens.download(cfg, opener)
    runs = ens.run_times(cfg)
    assert [ens.run_path(cfg.paths.raw, r).exists() for r in runs] == [True, False, True]
    assert ens.missing_runs(cfg.paths.raw) == [runs[1].isoformat()]

    table = ens.load(cfg)
    assert set(table["lead_h"]) == set(range(0, cfg.nwp.ensemble.max_lead_h + 1, 3))
    assert table["point"].nunique() == len(cfg.points) and table["member"].nunique() == 4
    row = table[(table["member"] == 2) & (table["lead_h"] == 24)].iloc[0]
    assert row["wind_u_100m"] == pytest.approx(2.024)
    assert row["wind_v_100m"] == pytest.approx(-2.024)
    hr1 = table[table["point"] == "horns_rev_1"].iloc[0]
    assert (hr1["grid_lat"], hr1["grid_lon"]) == (55.5, 8.0)  # nearest cell to 55.53, 7.91

    ens.download(cfg, opener)
    assert len(opened) == 1, "cached and missing runs must not be requested again"


def test_offline_refuses_ens_download(cfg, tmp_path):
    offline = dataclasses.replace(
        _config(cfg, tmp_path), data=dataclasses.replace(cfg.data, offline=True)
    )
    with pytest.raises(FileNotFoundError, match="offline"):
        ens.download(offline, lambda _: _archive())
