"""ECMWF IFS ENS point values from dynamical.org, cached one parquet file per run.

The dataset (``[nwp.ensemble] dataset``) is a global 0.25 deg Icechunk/Zarr store with 51
members, 3-hourly leads to 144 h and one 00 UTC run per day from 2024-04-01. All DK1 points
fall in one 8 x 8 deg storage chunk, so a run costs one chunk read per variable. Only the
nearest grid cell to each ``[points]`` entry and leads up to ``max_lead_h`` are kept, in
``<raw>/ens/<YYYY>/<YYYY-MM-DD>T<HH>.parquet`` (run_utc, lead_h, member, point, variables).
Runs not in the archive are listed in ``<raw>/ens/missing.json`` and not requested again.

Data licence: CC BY 4.0, "ECMWF IFS ENS forecast data processed by dynamical.org from ECMWF
Open Data".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wpforecast.config import Config

log = logging.getLogger(__name__)

MISSING = "missing.json"


def open_dataset(cfg: Config) -> xr.Dataset:
    import dynamical_catalog  # heavy import, network on open: only when downloading

    return dynamical_catalog.open(cfg.nwp.ensemble.dataset, chunks=None)


def run_times(cfg: Config) -> list[pd.Timestamp]:
    days = pd.date_range(cfg.nwp.ensemble.start, cfg.data.end, freq="D", tz="UTC")
    return list(days + pd.Timedelta(hours=cfg.nwp.cycle_utc))


def run_path(raw: Path, run: pd.Timestamp) -> Path:
    return raw / "ens" / f"{run:%Y}" / f"{run:%Y-%m-%dT%H}.parquet"


def _missing_path(raw: Path) -> Path:
    return raw / "ens" / MISSING


def missing_runs(raw: Path) -> list[str]:
    path = _missing_path(raw)
    return json.loads(path.read_text()) if path.exists() else []


def extract(cfg: Config, ds: xr.Dataset, run: pd.Timestamp) -> pd.DataFrame:
    """Point values of one run: nearest grid cell, leads 0..max_lead_h."""
    ens = cfg.nwp.ensemble
    names = list(cfg.points)
    lat = xr.DataArray([cfg.points[n].lat for n in names], dims="point")
    lon = xr.DataArray([cfg.points[n].lon for n in names], dims="point")
    sub = (
        ds[list(ens.variables)]
        .sel(init_time=run.tz_convert(None).to_datetime64())
        .sel(lead_time=slice(pd.Timedelta(0), pd.Timedelta(hours=ens.max_lead_h)))
        .sel(latitude=lat, longitude=lon, method="nearest")
        .load()
    )
    grid = pd.DataFrame(
        {
            "point": names,
            "grid_lat": sub["latitude"].to_numpy(),
            "grid_lon": sub["longitude"].to_numpy(),
        }
    )
    frame = (
        sub.drop_vars([c for c in sub.coords if c not in sub.dims])
        .to_dataframe()
        .reset_index()
        .rename(columns={"ensemble_member": "member"})
    )
    frame["point"] = np.asarray(names)[frame["point"].to_numpy()]
    frame["lead_h"] = (frame.pop("lead_time") / pd.Timedelta(hours=1)).astype(np.int16)
    frame = frame.merge(grid, on="point")
    frame.insert(0, "run_utc", run)
    columns = ["run_utc", "lead_h", "member", "point", "grid_lat", "grid_lon", *ens.variables]
    return frame[columns].astype({v: np.float32 for v in ens.variables})


def download(cfg: Config, opener: Callable[[Config], xr.Dataset] = open_dataset) -> None:
    """Fetch every run not yet cached; runs absent from the archive are recorded once."""
    raw = cfg.paths.raw
    skip = set(missing_runs(raw))
    todo = [
        r for r in run_times(cfg) if not run_path(raw, r).exists() and r.isoformat() not in skip
    ]
    if not todo:
        return
    if cfg.data.offline:
        raise FileNotFoundError(f"offline mode, but {len(todo)} ENS runs are missing")

    ds = opener(cfg)
    archived = pd.DatetimeIndex(ds["init_time"].values).tz_localize("UTC")
    absent = [r for r in todo if r not in archived]
    if absent:
        log.warning("ens: %d runs not in the archive, e.g. %s", len(absent), absent[0])
        path = _missing_path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(skip | {r.isoformat() for r in absent}), indent=2))
    todo = [r for r in todo if r in archived]
    log.info("ens: %d runs to download from %s", len(todo), cfg.nwp.ensemble.dataset)

    def fetch(run: pd.Timestamp) -> None:
        frame = extract(cfg, ds, run)
        path = run_path(raw, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)  # never leave a half-written run that looks cached

    with ThreadPoolExecutor(max_workers=cfg.nwp.ensemble.workers) as pool:
        for i, _ in enumerate(pool.map(fetch, todo), 1):
            if i % 50 == 0 or i == len(todo):
                log.info("ens: %d / %d runs downloaded", i, len(todo))


def load(cfg: Config) -> pd.DataFrame:
    """All cached runs as one long table."""
    paths = [run_path(cfg.paths.raw, r) for r in run_times(cfg)]
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"no ENS runs under {cfg.paths.raw / 'ens'}; run `make ens`")
    return pd.concat(frames, ignore_index=True)
