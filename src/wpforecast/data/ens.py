"""ECMWF IFS ENS point values from dynamical.org, cached one parquet file per run.

The dataset (``[nwp.ensemble] dataset``) is a global 0.25 deg Icechunk/Zarr store with 51
members, 3-hourly leads to 144 h and one 00 UTC run per day from 2024-04-01. All DK1 points
fall in one 8 x 8 deg storage chunk, so a run costs one chunk read per variable. Only the
nearest grid cell to each ``[points]`` entry and leads up to ``max_lead_h`` are kept, in
``<raw>/ens/<YYYY>/<YYYY-MM-DD>T<HH>.parquet`` (run_utc, lead_h, member, point, variables).
Runs not in the archive are listed in ``<raw>/ens/missing.json`` and not requested again.

``build`` turns the cached runs into the interim table ``nwp_ens`` (``hourly``): per run,
member and UTC hour, the regional hour means of 100 m wind speed and direction, as the HRES
features are built (``wpforecast.features``).

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
INTERIM = "nwp_ens"
U, V = "wind_u_100m", "wind_v_100m"
SPEED, DIRECTION = "wind_speed_100m", "wind_direction_100m"
REGIONS = ("onshore", "offshore")


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


def cached(cfg: Config) -> bool:
    return any(run_path(cfg.paths.raw, r).exists() for r in run_times(cfg))


def hourly(cfg: Config, run: pd.DataFrame) -> pd.DataFrame:
    """One run: per member and hour, regional hour means of 100 m speed and direction.

    The 3-hourly u, v are interpolated linearly to every hour. Speed is taken per point and
    instant; the hour [t, t + 1 h) is the mean of its two ends, as for HRES; regions average
    their points; the direction comes from the regional mean u, v."""
    points = list(cfg.points)
    leads = np.sort(run["lead_h"].unique()).astype(int)
    members = np.sort(run["member"].unique())
    index = pd.MultiIndex.from_product([members, points, leads])
    cube = run.set_index(["member", "point", "lead_h"])[[U, V]].reindex(index)
    shape = (len(members), len(points), len(leads))
    u, v = (cube[c].to_numpy(dtype=float).reshape(shape) for c in (U, V))

    hours = np.arange(leads[0], leads[-1] + 1)
    i = np.clip(np.searchsorted(leads, hours, side="right") - 1, 0, len(leads) - 2)
    w = (hours - leads[i]) / (leads[i + 1] - leads[i])
    u, v = (a[..., i] * (1 - w) + a[..., i + 1] * w for a in (u, v))
    speed = np.hypot(u, v)

    def hour_mean(a: np.ndarray) -> np.ndarray:
        return (a[..., :-1] + a[..., 1:]) / 2

    start = run["run_utc"].iloc[0]
    out = {
        "run_utc": start,
        "time_utc": np.tile(start + pd.to_timedelta(hours[:-1], unit="h"), len(members)),
        "member": np.repeat(members, len(hours) - 1),
    }
    for region in REGIONS:
        on = np.array([cfg.points[p].region == region for p in points])
        mean_u, mean_v = (hour_mean(a)[:, on].mean(axis=1) for a in (u, v))
        out[f"{SPEED}@{region}"] = hour_mean(speed)[:, on].mean(axis=1).ravel()
        out[f"{DIRECTION}@{region}"] = (np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360).ravel()
    frame = pd.DataFrame(out)
    values = [c for c in frame.columns if "@" in c]
    return frame.astype({c: np.float32 for c in values})


def build(cfg: Config) -> Path:
    """The interim table ``nwp_ens`` from every cached run."""
    paths = [run_path(cfg.paths.raw, r) for r in run_times(cfg)]
    frames = [hourly(cfg, pd.read_parquet(p)) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"no ENS runs under {cfg.paths.raw / 'ens'}; run `make ens`")
    table = pd.concat(frames, ignore_index=True)
    out = cfg.paths.interim / f"{INTERIM}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    log.info("ens: %d runs, %d rows -> %s", len(frames), len(table), out)
    return out
