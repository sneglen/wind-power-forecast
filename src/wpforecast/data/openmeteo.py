"""ECMWF IFS HRES runs from the Open-Meteo Single Runs API, cached one file per run.

Each 00 UTC run is fetched as two requests (onshore points on land grid cells, offshore
points on sea grid cells) and stored as received in
``<raw>/openmeteo/<model>/<YYYY>/<YYYY-MM-DD>T<HH>.json.gz``. A run the archive does not
have is stored as a marker with the API's reason, so it is not requested again.

The free tier allows 5000 calls / hour and 10000 / day, counting each location as a call.
A call ledger shared by every config on this machine (``~/.cache/wpforecast``) keeps the
download under the configured budgets. When the daily budget is spent the download stops
cleanly; running it again later resumes where it stopped.
Data licence: CC BY 4.0, "Weather data by Open-Meteo.com" (ECMWF open data).
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from wpforecast.config import Config, Point
from wpforecast.data import http

log = logging.getLogger(__name__)

CELL_SELECTION = {"onshore": "land", "offshore": "sea"}
HOUR, DAY = 3600.0, 86400.0


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    """Rolling-window call ledger persisted to disk, shared across runs of `make data`."""

    def __init__(
        self,
        path: Path,
        per_hour: float,
        per_day: float,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.path, self.per_hour, self.per_day, self.now = path, per_hour, per_day, now
        self.entries: list[list[float]] = json.loads(path.read_text()) if path.exists() else []

    def _used(self, window_s: float) -> float:
        cutoff = self.now() - window_s
        return sum(w for t, w in self.entries if t > cutoff)

    def acquire(self, weight: float) -> None:
        self.entries = [e for e in self.entries if e[0] > self.now() - DAY]
        if self._used(DAY) + weight > self.per_day:
            resume = dt.datetime.fromtimestamp(self.entries[0][0] + DAY, dt.UTC)
            raise BudgetExhausted(
                f"Open-Meteo daily budget ({self.per_day:g} calls) reached; "
                f"run `make data` again after {resume:%Y-%m-%d %H:%M} UTC to continue."
            )
        while self._used(HOUR) + weight > self.per_hour:
            oldest = min(t for t, _ in self.entries if t > self.now() - HOUR)
            wait_s = oldest + HOUR - self.now() + 1.0
            log.info("openmeteo: hourly budget reached, waiting %.0f min", wait_s / 60)
            http.SLEEP(wait_s)
        self.entries.append([self.now(), weight])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries))


def ledger_path() -> Path:
    """One ledger per machine: the API quota is per client, whichever config downloads."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "wpforecast" / "openmeteo_ledger.json"


def run_times(cfg: Config) -> list[pd.Timestamp]:
    days = pd.date_range(cfg.data.start, cfg.data.end, freq="D", tz="UTC")
    return list(days + pd.Timedelta(hours=cfg.nwp.cycle_utc))


def run_path(raw: Path, model: str, run: pd.Timestamp) -> Path:
    return raw / "openmeteo" / model / f"{run:%Y}" / f"{run:%Y-%m-%dT%H}.json.gz"


def _groups(points: dict[str, Point]) -> dict[str, list[str]]:
    return {
        region: [name for name, p in points.items() if p.region == region]
        for region in CELL_SELECTION
    }


def _params(cfg: Config, names: list[str], region: str, run: pd.Timestamp) -> dict[str, str]:
    pts = [cfg.points[n] for n in names]
    return {
        "latitude": ",".join(f"{p.lat:.4f}" for p in pts),
        "longitude": ",".join(f"{p.lon:.4f}" for p in pts),
        "run": f"{run:%Y-%m-%dT%H:%M}",
        "hourly": ",".join(cfg.nwp.variables),
        "models": cfg.nwp.model,
        "forecast_hours": str(cfg.nwp.max_lead_h + 1),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "cell_selection": CELL_SELECTION[region],
    }


def fetch_run(
    cfg: Config,
    session: httpx.Client,
    run: pd.Timestamp,
    budget: Budget,
    throttle: http.Throttle,
) -> dict:
    """Request one run for all points; a run missing from the archive returns a marker."""
    payload: dict = {
        "model": cfg.nwp.model,
        "run": run.isoformat(),
        "downloaded_at": dt.datetime.now(dt.UTC).isoformat(),
        "groups": {},
    }
    for region, names in _groups(cfg.points).items():
        if not names:
            continue
        budget.acquire(len(names) * cfg.data.openmeteo.calls_per_location)
        throttle.wait()
        response = http.get(session, cfg.data.openmeteo.base_url, _params(cfg, names, region, run))
        body = response.json()
        if response.status_code == 400 and "not available" in str(body.get("reason", "")):
            return {k: payload[k] for k in ("model", "run", "downloaded_at")} | {
                "missing": body["reason"]
            }
        response.raise_for_status()
        locations = body if isinstance(body, list) else [body]
        _validate(cfg, run, names, locations)
        payload["groups"][region] = {"points": names, "response": locations}
    return payload


def _validate(cfg: Config, run: pd.Timestamp, names: list[str], locations: list[dict]) -> None:
    if len(locations) != len(names):
        raise ValueError(f"run {run}: expected {len(names)} locations, got {len(locations)}")
    expected = cfg.nwp.max_lead_h + 1
    for name, loc in zip(names, locations, strict=True):
        times = loc["hourly"]["time"]
        if len(times) != expected or pd.Timestamp(times[0], tz="UTC") != run:
            raise ValueError(f"run {run}, {name}: unexpected time axis {times[0]}..({len(times)})")


def download(cfg: Config, client: httpx.Client | None = None) -> None:
    """Fetch every run not yet cached (oldest first) within the call budget."""
    raw, model = cfg.paths.raw, cfg.nwp.model
    missing = [r for r in run_times(cfg) if not run_path(raw, model, r).exists()]
    if not missing:
        return
    if cfg.data.offline:
        raise FileNotFoundError(f"offline mode, but {len(missing)} Open-Meteo runs are missing")

    om = cfg.data.openmeteo
    budget = Budget(ledger_path(), om.calls_per_hour, om.calls_per_day)
    throttle = http.Throttle(om.min_interval_s)
    log.info("openmeteo: %d runs to download", len(missing))
    with client or http.client() as session:
        for i, run in enumerate(missing, 1):
            try:
                payload = fetch_run(cfg, session, run, budget, throttle)
            except http.LimitExceeded as exc:
                raise BudgetExhausted(
                    f"Open-Meteo quota reached after {i - 1} runs ({exc}); "
                    "run `make data` again tomorrow to continue."
                ) from exc
            path = run_path(raw, model, run)
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as f:
                json.dump(payload, f)
            if "missing" in payload:
                log.warning("openmeteo: run %s not in archive: %s", run, payload["missing"])
            if i % 50 == 0 or i == len(missing):
                log.info("openmeteo: %d / %d runs downloaded", i, len(missing))


def _read(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def load(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Long NWP table, the grid cell actually used per point, and the runs missing upstream.

    NWP columns: run_utc, valid_utc, lead_h, point, then one column per variable.
    """
    frames, cells, missing = [], {}, []
    for run in run_times(cfg):
        payload = _read(run_path(cfg.paths.raw, cfg.nwp.model, run))
        if "missing" in payload:
            missing.append(run.isoformat())
            continue
        for group in payload["groups"].values():
            for name, loc in zip(group["points"], group["response"], strict=True):
                hourly = loc["hourly"]
                valid = pd.to_datetime(hourly["time"], utc=True)
                frame = pd.DataFrame(
                    {
                        "run_utc": run,
                        "valid_utc": valid,
                        "lead_h": ((valid - run) / pd.Timedelta(hours=1)).astype(np.int16),
                        "point": name,
                    }
                )
                for var in cfg.nwp.variables:
                    frame[var] = np.asarray(hourly[var], dtype=np.float32)
                frames.append(frame)
                cells[name] = (loc["latitude"], loc["longitude"], loc["elevation"])

    nwp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not nwp.empty:
        nwp["point"] = nwp["point"].astype("category")
    return nwp, _points_table(cfg, cells), missing


def _points_table(cfg: Config, cells: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    rows = []
    for name, p in cfg.points.items():
        if name not in cells:
            continue
        lat, lon, elevation = cells[name]
        dist = math.hypot(
            (lat - p.lat) * 111.2, (lon - p.lon) * 111.2 * math.cos(math.radians(p.lat))
        )
        rows.append(
            {
                "point": name,
                "region": p.region,
                "lat": p.lat,
                "lon": p.lon,
                "grid_lat": lat,
                "grid_lon": lon,
                "grid_elevation_m": elevation,
                "distance_km": round(dist, 2),
            }
        )
    return pd.DataFrame(rows)
