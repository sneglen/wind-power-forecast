"""Energi Data Service (Energinet) downloads, cached per dataset and calendar-year chunk.

Each chunk is stored as received (all columns) in
``<raw>/eds/<Dataset>/<first>_<last>.parquet`` (UTC days, inclusive), so a re-run touches
the network only for chunks that are missing. ``manifest.json`` records when each chunk
was downloaded: settlement data is revised for months, so the download date matters.
Data licence: CC BY 4.0, "Source: Energinet (www.energidataservice.dk)".
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from wpforecast.config import Config
from wpforecast.data import http

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    name: str
    time_col: str  # the dataset's filter column; UTC unless monthly
    area_filter: bool = True
    monthly: bool = False


DATASETS = {
    "production": Dataset("ProductionConsumptionSettlement", "HourUTC"),
    "energinet_forecast": Dataset("Forecasts_Hour", "HourUTC"),
    "elspot": Dataset("Elspotprices", "HourUTC"),  # hourly, discontinued after 2025-09-30
    "day_ahead": Dataset("DayAheadPrices", "TimeUTC"),  # 15 min, from 2025-10-01
    "imbalance": Dataset("ImbalancePrice", "TimeUTC"),  # 15 min, from 2025-03-04
    "capacity": Dataset("CapacityPerMunicipality", "Month", area_filter=False, monthly=True),
}


def chunks(dataset: Dataset, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Calendar-year pieces of [start, end]; monthly data starts a month early, in one piece."""
    if dataset.monthly:
        first = (start.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
        return [(first, end)]
    return [
        (max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


def chunk_path(raw: Path, dataset: Dataset, first: dt.date, last: dt.date) -> Path:
    return raw / "eds" / dataset.name / f"{first}_{last}.parquet"


def _params(cfg: Config, dataset: Dataset, first: dt.date, last: dt.date) -> dict[str, str]:
    params = {
        "start": f"{first}T00:00",
        "end": f"{last + dt.timedelta(days=1)}T00:00",  # exclusive
        "timezone": "UTC",
        "sort": f"{dataset.time_col} ASC",
        "limit": "0",  # all rows
    }
    if dataset.area_filter:
        params["filter"] = json.dumps({"PriceArea": [cfg.data.price_area]})
    return params


def _update_manifest(raw: Path, key: str, entry: dict) -> None:
    path = raw / "eds" / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[key] = entry
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def download(cfg: Config, client: httpx.Client | None = None) -> None:
    """Fetch every missing chunk of every dataset for [data.start, data.end]."""
    raw, throttle = cfg.paths.raw, http.Throttle(cfg.data.eds.min_interval_s)
    missing = [
        (dataset, first, last)
        for dataset in DATASETS.values()
        for first, last in chunks(dataset, cfg.data.start, cfg.data.end)
        if not chunk_path(raw, dataset, first, last).exists()
    ]
    if not missing:
        return
    if cfg.data.offline:
        names = [str(chunk_path(raw, *m).relative_to(raw)) for m in missing]
        raise FileNotFoundError(f"offline mode, but {len(names)} EDS chunks are missing: {names}")

    with client or http.client() as session:
        for dataset, first, last in missing:
            throttle.wait()
            url = f"{cfg.data.eds.base_url}/dataset/{dataset.name}"
            response = http.get(session, url, _params(cfg, dataset, first, last))
            response.raise_for_status()
            records = response.json()["records"]
            path = chunk_path(raw, dataset, first, last)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(records).to_parquet(path, index=False)
            _update_manifest(
                raw,
                str(path.relative_to(raw)),
                {"rows": len(records), "downloaded_at": dt.datetime.now(dt.UTC).isoformat()},
            )
            log.info("eds: %s %s..%s: %d rows", dataset.name, first, last, len(records))


def load(cfg: Config, key: str) -> pd.DataFrame:
    """All cached chunks of one dataset for the configured range, concatenated."""
    dataset = DATASETS[key]
    frames = [
        pd.read_parquet(chunk_path(cfg.paths.raw, dataset, first, last))
        for first, last in chunks(dataset, cfg.data.start, cfg.data.end)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
