"""Build the interim tables (UTC, tidy column names, parquet) from the download cache.

Tables written to ``paths.interim``:

- ``production``: DK1 wind settlement, the four columns and their total (MW, hourly)
- ``capacity``: installed wind capacity C(t) (MW, hourly step function of monthly data)
- ``energinet_forecast``: Energinet's day-ahead and intraday wind forecasts (MW, hourly)
- ``da_price``: day-ahead price (EUR/MWh, hourly; 15-min prices averaged from 2025-10-01)
- ``imbalance``: imbalance price components (EUR/MWh, 15 min)
- ``nwp_hres``: HRES runs, long format (run_utc, valid_utc, lead_h, point, variables)
- ``points``: requested coordinates and the grid cells used
- ``dataset``: hourly target table (power, capacity, normalised power y = P / C)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from wpforecast.config import Config
from wpforecast.data import capacity, eds, openmeteo

log = logging.getLogger(__name__)

PRODUCTION_COLUMNS = {
    "OffshoreWindLt100MW_MWh": "offshore_lt100mw_mw",
    "OffshoreWindGe100MW_MWh": "offshore_ge100mw_mw",
    "OnshoreWindLt50kW_MWh": "onshore_lt50kw_mw",
    "OnshoreWindGe50kW_MWh": "onshore_ge50kw_mw",
}
FORECAST_TYPES = {"Onshore Wind": "onshore", "Offshore Wind": "offshore"}
IMBALANCE_COLUMNS = {
    "ImbalancePriceEUR": "imbalance_price_eur_mwh",
    "SpotPriceEUR": "spot_price_eur_mwh",
    "DominatingDirection": "dominating_direction",
    "SatisfiedDemand": "satisfied_demand_mw",
    "mFRRMarginalPriceUpEUR": "mfrr_up_eur_mwh",
    "mFRRMarginalPriceDownEUR": "mfrr_down_eur_mwh",
    "aFRRVWAUpEUR": "afrr_vwa_up_eur_mwh",
    "aFRRVWADownEUR": "afrr_vwa_down_eur_mwh",
}


def _utc(values: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True), name="time_utc")


def hourly_index(cfg: Config) -> pd.DatetimeIndex:
    """Every UTC hour of the downloaded days [data.start, data.end]."""
    first = pd.Timestamp(cfg.data.start, tz="UTC")
    last = pd.Timestamp(cfg.data.end, tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(first, last, freq="h", name="time_utc")


def production(cfg: Config) -> pd.DataFrame:
    raw = eds.load(cfg, "production")
    df = raw[list(PRODUCTION_COLUMNS)].rename(columns=PRODUCTION_COLUMNS)
    df.index = _utc(raw["HourUTC"])
    df["wind_mw"] = df.sum(axis=1, min_count=len(PRODUCTION_COLUMNS))
    return df.sort_index()


def capacity_hourly(cfg: Config, index: pd.DatetimeIndex) -> pd.DataFrame:
    monthly = capacity.monthly(eds.load(cfg, "capacity"), cfg.data.price_area)
    return capacity.hourly(monthly, index)


def energinet_forecast(cfg: Config) -> pd.DataFrame:
    raw = eds.load(cfg, "energinet_forecast")
    raw = raw[raw["ForecastType"].isin(FORECAST_TYPES)]
    wide = raw.pivot_table(
        index="HourUTC",
        columns="ForecastType",
        values=["ForecastDayAhead", "ForecastIntraday"],
        aggfunc="first",
    )
    out = pd.DataFrame(index=_utc(wide.index.to_series()))
    for field, label in (("ForecastDayAhead", "da"), ("ForecastIntraday", "intraday")):
        for kind, name in FORECAST_TYPES.items():
            out[f"{name}_{label}_mw"] = wide[(field, kind)].to_numpy()
        out[f"total_{label}_mw"] = out[f"onshore_{label}_mw"] + out[f"offshore_{label}_mw"]
    return out.sort_index()


def da_price(cfg: Config) -> pd.DataFrame:
    """Hourly day-ahead price: Elspot (hourly) to 2025-09-30, then hourly means of 15-min."""
    parts = []
    elspot = eds.load(cfg, "elspot")
    if not elspot.empty:
        parts.append(
            pd.DataFrame(
                {"price_eur_mwh": elspot["SpotPriceEUR"].to_numpy(), "source": "Elspotprices"},
                index=_utc(elspot["HourUTC"]),
            )
        )
    quarter = eds.load(cfg, "day_ahead")
    if not quarter.empty:
        q = pd.Series(quarter["DayAheadPriceEUR"].to_numpy(), index=_utc(quarter["TimeUTC"]))
        hourly = q.resample("h").agg(["mean", "count"])
        parts.append(
            pd.DataFrame(
                {
                    "price_eur_mwh": hourly["mean"],
                    "source": "DayAheadPrices",
                    "n_15min": hourly["count"],
                }
            )
        )
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def imbalance(cfg: Config) -> pd.DataFrame:
    raw = eds.load(cfg, "imbalance")
    if raw.empty:
        return pd.DataFrame(columns=list(IMBALANCE_COLUMNS.values()), index=_utc(pd.Series([])))
    df = raw[list(IMBALANCE_COLUMNS)].rename(columns=IMBALANCE_COLUMNS)
    df.index = _utc(raw["TimeUTC"])
    return df.sort_index()


def dataset(prod: pd.DataFrame, cap: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Regular hourly grid over the downloaded range; missing hours stay NaN."""
    out = pd.DataFrame(index=index)
    out["power_mw"] = prod["wind_mw"].reindex(index)
    out["capacity_mw"] = cap["total_mw"].reindex(index)
    out["power_norm"] = (out["power_mw"] / out["capacity_mw"]).clip(0.0, 1.0)
    return out


def build(cfg: Config) -> dict[str, Path]:
    """Write all interim tables; returns their paths."""
    prod = production(cfg)
    nwp, points, missing_runs = openmeteo.load(cfg)
    index = hourly_index(cfg)
    cap = capacity_hourly(cfg, index)
    tables = {
        "production": prod,
        "capacity": cap,
        "energinet_forecast": energinet_forecast(cfg),
        "da_price": da_price(cfg),
        "imbalance": imbalance(cfg),
        "nwp_hres": nwp,
        "points": points,
        "dataset": dataset(prod, cap, index),
    }
    out_dir = cfg.paths.interim
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        paths[name] = out_dir / f"{name}.parquet"
        table.to_parquet(paths[name], index=name not in ("nwp_hres", "points"))
    (out_dir / "nwp_missing_runs.json").write_text(json.dumps(missing_runs, indent=2) + "\n")
    log.info(
        "build: %d production hours, %d NWP rows (%d runs missing) -> %s",
        int(prod["wind_mw"].notna().sum()),
        len(nwp),
        len(missing_runs),
        out_dir,
    )
    return paths


def read(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.interim / f"{name}.parquet")
