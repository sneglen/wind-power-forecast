"""Installed wind capacity C(t) for a price area from Energinet's CapacityPerMunicipality.

The dataset is monthly, per municipality, valid from local midnight on the first day of
the month. Municipality codes map to price areas by geography: DK1 is Jutland and Funen
(codes 410 and above, except 411 Christiansø), DK2 the rest. Offshore farms outside any
municipality get codes below 101; each one is mapped explicitly, so a new code fails
loudly instead of being guessed.
"""

from __future__ import annotations

import pandas as pd

OUTSIDE_MUNICIPALITIES = {1: "DK1"}  # code 1: Thor (72 turbines, 1058.4 MW), from 2026-03


def price_area(code: str | int) -> str:
    c = int(code)
    if c in OUTSIDE_MUNICIPALITIES:
        return OUTSIDE_MUNICIPALITIES[c]
    if c < 101:
        raise ValueError(f"municipality code {c} is not a municipality; map it explicitly")
    return "DK1" if c >= 410 and c != 411 else "DK2"


def monthly(raw: pd.DataFrame, area: str) -> pd.DataFrame:
    """Onshore/offshore capacity (MW) per month for one price area, indexed by month start UTC."""
    df = raw[raw["MunicipalityNo"].map(price_area) == area]
    out = df.groupby("Month")[["OnshoreWindCapacity", "OffshoreWindCapacity"]].sum()
    out.columns = ["onshore_mw", "offshore_mw"]
    out["total_mw"] = out["onshore_mw"] + out["offshore_mw"]
    local = pd.to_datetime(out.index).tz_localize("Europe/Copenhagen")
    out.index = local.tz_convert("UTC").rename("month_start_utc")
    return out.sort_index()


def hourly(monthly_mw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Step function: each month's capacity holds from its start until the next month's."""
    combined = monthly_mw.reindex(monthly_mw.index.union(index)).ffill()
    return combined.reindex(index).rename_axis("time_utc")
