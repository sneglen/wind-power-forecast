"""The forecast problem: one row per issue time x target hour, built from the information set.

A forecast for local day D is issued on D-1 at ``[issue] time_local``. Its rows run from the
first hour after the issue to the end of D: k = 1..13 on D-1, then the day-ahead product
k = 14..37 (36 or 38 on DST days). Horizon k counts to the END of the hour-beginning stamp.

Columns (``KEYS``, inputs, ``TARGETS``):

- keys: target day, period, issue time, target hour, horizon, ``in_product`` (hour of day D)
- inputs, known at issue time: ``y_issue`` and ``capacity_issue_mw`` at the last known hour
  ``last_known_utc``, ``run_utc`` and ``lead_h`` of the NWP run used, NWP columns
  ``<variable>@<point|region>`` for the target hour and ``<variable>@<region>@issue`` for
  the last known hour (same run)
- ensemble inputs, if the ``nwp_ens`` table exists (Step 8): ``ens_run_utc`` and, per ENS
  member m, ``ens<mm>_<variable>@<region>`` for 100 m speed and direction, plus the member
  mean and standard deviation of the regional speeds, ``ens_mean_...`` and ``ens_std_...``
- targets, revealed after the target hour: ``y`` = P / C, ``power_mw``, ``capacity_mw``,
  ``curtailed`` (day-ahead price <= ``[curtailment] da_price_max``)

Information set at issue time t0:

- an observation stamped s covers [s, s + 1 h) and is usable from s + 1 h + power_lag_h;
  capacity is read like production, at the last hour known at t0;
- an NWP run r is usable from r + availability_lag_h. The latest usable run that covers
  every target hour is taken: normally the 00 UTC run of D-1, else that of D-2; the same
  rule, with its own lag, picks the ENS run;
- day-ahead prices are published after t0, so they only flag curtailed target hours.

NWP values are instantaneous. The mean over the hour [t, t + 1 h) is approximated by the
mean of the values at t and t + 1 h; wind is averaged as u/v components and the direction
derived from the means. Gusts are maxima over the preceding hour: the value at t + 1 h.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast.config import Config
from wpforecast.data import build

log = logging.getLogger(__name__)

FEATURES = "features.parquet"
KEYS = ["target_day", "period", "issue_time_utc", "time_utc", "horizon_h", "in_product"]
TARGETS = ["y", "power_mw", "capacity_mw", "curtailed"]
HOUR = pd.Timedelta(hours=1)
SPEED, DIRECTION, GUSTS = "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m"
ENS = "nwp_ens"


# --- issue times, horizons and periods ---------------------------------------------------


def issue_time_utc(cfg: Config, target_day: dt.date) -> pd.Timestamp:
    """Instant at which the forecast for local ``target_day`` is issued (day before, local)."""
    local = dt.datetime.combine(target_day - dt.timedelta(days=1), cfg.issue.time_local)
    return pd.Timestamp(local, tz=cfg.issue.tz).tz_convert("UTC")


def target_hours_utc(cfg: Config, target_day: dt.date) -> pd.DatetimeIndex:
    """Hour-beginning UTC stamps of local ``target_day`` (23, 24 or 25 hours with DST)."""
    start = pd.Timestamp(target_day, tz=cfg.issue.tz)
    end = pd.Timestamp(target_day + dt.timedelta(days=1), tz=cfg.issue.tz)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def issue_hours_utc(cfg: Config, target_day: dt.date) -> pd.DatetimeIndex:
    """Every hour from the issue time to the end of ``target_day``: k = 1 .. end of D."""
    end = target_hours_utc(cfg, target_day)[-1]
    return pd.date_range(issue_time_utc(cfg, target_day), end, freq="h")


def horizons(issue: pd.Timestamp, targets: pd.DatetimeIndex) -> np.ndarray:
    """Horizon k (hours, to the end of each hour-beginning target stamp)."""
    return ((targets - issue) / HOUR).to_numpy() + 1


def period(cfg: Config, day: dt.date) -> str:
    """Which period a target day belongs to; days before ``train_start`` are warm-up."""
    d = cfg.dates
    if day < d.train_start:
        return "warmup"
    if day < d.validation_start:
        return "train"
    if day <= d.train_end:
        return "validation"
    if day <= d.test_end:
        return "test"
    return "stress"


def last_day(cfg: Config, index: pd.DatetimeIndex) -> dt.date:
    """Last target day: stress_end if set, else the last local day fully covered by data."""
    if cfg.dates.stress_end is not None:
        return cfg.dates.stress_end
    day = index[-1].tz_convert(cfg.issue.tz).date()
    return day if target_hours_utc(cfg, day)[-1] <= index[-1] else day - dt.timedelta(days=1)


def last_known_stamp(cfg: Config, issue: pd.Timestamp | pd.Series) -> pd.Timestamp | pd.Series:
    """Stamp of the last observed hour usable at ``issue``."""
    return issue - HOUR * (1 + cfg.issue.power_lag_h)


def skeleton(cfg: Config, days: list[dt.date]) -> pd.DataFrame:
    """Key columns for every issue time and target hour."""
    frames = []
    for day in days:
        issue = issue_time_utc(cfg, day)
        hours = issue_hours_utc(cfg, day)
        k = horizons(issue, hours)
        if k.max() > cfg.issue.max_horizon_h:
            raise ValueError(f"target day {day} needs k = {k.max():g} > max_horizon_h")
        frames.append(
            pd.DataFrame(
                {
                    "target_day": day,
                    "period": period(cfg, day),
                    "issue_time_utc": issue,
                    "time_utc": hours,
                    "horizon_h": k.astype(np.int16),
                    "in_product": hours >= target_hours_utc(cfg, day)[0],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- NWP --------------------------------------------------------------------------------


def _uv(speed: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Wind components from speed and meteorological direction (where the wind comes from)."""
    rad = np.deg2rad(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def _direction(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.rad2deg(np.arctan2(-u, -v)) % 360.0


def nwp_hourly(cfg: Config, nwp: pd.DataFrame) -> pd.DataFrame:
    """Hour means per (run_utc, time_utc): columns ``<variable>@<point>`` and ``@<region>``."""
    df = nwp.copy()
    df["u_100m"], df["v_100m"] = _uv(df[SPEED], df[DIRECTION])
    variables = [v for v in cfg.nwp.variables if v != DIRECTION] + ["u_100m", "v_100m"]
    long = df.set_index(["run_utc", "valid_utc", "point"])[variables]
    # A unique index, so a pure reshape: nothing to aggregate as pivot_table would.
    wide = long.unstack("point").astype(np.float64).sort_index()  # noqa: PD010
    following = wide.groupby(level="run_utc").shift(-1)  # values at t + 1 h, same run
    hourly = (wide + following) / 2
    if GUSTS in variables:
        hourly[GUSTS] = following[GUSTS]

    regions = {}
    for region in ("onshore", "offshore"):
        names = [n for n, p in cfg.points.items() if p.region == region]
        for var in variables:
            regions[(var, region)] = hourly[var][names].mean(axis=1)
    hourly = pd.concat([hourly, pd.DataFrame(regions)], axis=1)
    for name in hourly["u_100m"].columns:
        hourly[(DIRECTION, name)] = _direction(hourly[("u_100m", name)], hourly[("v_100m", name)])

    hourly.columns = [f"{var}@{name}" for var, name in hourly.columns]
    hourly.index = hourly.index.set_names(["run_utc", "time_utc"])
    return hourly.reset_index()


def choose_runs(
    cfg: Config, keys: pd.DataFrame, runs: pd.DatetimeIndex, ensemble: bool = False
) -> pd.Series:
    """Per issue time, the latest run usable at issue time that covers all its target hours."""
    nwp = cfg.nwp.ensemble if ensemble else cfg.nwp
    lag = pd.Timedelta(hours=nwp.availability_lag_h)
    reach = pd.Timedelta(hours=nwp.max_lead_h)
    runs = runs.sort_values()
    last_end = keys.groupby("issue_time_utc")["time_utc"].max() + HOUR
    chosen = {}
    for issue, end in last_end.items():
        usable = runs[(runs + lag <= issue) & (runs + reach >= end)]
        chosen[issue] = usable[-1] if len(usable) else pd.NaT
    return pd.Series(chosen, name="run_utc").rename_axis("issue_time_utc")


def member_prefix(member: int) -> str:
    return f"ens{member:02d}_"


def ensemble_members(columns) -> list[int]:
    """Members with columns in a feature table (empty without ENS inputs)."""
    found = {int(c[3:5]) for c in columns if c.startswith("ens") and c[3:5].isdigit()}
    return sorted(found)


def ensemble_inputs(cfg: Config, rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per row, the members of the ENS run usable at its issue time, plus their mean and
    standard deviation of the regional speeds."""
    runs = choose_runs(cfg, rows, pd.DatetimeIndex(table["run_utc"].unique()), ensemble=True)
    values = [c for c in table.columns if "@" in c]
    # A unique index, so a pure reshape: nothing to aggregate as pivot_table would.
    wide = table.set_index(["run_utc", "time_utc", "member"])[values].unstack("member")  # noqa: PD010
    wide.columns = [f"{member_prefix(m)}{var}" for var, m in wide.columns]
    for region in ("onshore", "offshore"):
        speeds = wide[[c for c in wide.columns if c.endswith(f"{SPEED}@{region}")]]
        wide[f"ens_mean_{SPEED}@{region}"] = speeds.mean(axis=1)
        wide[f"ens_std_{SPEED}@{region}"] = speeds.std(axis=1)
    wide = wide.astype(np.float32).rename_axis(["ens_run_utc", "time_utc"]).reset_index()
    keys = rows[["issue_time_utc", "time_utc"]].assign(
        ens_run_utc=rows["issue_time_utc"].map(runs).astype(table["run_utc"].dtype)
    )
    return keys.merge(wide, on=["ens_run_utc", "time_utc"], how="left").drop(
        columns=["issue_time_utc", "time_utc"]
    )


# --- the feature table -------------------------------------------------------------------


def build_features(
    cfg: Config,
    dataset: pd.DataFrame,
    nwp: pd.DataFrame,
    price: pd.DataFrame,
    ens: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pure function of the interim tables; see the module docstring for the rules."""
    first = cfg.data.start + dt.timedelta(days=1)
    days = list(pd.date_range(first, last_day(cfg, dataset.index), freq="D").date)
    rows = skeleton(cfg, days)

    known = last_known_stamp(cfg, rows["issue_time_utc"])
    rows["y_issue"] = dataset["power_norm"].reindex(known).to_numpy()
    rows["capacity_issue_mw"] = dataset["capacity_mw"].reindex(known).to_numpy()

    runs = choose_runs(cfg, rows, pd.DatetimeIndex(nwp["run_utc"].unique()))
    rows["run_utc"] = rows["issue_time_utc"].map(runs).astype(nwp["run_utc"].dtype)
    rows["lead_h"] = ((rows["time_utc"] + HOUR - rows["run_utc"]) / HOUR).astype("Int16")
    hourly = nwp_hourly(cfg, nwp)
    rows = rows.merge(hourly, on=["run_utc", "time_utc"], how="left")

    # The same run at the last known hour: lets a model compare its fit with y_issue.
    regional = [c for c in hourly.columns if c.endswith(("@onshore", "@offshore"))]
    at_issue = hourly[["run_utc", "time_utc", *regional]].rename(
        columns={"time_utc": "last_known_utc"} | {c: f"{c}@issue" for c in regional}
    )
    rows["last_known_utc"] = known.astype(hourly["time_utc"].dtype)
    rows = rows.merge(at_issue, on=["run_utc", "last_known_utc"], how="left")
    no_run = runs.isna().sum()
    if no_run:
        log.warning("features: %d issue times have no usable NWP run", no_run)

    if ens is not None:
        rows = pd.concat([rows, ensemble_inputs(cfg, rows, ens)], axis=1)

    rows["y"] = dataset["power_norm"].reindex(rows["time_utc"]).to_numpy()
    rows["power_mw"] = dataset["power_mw"].reindex(rows["time_utc"]).to_numpy()
    rows["capacity_mw"] = dataset["capacity_mw"].reindex(rows["time_utc"]).to_numpy()
    da = price["price_eur_mwh"].reindex(rows["time_utc"]).to_numpy()
    rows["curtailed"] = da <= cfg.curtailment.da_price_max  # NaN price -> False

    _assert_information_set(cfg, rows)
    return rows


def input_columns(features: pd.DataFrame) -> list[str]:
    """Columns a model may use at issue time: everything except the targets."""
    return [c for c in features.columns if c not in TARGETS]


def _assert_information_set(cfg: Config, rows: pd.DataFrame) -> None:
    issue = rows["issue_time_utc"]
    if not (rows["time_utc"] >= issue).all():
        raise RuntimeError("a target hour starts before its issue time")
    lag = pd.Timedelta(hours=cfg.nwp.availability_lag_h)
    has_run = rows["run_utc"].notna()
    if not (rows.loc[has_run, "run_utc"] + lag <= issue[has_run]).all():
        raise RuntimeError("an NWP run is used before it is available")
    if "ens_run_utc" in rows:
        lag = pd.Timedelta(hours=cfg.nwp.ensemble.availability_lag_h)
        has_run = rows["ens_run_utc"].notna()
        if not (rows.loc[has_run, "ens_run_utc"] + lag <= issue[has_run]).all():
            raise RuntimeError("an ENS run is used before it is available")
    if any("price" in c for c in input_columns(rows)):
        raise RuntimeError("prices are published after the issue time; never an input")


def run(cfg: Config) -> Path:
    """Build the feature table from the interim tables and write it next to them."""
    ens = cfg.paths.interim / f"{ENS}.parquet"
    table = build_features(
        cfg,
        build.read(cfg, "dataset"),
        build.read(cfg, "nwp_hres"),
        build.read(cfg, "da_price"),
        pd.read_parquet(ens) if ens.exists() else None,
    )
    out = cfg.paths.interim / FEATURES
    table.to_parquet(out, index=False)
    log.info(
        "features: %d issue times, %d rows, %d columns -> %s",
        table["issue_time_utc"].nunique(),
        len(table),
        table.shape[1],
        out,
    )
    return out


def read(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.interim / FEATURES
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make features` first.")
    return pd.read_parquet(path)
