"""Typed, validated access to ``config.toml``.

A config file may set ``base = "<path>"`` (relative to itself) to inherit from another
config and override only selected keys; tables are merged recursively. Data paths in
``[paths]`` are relative to the working directory, i.e. the repository root.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REGIMES = ("frozen", "walk_forward", "recursive")
QR_SOLVERS = ("irls", "highs")


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    outputs: Path
    figures: Path
    reports: Path


@dataclass(frozen=True)
class EDS:
    base_url: str
    min_interval_s: float


@dataclass(frozen=True)
class OpenMeteo:
    base_url: str
    calls_per_hour: float
    calls_per_day: float
    calls_per_location: float
    min_interval_s: float
    max_wait_min: float


@dataclass(frozen=True)
class Data:
    start: dt.date
    end: dt.date
    offline: bool
    price_area: str
    eds: EDS
    openmeteo: OpenMeteo


@dataclass(frozen=True)
class Dates:
    train_start: dt.date
    validation_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    stress_start: dt.date
    stress_end: dt.date | None = None  # None: up to the latest day in the data


@dataclass(frozen=True)
class Issue:
    timezone: str
    time_local: dt.time
    gate_closure_local: dt.time
    max_horizon_h: int
    power_lag_h: int

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class Ensemble:
    source: str
    dataset: str
    start: dt.date
    variables: tuple[str, ...]
    members: int
    resolution_deg: float
    step_h: int
    max_lead_h: int
    workers: int
    availability_lag_h: int


@dataclass(frozen=True)
class NWP:
    source: str
    model: str
    variables: tuple[str, ...]
    cycle_utc: int
    availability_lag_h: int
    max_lead_h: int
    ensemble: Ensemble


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float
    region: str  # "onshore" | "offshore"


@dataclass(frozen=True)
class Target:
    resolution: str


@dataclass(frozen=True)
class Curtailment:
    da_price_max: float


@dataclass(frozen=True)
class Quantiles:
    levels: tuple[float, ...]
    crps_levels: int
    tail_levels: tuple[float, ...]


@dataclass(frozen=True)
class Adaptivity:
    regimes: tuple[str, ...]
    walk_forward_refit: str


@dataclass(frozen=True)
class Baselines:
    min_error_samples: int


@dataclass(frozen=True)
class RLS:
    forgetting_grid: tuple[float, ...]
    delta: float
    spline_knots: int
    spline_degree: int
    speed_max_ms: dict[str, float]  # per region
    diurnal_harmonics: int
    horizon_blocks: tuple[int, ...]
    error_window_days: int


@dataclass(frozen=True)
class Probabilistic:
    min_train_rows: int


@dataclass(frozen=True)
class QuantReg:
    window_days: int
    refit_days: int
    spline_knots: int
    spline_degree: int
    solver: str


@dataclass(frozen=True)
class LogitNormal:
    nu: float
    epsilon: float


@dataclass(frozen=True)
class GBM:
    learning_rate: float
    num_leaves: int
    n_estimators: int
    min_child_samples: int
    num_threads: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class Model:
    baselines: Baselines
    rls: RLS
    probabilistic: Probabilistic
    quantreg: QuantReg
    logitnormal: LogitNormal
    gbm: GBM


@dataclass(frozen=True)
class Evaluation:
    bootstrap_block_days: int
    n_bootstrap: int
    band: float
    intervals: tuple[float, ...]
    pit_bins: int
    min_hours_per_horizon: int
    dm_max_lag: int
    seed: int
    spread_bins: int


@dataclass(frozen=True)
class Market:
    area: str
    pricing: str
    portfolio_mw: float
    start: dt.date
    end: dt.date
    tau_lookback_days: int
    tau_min_days: int
    reference_model: str
    spike_eur_mwh: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: Data
    dates: Dates
    issue: Issue
    nwp: NWP
    points: dict[str, Point]
    target: Target
    curtailment: Curtailment
    quantiles: Quantiles
    adaptivity: Adaptivity
    model: Model
    evaluation: Evaluation
    market: Market


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _merge(_read((path.parent / base).resolve()), raw)


def _table(raw: dict[str, Any], dotted: str) -> dict[str, Any]:
    table = raw
    for key in dotted.split("."):
        if not isinstance(table.get(key), dict):
            raise ValueError(f"config is missing table [{dotted}]")
        table = table[key]
    return table


def _build[T](cls: type[T], raw: dict[str, Any], dotted: str, /, **override: Any) -> T:
    table = {k: v for k, v in _table(raw, dotted).items() if not isinstance(v, dict)}
    try:
        return cls(**(table | override))
    except TypeError as exc:
        raise ValueError(f"[{dotted}] has unexpected or missing keys: {exc}") from exc


def _parse_time(value: str, key: str) -> dt.time:
    try:
        return dt.time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"[issue] {key} must be HH:MM, got {value!r}") from exc


def _parse_points(raw: dict[str, Any]) -> dict[str, Point]:
    points: dict[str, Point] = {}
    for region in ("onshore", "offshore"):
        for name, latlon in _table(raw, f"points.{region}").items():
            if name in points:
                raise ValueError(f"[points] duplicate point name {name!r}")
            points[name] = Point(float(latlon[0]), float(latlon[1]), region)
    return points


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _issue_hours_utc(issue: Issue) -> list[float]:
    """Issue time in UTC hours of day, in winter and in summer."""
    hours = []
    for day in (dt.date(2025, 1, 15), dt.date(2025, 7, 15)):
        local = dt.datetime.combine(day, issue.time_local, tzinfo=issue.tz)
        utc = local.astimezone(dt.UTC)
        hours.append(utc.hour + utc.minute / 60)
    return hours


def _validate(cfg: Config) -> None:
    d = cfg.dates
    _check(
        d.train_start
        <= d.validation_start
        <= d.train_end
        < d.test_start
        <= d.test_end
        < d.stress_start
        <= (d.stress_end or d.stress_start),
        "[dates] must satisfy train_start <= validation_start <= train_end < test_start"
        " <= test_end < stress_start <= stress_end",
    )

    data = cfg.data
    _check(data.start < d.train_start, "[data] start must precede [dates] train_start")
    _check(data.start <= data.end, "[data] start must not be after end")
    _check(
        d.stress_end is None or d.stress_end <= data.end, "[dates] stress_end is after [data] end"
    )
    om = data.openmeteo
    _check(0 < om.calls_per_hour <= om.calls_per_day, "[data.openmeteo] call budgets inconsistent")

    issue = cfg.issue
    try:
        issue.tz  # noqa: B018 — resolve the zone to validate it
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"[issue] unknown timezone {issue.timezone!r}") from exc
    _check(issue.time_local < issue.gate_closure_local, "[issue] must precede gate closure")
    _check(issue.max_horizon_h >= 1, "[issue] max_horizon_h must be >= 1")
    _check(issue.power_lag_h >= 0, "[issue] power_lag_h must be >= 0")

    nwp = cfg.nwp
    _check(0 <= nwp.cycle_utc < 24, "[nwp] cycle_utc must be in 0..23")
    _check(
        {"wind_speed_100m", "wind_direction_100m"} <= set(nwp.variables),
        "[nwp] variables must include wind_speed_100m and wind_direction_100m",
    )
    hours = _issue_hours_utc(issue)
    _check(
        nwp.cycle_utc + nwp.availability_lag_h <= min(hours),
        "[nwp] the run must be available by the issue time (summer and winter)",
    )
    _check(
        max(hours) - nwp.cycle_utc + issue.max_horizon_h <= nwp.max_lead_h,
        "[nwp] max_lead_h does not cover max_horizon_h",
    )

    ens = nwp.ensemble
    _check(len(ens.variables) > 0, "[nwp.ensemble] variables must not be empty")
    _check(ens.start <= data.end, "[nwp.ensemble] start is after [data] end")
    _check(ens.max_lead_h % ens.step_h == 0, "[nwp.ensemble] max_lead_h must be k * step_h")
    _check(ens.workers >= 1, "[nwp.ensemble] workers must be >= 1")
    _check(ens.availability_lag_h >= 0, "[nwp.ensemble] availability_lag_h must be >= 0")

    _check(any(p.region == "onshore" for p in cfg.points.values()), "[points] need onshore points")
    for name, p in cfg.points.items():
        _check(
            -90 <= p.lat <= 90 and -180 <= p.lon <= 180, f"[points] {name} is not a valid lat/lon"
        )

    q = cfg.quantiles.levels
    _check(len(q) > 0, "[quantiles] levels must not be empty")
    _check(all(0 < x < 1 for x in q), "[quantiles] levels must lie strictly inside (0, 1)")
    _check(all(a < b for a, b in zip(q, q[1:], strict=False)), "[quantiles] must be increasing")
    n = cfg.quantiles.crps_levels + 1
    _check(cfg.quantiles.crps_levels >= 2, "[quantiles] crps_levels must be >= 2")
    tails = cfg.quantiles.tail_levels
    _check(
        all(abs(x * n - round(x * n)) < 1e-9 and 0 < x < 1 for x in q + tails),
        "[quantiles] levels and tail_levels must lie on the CRPS grid 1/n .. (n-1)/n,"
        " n = crps_levels + 1",
    )
    grid = (1 / n, (n - 1) / n)
    fitted = sorted(set(q) | set(tails))
    _check(
        abs(fitted[0] - grid[0]) < 1e-9 and abs(fitted[-1] - grid[1]) < 1e-9,
        "[quantiles] tail_levels must include the ends of the CRPS grid (no extrapolation)",
    )

    _check(
        set(cfg.adaptivity.regimes) <= set(REGIMES), f"[adaptivity] regimes must be in {REGIMES}"
    )

    m = cfg.model
    _check(m.baselines.min_error_samples >= 1, "[model.baselines] min_error_samples must be >= 1")
    rls = m.rls
    _check(len(rls.forgetting_grid) > 0, "[model.rls] forgetting_grid must not be empty")
    _check(all(0 < x <= 1 for x in rls.forgetting_grid), "[model.rls] forgetting in (0, 1]")
    _check(rls.delta > 0, "[model.rls] delta must be > 0")
    _check(rls.spline_knots >= 1 and rls.spline_degree >= 1, "[model.rls] spline too small")
    _check(
        set(rls.speed_max_ms) == {"onshore", "offshore"}
        and all(v > 0 for v in rls.speed_max_ms.values()),
        "[model.rls] speed_max_ms needs positive onshore and offshore values",
    )
    blocks = rls.horizon_blocks
    _check(
        len(blocks) > 0
        and blocks[0] == 1
        and all(a < b for a, b in zip(blocks, blocks[1:], strict=False)),
        "[model.rls] horizon_blocks must start at 1 and increase",
    )
    _check(rls.error_window_days >= 1, "[model.rls] error_window_days must be >= 1")
    _check(m.probabilistic.min_train_rows >= 1, "[model.probabilistic] min_train_rows >= 1")
    _check(m.quantreg.refit_days <= m.quantreg.window_days, "[model.quantreg] refit > window")
    _check(m.quantreg.spline_knots >= 1, "[model.quantreg] spline_knots must be >= 1")
    _check(m.quantreg.solver in QR_SOLVERS, f"[model.quantreg] solver must be one of {QR_SOLVERS}")
    _check(len(m.gbm.features) > 0, "[model.gbm] features must not be empty")
    _check(m.logitnormal.nu > 0, "[model.logitnormal] nu must be > 0")
    _check(0 < m.logitnormal.epsilon < 0.5, "[model.logitnormal] epsilon must be in (0, 0.5)")

    ev = cfg.evaluation
    _check(
        ev.bootstrap_block_days >= 1 and ev.n_bootstrap >= 10, "[evaluation] bootstrap too small"
    )
    _check(0 < ev.band < 1, "[evaluation] band must be in (0, 1)")
    _check(all(0 < x < 1 for x in ev.intervals), "[evaluation] intervals must be in (0, 1)")
    _check(
        all(
            any(abs(level - (1 - x) / 2) < 1e-9 for level in cfg.quantiles.levels)
            for x in ev.intervals
        ),
        "[evaluation] each interval's lower end must be a reporting quantile level",
    )
    _check(ev.pit_bins >= 2 and ev.dm_max_lag >= 0, "[evaluation] pit_bins >= 2, dm_max_lag >= 0")
    _check(ev.spread_bins >= 2, "[evaluation] spread_bins must be >= 2")

    mk = cfg.market
    _check(mk.pricing == "single", "[market] only single pricing is supported")
    _check(mk.portfolio_mw > 0, "[market] portfolio_mw must be > 0")
    _check(
        d.test_start <= mk.start <= mk.end <= d.test_end,
        "[market] window must lie inside the test period",
    )
    _check(
        1 <= mk.tau_min_days <= mk.tau_lookback_days,
        "[market] need 1 <= tau_min_days <= tau_lookback_days",
    )
    _check(mk.spike_eur_mwh > 0, "[market] spike_eur_mwh must be > 0")


def load_config(path: str | Path) -> Config:
    """Load, merge (via ``base``) and validate a config file."""
    raw = _read(Path(path).resolve())
    issue = _table(raw, "issue")
    cfg = Config(
        paths=_build(Paths, raw, "paths", **{k: Path(v) for k, v in _table(raw, "paths").items()}),
        data=_build(
            Data,
            raw,
            "data",
            eds=_build(EDS, raw, "data.eds"),
            openmeteo=_build(OpenMeteo, raw, "data.openmeteo"),
        ),
        dates=_build(Dates, raw, "dates"),
        issue=_build(
            Issue,
            raw,
            "issue",
            time_local=_parse_time(issue.get("time_local", ""), "time_local"),
            gate_closure_local=_parse_time(
                issue.get("gate_closure_local", ""), "gate_closure_local"
            ),
        ),
        nwp=_build(
            NWP,
            raw,
            "nwp",
            variables=tuple(_table(raw, "nwp").get("variables", [])),
            ensemble=_build(
                Ensemble,
                raw,
                "nwp.ensemble",
                variables=tuple(_table(raw, "nwp.ensemble").get("variables", [])),
            ),
        ),
        points=_parse_points(raw),
        target=_build(Target, raw, "target"),
        curtailment=_build(Curtailment, raw, "curtailment"),
        quantiles=_build(
            Quantiles,
            raw,
            "quantiles",
            levels=tuple(float(x) for x in _table(raw, "quantiles").get("levels", [])),
            tail_levels=tuple(float(x) for x in _table(raw, "quantiles").get("tail_levels", [])),
        ),
        adaptivity=_build(
            Adaptivity,
            raw,
            "adaptivity",
            regimes=tuple(_table(raw, "adaptivity").get("regimes", [])),
        ),
        model=Model(
            baselines=_build(Baselines, raw, "model.baselines"),
            rls=_build(
                RLS,
                raw,
                "model.rls",
                forgetting_grid=tuple(_table(raw, "model.rls").get("forgetting_grid", [])),
                speed_max_ms={
                    k: float(v) for k, v in _table(raw, "model.rls.speed_max_ms").items()
                },
                horizon_blocks=tuple(_table(raw, "model.rls").get("horizon_blocks", [])),
            ),
            probabilistic=_build(Probabilistic, raw, "model.probabilistic"),
            quantreg=_build(QuantReg, raw, "model.quantreg"),
            logitnormal=_build(LogitNormal, raw, "model.logitnormal"),
            gbm=_build(
                GBM,
                raw,
                "model.gbm",
                features=tuple(_table(raw, "model.gbm").get("features", [])),
            ),
        ),
        evaluation=_build(
            Evaluation,
            raw,
            "evaluation",
            intervals=tuple(float(x) for x in _table(raw, "evaluation").get("intervals", [])),
        ),
        market=_build(Market, raw, "market"),
    )
    _validate(cfg)
    return cfg
