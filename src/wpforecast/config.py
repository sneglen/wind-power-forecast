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


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    outputs: Path
    figures: Path


@dataclass(frozen=True)
class Dates:
    start: dt.date
    train_end: dt.date
    test_start: dt.date
    end: dt.date


@dataclass(frozen=True)
class Issue:
    timezone: str
    time_local: dt.time
    gate_closure_local: dt.time
    nwp_cycle_utc: int
    nwp_availability_lag_h: int
    power_lag_h: int

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class Climatology:
    window_days: int


@dataclass(frozen=True)
class RLS:
    forgetting: float
    delta: float


@dataclass(frozen=True)
class AdaptiveQuantile:
    forgetting: float
    spline_knots: int


@dataclass(frozen=True)
class Bounded:
    nu: float
    epsilon: float


@dataclass(frozen=True)
class Model:
    climatology: Climatology
    rls: RLS
    adaptive_quantile: AdaptiveQuantile
    bounded: Bounded


@dataclass(frozen=True)
class Market:
    area: str
    pricing: str
    capacity_normalise: bool


@dataclass(frozen=True)
class Config:
    paths: Paths
    dates: Dates
    issue: Issue
    points: dict[str, tuple[float, float]]
    quantiles: tuple[float, ...]
    model: Model
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


def _parse_time(value: str, key: str) -> dt.time:
    try:
        return dt.time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"[issue] {key} must be HH:MM, got {value!r}") from exc


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _validate(cfg: Config) -> None:
    d = cfg.dates
    _check(
        d.start <= d.train_end < d.test_start <= d.end,
        "[dates] must satisfy start <= train_end < test_start <= end",
    )

    try:
        cfg.issue.tz  # noqa: B018 — resolve the zone to validate it
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"[issue] unknown timezone {cfg.issue.timezone!r}") from exc
    _check(
        cfg.issue.time_local < cfg.issue.gate_closure_local,
        "[issue] time_local must precede gate_closure_local",
    )
    _check(0 <= cfg.issue.nwp_cycle_utc < 24, "[issue] nwp_cycle_utc must be in 0..23")
    _check(cfg.issue.nwp_availability_lag_h >= 0, "[issue] nwp_availability_lag_h must be >= 0")
    _check(cfg.issue.power_lag_h >= 0, "[issue] power_lag_h must be >= 0")

    _check(len(cfg.points) > 0, "[points] needs at least one point")
    for name, (lat, lon) in cfg.points.items():
        _check(-90 <= lat <= 90 and -180 <= lon <= 180, f"[points] {name} is not a valid lat/lon")

    q = cfg.quantiles
    _check(len(q) > 0, "[quantiles] levels must not be empty")
    _check(all(0 < x < 1 for x in q), "[quantiles] levels must lie strictly inside (0, 1)")
    _check(all(a < b for a, b in zip(q, q[1:], strict=False)), "[quantiles] must be increasing")

    m = cfg.model
    _check(m.climatology.window_days >= 1, "[model.climatology] window_days must be >= 1")
    for name, lam in (
        ("rls", m.rls.forgetting),
        ("adaptive_quantile", m.adaptive_quantile.forgetting),
    ):
        _check(0 < lam <= 1, f"[model.{name}] forgetting must be in (0, 1]")
    _check(m.rls.delta > 0, "[model.rls] delta must be > 0")
    _check(m.adaptive_quantile.spline_knots >= 2, "[model.adaptive_quantile] spline_knots >= 2")
    _check(m.bounded.nu > 0, "[model.bounded] nu must be > 0")
    _check(0 < m.bounded.epsilon < 0.5, "[model.bounded] epsilon must be in (0, 0.5)")

    _check(cfg.market.pricing == "single", "[market] only single pricing is supported")


def load_config(path: str | Path) -> Config:
    """Load, merge (via ``base``) and validate a config file."""
    raw = _read(Path(path).resolve())
    try:
        issue = raw["issue"]
        model = raw["model"]
        cfg = Config(
            paths=Paths(**{k: Path(v) for k, v in raw["paths"].items()}),
            dates=Dates(**raw["dates"]),
            issue=Issue(
                timezone=issue["timezone"],
                time_local=_parse_time(issue["time_local"], "time_local"),
                gate_closure_local=_parse_time(issue["gate_closure_local"], "gate_closure_local"),
                nwp_cycle_utc=issue["nwp_cycle_utc"],
                nwp_availability_lag_h=issue["nwp_availability_lag_h"],
                power_lag_h=issue["power_lag_h"],
            ),
            points={name: (float(ll[0]), float(ll[1])) for name, ll in raw["points"].items()},
            quantiles=tuple(float(x) for x in raw["quantiles"]["levels"]),
            model=Model(
                climatology=Climatology(**model["climatology"]),
                rls=RLS(**model["rls"]),
                adaptive_quantile=AdaptiveQuantile(**model["adaptive_quantile"]),
                bounded=Bounded(**model["bounded"]),
            ),
            market=Market(**raw["market"]),
        )
    except KeyError as exc:
        raise ValueError(f"config {path} is missing key {exc}") from exc
    except TypeError as exc:
        raise ValueError(f"config {path} has unexpected or missing keys: {exc}") from exc
    _validate(cfg)
    return cfg
