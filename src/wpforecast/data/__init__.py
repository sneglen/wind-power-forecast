"""Data layer: download (cached, resumable), build interim tables, check them."""

from __future__ import annotations

from wpforecast.config import Config
from wpforecast.data import build, checks, eds, openmeteo


def acquire(cfg: Config) -> None:
    """`make data`: fetch what is missing, rebuild the interim tables, write the check report."""
    eds.download(cfg)
    openmeteo.download(cfg)
    build.build(cfg)
    checks.run(cfg)
