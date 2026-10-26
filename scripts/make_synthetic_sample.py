"""Generate the committed synthetic sample used by the CI smoke run.

The sample has the same schema as the real DK1 dataset (hourly, UTC, hour-beginning):
``time_utc, power_mw, capacity_mw, wind_speed_<point>`` for every point in the config.
It is NOT real data; it only exercises the pipeline. Deterministic for a given seed.

    uv run python scripts/make_synthetic_sample.py [--config config.toml] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wpforecast.config import load_config

START, END = "2025-01-01", "2025-01-22"  # [START, END) in UTC
CAPACITY_MW = 6500.0
SEED = 20240101


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma / np.sqrt(1 - phi**2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)
    return x


def make_sample(points: list[str], seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range(START, END, freq="h", inclusive="left", tz="UTC")
    n = len(time)

    regional = 8.0 + ar1(rng, n, phi=0.97, sigma=0.7)
    diurnal = 0.6 * np.sin(2 * np.pi * (time.hour.to_numpy() - 9) / 24)
    speeds = {
        f"wind_speed_{p}": np.clip(
            regional + diurnal + rng.normal(0.0, 0.8) + ar1(rng, n, phi=0.9, sigma=0.4), 0.0, None
        )
        for p in points
    }

    mean_speed = np.mean(list(speeds.values()), axis=0)
    curve = 1.0 / (1.0 + np.exp(-(mean_speed - 8.0) / 1.5))
    power = np.clip(curve + rng.normal(0.0, 0.03, n), 0.0, 1.0) * CAPACITY_MW

    df = pd.DataFrame({"time_utc": time, "power_mw": power, "capacity_mw": CAPACITY_MW, **speeds})
    df["time_utc"] = df["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--out", default="data/sample/dk1_synthetic.csv", type=Path)
    args = parser.parse_args()

    df = make_sample(list(load_config(args.config).points))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
