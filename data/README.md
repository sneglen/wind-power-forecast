# Data

`make data` downloads every input into a local cache, builds clean UTC tables and writes
the data-check report ([reports/data_check.md](../reports/data_check.md)). Downloads are
cached and resumable: a re-run fetches only what is missing. Only the small smoke sample
in `data/sample/` is committed.

The raw cache the results were made with is published as a frozen snapshot: the GitHub
Release [`data-2026-10-25`](https://github.com/sneglen/wind-power-forecast/releases/tag/data-2026-10-25)
(93 MB, checksum-verified, with the attribution of all three sources). `make data-snapshot`
unpacks it into `data/raw/`; `make data` then builds every table without downloading.

| Directory | Contents | In git |
|---|---|---|
| `data/raw/` | responses as received, one file per request chunk | no |
| `data/interim/` | cleaned tables, UTC, parquet | no |
| `data/sample/` | a real 24-day slice (2025-03-08 .. 2025-03-31) of `data/raw/` + `smoke.toml` | yes |

## Sources

| Source | Dataset | Resolution | Used for |
|---|---|---|---|
| Energinet, Energi Data Service | [ProductionConsumptionSettlement](https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement) | 1 h | target: DK1 wind production |
| | [CapacityPerMunicipality](https://api.energidataservice.dk/meta/dataset/CapacityPerMunicipality) | 1 month | installed capacity C(t) |
| | [Forecasts_Hour](https://api.energidataservice.dk/meta/dataset/Forecasts_Hour) | 1 h | benchmark: Energinet's day-ahead forecast |
| | [Elspotprices](https://api.energidataservice.dk/meta/dataset/Elspotprices) | 1 h, to 2025-09-30 | day-ahead price |
| | [DayAheadPrices](https://api.energidataservice.dk/meta/dataset/DayAheadPrices) | 15 min, from 2025-10-01 | day-ahead price |
| | [ImbalancePrice](https://api.energidataservice.dk/meta/dataset/ImbalancePrice) | 15 min, from 2025-03-04 | market backtest |
| Open-Meteo | [Single Runs API](https://open-meteo.com/en/docs/single-runs-api), model `ecmwf_ifs` | 1 h, leads 0–72 h | NWP: ECMWF IFS HRES 9 km, 00 UTC runs |
| dynamical.org | [ECMWF IFS ENS, 15 day, 0.25°](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) | 3 h, leads 0–72 h | Extension A: 51-member ensemble, 100 m u/v wind, 00 UTC runs from 2024-04-01 (`make ens`) |

All requests are for price area DK1 and UTC days 2024-03-14 .. 2026-10-25 (`[data]` in
`config.toml`).

## Interim tables (`data/interim/*.parquet`)

Timestamps are UTC and hour-beginning: `2025-03-08 10:00` is the interval 10:00–11:00.
Energy per hour (MWh/h) is written as average power (MW).

| Table | Index / keys | Columns |
|---|---|---|
| `production` | `time_utc` (1 h) | `offshore_lt100mw_mw`, `offshore_ge100mw_mw`, `onshore_lt50kw_mw`, `onshore_ge50kw_mw` (the four settlement columns), `wind_mw` (their sum) |
| `capacity` | `time_utc` (1 h) | `onshore_mw`, `offshore_mw`, `total_mw`; monthly values held from local midnight on the 1st |
| `dataset` | `time_utc` (1 h, regular) | `power_mw`, `capacity_mw`, `power_norm` = y = P / C clipped to [0, 1] |
| `energinet_forecast` | `time_utc` (1 h) | `{onshore,offshore,total}_{da,intraday}_mw`; day-ahead issued 18:00 local D−1 (after our issue time), excludes Thor |
| `da_price` | `time_utc` (1 h) | `price_eur_mwh`, `source`, `n_15min` (hourly mean of 15-min prices from 2025-10-01) |
| `imbalance` | `time_utc` (15 min) | `imbalance_price_eur_mwh`, `spot_price_eur_mwh`, `dominating_direction`, `satisfied_demand_mw`, `mfrr_{up,down}_eur_mwh`, `afrr_vwa_{up,down}_eur_mwh` |
| `nwp_hres` | `run_utc`, `valid_utc`, `point` | `lead_h` (0–72) and `wind_speed_100m`, `wind_direction_100m`, `wind_speed_10m`, `wind_speed_120m`, `wind_gusts_10m` (m/s, deg), `temperature_2m` (°C), `surface_pressure` (hPa) |
| `nwp_ens` | `run_utc`, `time_utc`, `member` | 51 ENS members (Step 8): regional hour means `wind_speed_100m@{onshore,offshore}` and `wind_direction_100m@...`, from 3-hourly u/v interpolated to hourly; built by `make ens` (or `make data` once runs are cached) |
| `points` | `point` | requested `lat`, `lon`; grid cell used `grid_lat`, `grid_lon`, `grid_elevation_m`, `distance_km` |

## Caveats

- **Settlement revisions.** Energinet revises settlement data for up to two years
  (99.9 % final after three months). `data/raw/eds/manifest.json` records when each chunk
  was downloaded; that date defines the snapshot.
- **Capacity.** `CapacityPerMunicipality` counts a wind farm from its registration. Thor
  (municipality code 1) appears with its full 1058.4 MW from March 2026, while its
  turbines were still being installed (36 of 72 by July 2026), so y = P / C is biased
  low during the ramp-up. Scores in the stress period are therefore reported in MW.
- **Energinet's forecast** is published at 18:00 local on D−1, seven hours after our issue
  time, from later NWP. It is a benchmark with an information edge, not a competitor on
  equal terms. It has 217 missing hours, the longest gap 72 h.
- **Production lags.** Settlement data arrives about ten days late: the download of
  2026-10-26 ends on 2026-10-16, so the last days of the window have no target.
- **Imbalance price method** changed on 2025-03-19 (single price) and 2026-03-04
  (dominating direction from satisfied demand); the market backtest stays inside one method.
- **Missing NWP runs.** Five consecutive 00 UTC runs are unusable: Open-Meteo's archive lacks
  2025-08-05, 08-06, 08-08 and 08-09, and returns 2025-08-07 without any values. Target days
  2025-08-07 to 08-10 therefore have no usable run, so no model forecasts them; on 08-06 the
  previous day's run is used, with longer leads. The run of 2026-06-23 lacks 10 m wind,
  gusts, temperature and pressure but has complete 100 m wind, so it is kept.
- **NWP grid cells.** Onshore points take the nearest land cell with similar elevation,
  offshore points the nearest sea cell (Open-Meteo `cell_selection`); all are within
  7.3 km of the requested coordinates.

## Licences and attribution

- Energinet data: [CC BY 4.0](https://www.energidataservice.dk/terms-and-conditions).
  Source: Energinet (www.energidataservice.dk). Energinet does not endorse this project.
- Weather data by [Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://open-meteo.com/en/licence),
  from ECMWF IFS HRES open data. Downloaded through the free, non-commercial API tier for
  this non-commercial portfolio project.
- ENS data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and the ECMWF terms
  of use. ECMWF IFS ENS forecast data processed by dynamical.org from ECMWF Open Data.
- The committed sample in `data/sample/raw/` is a verbatim extract of Energinet and
  Open-Meteo data under the same licences.
