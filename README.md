# wind-power-forecast

[![CI](https://github.com/sneglen/wind-power-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/sneglen/wind-power-forecast/actions/workflows/ci.yml)

Day-ahead forecasts of DK1 wind power as calibrated, self-adapting predictive distributions
(recursive least squares, time-adaptive quantile regression, bounded distributions), and what
that uncertainty is worth in the DK1 single-price imbalance market. Data: 2024–2026.

> **Status:** Step 1 — data acquisition and verification. The pipeline runs end to end on
> real DK1 data with a climatology benchmark; models follow.

Developed with AI assistance (Claude Code); modelling choices and their reasons are mine and
recorded in [DECISIONS.md](DECISIONS.md).

## Results at a glance

_Pending._

## Quick start

Requires Linux, [uv](https://docs.astral.sh/uv/) and `make`. uv installs the pinned Python (3.12.15) itself.

```bash
git clone git@github.com:sneglen/wind-power-forecast.git && cd wind-power-forecast
make install   # locked environment + pre-commit hooks
make smoke     # full pipeline, offline, on a committed 24-day slice of real data
make test      # test suite
make data      # download all inputs (resumable; ~2 days on the Open-Meteo free tier)
make all       # full pipeline on the downloaded data
```

Individual stages: `make data`, `make backtest`, `make evaluate`, `make figures`.
Any target accepts another config: `make all CONFIG=path/to/config.toml`.

Making a change: `make lint test`, commit, then `git push && make ci-watch` to confirm the
GitHub Actions run for that commit is green (needs the [GitHub CLI](https://cli.github.com/)).

## Problem and data

**Target.** Hourly DK1 wind production (onshore + offshore settlement data) divided by
installed capacity: y = P / C ∈ [0, 1], modelled as one total. DK1 has about 5.7 GW of wind
(4.1 GW onshore, 1.6 GW offshore) until Thor adds 1.06 GW from March 2026.

**Forecast.** Issued at 11:00 Europe/Copenhagen on D−1, before the 12:00 day-ahead gate
closure, for every hour of local day D (horizons k = 14–37 h), using the 00 UTC ECMWF run of
that day and production known up to the issue time.

**Inputs** (UTC days 2024-03-14 .. 2026-10-25, all cached and resumable with `make data`):

| Input | Source | Resolution |
|---|---|---|
| Wind production, 4 settlement columns | Energinet | 1 h |
| Installed capacity C(t) | Energinet, per municipality | 1 month |
| Energinet's day-ahead wind forecast (benchmark) | Energinet | 1 h |
| Day-ahead and imbalance prices | Energinet | 1 h / 15 min |
| ECMWF IFS HRES 00 UTC runs, 19 points, leads 0–72 h | Open-Meteo | 1 h |

The 19 NWP points are 12 onshore cells on a 0.5° grid over Jutland and Funen plus the
offshore farms Horns Rev 1–3, Vesterhav Nord/Syd, Anholt and Thor. Tables, units and
caveats: [data/README.md](data/README.md). Coverage, DST and unit checks:
[reports/data_check.md](reports/data_check.md).

Data: Energinet ([energidataservice.dk](https://www.energidataservice.dk)), CC BY 4.0.
Weather data by [Open-Meteo.com](https://open-meteo.com/) (ECMWF), CC BY 4.0.

## Issue-time discipline

_Pending._

## Methods

### Benchmarks

_Pending._

### Recursive least squares

_Pending._

### Time-adaptive quantile regression

_Pending._

### Bounded predictive distributions

_Pending._

## Evaluation

_Pending._

## DK1 imbalance-market backtest

_Pending._

## Extensions

### ECMWF ensemble mean vs HRES

_Pending._

### Thor: adapting to a regime change

_Pending._

## Reproducibility

- **Environment:** Python 3.12.15 pinned in `.python-version` (`pyproject.toml` allows 3.12.x); every
  dependency pinned in `uv.lock`. CI installs with `uv sync --locked`.
- **Configuration:** every date, NWP point, quantile level and hyper-parameter lives in
  [`config.toml`](config.toml). Variant configs inherit from it with `base = "..."` and
  override only what changes (see [`data/sample/smoke.toml`](data/sample/smoke.toml)).
- **Pipeline:** `make all` regenerates every artefact from raw data; stages are
  `data → backtest → evaluate → figures`.
- **CI:** GitHub Actions runs lint (pre-commit: ruff, formatting, lock check), the test suite,
  and a smoke run of the full pipeline on a committed sample, whose outputs are uploaded as
  a build artefact. A change counts as done only when CI is green for the pushed commit:
  `git push && make ci-watch` waits for that run (it appears a few seconds after the push),
  follows it to the end and fails if any job fails.
- **Data:** `make data` caches every response in `data/raw/` and records download dates
  (settlement data is revised for months). CI runs offline on `data/sample/`, a committed
  24-day slice of real data covering the single-price switch and a DST day.

## Repository layout

```
config.toml               dates, points, quantile grid, hyper-parameters
Makefile                  data | backtest | evaluate | figures | all | smoke | test | lint
src/wpforecast/
  config.py               typed, validated config loading
  pipeline.py             pipeline stages
  data/                   eds.py · openmeteo.py · capacity.py · build.py · checks.py
  __main__.py             CLI: wpforecast <stage> --config ...
scripts/                  utilities: CI watcher
tests/                    unit tests, look-ahead test, end-to-end smoke test
data/README.md            data dictionary, caveats, licences
data/sample/              committed slice of real data + smoke config
data/raw, data/interim    downloaded / derived data (git-ignored)
reports/                  data-check report and figures
outputs/, figures/        generated artefacts (git-ignored)
DECISIONS.md              running decision log
CLAUDE.md                 project rules for AI-assisted work
```

## Decision log

Design choices, with reasons and rejected alternatives, are recorded in
[DECISIONS.md](DECISIONS.md) as the project progresses.

## References

- H. A. Nielsen, H. Madsen, T. S. Nielsen (2006). Using quantile regression to extend an
  existing wind power forecasting system with probabilistic forecasts. *Wind Energy* 9(1–2).
- J. K. Møller, H. A. Nielsen, H. Madsen (2008). Time-adaptive quantile regression.
  *Computational Statistics & Data Analysis* 52(3).
- P. Pinson (2012). Very-short-term probabilistic forecasting of wind power with
  generalized logit–normal distributions. *JRSS Series C* 61(4).
- P. Pinson, C. Chevallier, G. Kariniotakis (2007). Trading wind generation from short-term
  probabilistic forecasts of wind power. *IEEE Transactions on Power Systems* 22(3).
- T. Gneiting, A. E. Raftery (2007). Strictly proper scoring rules, prediction, and
  estimation. *JASA* 102(477).

## Licence

[MIT](LICENSE) © 2026 Enrique Vidal Sanchez.
