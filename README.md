# wind-power-forecast

[![CI](https://github.com/sneglen/wind-power-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/sneglen/wind-power-forecast/actions/workflows/ci.yml)

Day-ahead forecasts of DK1 wind power as calibrated, self-adapting predictive distributions
(recursive least squares, time-adaptive quantile regression, bounded distributions), and what
that uncertainty is worth in the DK1 single-price imbalance market. Data: 2024–2026.

> **Status:** Step 0 — reproducible skeleton. The pipeline runs end to end on a
> synthetic sample with a climatology benchmark; real data and models follow.

Developed with AI assistance (Claude Code); modelling choices and their reasons are mine and
recorded in [DECISIONS.md](DECISIONS.md).

## Results at a glance

_Pending._

## Quick start

Requires Linux, [uv](https://docs.astral.sh/uv/) and `make`. uv installs the pinned Python (3.12.15) itself.

```bash
git clone git@github.com:sneglen/wind-power-forecast.git && cd wind-power-forecast
make install   # locked environment + pre-commit hooks
make smoke     # full pipeline on the committed synthetic sample (seconds)
make test      # test suite
make all       # full pipeline on real data (data/raw/dk1.csv, from Step 1)
```

Individual stages: `make data`, `make backtest`, `make evaluate`, `make figures`.
Any target accepts another config: `make all CONFIG=path/to/config.toml`.

Making a change: `make lint test`, commit, then `git push && make ci-watch` to confirm the
GitHub Actions run for that commit is green (needs the [GitHub CLI](https://cli.github.com/)).

## Problem and data

_Pending._

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
- **Synthetic sample:** `data/sample/dk1_synthetic.csv` is generated deterministically by
  `make sample`; a test checks the committed file matches the generator. It is not real data.

## Repository layout

```
config.toml               dates, points, quantile grid, hyper-parameters
Makefile                  data | backtest | evaluate | figures | all | smoke | test | lint
src/wpforecast/
  config.py               typed, validated config loading
  pipeline.py             pipeline stages
  __main__.py             CLI: wpforecast <stage> --config ...
scripts/                  utilities: synthetic sample generator, CI watcher
tests/                    unit tests, look-ahead test, end-to-end smoke test
data/sample/              committed synthetic sample + smoke config
data/raw, data/interim    downloaded / derived data (git-ignored)
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
