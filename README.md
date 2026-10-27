# wind-power-forecast

[![CI](https://github.com/sneglen/wind-power-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/sneglen/wind-power-forecast/actions/workflows/ci.yml)

Day-ahead forecasts of DK1 wind power as calibrated, self-adapting predictive distributions
(recursive least squares, time-adaptive quantile regression, bounded distributions), and what
that uncertainty is worth in the DK1 single-price imbalance market. Data: 2024–2026.

> **Status:** Step 4 — adaptive power curve (RLS). The pipeline runs end to end on real DK1
> data with three references and frozen, walk-forward and recursive power curves;
> probabilistic models follow.

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

Individual stages: `make data`, `make features`, `make backtest`, `make evaluate`,
`make figures`.
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

Every forecast is computed only from what was known at its issue time, t0 = 11:00 local on
D−1. Each issue has 37 rows (`make features`), one per target hour from k = 1 to the end of
day D. The day-ahead product is k = 14–37.

| Information | Usable at t0 if | In practice |
|---|---|---|
| production, capacity | stamp + 1 h ≤ t0 | everything up to the hour ending at t0 |
| NWP run r | r + 7 h ≤ t0 | 00 UTC run of D−1, or D−2's if it is missing |
| day-ahead price | never (published ~13:00) | only flags curtailed target hours |

The backtest is an event loop over issue times: **observe** the targets that have become
known, **fit or update** the model (frozen, monthly walk-forward refit, or a daily recursive
update), then **predict** from the issue's input columns. The harness enforces the
information set, not the models. Hours with a day-ahead price ≤ 0 (curtailment) are never
used for fitting and are scored both with and without.

Tests corrupt every value stamped after the information set (production, capacity, later
NWP runs, prices) and require the issue's inputs and forecasts to stay identical. The same
tests require a change to the last known hour to move them, so the boundary is exact.

With one issue per day, horizon and hour of day are tied (k = 14 is always local midnight),
so horizon and diurnal effects cannot be separated.

## Methods

### Benchmarks

Every skill score is measured against three references, refit monthly on everything known
at the issue time (`src/wpforecast/models/baselines.py`):

| Reference | Point forecast ŷ(t+k \| t) | Predictive distribution |
|---|---|---|
| Climatology | — | empirical quantiles of all past y |
| Persistence | y(t), the last known hour | ŷ + empirical quantiles of past errors at horizon k |
| New reference (Nielsen et al., 1998) | a_k y(t) + (1 − a_k) ȳ, a_k = corr(y(t), y(t+k)) | ŷ + error quantiles at k |

The new reference is ENFOR's own benchmark: it blends persistence and the mean with the
weight that minimises the squared error at each horizon. With the true a_k it is never worse
than either of them in mean squared error, which makes it the fairest reference at short
horizons.

Scores (`src/wpforecast/scoring.py`) cover the day-ahead product hours of the test year:
- CRPS, approximated from 99 quantiles (τ = 0.01 … 0.99);
- pinball loss and coverage on the 19 reporting levels;
- MAE of the median;
- skill = 1 − CRPS / CRPS(climatology), overall and by horizon.

All models are scored on the same hours, with and without curtailed hours. `make evaluate`
writes `outputs/scores.md`.

_First scores on the full data: pending the download (Step 1)._

### Recursive least squares

The point forecast is a power curve in the spirit of WPPT, linear in its parameters after a
basis expansion (`src/wpforecast/models/power_curve.py`):

p̂(t+k | t) = φ(t+k)ᵀ β̂ₜ + a_k · (pₜ − φ(t)ᵀ β̂ₜ)

- **φ:** cubic B-splines of the onshore and offshore mean 100 m wind speed, each times
  [1, cos θ, sin θ] of the regional direction, plus diurnal harmonics. That is 64
  parameters, one curve pooled over all horizons.
- **a_k:** corrects the curve's error at the last known hour pₜ. It is estimated per block
  of horizons and fades with k.

β̂ is updated by recursive least squares with exponential forgetting:

- K = P φ / (λ + φᵀ P φ)
- β̂ ← β̂ + K (y − φᵀ β̂)
- P ← (P − K φᵀ P) / λ

This is a Kalman filter for parameters that drift as a random walk. Dividing P by λ inflates
the parameter uncertainty at every step, so old observations lose weight geometrically, and
the effective memory is about 1/(1 − λ) observations. With ~37 hourly rows per day,
λ = 0.9995 remembers about 54 days. The start is regularised (P₀ = δI), and the trace of P is
bounded so that the covariance cannot wind up in calm spells.

With λ = 1 the recursion gives exactly the ridge-regularised least-squares fit. That is the
frozen twin (fitted once on the training and validation data) and the monthly walk-forward
refit, so all three adaptivity regimes run the same code. λ is chosen on the validation
period.

_Results on the full data: pending the download (Step 1)._

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
  `data → features → backtest → evaluate → figures`.
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
Makefile                  data | features | backtest | evaluate | figures | all | smoke | test
src/wpforecast/
  config.py               typed, validated config loading
  pipeline.py             pipeline stages
  data/                   eds.py · openmeteo.py · capacity.py · build.py · checks.py
  features.py             issue times, horizons, periods; the issue-time feature table
  backtest.py             event loop: observe -> fit/update -> predict
  models/                 baselines.py (climatology, persistence, new reference),
                          power_curve.py (B-spline curve, RLS with forgetting)
  scoring.py              CRPS, pinball loss, coverage, MAE, skill
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

- T. S. Nielsen, A. Joensen, H. Madsen, L. Landberg, G. Giebel (1998). A new reference for
  wind power forecasting. *Wind Energy* 1(1).
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
