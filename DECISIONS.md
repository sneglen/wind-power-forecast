# Decision log

A running record of design decisions. Each entry states the decision, why it was taken and
what was rejected. Newest entries at the bottom; superseded entries are marked, not deleted.

---

### D-001 · 2026-10-26 · Python 3.12, pinned
**Decision:** `requires-python = ">=3.12,<3.13"`, `.python-version = 3.12`.
**Why:** `dynamical-catalog` (ECMWF ENS access, Step 8) needs ≥ 3.12; an upper bound keeps CI
and local runs on the same minor version.
**Rejected:** 3.13/3.14, because the scientific stack lags new minors and gains nothing here.

### D-002 · 2026-10-26 · uv + committed `uv.lock`
**Decision:** uv manages Python, the venv and the lockfile; CI uses `uv sync --locked`.
**Why:** one tool, fast, fully reproducible; the `uv-lock` pre-commit hook keeps the lock in
sync with `pyproject.toml`.
**Rejected:** pip-tools/Poetry (slower, more moving parts); conda (heavier, not needed:
everything installs from wheels).

### D-003 · 2026-10-26 · src layout, package `wpforecast`
**Decision:** `src/wpforecast/` with a `wpforecast` CLI entry point; Make targets call the CLI.
**Why:** the src layout forces tests to run against the installed package; one CLI means Make,
CI and users run the same code path.

### D-004 · 2026-10-26 · pandas + xarray
**Decision:** pandas for tabular series (power, prices, forecasts), xarray for gridded NWP.
**Why:** xarray is what the NWP catalogue returns; pandas is the lingua franca for reviewers.
**Rejected:** polars (faster, but less familiar to the audience and with no data volume here
that needs it).

### D-005 · 2026-10-26 · One TOML config with inheritance
**Decision:** all dates, points, quantiles and hyper-parameters in `config.toml`, parsed with
stdlib `tomllib` into frozen, validated dataclasses. Variant configs set `base = "..."` and
override only what changes.
**Why:** a single source of truth, no extra dependency, and invalid settings fail at load
time rather than mid-run.
**Rejected:** Hydra/pydantic-settings (more power than needed); duplicated configs (drift).

### D-006 · 2026-10-26 · Time conventions and issue-time discipline — *superseded by D-011, D-012*
**Decision:** all timestamps UTC, hour-beginning. Forecasts for local day D are issued on
D−1 at 10:00 Europe/Copenhagen, before the 12:00 day-ahead gate closure, from the 00 UTC NWP
cycle. A power observation stamped `t` becomes usable at `t + 1h + power_lag_h`. Target days
are local, so DST days have 23 or 25 hours; lead times run 14–38 h.
**Why:** the backtest must use only what an operator would have had at issue time. This is
enforced in code (a runtime check per issued day) and by a test that corrupts post-issue data
and asserts the forecast is unchanged.

### D-007 · 2026-10-26 · Quantile grid 0.05–0.95, step 0.05
**Decision:** 19 quantile levels.
**Why:** dense enough for reliability diagrams and a CRPS approximation, and the outer levels
are still estimable from a few months of adaptive data. More extreme levels can be added once
the bounded-distribution work shows they are calibrated.

### D-008 · 2026-10-26 · Synthetic smoke sample, replaced by real data in Step 1 — *superseded by D-014*
**Decision:** commit a deterministic synthetic sample (3 weeks, hourly, final schema) and run
the full pipeline on it in CI.
**Why:** Step 0 comes before any data is touched, but CI should run the pipeline end to end
from day one. A test guarantees the committed file matches its generator.
**Next:** swap for a small real DK1 slice in Step 1.

### D-009 · 2026-10-26 · Climatology benchmark as the Step 0 model — *superseded by D-026*
**Decision:** the backtest stage issues rolling empirical quantiles of the last
`window_days` of observed normalised power.
**Why:** it is the natural reference model for skill scores, and it exercises the issue-time
machinery without any modelling decisions.

### D-010 · 2026-10-26 · MIT licence, English
**Decision:** MIT licence, all docs and code in English.
**Why:** permissive and standard for a public portfolio project.

### D-011 · 2026-10-26 · Config aligned with the project plan's key decisions
**Decision:**
- **Issue time** 11:00 Europe/Copenhagen on D−1, before the 12:00 gate closure. Production is
  known up to the issue time (real-time metering, `power_lag_h = 0`).
- **Periods** (target days): train 2024-04-01 → 2025-02-28, of which the last three months
  (from 2024-12-01) are validation; test 2025-03-01 → 2026-02-28; stress 2026-03-01 → latest.
- **NWP:** ECMWF IFS HRES 9 km, 00 UTC runs, 100 m wind via the Open-Meteo Single Runs API;
  ECMWF IFS ENS (51 members, 0.25°) via dynamical.org only for Extension A.
- **Points:** 12 onshore on a 0.5° grid over Jutland and Funen, plus Horns Rev 1–3, Vesterhav
  Nord/Syd, Anholt and Thor (19 in total, approximate; checked against a land/sea mask in Step 1).
- **Target:** DK1 onshore + offshore production ÷ installed capacity, one total, hourly.
- **Curtailment:** hours with day-ahead price ≤ 0 are flagged, dropped from fitting and scored
  with and without. DA prices are published after issue, so never a feature.
- **Scoring:** pinball on the 19-level grid; CRPS from 99 quantiles.
- **Adaptivity:** three regimes compared — frozen, monthly walk-forward refit, daily recursive.
- **Market:** DK1 single price, 100 MW price-taker, window 2025-03-19 → 2026-02-28 (one
  imbalance-price method).

**Why:** the Step 0 skeleton was first built from the Step 0 brief alone; these values come
from the project plan so every later step reads them from one place. Hyper-parameters remain
placeholders until their step tunes them on the validation set.

### D-012 · 2026-10-26 · Horizon convention
**Decision:** timestamps are hour-beginning; horizon k is the number of hours from the issue
time to the END of the target hour, so k = 1 is the first hour after issue. The day-ahead
product (local day D) is k = 14–37, i.e. lead times 13–37 h; DST days give 14–36 (spring) and
14–38 (autumn). `max_horizon_h = 38`: the plan's 1–37 plus one hour for the autumn DST day.
**Why:** matches the WPPT-style notation p̂(t+k | t) for an hourly mean ending at t+k, and makes
the DST edge case explicit rather than silently truncated.
**Rejected:** k counted to the start of the hour (product k = 13–36), which loses the plan's
1–37 range.

### D-013 · 2026-10-26 · Exact Python patch pin; CI checked after every push
**Decision:** `.python-version` pins 3.12.15 (`pyproject.toml` still allows 3.12.x). Every push
is followed by `make ci-watch`, and a step is not finished until its CI run is green.
**Why:** with only `3.12`, CI used the runner's system Python 3.12.3 while local runs used
3.12.15; outputs matched, but the environments did not. The CI run appears a few seconds after
a push, so `ci-watch` polls for the run of the pushed commit before following it.
**Rejected:** relying on the commit badge (easy to miss a red run), or pinning only in CI.

### D-014 · 2026-10-26 · Download cache, offline mode and a real smoke sample
**Decision:** every response is cached as received: Energi Data Service per dataset and
calendar year (`data/raw/eds/<Dataset>/<first>_<last>.parquet`, with a manifest of download
dates), Open-Meteo per run (`data/raw/openmeteo/ecmwf_ifs/<YYYY>/<date>T00.json.gz`). A
re-run fetches only missing pieces; `offline = true` forbids network access. CI runs the whole
pipeline offline on a committed 24-day slice of real data (2025-03-08 .. 2025-03-31, 1 MB).
**Why:** the free API tiers need a resumable download; settlement data is revised, so a
dated cache is the snapshot; the slice covers the single-price switch (2025-03-19) and a
23-hour DST day, which a synthetic sample could not.
**Rejected:** the synthetic sample (D-008); calling the APIs from CI (slow, rate-limited,
and results would drift with revisions).

### D-015 · 2026-10-26 · Capacity from Energinet's CapacityPerMunicipality
**Decision:** C(t) = monthly DK1 onshore + offshore capacity from `CapacityPerMunicipality`,
held from local midnight on the 1st. Municipalities map to DK1 by code (≥ 410, except 411
Christiansø); non-municipal code 1 is Thor and is mapped explicitly; any other such code
raises an error.
**Why:** an API with monthly history, from the same source as the target. Instead of the
Danish Energy Agency's turbine register named in the plan, which is a spreadsheet download.
**Caveat:** Thor is registered at its full 1058.4 MW from March 2026 while still being built
(36 of 72 turbines by July 2026), so y = P / C is biased low in the stress period; Step 9
scores in MW, as planned.

### D-016 · 2026-10-26 · NWP points, grid cells and availability
**Decision:** offshore points were checked against published farm locations and moved where
needed (Vesterhav Nord/Syd, Thor). Onshore points request land grid cells and offshore points
sea cells (`cell_selection`), two requests per run; every cell is within 7.3 km.
`availability_lag_h = 7`.
**Why:** Open-Meteo's default picks a land cell of similar elevation, which put the Vesterhav
Nord point 18 km away on the coast. Run availability on Open-Meteo was observed 6.1 h after
initialisation for the 06 UTC run of 2026-10-26 and 6.5 h for the 00 UTC run of 2026-10-27,
so the 00 UTC run is usable by 07:00 UTC, two hours before the 09:00 UTC summer issue time.

### D-017 · 2026-10-26 · Seven NWP variables instead of two
**Decision:** besides 100 m speed and direction, download 10 m and 120 m speed, 10 m gusts,
2 m temperature and surface pressure.
**Why:** Open-Meteo counts extra calls only above 10 variables per request, so these cost
nothing now and avoid a two-day re-download if Step 5 (gradient boosting) or Step 8 needs
shear, gusts or air density. The core models use only the 100 m fields.

### D-018 · 2026-10-26 · Open-Meteo free tier with a conservative call budget
**Decision:** use the free, non-commercial tier with CC BY 4.0 attribution. The downloader
assumes every location in a request counts as one call (19 per run, undocumented) and keeps
a rolling ledger under 4500 calls/hour and 9800/day, shared by every config on the machine
(`~/.cache/wpforecast/`); when the day is spent it stops, and the next `make data` resumes.
The full archive (956 runs, 18,164 calls) therefore takes two days. An Open-Meteo 429 for the
hourly quota waits for the next hour; the daily quota stops the download cleanly.
**Evidence:** the first full run hit Open-Meteo's hourly limit at ~5000 calls when two
configs kept separate ledgers, which confirmed that every location counts as a call.
**Why:** the portfolio project is non-commercial; staying well inside the limits avoids being
blocked. The paid Professional plan (≈ €99/month) stays the fallback if data is missing.

### D-019 · 2026-10-26 · Data check as a script and a Markdown report
**Decision:** `make data` ends with `wpforecast.data.checks`, which writes
`reports/data_check.md` and figures: coverage and gaps, missing NWP runs, UTC/DST
consistency, units and ranges, grid cells, and normalised power vs NWP wind speed.
**Why:** reproducible from `make`, tested in CI on the sample, readable on GitHub.
**Rejected:** a Jupyter notebook (the plan's wording): harder to test and to diff.

### D-020 · 2026-10-27 · Feature rows cover k = 1 to the end of day D
**Decision:** `features.py` writes one row per issue time and target hour, from the first
hour after the issue to the end of local day D: k = 1–13 on D−1 plus the day-ahead product
(`in_product`, k = 14–37, 36/38 on DST days); 37 rows per issue. Scores use product hours
only. Target days before `train_start` are labelled `warmup` (history for the first fits).
**Why:** the plan's horizons are k = 1–37; the short horizons give the Step 4 correction
term and persistence-type baselines their data, at no cost to the product evaluation.
**Note:** with one issue per day, horizon and hour of day are tied (k = 14 is always local
midnight), so horizon effects and diurnal effects cannot be separated.

### D-021 · 2026-10-27 · NWP: latest usable run, hour means
**Decision:** each issue takes the latest run with `run + availability_lag_h ≤ issue` that
covers all its target hours: the 00 UTC run of D−1, or D−2's (leads up to 71 h) if D−1's is
missing upstream. The hourly mean over [t, t + 1 h) is the mean of the instantaneous values
at t and t + 1 h; wind is averaged as u/v components and direction derived from the means;
gusts (maximum over the preceding hour) take the value at t + 1 h. Regional `@onshore` and
`@offshore` columns average the points of each region. `lead_h` counts to the end of the hour.
**Why:** production is an hourly mean, NWP fields are instantaneous; the trapezoid is the
simplest unbiased match. A fallback run keeps a day forecastable instead of dropping it.
**Rejected:** the value at t or t + 1 h alone (half-hour shift); averaging directions in
degrees (wrong across north).

### D-022 · 2026-10-27 · Capacity and persistence at the last known hour
**Decision:** `y_issue` and `capacity_issue_mw` are read at the last hour known at the issue
time (stamp `issue − 1 h − power_lag_h`). The target stays y = P(t) / C(t), with C at the
target hour; conversions to MW at forecast time use `capacity_issue_mw`.
**Why:** C(t) for a future month is not known at issue time, so it must not reach the
inputs; as a target normaliser it is a definition, not information. Tests corrupt every
value stamped after the last known hour and require the inputs to stay unchanged.
**Caveat:** the register itself is published with a lag; reading it at the issue hour treats
the monthly value as known from the start of its month.

### D-023 · 2026-10-27 · Backtest harness: observe, fit/update, predict
**Decision:** `backtest.run` loops over issue times. It first releases rows whose target is
known (stamp + 1 h + `power_lag_h` ≤ issue), then fits or updates by regime — `frozen`: fit
once at the start; `walk_forward`: refit at the first issue of each month on everything
known; `recursive`: fit once, then `update` with each day's released rows — and only then
calls `predict` with the input columns of that issue. Quantile crossing is fixed by sorting.
**Why:** the harness, not each model, enforces the information set, so leakage is
structurally impossible; one loop serves all three adaptivity regimes.
**Rejected:** per-model backtest code (each model would have to re-implement the timing).

### D-024 · 2026-10-27 · Curtailed hours never reach a model
**Decision:** `curtailed` = day-ahead price ≤ `[curtailment] da_price_max` at the target
hour (a missing price counts as not curtailed). The harness drops curtailed hours and missing
targets from everything it releases to models; evaluation reports scores with and without
them. Prices are never inputs, and the feature builder refuses any input column named price.
**Why:** negative prices trigger curtailment, so those hours measure a market decision, not
the weather; the price is only known after the issue time.

### D-025 · 2026-10-27 · ENS point extract from dynamical.org, a separate `make ens`
**Decision:** `make ens` reads the dynamical.org dataset
`ecmwf-ifs-ens-forecast-15-day-0-25-degree` (Icechunk/Zarr via `dynamical-catalog`) and
keeps, per 00 UTC run from 2024-04-01, the 51 members of 100 m u/v wind at the grid cell
nearest each of the 19 points, leads 0–72 h (3-hourly): one parquet file per run (~80 kB),
resumable, four runs in parallel. Runs absent from the archive are recorded once. It is not
part of `make data` or `make all`.
**Why:** all DK1 points sit in one 8° storage chunk, so a run costs two chunk reads (~2 s);
the whole archive takes about 30 minutes, so no overnight download is needed despite the
plan's caution. Point values are enough for the plan's features (ensemble mean, spread,
member-wise power). Keeping it out of `make data` leaves the core pipeline independent of
the extension.
**Rejected:** regridding or keeping the DK1 box (unneeded volume); Open-Meteo's ensemble
API (counts each member as calls against the free-tier budget).
**Open (Step 8):** verify that the 00 UTC ENS run is published before the issue time.
Closed by D-039.

### D-026 · 2026-10-27 · Three baselines, refit monthly on all known data
**Decision:** `models/baselines.py` holds the references, all run in the `walk_forward`
regime (refit at the first issue of each month on everything known then):
- **climatology**: empirical quantiles of every known hourly y, the same for every hour;
- **persistence**: y at the last known hour, plus empirical quantiles of its past errors at
  the same horizon k;
- **new reference** (Nielsen et al., 1998): a_k y(t) + (1 − a_k) ȳ with a_k the lag-k
  autocorrelation of the hourly series (lag k + `power_lag_h`), plus error quantiles by k.

Quantiles are clipped to [0, 1]. A horizon with fewer than `min_error_samples` past errors
borrows the nearest horizon that has enough (k = 38 occurs only on autumn DST days).
**Why:** the plan's assumptions (training data only, updated walk-forward). The 28-day
rolling climatology of Step 0 is a different, recency-weighted reference. Estimating a_k from
the hourly series uses every hour rather than one pair per day and matches the paper's
definition.
**Rejected:** month-of-year climatology (the plan's option): with a year of training, each
month's quantiles would rest on a single month.

### D-027 · 2026-10-27 · One quantile grid for CRPS and reporting; common hours
**Decision:** every model predicts the 99 quantiles τ = 0.01 … 0.99 (`crps_levels`), which
contain the 19 reporting levels. `scoring.py` computes CRPS with scoringrules
`crps_quantile` (= 2 × mean pinball loss over the grid), pinball loss and coverage on the
19 levels, and MAE of the median. Models are scored on the test period's product hours
that every model forecast, with and without curtailed hours; skill is
1 − CRPS / CRPS(climatology), overall and by horizon.
**Why:** one forecast serves both scores without interpolating quantiles; comparing models
on different hours would bias skill scores.

### D-028 · 2026-10-27 · Power curve: B-splines x direction, RLS with forgetting — *covariance bound corrected by D-043*
**Decision:** `models/power_curve.py` models y = φᵀβ with φ = cubic B-splines of the
regional 100 m speed (6 interior knots over 0–20 m/s onshore, 0–28 m/s offshore), each times
[1, cos θ, sin θ] of the regional direction, plus two diurnal harmonics (64 parameters), one
curve pooled over horizons. β is estimated by RLS with exponential forgetting λ per hourly
row, P0 = δI (δ = 100), and trace(P) ≤ trace(P0). The frozen and walk-forward twins are the
same code with λ = 1, which equals ridge regression with penalty 1/δ (unit-tested). λ is
chosen on the validation period from `forgetting_grid` by point RMSE (`outputs/tuning.json`).
Point forecasts are clipped to [0, 1].
**Why:** the plan's WPPT-style curve; one implementation for the three adaptivity regimes
makes the comparison exact. Separate speed ranges put the knots where the data are: regional
means pass 17.7 m/s onshore in only 0.1 % of hours (2024-03 to 2025-06), but reach 26 m/s
offshore. A common 0–25 m/s range left the onshore splines above 18 m/s to the prior.
**Evidence:** on the HRES data downloaded so far (test 2025-03-01 to 06-08) the curves
reach an RMSE of 7.7 % of capacity and 64 % CRPS skill over climatology. λ = 1 wins on
validation, so before Thor there is no drift for the forgetting to follow.
**Rejected:** the conditional-parametric model of Nielsen et al. (2000), the plan's optional
variant (more machinery for a direction effect that turns out small); knots at data
quantiles (they would move between refits).

### D-029 · 2026-10-27 · Short-horizon correction and error quantiles from prequential residuals
**Decision:** p̂(t+k|t) = φ(t+k)ᵀβ̂ + a_k (p_t − φ(t)ᵀβ̂), where φ(t) uses the same run's NWP
at the last known hour (new feature columns `<variable>@<region>@issue`). a_k is the
least-squares slope of the curve's target-hour residual on its issue-hour residual, one per
horizon block (k = 1–6, 7–12, … 31–38), accumulated with the same forgetting. Both a_k and the
empirical error quantiles (by k, last 90 days) use residuals computed with the parameters as
they were before those hours were observed.
**Why:** the plan's formula; prequential residuals keep the correction and the interim
predictive distribution out-of-sample. Step 5 replaces the error quantiles.

### D-030 · 2026-10-27 · Probabilistic layers on the RLS curve, trained on issued p̂
**Decision:** `models/probabilistic.py` wraps a power curve (the tuned λ, updated daily) in
three predictive distributions of y given p̂ and k:
- **quantile regression** on [B(p̂), B(p̂)·k/38], refit weekly on 180 days (statsmodels);
- **generalised logit-normal**, with μ linear in z(p̂) and k and log σ linear in z(p̂),
  z(p̂)² and k, fitted by maximum likelihood with y ≤ ε and y ≥ 1 − ε as censored
  observations, refit monthly on all known data;
- **LightGBM quantiles** on p̂, k, calendar terms and ten NWP features, refit monthly.

A layer learns from (p̂, k, y) pairs in which p̂ is the forecast as issued, or the curve's
prequential point for hours before the first forecast. It forecasts only once it has
`min_train_rows` pairs.
**Why:** the plan's three complementary distributions (nonparametric, parametric,
machine-learned). Training on issued p̂ keeps the distribution honest about the point
model's real errors.
**Deviation:** LightGBM refuses monotone constraints with its quantile objective, so the
plan's monotonicity in wind speed is not imposed; p̂ itself is monotone in speed.
**Deferred:** the recursive-likelihood logit-normal (Pierrot & Pinson) and the beta check,
both optional in the plan.
**Evidence:** on the HRES data downloaded by 2026-10-27 (test 2025-03-01 to 06-08), all three
layers lower CRPS slightly against the curve's own error quantiles (0.0401 → 0.0388–0.0393).
LightGBM is the sharpest but too narrow in the tails (5 % / 95 % quantiles cover 14 % / 89 %).
LightGBM is pinned to `num_threads = 4`: with all 16 cores its OpenMP threads spin-wait when
another process shares the CPU (114 s instead of 2 s on the smoke sample).

### D-031 · 2026-10-27 · Quantile models fitted at 23 levels, interpolated to 99
**Decision:** quantile regression and LightGBM are fitted at the 19 reporting levels plus
`tail_levels` (0.01, 0.02, 0.98, 0.99), sorted, and interpolated linearly in τ onto the
99-level CRPS grid; the logit-normal is evaluated at the same levels for consistency. The
tail levels must include the grid's ends, so nothing is extrapolated.
**Why:** 99 quantile regressions per weekly refit would take ~20 min on the full data
(87 ms each); 23 take ~5 min. Between neighbouring levels the quantile function is close to
linear, and CRPS is insensitive to that interpolation.
**Rejected:** scipy's HiGHS LP for quantile regression (16× slower than statsmodels here).

### D-032 · 2026-10-28 · Wait for the rolling daily budget instead of stopping
**Decision:** when Open-Meteo's daily budget is spent, the downloader computes when enough
calls leave the rolling 24 h window. If that is within `max_wait_min` (60), it waits;
otherwise it stops and names that time.
**Why:** the session of 2026-10-28 stopped at 08:09, one minute after the first of the
previous morning's calls expired. The message named that minute, but those calls expire
over 40 minutes. Waiting lets a session started at the named time run to the end.

### D-033 · 2026-10-28 · A run without complete 100 m wind counts as missing
**Decision:** when the interim tables are built, a run whose 100 m speed or direction has any
gap after lead 0 is dropped and listed with the missing runs, so the feature builder falls
back to an older run. Gaps in other variables are reported but kept (only LightGBM uses
them, and it handles missing values).
**Evidence:** the archive lacks the 00 UTC runs of 2025-08-05, 08-06, 08-08 and 08-09
("The requested model run is not available") and returns 2025-08-07 with every value null.
Counted as present, that empty run would have been chosen over a fallback. Target days
2025-08-07 to 08-10 have no usable run and are not forecast; 08-06 uses the run of 08-04.
The run of 2026-06-23 lacks 10 m wind, gusts, temperature and pressure and is kept.
**Also:** the range checks now ignore missing values, which the missing-value check
reports; before, a gap made them warn although every value was in range.

### D-034 · 2026-10-28 · No forecast without a usable weather run
**Decision:** at an issue time without a usable run, the power curve and the three
distribution layers return no forecast (NaN), and the harness drops those rows. LightGBM
could forecast from missing inputs, but is not allowed to.
**Why:** the first full run crashed on the four such days (2025-08-07 to 08-10): the spline
basis received no valid speeds. Without weather, any "forecast" is a disguised climatology,
and every model must skip the same hours for scores to stay comparable.
**Evidence (full data, test year):** the curves reach 66–67 % CRPS skill over climatology;
RLS chose λ = 1 again. Skill dips at k = 25–30 (midday) because up to 22 % of those hours
are curtailed; without them, skill is flat at 0.69–0.75.
**Open for Step 6:** every model has too many observations below its 5 % quantile (7–12 %);
quantile regression's lower tail is erratic at high p̂, where data are sparse; LightGBM's
distribution jumps at its monthly refit.

### D-035 · 2026-10-28 · Raw data published as a release snapshot, not committed
**Decision:** the raw cache (Energinet, HRES, ENS; 93 MB compressed) is a GitHub Release
asset, tag `data-2026-10-25`, with a SHA-256 checksum and an attribution file;
`make data-snapshot` fetches, verifies and unpacks it. Git still holds only the smoke sample.
**Why:** the full download takes two days on Open-Meteo's free tier, and Energinet revises
settlement data, so a fresh download would not reproduce the results exactly. In git, every
refresh would add another copy to the history of every clone. All three licences (CC BY 4.0)
allow redistribution with attribution.
**Rejected:** committing the data (history bloat); Git LFS (bandwidth quotas on a public
repository); a Zenodo DOI (permanent, not needed now).

### D-036 · 2026-10-28 · Test-year evaluation protocol
**Decision:** `evaluation.py` scores the test year once, on the hours every model forecast,
with and without curtailed hours: CRPS and skill with a day-block bootstrap band (1,000
draws, 90 %), MAE and RMSE in % of capacity, the coverage and width of central 50/80/90 %
intervals, reliability with bootstrap bands, randomised-rank PIT histograms, and
Diebold–Mariano tests on daily mean CRPS (Newey–West, 7 lags). The report and its figures
are committed under `reports/`. Horizons with fewer than 30 test hours (k = 38) are left out
of the by-horizon plots.
**Why:** the protocol of Madsen et al. (2005) and Pinson et al. (2007). Days are the
resampling and testing unit because errors within a day are strongly dependent; resampling
hours would give bands that are far too narrow.
**Energinet benchmark:** as published it runs 16 % low against settled production (slope
0.68; correct time alignment checked at lags −2…2), so it does not cover the same plants.
It is reported raw and after a linear recalibration fitted on the training period only
(P = 1.33 F − 227 MW). Raw, it looks worse than this project's forecast (RMSE 10.0 % against
8.2 %); recalibrated, it is better (6.9 %), as its later issue time predicts.
**Not done:** model changes after seeing the test year. The weaknesses found (too narrow
LightGBM, a high logit-normal median) are reported, not tuned away.

### D-037 · 2026-10-28 · Market backtest: settlement, bids and the trailing window
**Decision:** `market.py` prices day-ahead bids for a price-taking 100 MW portfolio producing
y × 100 MW over 2025-03-19 to 2026-02-28 (one single-price method). The cost is the revenue
lost against a perfect bid, Σ (λ_DA − λ_imb)(E − b) / Σ E. Hourly bids are settled against
the hourly mean of the four 15-min imbalance prices; hours missing a quarter are dropped.
The bids, per model, are the mean of the 99 quantiles, the median, τ* (the grid level with
the lowest cost over the trailing 60 days) and the newsvendor level τ = ĉ↓ / (ĉ↑ + ĉ↓).
ĉ↑ and ĉ↓ are the MWh-weighted mean premium paid per MWh short and forgone per MWh long,
over the trailing days, with positions taken from the median bid. The trailing days end
with D−2, the last local day complete at the issue time on D−1. They never start before the
market window, and the median is bid until 14 days exist. Energinet's forecast is bid as
published and recalibrated (D-036), clipped to [0, 1] of capacity. All bids are compared on
the same hours, with curtailed hours included and also reported without. Bands use the
day-block bootstrap of D-036, with savings computed paired by day.
**Why:** the plan's assumptions. Flat production within the hour makes the hourly mean price
exact, since settlement data is hourly. MWh weights make ĉ the cost per MWh that the
newsvendor rule needs. Using D−2 needs no assumption about how fast imbalance prices are
published. The first days after 2025-03-19 have no single-price history, hence the median.
**Rejected:** unweighted hourly means for ĉ, which overweight small imbalances; using the
morning of D−1, which depends on publication lags; a continuous τ, since the 99-level grid is
finer than the noise; intraday trading, outside the plan.

### D-038 · 2026-10-28 · Price spikes shown apart; cost split into covariance and bet
**Decision:** hours with |λ_DA − λ_imb| > 1,000 EUR/MWh (`spike_eur_mwh`; 30 hours, 0.4 %)
stay in every cost. The cost-vs-τ figure and table add a view without them. The median
bid's cost is split into a covariance part, Σ (s − s̄)(e − ē), and a mean-spread part,
n s̄ ē, with s = λ_DA − λ_imb and e = E − b.
**Why:** those 30 hours flip the sign of the mean spread (−3.2 to +2.8 EUR/MWh), and with it
the best fixed bid level, from τ = 0.01 to τ = 0.99. A single-price cost is linear in the
bid, so any level other than the median is a bet on that sign. The split shows that the
models' costs are almost all covariance: their errors co-move with the system's.
**Rejected:** removing or winsorising spike hours, which are real settlements that a bidder
pays; reporting only the hindsight-optimal τ, which would present a bet as a forecast
property.

### D-039 · 2026-10-28 · ENS inputs: availability, hourly members, feature columns
**Decision:** the 00 UTC ENS run counts as usable 9 h after its start
(`[nwp.ensemble] availability_lag_h`). Before the 11:00 issue, that is always the run of D−1:
9 h ahead in summer (09:00 UTC) and 10 h in winter. `make ens` (and `make data`, when runs are
cached) builds `nwp_ens`: per run, member and UTC hour, the regional hour means of 100 m speed
and direction, built like the HRES features. The 3-hourly u, v are linearly interpolated to
every hour; speed is taken per point and instant; each hour is the mean of its two ends;
direction comes from the mean u, v. The feature table gets `ens_run_utc`, the 51 members as
`ens<mm>_<variable>@<region>` and the members' mean and standard deviation of regional speed.
The run is chosen by the HRES rule, with the ENS lag.
**Why:** ECMWF's dissemination schedule puts the 00 UTC ENS to step 144 at 06:55–07:04 UTC
and the last step at 07:40; open data follow one hour later (checked 2026-10-28). Nine hours
also covers the last step, with margin. Keeping all members, not just their mean and spread,
lets each member pass through the power curve. The ensemble mean correlates 0.993 with the
HRES regional speed.
**Rejected:** dynamical.org's ingestion time as the availability time. It is an archive; in
operation the run would come from ECMWF open data. Also rejected: interpolating speed
instead of u, v, which is inconsistent with the HRES direction handling.

### D-040 · 2026-10-28 · ENS models: raw-ensemble benchmark and spread as a covariate
**Decision:** four models are added when the feature table has ENS inputs:
- `ens_raw`: every member through its own RLS curve, fitted on HRES like `curve_rls`, with
  the HRES issue-time correction. The members' quantiles are the forecast and their mean is
  the point.
- `quantreg_ens`, `logitnormal_ens` and `lightgbm_ens`: the Step 5 layers plus one covariate,
  s, the standard deviation of the 51 member powers through the layer's curve. Quantile
  regression adds B(p) s to its basis, the logit-normal adds θ s to log σ, and LightGBM takes
  s as a feature.

s is issued and stored like p̂. Before the first forecast, the curve's prequential spread
(before each update) fills in. The evaluation adds the CRPS change of each layer against
its twin without s, with Diebold–Mariano tests, and spread–skill curves.
**Why:** the plan's Extension A. Spread in power units, not wind, carries the curve's
nonlinearity: a wide wind spread near cut-in or rated power means little power uncertainty.
**Not done:** the optional adaptive HRES/ENS-mean combination; it is left for later.

### D-041 · 2026-10-28 · Quantile regression solver: IRLS or HiGHS, IRLS by default — *default superseded by D-042*
**Decision:** `[model.quantreg] solver` selects statsmodels' IRLS (`"irls"`, the default) or
the exact linear program solved by SciPy's HiGHS (`"highs"`). HiGHS solves the dual LP:
n variables bounded in [0, 1] and p equality constraints. The coefficients are the
constraint multipliers.
**Why:** IRLS made `quantreg_ens` take 33 minutes in the full backtest. On validation, HiGHS
is 2.8× faster for `quantreg` (65 s against 181 s) and 5× faster for `quantreg_ens` (123 s
against 629 s). It reaches the same pinball loss, with quantiles within 0.004 of IRLS on
synthetic data. The dual is about 15× faster than the primal (2n + p variables).
**Why IRLS stays the default:** validation CRPS is not significantly different (0.0344
against 0.0341, DM p = 0.27), but 99 % of that gap comes from one storm hour (2025-01-24
13:00 UTC, p̂ = 0.85). There, both solvers give nonsense quantiles for k ≥ 16; IRLS's
median, for example, is −1.17. The cause is the model, not the solver: the B-spline basis
in p spans [0, 1] with uniform knots, while the training p̂ stop near 0.88, so the top basis
functions rest on a handful of points. IRLS's early stop hides this less badly than the
exact optimum does. Switching the default would change the committed results without
fixing that.
**Open:** a quantile regression basis that stays identified at high p̂, e.g. knots over the
training range of p̂ with constant extrapolation beyond it, or a ridge penalty.

### D-042 · 2026-10-28 · Quantile regression knots at quantiles of the training p̂; HiGHS by default
**Decision:** the interior knots of the quantile regression's B-spline basis in p̂ sit at
equally spaced quantiles of the p̂ in each training window. The basis ends at the largest p̂
seen; beyond it, every quantile is constant in p̂. `[model.quantreg] solver` defaults to
`"highs"`.
**What was wrong:** the model specification, not the code and not the tuning. The basis
spanned [0, 1] with uniform knots, but in a 180-day window p̂ rarely exceeds 0.9: before
2025-01-23 its maximum was 0.884, and the largest y was 0.858. The top basis functions
therefore rested on a handful of rows, and their coefficients, which interact with the
horizon, were not identified. At p̂ = 0.85 and k = 28 the fitted median was −1.17 with
both solvers. Sorting and clipping in the backtest hid this except in storm hours. That is
the "erratic tail at high p̂" reported in Step 6 (D-036). No setting of the existing
parameters removes it: fewer knots only shrink the empty region. It was found on
validation, while comparing solvers (D-041).
**Effect on validation:** mean CRPS is unchanged (0.0341 against 0.0341, DM p = 0.87). The
storm hour of 2025-01-24 13:00 UTC improves from 0.281 to 0.079, and the quantiles at
p̂ = 0.85 become 0.51–0.80 (1–99 %) instead of −1.2 or worse. With the defect gone, the two
solvers agree to four decimals (0.0341 each). HiGHS becomes the default because it is
exact, has no iteration cap and is twice as fast (59 s against 122 s).
**Rejected:** a ridge penalty, which adds a hyper-parameter to tune; dropping the
horizon interaction, which loses the growth of spread with k.
**Effect on the test year** (full rerun, still with λ = 1; every other model unchanged): `quantreg` CRPS
0.0419 → 0.0414 (skill +66.7 % → +67.1 %); `quantreg_ens` 0.0412 → 0.0410. The ENS spread's
gain for quantile regression shrinks from −1.6 % (DM p = 0.007) to −0.9 % (p = 0.17, not
significant): part of the earlier gain was the spread compensating for the broken basis.
The full backtest takes 17 minutes instead of 55.
**Robustness:** HiGHS solves the dual LP; on small samples a rank-deficient design (e.g. 203
rows, rank 25 of 27 in the smoke run) can make its redundant constraints numerically
"infeasible". `quantile_lp` then solves the primal, which is always feasible. The full-data
run never needed the fallback.

### D-043 · 2026-10-28 · RLS covariance bounded per direction, not by its trace
**Decision:** after each daily release of rows, `RLS.bound` clips the eigenvalues of P at δ,
so P ≤ δI per direction (bounded-covariance RLS; Salgado, Goodwin & Middleton, 1988). This
replaces scaling the whole of P down whenever trace(P) > trace(P₀).
**What was wrong:** the implementation of the windup safeguard, not the model and not the
tuning.
- Some of the 64 directions get almost no data (rare speed × direction cells): even with
  λ = 1, trace(P) is ~646, mostly from them.
- With λ < 1 they inflate by 1/λ per row until the trace reaches the cap (6,400). Uniform
  scaling then also shrinks the well-observed directions and their gain, so forgetting
  stopped tracking anything.
- With λ = 0.999, the curve's April 2026 level stayed at 0.257 against 0.222 in the data
  (0.229 after the fix).

λ = 1 never reaches the cap, so the frozen and walk-forward curves are unchanged. Every
earlier λ < 1 result, including the validation tuning, compared crippled variants. Found
in Step 9, when λ = 0.999 stayed flat through Thor's registration.
**Effect:** validation now picks λ = 0.9999, about 270 days of memory (RMSE 0.06145 against
0.06149 for λ = 1). `curve_rls`, the three layers and the ENS models are rebuilt on it.
Test-year changes:

| Model | CRPS before → after |
|---|---|
| curve_rls | 0.0421 → 0.0419 (point RMSE 8.18 → 8.09 %) |
| quantreg | 0.0414 → 0.0411 |
| logitnormal | 0.0416 → 0.0411 |
| lightgbm | 0.0413 → 0.0413 |
| quantreg_ens | 0.0410 → 0.0409 |
| logitnormal_ens | 0.0416 → 0.0411 |
| lightgbm_ens | 0.0407 → 0.0406 |
| ens_raw | 0.0406 → 0.0406 |

The ENS mean is no longer a better point forecast than HRES (8.08 against 8.09 %). A unit
test now fails if an unexcited direction can stop an excited one from tracking.
**Rejected:** clipping per row (one 64 × 64 eigendecomposition per row; within a day the
inflation is only 1/λ³⁷ ≈ 1.004 at λ = 0.9999); directional forgetting (more machinery for
the same effect here).

### D-044 · 2026-10-28 · Regime-change demo: Thor and IFS 50r1, scored in MW
**Decision:** `stress.py` (`make stress`) compares, over the test year and the stress period
(2026-03-01 to the last settled day), the frozen and walk-forward curves with the RLS curve
for every λ of the grid and with `rls_reset`: tuned λ, P reset to δI at each `[stress]`
event. Forecasts in MW are the quantiles times the capacity known at issue time, against
settled production. It reports bias, MAE, RMSE and CRPS in MW, trailing 14-day bias and
CRPS, and the curve's level at 10 m/s over time.

Events, dated in advance:
- 2026-03-01: Thor in the capacity register, +1,058 MW;
- 2026-05-13: the first 00 UTC run of IFS 50r1, implemented from the 06 UTC run of 12 May,
  as confirmed by ECMWF.

New offshore output first shows on 2026-05-15: offshore production above its maximum of
the year before, plus 5 %. It is marked from data and never used by a model.
**Why:** the plan's Extension B. The register counts Thor ~2.5 months before it produces,
so y = P/C is biased low in between; MW sidesteps that. The settlement data were checked:
DK1 offshore output reaches ~2,000–2,090 MW from June 2026, against ≤ 1,650 MW before, which
is about half of Thor, consistent with 36 of 72 turbines by July.
**Findings:**
- Shorter memory tracks the change: stress bias falls from +124 MW (λ = 1) to +58 MW
  (λ = 0.999), at a cost of 240 → 245 MW test-year CRPS.
- The best λ by CRPS is 0.9999 on the stable test year and 0.9995 under change.
- The full covariance reset removes the bias (+17 MW) but has the worst stress CRPS (289 MW):
  it forgets the precision of all 64 parameters and the curve swings.
- Thor's first output and the IFS change are two days apart, so their effects cannot be
  separated here.
**Rejected:** tuning λ or the reset size on the stress period, which is the evaluation;
scoring the stress period in units of capacity.
