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

### D-009 · 2026-10-26 · Climatology benchmark as the Step 0 model
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
The full archive (957 runs, 18,183 calls) therefore takes two days. An Open-Meteo 429 for the
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
