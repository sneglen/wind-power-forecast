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

### D-008 · 2026-10-26 · Synthetic smoke sample, replaced by real data in Step 1
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
