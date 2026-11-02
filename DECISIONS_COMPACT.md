# Decisions, compacted

What holds at v1.0, without the history. [DECISIONS.md](DECISIONS.md) remains the full log,
with each decision's reasons, rejected alternatives and evidence; `D-NNN` refers to its
entries. The first section lists what was later changed or reversed.

## Changed along the way

| Earlier | Replaced by | What changed |
|---|---|---|
| D-006: issue at 10:00, lead times 14–38 h | D-011, D-012 | Issue at 11:00, from the project plan; k counts to the end of the hour, product k = 14–37. |
| D-008: synthetic smoke sample | D-014 | A committed 24-day slice of real data, covering the single-price switch and a DST day. |
| D-009: 28-day rolling climatology | D-026 | Three baselines, refit monthly on all known data. |
| D-018: stop when the daily call budget is spent | D-032 | Wait up to 60 min for the rolling 24 h window to free calls. |
| D-025: open question, is the ENS run out before the issue? | D-039 | Yes: usable 9 h after its start. |
| D-028: RLS covariance capped by its trace | D-043 | Eigenvalues clipped at δ per direction. The trace cap shrank every gain, so forgetting tracked nothing (a code defect). |
| D-028, D-034: "λ = 1 wins on validation; no drift before Thor" | D-043 | An artefact of the trace cap. After the fix, validation picks λ = 0.9999. |
| D-030: quantile regression by statsmodels, uniform knots over [0, 1] | D-042 | Knots at quantiles of the training p̂, constant beyond its maximum. Uniform knots left the top basis functions unidentified (a model defect). |
| D-031: HiGHS rejected as 16× slower than statsmodels | D-041, D-042 | Solving the dual LP made HiGHS 2× faster than IRLS; it is the default since D-042. |
| D-041: IRLS as the default solver | D-042 | With the knots fixed, both solvers agree; HiGHS is exact and faster. |
| D-034: open calibration items for Step 6 | D-036, D-042 | Reported in the evaluation; quantile regression's erratic tail at high p̂ fixed by D-042. |
| D-028, D-030: evidence on data to 2025-06-08 | D-036, D-043 | Superseded by the full test-year numbers, e.g. RMSE 8.2 % → 8.09 %. |

**On the test protocol.** D-036 scores the test year once with every hyper-parameter frozen.
The two defects (D-042, D-043) were found afterwards, on validation and in the stress period,
not on the test year. After each fix the test year was rerun with nothing else changed, and
λ was re-chosen on validation.

## Current decisions

### Tooling and reproducibility
- Python 3.12.15 pinned; uv with a committed `uv.lock`; CI installs with `uv sync --locked`
  (D-001, D-002, D-013).
- src layout, package `wpforecast`, one CLI that Make, CI and users all call (D-003).
- pandas for series, xarray for NWP (D-004). MIT licence, English (D-010).
- One `config.toml`, parsed by `tomllib` into frozen, validated dataclasses; variants inherit
  with `base = "..."` (D-005).
- CI: lint, tests and an offline smoke run on a committed real slice, 2025-03-08 to 03-31.
  Every push is followed by `make ci-watch` (D-013, D-014).
- Raw data as a release snapshot (`data-2026-10-25`, SHA-256, `make data-snapshot`), never in
  git (D-035). `make all` reproduces every committed report and figure byte for byte (D-045).

### Time and issue-time discipline
- UTC internally, hour-beginning stamps. Issue at 11:00 Europe/Copenhagen on D−1, before the
  12:00 gate closure; production is known up to the issue time (D-011).
- k counts from the issue to the end of the target hour. The product, local day D, is
  k = 14–37 (14–36 and 14–38 on DST days); `max_horizon_h = 38` (D-012).
- 37 feature rows per issue, k = 1 to the end of D; scores use product hours only (D-020).
- The harness releases known rows, then fits or updates, then predicts, so leakage is
  structurally impossible; tests corrupt post-issue data and require unchanged forecasts
  (D-022, D-023).
- Capacity and persistence are read at the last known hour; C(t) at the target hour only
  normalises the target (D-022).
- Without a usable weather run, no weather-based model forecasts, LightGBM included (D-034).

### Data
- Target days: training 2024-04-01 to 2025-02-28, its last three months validation; test
  2025-03-01 to 2026-02-28; stress from 2026-03-01 (D-011).
- Target: DK1 onshore + offshore production ÷ capacity, one hourly total (D-011). Monthly
  capacity from Energinet's CapacityPerMunicipality; Thor counts in full from March 2026,
  ahead of its output (D-015).
- ECMWF IFS HRES, 00 UTC, via Open-Meteo, usable 7 h after the run (D-011, D-016).
  19 points: 12 onshore on a 0.5° grid, 7 offshore farms, on land or sea cells within
  7.3 km (D-016). Seven variables; the curve uses 100 m speed and direction (D-017).
- Free Open-Meteo tier: shared call ledger below 4,500/h and 9,800/day, waiting up to
  60 min for the daily window (D-018, D-032). Responses cached as received; `offline = true`
  forbids the network (D-014).
- Data check script with a Markdown report (D-019).
- Each issue takes the latest usable run that covers its hours, falling back to D−2's; hour
  means are trapezoids of u and v (D-021). A run with any gap in 100 m wind counts as
  missing (D-033).
- Curtailment: hours with a day-ahead price ≤ 0 never reach a model and are scored with and
  without; prices are never inputs (D-024).

### Models
- Quantile grids: 19 reporting levels, 0.05–0.95; 99 levels for CRPS. Quantile regression
  and LightGBM fit 23 levels (adding 0.01, 0.02, 0.98, 0.99) and interpolate in τ
  (D-007, D-027, D-031).
- Baselines: climatology, persistence and the new reference, refit monthly (D-026).
- Power curve: cubic B-splines of regional 100 m speed (6 interior knots; 0–20 m/s onshore,
  0–28 m/s offshore) × [1, cos θ, sin θ], plus two diurnal harmonics: 64 parameters (D-028).
  - RLS with forgetting λ, P₀ = δI (δ = 100), eigenvalues of P clipped at δ once a day
    (D-028, D-043).
  - λ = 1 gives the frozen and walk-forward twins and equals ridge regression (D-028).
  - λ is chosen on validation by point RMSE: 0.9999, about 270 days of memory (D-043).
- Short-horizon correction a_k per horizon block; empirical error quantiles from prequential
  residuals over 90 days (D-029).
- Distribution layers, trained on p̂ as issued (D-030):
  - quantile regression on [B(p̂), B(p̂)·k/38], knots at quantiles of the training p̂,
    constant beyond its maximum; 180-day window, weekly refit; HiGHS dual LP with a primal
    fallback, IRLS selectable (D-041, D-042);
  - censored generalised logit-normal, maximum likelihood, monthly refit;
  - LightGBM quantiles on p̂, k, calendar terms and ten NWP features; no monotone constraint;
    4 threads; monthly refit.
- Ensemble: 51 ENS members at the 19 points from dynamical.org (`make ens`), usable 9 h after
  the 00 UTC run, interpolated to hourly in u and v (D-025, D-039). `ens_raw` passes every
  member through the RLS curve; the `*_ens` layers add the spread of member power (D-040).

### Evaluation
- The test year on hours every model forecast, with and without curtailed hours: CRPS and
  skill with day-block bootstrap bands, MAE and RMSE, interval coverage and width,
  reliability, PIT, and Diebold–Mariano tests on daily CRPS (D-027, D-036).
- Energinet's forecast as published and recalibrated on the training period (D-036).

### Market
- A 100 MW price-taker over 2025-03-19 to 2026-02-28, settled at the hourly mean of the
  15-min imbalance prices. Cost is the revenue lost against a perfect bid. Bids: mean,
  median, trailing-best τ and the newsvendor τ, from the 60 days ending on D−2 (D-037).
- Spike hours (|λ_DA − λ_imb| > 1,000 EUR/MWh) stay in and are also shown apart. The cost is
  split into a covariance part and a mean-spread bet (D-038).

### Regime change and report
- Thor and IFS 50r1, scored in MW, events dated in advance; frozen, walk-forward, RLS over
  the λ grid and a covariance reset. Nothing is tuned on the stress period (D-044).
- Headline figures committed in `reports/figures/`; THEORY.md; v1.0 (D-045).

### Deferred
- The conditional-parametric curve (D-028); the recursive-likelihood logit-normal and the
  beta check (D-030); the adaptive HRES/ENS-mean combination (D-040); a partial covariance
  reset (D-044).
