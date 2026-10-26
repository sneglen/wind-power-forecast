# Project rules

DK1 day-ahead wind power as calibrated, self-adapting predictive distributions, plus a DK1
single-price imbalance backtest. The project is built in numbered steps; each step's README
section is filled in when the step finishes.

## Commands
- `make install` · `make lint` · `make test` · `make smoke` · `make all`
- After every push: `git push && make ci-watch`. A step is done only when CI is green for
  the pushed commit; on failure, read `gh run view <id> --log-failed`, fix, push, watch again.

## Rules
- **Issue-time discipline:** a forecast may use only data available at its issue time
  (11:00 Europe/Copenhagen on D−1). Every new input gets a test that fails if a value stamped
  at or after the issue time can reach the forecast.
- **Time:** UTC internally, hour-beginning stamps; horizon k counts to the end of the target
  hour (DECISIONS.md D-012).
- **Config:** dates, points, quantiles and hyper-parameters live in `config.toml` only; no
  magic numbers in code. New keys are typed and validated in `src/wpforecast/config.py`.
- **Decisions:** every design choice gets a `DECISIONS.md` entry (decision, why, rejected).
  Superseded entries are marked, not deleted.
- **Data:** nothing under `data/raw` or `data/interim` is committed; only the small sample in
  `data/sample/`, which must match its generator.
- **Code:** lean and tested; match the surrounding style (ruff, line length 100).
