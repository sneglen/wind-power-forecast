# Reproducible pipeline entry points. Override the config with `make all CONFIG=path.toml`.
CONFIG ?= config.toml
RUN := uv run wpforecast

.PHONY: help install data check backtest evaluate figures all smoke sample lint test ci-watch clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Create the locked environment and install git hooks
	uv sync --locked
	uv run pre-commit install

data:  ## Download (cached, resumable), build interim tables, write data check
	$(RUN) data --config $(CONFIG)

check:  ## Re-run the data check report on the interim tables
	$(RUN) check --config $(CONFIG)

backtest:  ## Issue day-ahead forecasts over [dates]
	$(RUN) backtest --config $(CONFIG)

evaluate:  ## Score out-of-sample forecasts
	$(RUN) evaluate --config $(CONFIG)

figures:  ## Render figures from the evaluation
	$(RUN) figures --config $(CONFIG)

all:  ## Run data -> backtest -> evaluate -> figures
	$(RUN) all --config $(CONFIG)

smoke:  ## Full pipeline on the committed synthetic sample
	$(MAKE) all CONFIG=data/sample/smoke.toml

sample:  ## Re-download the committed smoke sample (real DK1 slice, needs network)
	rm -rf data/sample/raw
	$(RUN) data --config data/sample/smoke.toml --online

lint:  ## Run all pre-commit hooks on every file
	uv run pre-commit run --all-files

test:  ## Run the test suite
	uv run pytest

ci-watch:  ## After `git push`: wait for the CI run of HEAD, fail if any job fails
	scripts/ci_watch.sh

clean:  ## Remove generated artefacts (keeps data/raw)
	rm -rf data/interim outputs figures .pytest_cache .ruff_cache
