# Reproducible pipeline entry points. Override the config with `make all CONFIG=path.toml`.
CONFIG ?= config.toml
RUN := uv run wpforecast

.PHONY: help install data backtest evaluate figures all smoke sample lint test ci-watch clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Create the locked environment and install git hooks
	uv sync --locked
	uv run pre-commit install

data:  ## Load and validate raw data -> interim dataset
	$(RUN) data --config $(CONFIG)

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

sample:  ## Regenerate the committed synthetic sample
	uv run python scripts/make_synthetic_sample.py --config $(CONFIG)

lint:  ## Run all pre-commit hooks on every file
	uv run pre-commit run --all-files

test:  ## Run the test suite
	uv run pytest

ci-watch:  ## After `git push`: wait for the CI run of HEAD, fail if any job fails
	scripts/ci_watch.sh

clean:  ## Remove generated artefacts (keeps data/raw)
	rm -rf data/interim outputs figures .pytest_cache .ruff_cache
