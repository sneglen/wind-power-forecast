"""Command-line entry point: ``wpforecast <stage> --config config.toml``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from wpforecast.config import load_config
from wpforecast.pipeline import STAGES, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wpforecast", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--config", default="config.toml", help="path to config TOML")
    parser.add_argument(
        "--online", action="store_true", help="allow downloads even if [data] offline = true"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("matplotlib", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
        if args.online:
            cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, offline=False))
        run_all(cfg) if args.stage == "all" else STAGES[args.stage](cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.getLogger("wpforecast").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
