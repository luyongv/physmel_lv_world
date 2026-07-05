from __future__ import annotations

import argparse
import logging

from physmel_lv_world.trial.loop import train_and_save
from physmel_lv_world.vault.config import load_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physmel-train")
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    run = load_run(args.config)
    metrics = train_and_save(run, args.steps)
    logging.getLogger(__name__).info("metrics=%s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
