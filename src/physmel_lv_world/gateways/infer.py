from __future__ import annotations

import argparse
import logging

from physmel_lv_world.engines.latent import PhysMelWorld
from physmel_lv_world.trial.checkpoint import load_model_state
from physmel_lv_world.vault.config import load_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physmel-infer")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    run = load_run(args.config)
    model = PhysMelWorld(run.model)
    payload = load_model_state(args.checkpoint, model)
    logging.getLogger(__name__).info("loaded_step=%s", payload.get("step"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
