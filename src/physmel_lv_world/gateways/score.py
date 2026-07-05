from __future__ import annotations

import argparse
import logging

from physmel_lv_world.atlas.records import synthetic_batch
from physmel_lv_world.engines.latent import PhysMelWorld
from physmel_lv_world.trial.checkpoint import load_model_state
from physmel_lv_world.trial.loop import evaluate_batch
from physmel_lv_world.vault.config import load_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physmel-eval")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    run = load_run(args.config)
    model = PhysMelWorld(run.model)
    load_model_state(args.checkpoint, model)
    batch = synthetic_batch(
        run.train.batch_size,
        run.model.path_dim,
        run.model.gene_dim,
        run.model.immune_dim,
        run.seed + 202,
    )
    metrics = evaluate_batch(model, batch)
    logging.getLogger(__name__).info("metrics=%s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
