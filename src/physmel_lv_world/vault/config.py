from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from physmel_lv_world.vault.types import ModelShape, RunShape, TrainShape


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    return loaded


def load_run(path: str | Path) -> RunShape:
    raw = read_yaml(path)
    model_raw = raw.get("model", {})
    train_raw = raw.get("train", {})
    output_raw = raw.get("outputs", {})
    model = ModelShape(
        path_dim=int(model_raw.get("path_dim", 768)),
        gene_dim=int(model_raw.get("gene_dim", 512)),
        immune_dim=int(model_raw.get("immune_dim", 8)),
        latent_dim=int(model_raw.get("latent_dim", 128)),
        fusion_hidden=int(model_raw.get("fusion_hidden", 256)),
        residual_hidden=int(model_raw.get("residual_hidden", 64)),
        trajectory_points=int(model_raw.get("trajectory_points", 100)),
        horizon_days=float(model_raw.get("horizon_days", 365.0)),
        dropout=float(model_raw.get("dropout", 0.05)),
        use_free_ode=bool(model_raw.get("use_free_ode", False)),
    )
    seeds_raw = train_raw.get("seeds", [int(raw.get("seed", 0))])
    seeds = tuple(int(item) for item in seeds_raw)
    train = TrainShape(
        batch_size=int(train_raw.get("batch_size", 512)),
        grad_accum=int(train_raw.get("grad_accum", 1)),
        epochs=int(train_raw.get("epochs", 4000)),
        steps_per_epoch=int(train_raw.get("steps_per_epoch", 256)),
        learning_rate=float(train_raw.get("learning_rate", 1e-7)),
        weight_decay=float(train_raw.get("weight_decay", 1e-5)),
        warmup_epochs=int(train_raw.get("warmup_epochs", 200)),
        scheduler=str(train_raw.get("scheduler", "cosine")),
        optimizer=str(train_raw.get("optimizer", "AdamW")),
        grad_clip=float(train_raw.get("grad_clip", 1.0)),
        physics_weight=float(train_raw.get("physics_weight", 0.1)),
        residual_weight=float(train_raw.get("residual_weight", 0.01)),
        folds=int(train_raw.get("folds", 5)),
        seeds=seeds,
    )
    return RunShape(
        seed=int(raw.get("seed", 0)),
        model=model,
        train=train,
        run_dir=Path(output_raw.get("run_dir", "runs/main")),
        checkpoint_dir=Path(output_raw.get("checkpoint_dir", "checkpoints")),
    )
