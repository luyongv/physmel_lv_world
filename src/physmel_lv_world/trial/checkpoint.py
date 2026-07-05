from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.optim import Optimizer

from physmel_lv_world.vault.types import RunShape

PlainValue = str | int | float | bool | None | list["PlainValue"] | dict[str, "PlainValue"]


def plain_value(value: object) -> PlainValue:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_value(item) for item in value]
    if isinstance(value, list):
        return [plain_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    run: RunShape,
    step: int,
    metrics: dict[str, float],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": run.seed,
        "step": step,
        "metrics": metrics,
        "config": plain_value(asdict(run)),
    }
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_model_state(path: str | Path, model: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    return cast(dict[str, Any], payload)
