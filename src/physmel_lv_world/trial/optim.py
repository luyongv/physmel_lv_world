from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch.optim import AdamW, Optimizer

from physmel_lv_world.vault.types import TrainShape


def build_optimizer(parameters: object, train: TrainShape) -> Optimizer:
    typed = cast(Iterable[torch.Tensor] | Iterable[dict[str, Any]], parameters)
    return AdamW(typed, lr=train.learning_rate, weight_decay=train.weight_decay)


class WarmCosine:
    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, total_steps: int, base_lr: float
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.base_lr = base_lr
        self.step_index = 0

    def step(self) -> float:
        self.step_index += 1
        if self.step_index <= self.warmup_steps:
            lr = self.base_lr * self.step_index / self.warmup_steps
        else:
            progress = (self.step_index - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def build_scheduler(optimizer: Optimizer, train: TrainShape) -> WarmCosine:
    warmup = train.warmup_epochs * train.steps_per_epoch
    total = train.epochs * train.steps_per_epoch
    return WarmCosine(optimizer, warmup, total, train.learning_rate)
