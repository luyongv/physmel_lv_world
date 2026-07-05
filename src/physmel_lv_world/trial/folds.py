from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldRun:
    fold: int
    seed: int
    train_name: str
    valid_name: str


def fold_plan(folds: int, seeds: tuple[int, ...]) -> list[FoldRun]:
    plan: list[FoldRun] = []
    for fold in range(folds):
        for seed in seeds:
            plan.append(
                FoldRun(
                    fold=fold,
                    seed=seed,
                    train_name=f"fold{fold}_seed{seed}_train",
                    valid_name=f"fold{fold}_seed{seed}_valid",
                )
            )
    return plan


def effective_batch(batch_size: int, grad_accum: int, world_size: int) -> int:
    return batch_size * grad_accum * world_size
