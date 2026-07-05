from __future__ import annotations

import logging

import torch

from physmel_lv_world.atlas.records import synthetic_batch
from physmel_lv_world.engines.latent import PhysMelWorld
from physmel_lv_world.riskcraft.measures import metric_bundle
from physmel_lv_world.riskcraft.objectives import composite_loss
from physmel_lv_world.trial.checkpoint import save_checkpoint
from physmel_lv_world.trial.optim import build_optimizer, build_scheduler
from physmel_lv_world.vault.seed import set_seed
from physmel_lv_world.vault.types import Batch, RunShape

LOGGER = logging.getLogger(__name__)


def to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        pathology=batch.pathology.to(device),
        genomics=batch.genomics.to(device),
        immune=batch.immune.to(device),
        time=batch.time.to(device),
        event=batch.event.to(device),
        case_id=batch.case_id,
    )


def fit_synthetic(run: RunShape, steps: int | None = None) -> tuple[PhysMelWorld, list[float]]:
    set_seed(run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysMelWorld(run.model).to(device)
    optimizer = build_optimizer(model.parameters(), run.train)
    scheduler = build_scheduler(optimizer, run.train)
    total_steps = steps if steps is not None else run.train.epochs * run.train.steps_per_epoch
    losses: list[float] = []
    model.train()
    for step in range(total_steps):
        batch = synthetic_batch(
            run.train.batch_size,
            run.model.path_dim,
            run.model.gene_dim,
            run.model.immune_dim,
            run.seed + step,
        )
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        bundle = model(batch)
        loss = composite_loss(
            bundle, batch.time, batch.event, run.train.physics_weight, run.train.residual_weight
        )
        torch.autograd.backward(loss.total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), run.train.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.total.detach().cpu().item()))
        LOGGER.info("step=%s loss=%.6f", step, losses[-1])
    return model, losses


def evaluate_batch(model: PhysMelWorld, batch: Batch) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        moved = to_device(batch, device)
        bundle = model(moved)
        metrics = metric_bundle(bundle.risk.cpu(), moved.time.cpu(), moved.event.cpu())
    return {
        "c_index": metrics.c_index,
        "brier": metrics.brier,
        "auc_1y": metrics.auc_1y,
        "auc_3y": metrics.auc_3y,
        "auc_5y": metrics.auc_5y,
    }


def train_and_save(run: RunShape, steps: int | None = None) -> dict[str, float]:
    model, losses = fit_synthetic(run, steps)
    batch = synthetic_batch(
        run.train.batch_size,
        run.model.path_dim,
        run.model.gene_dim,
        run.model.immune_dim,
        run.seed + 999,
    )
    metrics = evaluate_batch(model, batch)
    metrics["final_loss"] = losses[-1]
    optimizer = build_optimizer(model.parameters(), run.train)
    save_checkpoint(run.checkpoint_dir / "main.pt", model, optimizer, run, len(losses), metrics)
    return metrics
