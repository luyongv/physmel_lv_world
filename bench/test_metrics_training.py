import torch

from physmel_lv_world.atlas.records import synthetic_batch
from physmel_lv_world.riskcraft.measures import (
    bootstrap_cindex,
    concordance_index,
    metric_bundle,
)
from physmel_lv_world.trial.loop import fit_synthetic
from physmel_lv_world.vault.config import load_run


def test_metrics_return_values() -> None:
    risk = torch.tensor([0.9, 0.2, 0.7, -0.1])
    time = torch.tensor([1.0, 5.0, 2.0, 7.0])
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])
    assert concordance_index(risk, time, event) >= 0.0
    metrics = metric_bundle(risk, time, event)
    assert metrics.brier >= 0.0


def test_bootstrap_cindex() -> None:
    batch = synthetic_batch(10, 6, 5, 3, 2)
    risk = torch.linspace(-1.0, 1.0, 10)
    mean, low, high = bootstrap_cindex(risk, batch.time, batch.event, 8, 1)
    assert low <= mean <= high


def test_tiny_training_runs() -> None:
    run = load_run("settings/checks/tiny.yaml")
    model, losses = fit_synthetic(run, steps=2)
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert sum(p.numel() for p in model.parameters()) > 0
