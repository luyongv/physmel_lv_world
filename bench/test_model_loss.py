import torch

from physmel_lv_world.atlas.records import synthetic_batch
from physmel_lv_world.engines.latent import PhysMelWorld
from physmel_lv_world.riskcraft.objectives import composite_loss, cox_partial_loss
from physmel_lv_world.vault.types import ModelShape


def small_shape() -> ModelShape:
    return ModelShape(
        path_dim=12,
        gene_dim=10,
        immune_dim=4,
        latent_dim=16,
        fusion_hidden=20,
        residual_hidden=12,
        trajectory_points=12,
        horizon_days=30.0,
        dropout=0.0,
        use_free_ode=False,
    )


def test_forward_bundle_shapes() -> None:
    model = PhysMelWorld(small_shape())
    batch = synthetic_batch(5, 12, 10, 4, 3)
    bundle = model(batch)
    assert bundle.risk.shape == (5,)
    assert bundle.trajectory.states.shape == (5, 12, 2)
    assert bundle.features.shape == (5, 6)


def test_cox_loss_is_finite() -> None:
    risk = torch.tensor([0.1, 0.4, -0.2, 0.7])
    time = torch.tensor([4.0, 3.0, 2.0, 1.0])
    event = torch.tensor([1.0, 0.0, 1.0, 1.0])
    loss = cox_partial_loss(risk, time, event)
    assert torch.isfinite(loss)


def test_composite_loss_backward() -> None:
    model = PhysMelWorld(small_shape())
    batch = synthetic_batch(6, 12, 10, 4, 4)
    bundle = model(batch)
    loss = composite_loss(bundle, batch.time, batch.event, 0.1, 0.01)
    loss.total.backward()
    assert torch.isfinite(loss.total)
