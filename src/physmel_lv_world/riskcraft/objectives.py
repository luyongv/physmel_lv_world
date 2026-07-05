from __future__ import annotations

import torch

from physmel_lv_world.riskcraft.trajectory import (
    bounded_penalty,
    positivity_penalty,
    smoothness_penalty,
)
from physmel_lv_world.vault.types import LossBundle, RiskBundle


def cox_partial_loss(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk_ordered = risk[order]
    event_ordered = event[order]
    log_cumsum = torch.logcumsumexp(risk_ordered, dim=0)
    uncensored = event_ordered > 0.5
    if not torch.any(uncensored):
        return risk.sum() * 0.0
    losses = -(risk_ordered[uncensored] - log_cumsum[uncensored])
    return losses.mean()


def physics_loss(bundle: RiskBundle) -> torch.Tensor:
    states = bundle.trajectory.states
    params = bundle.trajectory.params
    return (
        positivity_penalty(states)
        + bounded_penalty(states, params)
        + smoothness_penalty(states)
    )


def composite_loss(
    bundle: RiskBundle,
    time: torch.Tensor,
    event: torch.Tensor,
    physics_weight: float,
    residual_weight: float,
) -> LossBundle:
    cox = cox_partial_loss(bundle.risk, time, event)
    physics = physics_loss(bundle)
    residual = bundle.trajectory.residual_energy.mean()
    total = cox + physics_weight * physics + residual_weight * residual
    return LossBundle(total=total, cox=cox, physics=physics, residual=residual)
