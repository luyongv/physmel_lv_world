from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from physmel_lv_world.vault.types import OdeParameters


class ResidualField(nn.Module):
    def __init__(self, latent: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent + 2, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, latent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(torch.cat([latent, state], dim=-1)))


class LotkaVolterraField(nn.Module):
    def __init__(self, latent: int, hidden: int, free_form: bool = False) -> None:
        super().__init__()
        self.residual = ResidualField(latent, hidden)
        self.free_form = free_form
        self.free = nn.Sequential(
            nn.Linear(latent + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def mechanistic(self, state: torch.Tensor, params: OdeParameters) -> torch.Tensor:
        tumor = state[:, 0].clamp_min(0.0)
        immune = state[:, 1].clamp_min(0.0)
        d_tumor = (
            params.r * tumor * (1.0 - tumor / params.K.clamp_min(1e-4))
            - params.k * tumor * immune
        )
        d_immune = (
            params.alpha * tumor * immune
            - params.beta * immune
            + params.gamma
            - params.delta * tumor * immune
        )
        return torch.stack([d_tumor, d_immune], dim=-1)

    def forward(
        self, latent: torch.Tensor, state: torch.Tensor, params: OdeParameters
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.free_form:
            residual = self.free(torch.cat([latent, state], dim=-1))
            return residual, residual.square().mean(dim=-1)
        base = self.mechanistic(state, params)
        residual = self.residual(latent, state)
        return base + residual, residual.square().mean(dim=-1)


def euler_integrate(
    field: LotkaVolterraField,
    latent: torch.Tensor,
    initial: torch.Tensor,
    params: OdeParameters,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = [initial]
    energies = []
    current = initial
    for index in range(1, times.numel()):
        dt = times[index] - times[index - 1]
        velocity, energy = field(latent, current, params)
        current = F.softplus(current + dt * velocity)
        states.append(current)
        energies.append(energy)
    if energies:
        residual_energy = torch.stack(energies, dim=0).mean(dim=0)
    else:
        residual_energy = torch.zeros(latent.shape[0], device=latent.device)
    return torch.stack(states, dim=1), residual_energy
