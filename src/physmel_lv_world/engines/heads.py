from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from physmel_lv_world.vault.types import OdeParameters


class OdeParameterHead(nn.Module):
    def __init__(self, latent: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent), nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 7)
        )

    def forward(self, z: torch.Tensor) -> OdeParameters:
        raw = self.net(z)
        positive = F.softplus(raw) + 1e-4
        return OdeParameters(
            r=positive[:, 0],
            K=positive[:, 1] + 1.0,
            k=positive[:, 2],
            alpha=positive[:, 3],
            beta=positive[:, 4],
            delta=positive[:, 5],
            gamma=positive[:, 6],
        )


class CoxHead(nn.Module):
    def __init__(self, latent: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent + feature_dim),
            nn.Linear(latent + feature_dim, latent),
            nn.GELU(),
            nn.Linear(latent, 1),
        )

    def forward(self, latent: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(torch.cat([latent, features], dim=-1))).squeeze(-1)
