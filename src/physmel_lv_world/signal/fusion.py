from __future__ import annotations

from typing import cast

import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(
        self,
        path_dim: int,
        gene_dim: int,
        immune_dim: int,
        hidden: int,
        latent: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.path = nn.Sequential(
            nn.LayerNorm(path_dim), nn.Linear(path_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.gene = nn.Sequential(
            nn.LayerNorm(gene_dim), nn.Linear(gene_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.immune = nn.Sequential(
            nn.LayerNorm(immune_dim),
            nn.Linear(immune_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent),
        )

    def forward(
        self, pathology: torch.Tensor, genomics: torch.Tensor, immune: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.path(pathology)
        g = self.gene(genomics)
        i = self.immune(immune)
        joined = torch.cat([p, g, i], dim=-1)
        weights = torch.softmax(self.gate(joined), dim=-1)
        fused = torch.cat(
            [p * weights[:, 0:1], g * weights[:, 1:2], i * weights[:, 2:3]], dim=-1
        )
        return cast(torch.Tensor, self.out(fused)), weights


class MaskedFusion(nn.Module):
    def __init__(self, base: GatedFusion) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        pathology: torch.Tensor,
        genomics: torch.Tensor,
        immune: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pathology = pathology * mask[:, 0:1]
        genomics = genomics * mask[:, 1:2]
        immune = immune * mask[:, 2:3]
        return cast(tuple[torch.Tensor, torch.Tensor], self.base(pathology, genomics, immune))
