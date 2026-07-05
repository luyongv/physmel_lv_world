from __future__ import annotations

import torch
from torch import nn

from physmel_lv_world.engines.heads import CoxHead, OdeParameterHead
from physmel_lv_world.engines.lotka import LotkaVolterraField, euler_integrate
from physmel_lv_world.riskcraft.trajectory import trajectory_features
from physmel_lv_world.signal.fusion import GatedFusion
from physmel_lv_world.vault.types import Batch, ModelShape, RiskBundle, TrajectoryBundle


class PhysMelWorld(nn.Module):
    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.shape = shape
        self.fusion = GatedFusion(
            shape.path_dim,
            shape.gene_dim,
            shape.immune_dim,
            shape.fusion_hidden,
            shape.latent_dim,
            shape.dropout,
        )
        self.initial = nn.Sequential(
            nn.Linear(shape.latent_dim, shape.residual_hidden),
            nn.GELU(),
            nn.Linear(shape.residual_hidden, 2),
        )
        self.param_head = OdeParameterHead(shape.latent_dim, shape.residual_hidden)
        self.field = LotkaVolterraField(
            shape.latent_dim, shape.residual_hidden, shape.use_free_ode
        )
        self.gru = nn.GRUCell(2, shape.latent_dim)
        self.cox = CoxHead(shape.latent_dim, 6)

    def forward(self, batch: Batch) -> RiskBundle:
        latent, _ = self.fusion(batch.pathology, batch.genomics, batch.immune)
        params = self.param_head(latent)
        initial = torch.nn.functional.softplus(self.initial(latent)) + 1e-3
        times = torch.linspace(
            0.0, self.shape.horizon_days, self.shape.trajectory_points, device=latent.device
        )
        states, residual_energy = euler_integrate(self.field, latent, initial, params, times)
        h = latent
        stride = max(1, states.shape[1] // 8)
        for step in range(0, states.shape[1], stride):
            h = self.gru(states[:, step, :], h)
        features = trajectory_features(states, times, params)
        risk = self.cox(h, features)
        bundle = TrajectoryBundle(
            states=states, times=times, params=params, residual_energy=residual_energy
        )
        return RiskBundle(risk=risk, latent=h, trajectory=bundle, features=features)
