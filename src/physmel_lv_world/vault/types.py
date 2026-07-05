from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

Endpoint = Literal["os", "pfs", "mfs"]
Regime = Literal["control", "escape", "oscillatory"]


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    version: str
    license_or_policy: str
    access: str


@dataclass(frozen=True)
class ManifestRow:
    case_id: str
    path_embed: Path | None
    gene_embed: Path | None
    immune_fraction: Path | None
    time: float
    event: int
    fold: int
    endpoint: Endpoint
    site: str
    region: str


@dataclass(frozen=True)
class Batch:
    pathology: torch.Tensor
    genomics: torch.Tensor
    immune: torch.Tensor
    time: torch.Tensor
    event: torch.Tensor
    case_id: tuple[str, ...]


@dataclass(frozen=True)
class ModelShape:
    path_dim: int
    gene_dim: int
    immune_dim: int
    latent_dim: int
    fusion_hidden: int
    residual_hidden: int
    trajectory_points: int
    horizon_days: float
    dropout: float
    use_free_ode: bool


@dataclass(frozen=True)
class TrainShape:
    batch_size: int
    grad_accum: int
    epochs: int
    steps_per_epoch: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    scheduler: str
    optimizer: str
    grad_clip: float
    physics_weight: float
    residual_weight: float
    folds: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class RunShape:
    seed: int
    model: ModelShape
    train: TrainShape
    run_dir: Path
    checkpoint_dir: Path


@dataclass(frozen=True)
class OdeParameters:
    r: torch.Tensor
    K: torch.Tensor
    k: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor


@dataclass(frozen=True)
class TrajectoryBundle:
    states: torch.Tensor
    times: torch.Tensor
    params: OdeParameters
    residual_energy: torch.Tensor


@dataclass(frozen=True)
class RiskBundle:
    risk: torch.Tensor
    latent: torch.Tensor
    trajectory: TrajectoryBundle
    features: torch.Tensor


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    cox: torch.Tensor
    physics: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class MetricBundle:
    c_index: float
    brier: float
    auc_1y: float
    auc_3y: float
    auc_5y: float
