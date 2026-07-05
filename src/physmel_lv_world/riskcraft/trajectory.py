from __future__ import annotations

import torch

from physmel_lv_world.vault.types import OdeParameters


def trapezoid(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    widths = x[1:] - x[:-1]
    mids = 0.5 * (y[:, 1:] + y[:, :-1])
    return (mids * widths.view(1, -1)).sum(dim=1)


def critical_time(
    states: torch.Tensor, times: torch.Tensor, params: OdeParameters
) -> torch.Tensor:
    tumor = states[:, :, 0]
    threshold = 0.5 * params.K.view(-1, 1)
    crossed = tumor >= threshold
    index = crossed.float().argmax(dim=1)
    any_crossed = crossed.any(dim=1)
    selected = times[index]
    terminal = float(times[-1].detach().cpu().item())
    return torch.where(any_crossed, selected, torch.full_like(selected, terminal))


def regime_one_hot(states: torch.Tensor, params: OdeParameters) -> torch.Tensor:
    endpoint = states[:, -1, :]
    tumor = endpoint[:, 0]
    immune = endpoint[:, 1]
    escape = tumor > 0.5 * params.K
    control = (tumor <= 0.5 * params.K) & (immune >= immune.median())
    oscillatory = ~(escape | control)
    return torch.stack([control.float(), escape.float(), oscillatory.float()], dim=-1)


def trajectory_features(
    states: torch.Tensor, times: torch.Tensor, params: OdeParameters
) -> torch.Tensor:
    tumor = states[:, :, 0]
    immune = states[:, :, 1]
    auct = trapezoid(tumor, times).unsqueeze(-1)
    auci = trapezoid(immune, times).unsqueeze(-1)
    regime = regime_one_hot(states, params)
    tau = critical_time(states, times, params).unsqueeze(-1) / times[-1].clamp_min(1.0)
    return torch.cat([auct, auci, regime, tau], dim=-1)


def positivity_penalty(states: torch.Tensor) -> torch.Tensor:
    return torch.relu(-states).square().mean()


def bounded_penalty(states: torch.Tensor, params: OdeParameters) -> torch.Tensor:
    tumor = states[:, :, 0]
    immune = states[:, :, 1]
    tumor_bound = torch.relu(tumor - params.K.view(-1, 1)).square().mean()
    immune_bound = (
        torch.relu(immune - immune.detach().quantile(0.95).clamp_min(1.0)).square().mean()
    )
    return tumor_bound + immune_bound


def smoothness_penalty(states: torch.Tensor) -> torch.Tensor:
    if states.shape[1] < 3:
        return states.new_tensor(0.0)
    second = states[:, 2:] - 2.0 * states[:, 1:-1] + states[:, :-2]
    return second.square().mean()
