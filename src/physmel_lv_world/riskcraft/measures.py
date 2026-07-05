from __future__ import annotations

import math

import torch

from physmel_lv_world.vault.types import MetricBundle


def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    count = 0.0
    score = 0.0
    n = risk.numel()
    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] > 0.5:
                count += 1.0
                if risk[i] > risk[j]:
                    score += 1.0
                elif risk[i] == risk[j]:
                    score += 0.5
    if count == 0.0:
        return float("nan")
    return score / count


def brier_score(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, horizon: float
) -> float:
    prob = torch.sigmoid(risk)
    observed = ((time <= horizon) & (event > 0.5)).float()
    return float((prob - observed).square().mean().item())


def binary_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    pos = score[label > 0.5]
    neg = score[label <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    pairs = (pos.view(-1, 1) > neg.view(1, -1)).float()
    ties = (pos.view(-1, 1) == neg.view(1, -1)).float() * 0.5
    return float((pairs + ties).mean().item())


def time_auc(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, horizon: float
) -> float:
    label = ((time <= horizon) & (event > 0.5)).float()
    return binary_auc(risk, label)


def metric_bundle(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> MetricBundle:
    cindex = concordance_index(risk, time, event)
    brier = brier_score(risk, time, event, 1095.0)
    auc1 = time_auc(risk, time, event, 365.0)
    auc3 = time_auc(risk, time, event, 1095.0)
    auc5 = time_auc(risk, time, event, 1825.0)
    return MetricBundle(c_index=cindex, brier=brier, auc_1y=auc1, auc_3y=auc3, auc_5y=auc5)


def bootstrap_cindex(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, draws: int, seed: int
) -> tuple[float, float, float]:
    gen = torch.Generator().manual_seed(seed)
    values: list[float] = []
    n = risk.numel()
    for _ in range(draws):
        index = torch.randint(0, n, (n,), generator=gen)
        value = concordance_index(risk[index], time[index], event[index])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), float("nan")
    tensor = torch.tensor(values)
    return (
        float(tensor.mean().item()),
        float(tensor.quantile(0.025).item()),
        float(tensor.quantile(0.975).item()),
    )
