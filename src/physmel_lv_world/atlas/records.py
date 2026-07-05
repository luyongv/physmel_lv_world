from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset

from physmel_lv_world.vault.types import Batch, Endpoint, ManifestRow


def parse_endpoint(value: str) -> Endpoint:
    if value == "os" or value == "pfs" or value == "mfs":
        return cast(Endpoint, value)
    raise ValueError("endpoint must be os, pfs, or mfs")


def parse_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                ManifestRow(
                    case_id=str(raw["case_id"]),
                    path_embed=Path(raw["path_embed"]) if raw.get("path_embed") else None,
                    gene_embed=Path(raw["gene_embed"]) if raw.get("gene_embed") else None,
                    immune_fraction=(
                        Path(raw["immune_fraction"]) if raw.get("immune_fraction") else None
                    ),
                    time=float(raw["time"]),
                    event=int(raw["event"]),
                    fold=int(raw.get("fold", 0)),
                    endpoint=parse_endpoint(str(raw.get("endpoint", "os"))),
                    site=str(raw.get("site", "unknown")),
                    region=str(raw.get("region", "unknown")),
                )
            )
    return rows


def filter_endpoint(rows: Iterable[ManifestRow], endpoint: Endpoint) -> list[ManifestRow]:
    return [row for row in rows if row.endpoint == endpoint]


def filter_fold(rows: Iterable[ManifestRow], fold: int, holdout: bool) -> list[ManifestRow]:
    if holdout:
        return [row for row in rows if row.fold == fold]
    return [row for row in rows if row.fold != fold]


def load_tensor(path: Path | None, width: int) -> torch.Tensor:
    if path is None:
        return torch.zeros(width, dtype=torch.float32)
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "embedding" in value:
        tensor = value["embedding"]
    else:
        tensor = value
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
    tensor = tensor.float().flatten()
    if tensor.numel() == width:
        return cast(torch.Tensor, tensor)
    if tensor.numel() > width:
        return cast(torch.Tensor, tensor[:width])
    pad = torch.zeros(width - tensor.numel(), dtype=torch.float32)
    return torch.cat([tensor, pad], dim=0)


class FeatureManifestDataset(Dataset[dict[str, object]]):
    def __init__(
        self, rows: list[ManifestRow], path_dim: int, gene_dim: int, immune_dim: int
    ) -> None:
        self.rows = rows
        self.path_dim = path_dim
        self.gene_dim = gene_dim
        self.immune_dim = immune_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "case_id": row.case_id,
            "pathology": load_tensor(row.path_embed, self.path_dim),
            "genomics": load_tensor(row.gene_embed, self.gene_dim),
            "immune": load_tensor(row.immune_fraction, self.immune_dim),
            "time": torch.tensor(row.time, dtype=torch.float32),
            "event": torch.tensor(row.event, dtype=torch.float32),
        }


def collate_feature_batch(items: list[dict[str, object]]) -> Batch:
    return Batch(
        pathology=torch.stack(
            [item["pathology"] for item in items if isinstance(item["pathology"], torch.Tensor)]
        ),
        genomics=torch.stack(
            [item["genomics"] for item in items if isinstance(item["genomics"], torch.Tensor)]
        ),
        immune=torch.stack(
            [item["immune"] for item in items if isinstance(item["immune"], torch.Tensor)]
        ),
        time=torch.stack(
            [item["time"] for item in items if isinstance(item["time"], torch.Tensor)]
        ),
        event=torch.stack(
            [item["event"] for item in items if isinstance(item["event"], torch.Tensor)]
        ),
        case_id=tuple(str(item["case_id"]) for item in items),
    )


def synthetic_batch(
    size: int, path_dim: int, gene_dim: int, immune_dim: int, seed: int
) -> Batch:
    gen = torch.Generator().manual_seed(seed)
    pathology = torch.randn(size, path_dim, generator=gen)
    genomics = torch.randn(size, gene_dim, generator=gen)
    immune = torch.rand(size, immune_dim, generator=gen)
    linear = pathology[:, 0] * 0.4 + genomics[:, 0] * 0.2 - immune[:, 0] * 0.3
    time = torch.clamp(60.0 + 20.0 * torch.randn(size, generator=gen) - 8.0 * linear, min=1.0)
    event = (linear + 0.2 * torch.randn(size, generator=gen) > linear.median()).float()
    return Batch(
        pathology=pathology,
        genomics=genomics,
        immune=immune,
        time=time,
        event=event,
        case_id=tuple(f"case_{idx}" for idx in range(size)),
    )
