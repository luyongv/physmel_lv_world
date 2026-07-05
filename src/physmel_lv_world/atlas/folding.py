from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from physmel_lv_world.vault.types import ManifestRow


def assign_round_robin(rows: Iterable[ManifestRow], folds: int) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (row.endpoint, row.region, row.site, row.case_id))
    return {row.case_id: index % folds for index, row in enumerate(ordered)}


def group_counts(rows: Iterable[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row.endpoint}:{row.region}:{row.site}"] += 1
    return dict(counts)


def fold_sizes(rows: Iterable[ManifestRow], folds: int) -> list[int]:
    sizes = [0 for _ in range(folds)]
    for row in rows:
        if 0 <= row.fold < folds:
            sizes[row.fold] += 1
    return sizes
