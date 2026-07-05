from __future__ import annotations

from dataclasses import asdict

from physmel_lv_world.vault.types import MetricBundle


def metric_row(name: str, metrics: MetricBundle) -> dict[str, float | str]:
    row: dict[str, float | str] = {"name": name}
    row.update(asdict(metrics))
    return row


def format_table(rows: list[dict[str, float | str]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    widths = {
        key: max(
            len(str(key)),
            *(
                len(f"{row[key]:.4f}") if isinstance(row[key], float) else len(str(row[key]))
                for row in rows
            ),
        )
        for key in keys
    }
    lines = [" | ".join(str(key).ljust(widths[key]) for key in keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value:.4f}".ljust(widths[key]))
            else:
                cells.append(str(value).ljust(widths[key]))
        lines.append(" | ".join(cells))
    return "\n".join(lines)
