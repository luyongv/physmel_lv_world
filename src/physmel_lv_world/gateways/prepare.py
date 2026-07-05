from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physmel-prepare")
    parser.add_argument("output")
    parser.add_argument("--rows", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "path_embed",
                "gene_embed",
                "immune_fraction",
                "time",
                "event",
                "fold",
                "endpoint",
                "site",
                "region",
            ],
        )
        writer.writeheader()
        for index in range(args.rows):
            writer.writerow(
                {
                    "case_id": f"case_{index}",
                    "path_embed": "",
                    "gene_embed": "",
                    "immune_fraction": "",
                    "time": 30 + index,
                    "event": index % 2,
                    "fold": index % 5,
                    "endpoint": "os",
                    "site": "site_a",
                    "region": "region_a",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
