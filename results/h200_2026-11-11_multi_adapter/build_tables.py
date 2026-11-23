"""Build simple Markdown tables from CSV files in source/."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).parent


def main() -> None:
    tables = ROOT / "tables"
    tables.mkdir(exist_ok=True)
    for source in sorted((ROOT / "source").glob("*.csv")):
        with source.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows:
            continue
        header, *body = rows
        lines = [
            f"# {source.stem} — NVIDIA H200, PACE ICE",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        (tables / f"{source.stem}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
