"""Wrapper that invokes results/h200_2026-11-12_multi_adapter/build_tables.py."""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parent / "results" / "h200_2026-11-12_multi_adapter" / "build_tables.py"
runpy.run_path(str(target), run_name="__main__")
