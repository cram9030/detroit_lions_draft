"""Project-wide path constants for NFL draft analysis."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_RAW_DIR: Path = PROJECT_ROOT / "data/raw/stathead/annual_av"
DEFAULT_EAVAR_PATH: Path = PROJECT_ROOT / "data/processed/expected_av_above_replacement.csv"
DEFAULT_MODELS_DIR: Path = PROJECT_ROOT / "models"
DEFAULT_PROCESSED_DIR: Path = PROJECT_ROOT / "data/processed"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
