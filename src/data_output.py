"""Data output utilities for the Detroit Lions draft analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl


def save_csv(df: pl.DataFrame, output_path: str | Path) -> None:
    """Write an eager DataFrame to a CSV file.

    Creates parent directories if they do not exist. Overwrites any existing
    file at ``output_path`` without warning.

    Args:
        df: Eager Polars DataFrame to save.
        output_path: Destination path for the CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def save_parquet(df: pl.DataFrame, output_path: str | Path) -> None:
    """Write an eager DataFrame to a Parquet file.

    Creates parent directories if they do not exist. Overwrites any existing
    file at ``output_path`` without warning.

    Args:
        df: Eager Polars DataFrame to save.
        output_path: Destination path for the Parquet file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def save_data(
    df: pl.DataFrame,
    output_path: str | Path,
    format: Literal["csv", "parquet"] = "parquet",
) -> None:
    """Save a DataFrame to either CSV or Parquet format.

    Dispatches to :func:`save_csv` or :func:`save_parquet` based on
    ``format``. Parent directories are created automatically.

    Args:
        df: Eager Polars DataFrame to save.
        output_path: Destination path for the output file.
        format: Output format — ``"csv"`` or ``"parquet"`` (default).

    Raises:
        ValueError: If ``format`` is not ``"csv"`` or ``"parquet"``.
    """
    if format == "csv":
        save_csv(df, output_path)
    elif format == "parquet":
        save_parquet(df, output_path)
    else:
        raise ValueError(f"Unsupported format: {format!r}. Use 'csv' or 'parquet'.")
