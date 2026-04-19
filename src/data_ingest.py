"""Data ingestion utilities for the Detroit Lions draft analysis pipeline."""

from pathlib import Path

import polars as pl


def load_csv(file_path: str | Path) -> pl.DataFrame:
    """Load a single CSV file into an eager Polars DataFrame.

    All columns are returned with Polars-inferred types. No type coercion is
    applied — callers are responsible for any necessary casting.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Eager DataFrame with all columns from the CSV.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pl.read_csv(path)


def load_parquet(
    file_path: str | Path,
    lazy: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """Load a single Parquet file.

    Args:
        file_path: Path to the Parquet file.
        lazy: When ``True`` (default) returns a ``pl.LazyFrame`` via
            ``pl.scan_parquet``; when ``False`` returns an eager
            ``pl.DataFrame`` via ``pl.read_parquet``.

    Returns:
        LazyFrame or eager DataFrame depending on ``lazy``.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    if lazy:
        return pl.scan_parquet(str(path))
    return pl.read_parquet(path)


def load_parquets_from_dir(
    directory: str | Path,
    lazy: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """Scan all ``.parquet`` files in a directory into a single combined frame.

    Uses a glob pattern so all files are processed in a single Polars scan
    node rather than 210 separate ones. The combined frame contains all rows
    from every matching file without deduplication.

    Expected columns for the annual AV parquets (all ``String`` dtype):
        - ``Rk``: Row rank within the source query.
        - ``Player``: Player name.
        - ``AV``: Career approximate value (from source header).
        - ``Draft Team``: Three-letter team code that drafted the player.
        - ``Round``: Draft round.
        - ``Pick``: Overall pick number.
        - ``Draft Year``: Year the player was drafted.
        - ``College``: College or university attended.
        - ``Season``: NFL season year for this row.
        - ``Age``: Player age during the season.
        - ``Team``: Team(s) played for that season (may be comma-separated).
        - ``G``: Games played.
        - ``GS``: Games started.
        - ``AV.1``: Approximate Value for this specific season.
        - ``Pos``: Position.

    Args:
        directory: Directory containing the ``.parquet`` files.
        lazy: When ``True`` (default) returns a ``pl.LazyFrame``. When
            ``False`` collects and returns an eager ``pl.DataFrame``.

    Returns:
        Combined LazyFrame or DataFrame of all parquet files.

    Raises:
        FileNotFoundError: If ``directory`` does not exist or contains no
            ``.parquet`` files.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    parquet_files = list(dir_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in: {dir_path}")

    glob_pattern = str(dir_path / "*.parquet")
    lf = pl.scan_parquet(glob_pattern)

    if lazy:
        return lf
    return lf.collect()
