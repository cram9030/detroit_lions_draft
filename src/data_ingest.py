"""Data ingestion utilities for the Detroit Lions draft analysis pipeline."""

from pathlib import Path

import polars as pl
import nflreadpy


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
    return pl.read_csv(path, infer_schema_length=None)


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


def load_nflreadr_draft_picks(seasons: list[int] | None = None) -> pl.DataFrame:
    """Load NFL draft pick data from nflreadpy, including ``dr_av`` per player.

    Uses ``nflreadpy.load_draft_picks()`` and normalises column names to match
    project conventions. ``dr_av`` (Approximate Value produced for the team
    that drafted the player) is cast to Float64; rows where ``dr_av`` is null
    are retained — callers may drop them with ``.drop_nulls(subset=["dr_av"])``.

    Args:
        seasons: Optional list of draft years to include (e.g. ``[1980, 1981]``).
            If ``None``, all available years are returned.

    Returns:
        Eager DataFrame with columns including ``Pick`` (Int64),
        ``Draft Year`` (Int64), ``Player`` (String), ``dr_av`` (Float64),
        plus all other columns returned by nflreadpy.
    """
    df = (
        nflreadpy.load_draft_picks()
        .rename({"season": "Draft Year", "pick": "Pick", "pfr_player_name": "Player"})
        .with_columns([
            pl.col("Pick").cast(pl.Int64),
            pl.col("Draft Year").cast(pl.Int64),
            pl.col("dr_av").cast(pl.Float64, strict=False),
        ])
    )
    if seasons is not None:
        df = df.filter(pl.col("Draft Year").is_in(seasons))
    return df
