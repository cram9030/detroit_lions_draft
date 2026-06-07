"""Annual Approximate Value (AV) analysis functions for NFL draft pick evaluation.

Core data preparation and aggregation. Position-level, pick-level, and
curve-fitting functions have been extracted to focused modules:
- src.positions  -- position normalization constants and helpers
- src.av_stats   -- pick-level distribution analysis
- src.career_av  -- career trajectory aggregation
- src.pick_curves -- exponential/logarithmic decay wrappers
"""

from __future__ import annotations

import polars as pl

from src.data_ingest import load_parquets_from_dir
from src.positions import _GENERALIST, _POSITION_GROUPS, _SPECALIST  # noqa: F401


# ---------------------------------------------------------------------------
# Data preparation and aggregation (public)
# ---------------------------------------------------------------------------


def prepare_av_data(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """Cast raw string columns to analysis-ready types and drop unusable rows."""
    return (
        lazy_frame.with_columns(
            [
                pl.col("Pick").cast(pl.Int64),
                pl.col("AV.1").cast(pl.Float64, strict=False),
                pl.col("Draft Year").cast(pl.Int64),
                pl.col("Season").cast(pl.Int64),
                pl.col("Pos").str.strip_chars(),
            ]
        )
        .drop_nulls(subset=["AV.1"])
    )


def aggregate_player_av(
    lazy_frame: pl.LazyFrame,
    max_seasons_from_draft: int = 4,
    min_season_av: float = 0,
) -> pl.LazyFrame:
    """Aggregate season-level AV into one ``rookie_contract_av`` value per player."""
    return (
        lazy_frame
        .filter(pl.col("Season") - pl.col("Draft Year") < max_seasons_from_draft)
        .filter(pl.col("AV.1") >= min_season_av)
        .group_by(["Player", "Pick", "Draft Year", "Draft Team"])
        .agg(pl.col("AV.1").sum().alias("rookie_contract_av"))
    )


def lookup_player_av(
    player_name: str,
    player_av_data: pl.LazyFrame | pl.DataFrame,
) -> pl.DataFrame:
    """Return the ``rookie_contract_av`` row(s) for a named player."""
    return (
        player_av_data.lazy()
        .filter(pl.col("Player").str.to_lowercase().str.contains(player_name.lower()))
        .select(["Player", "Draft Year", "Pick", "Draft Team", "rookie_contract_av"])
        .sort("Draft Year")
        .collect()
    )
