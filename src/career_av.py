"""Career AV trajectory analysis functions."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.annual_av_analysis import prepare_av_data
from src.data_ingest import load_parquets_from_dir
from src.positions import _GENERALIST, _POSITION_GROUPS, _SPECALIST


def aggregate_career_av_by_position(
    lazy_frame: pl.LazyFrame,
    normalize: bool,
    rounds: list[int] | None = None,
) -> pl.LazyFrame:
    """Return season-level AV annotated with career year and (optionally) normalized position."""
    lf = (
        lazy_frame
        .with_columns(
            [
                pl.col("Round").cast(pl.Int64, strict=False),
                (pl.col("Season") - pl.col("Draft Year")).alias("years_from_draft"),
            ]
        )
        .filter(pl.col("years_from_draft") >= 0)
        .drop_nulls(subset=["Pos"])
    )

    if rounds is not None:
        lf = lf.filter(pl.col("Round").is_in(rounds))

    if normalize:
        lf = (
            lf.with_columns(
                pl.col("Pos")
                .str.replace_all("-", "/")
                .str.split("/")
                .list.eval(
                    pl.element().replace(_POSITION_GROUPS, default=pl.element())
                )
            )
            .explode("Pos")
            .unique(subset=["Player", "Draft Year", "years_from_draft", "Pos"])
            .filter(~pl.col("Pos").is_in(_SPECALIST)).filter(~pl.col("Pos").is_in(_GENERALIST))
        )

    base_cols = ["Player", "Pos", "Draft Year", "years_from_draft", "AV.1"]
    if "Pick" in lf.collect_schema():
        base_cols = ["Player", "Pos", "Pick", "Draft Year", "years_from_draft", "AV.1"]
    return lf.select(base_cols)


def _compute_group_year_describe(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Compute descriptive statistics of ``AV.1`` grouped by an arbitrary column and career year."""
    return (
        df.group_by([group_col, "years_from_draft"])
        .agg(
            [
                pl.col("AV.1").count().alias("count"),
                pl.col("AV.1").mean().alias("mean"),
                pl.col("AV.1").std().alias("std"),
                pl.col("AV.1").min().alias("min"),
                pl.col("AV.1").quantile(0.25, interpolation="linear").alias("25%"),
                pl.col("AV.1").quantile(0.50, interpolation="linear").alias("50%"),
                pl.col("AV.1").quantile(0.75, interpolation="linear").alias("75%"),
                pl.col("AV.1").max().alias("max"),
            ]
        )
        .sort([group_col, "years_from_draft"])
    )


def position_career_stats(
    directory: str | Path,
    normalize: bool = True,
    rounds: list[int] | None = None,
) -> pl.DataFrame:
    """Compute per-position, per-career-year descriptive statistics of annual AV."""
    lf = load_parquets_from_dir(directory, lazy=True)
    lf = prepare_av_data(lf)
    lf = aggregate_career_av_by_position(lf, normalize=normalize, rounds=rounds)
    df = lf.collect()
    return _compute_group_year_describe(df, "Pos")


def round_career_stats(directory: str | Path) -> pl.DataFrame:
    """Compute per-draft-round, per-career-year descriptive statistics of annual AV."""
    lf = load_parquets_from_dir(directory, lazy=True)
    df = (
        prepare_av_data(lf)
        .with_columns(
            [
                pl.col("Round").cast(pl.Int64, strict=False),
                (pl.col("Season") - pl.col("Draft Year")).alias("years_from_draft"),
            ]
        )
        .filter(pl.col("years_from_draft") >= 0)
        .drop_nulls(subset=["Round"])
        .select(["Player", "Round", "Draft Year", "years_from_draft", "AV.1"])
        .collect()
    )
    return _compute_group_year_describe(df, "Round")
