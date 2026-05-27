"""Pick-level AV distribution analysis functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
from scipy.stats import skewnorm

from src.annual_av_analysis import aggregate_player_av, prepare_av_data
from src.data_ingest import load_parquets_from_dir


def filter_top_percentile_per_pick(
    df: pl.DataFrame,
    av_col: str,
    percentile: float = 0.10,
) -> pl.DataFrame:
    """Keep only the top ``percentile`` fraction of players by AV within each pick."""
    return (
        df.with_columns([
            pl.col(av_col)
            .rank(method="ordinal", descending=True)
            .over("Pick")
            .alias("_rank"),
            pl.len().over("Pick").alias("_count"),
        ])
        .filter(
            pl.col("_rank")
            <= (pl.col("_count").cast(pl.Float64) * percentile).ceil().cast(pl.Int64)
        )
        .drop(["_rank", "_count"])
    )


def _compute_pick_describe(player_av_df: pl.DataFrame) -> pl.DataFrame:
    """Compute descriptive statistics of ``rookie_contract_av`` grouped by pick."""
    return (
        player_av_df.group_by("Pick")
        .agg(
            [
                pl.col("rookie_contract_av").count().alias("count"),
                pl.col("rookie_contract_av").null_count().alias("null_count"),
                pl.col("rookie_contract_av").mean().alias("mean"),
                pl.col("rookie_contract_av").std().alias("std"),
                pl.col("rookie_contract_av").min().alias("min"),
                pl.col("rookie_contract_av")
                .quantile(0.25, interpolation="linear")
                .alias("25%"),
                pl.col("rookie_contract_av")
                .quantile(0.50, interpolation="linear")
                .alias("50%"),
                pl.col("rookie_contract_av")
                .quantile(0.75, interpolation="linear")
                .alias("75%"),
                pl.col("rookie_contract_av").max().alias("max"),
            ]
        )
        .sort("Pick")
    )


def _fit_skewnorm_on_df(
    df: pl.DataFrame,
    min_samples: int = 5,
) -> pl.DataFrame:
    """Fit a skew-normal distribution to ``rookie_contract_av`` for each pick."""
    records: list[dict] = []
    for (pick,), group_df in df.group_by(["Pick"]):
        values = group_df["rookie_contract_av"].drop_nulls().to_numpy()
        if len(values) < min_samples:
            continue
        try:
            a, loc, scale = skewnorm.fit(values)
            records.append({"Pick": pick, "a": float(a), "loc": float(loc), "scale": float(scale)})
        except Exception:
            continue

    if not records:
        return pl.DataFrame(
            schema={"Pick": pl.Int64, "a": pl.Float64, "loc": pl.Float64, "scale": pl.Float64}
        )

    return pl.DataFrame(
        records,
        schema={"Pick": pl.Int64, "a": pl.Float64, "loc": pl.Float64, "scale": pl.Float64},
    ).sort("Pick")


def _rolling_window_iter(
    df_all: pl.DataFrame,
    half: int,
) -> Iterator[tuple[int, pl.DataFrame]]:
    """Yield ``(center_year, window_df)`` for every valid rolling window."""
    min_year: int = df_all["Draft Year"].min()
    max_year: int = df_all["Draft Year"].max()
    for center in range(min_year + half, max_year - half + 1):
        window_df = df_all.filter(
            (pl.col("Draft Year") >= center - half)
            & (pl.col("Draft Year") <= center + half)
        )
        yield center, window_df


def pick_based_stats(
    directory: str | Path,
    max_seasons_from_draft: int = 4,
    draft_year_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """Compute per-pick descriptive statistics across all available draft years."""
    lf = load_parquets_from_dir(directory, lazy=True)
    lf = prepare_av_data(lf)
    if draft_year_range is not None:
        lf = lf.filter(
            (pl.col("Draft Year") >= draft_year_range[0])
            & (pl.col("Draft Year") <= draft_year_range[1])
        )
    lf = aggregate_player_av(lf, max_seasons_from_draft=max_seasons_from_draft)
    df = lf.collect()
    return _compute_pick_describe(df)


def skew_normal_fit(
    player_av_data: pl.LazyFrame,
    min_samples: int = 5,
) -> pl.DataFrame:
    """Fit a skew-normal distribution to ``rookie_contract_av`` per pick."""
    df = player_av_data.collect()
    return _fit_skewnorm_on_df(df, min_samples=min_samples)


def rolling_window_pick_stats(
    directory: str | Path,
    window_length: int,
    max_seasons_from_draft: int = 4,
) -> dict[int, pl.DataFrame]:
    """Compute per-pick descriptive statistics for each rolling window of draft years."""
    if window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd for symmetric centering, got {window_length}."
        )

    half = window_length // 2
    lf = load_parquets_from_dir(directory, lazy=True)
    df_all = aggregate_player_av(
        prepare_av_data(lf), max_seasons_from_draft=max_seasons_from_draft
    ).collect()

    return {
        center: _compute_pick_describe(window_df)
        for center, window_df in _rolling_window_iter(df_all, half)
    }


def rolling_window_skew_fit(
    directory: str | Path,
    window_length: int,
    min_samples: int = 5,
    max_seasons_from_draft: int = 4,
) -> dict[int, pl.DataFrame]:
    """Fit skew-normal distributions per pick for each rolling window of draft years."""
    if window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd for symmetric centering, got {window_length}."
        )

    half = window_length // 2
    lf = load_parquets_from_dir(directory, lazy=True)
    df_all = aggregate_player_av(
        prepare_av_data(lf), max_seasons_from_draft=max_seasons_from_draft
    ).collect()

    return {
        center: _fit_skewnorm_on_df(window_df, min_samples=min_samples)
        for center, window_df in _rolling_window_iter(df_all, half)
    }
