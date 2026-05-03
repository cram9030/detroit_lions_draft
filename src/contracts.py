"""Contract data utilities for replacement-level AV analysis."""

from pathlib import Path

import nflreadpy
import polars as pl

from src.data_ingest import load_csv, load_parquets_from_dir

_MIN_SALARIES_PATH = Path(__file__).resolve().parent.parent / "data/processed/nfl_minimum_salaries.csv"

_EXPERIENCE_COLS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]


def _load_min_salary_long() -> pl.DataFrame:
    """Return minimum salary table in long form with numeric columns.

    Columns: Year (Int32), experience (Int32, capped at 10), min_salary_m (Float64, in millions).
    """
    df = load_csv(_MIN_SALARIES_PATH)
    salary_exprs = [
        pl.col(c).str.replace_all(r"[$,]", "").cast(pl.Float64).alias(c)
        for c in _EXPERIENCE_COLS
    ]
    df = df.with_columns([pl.col("Year").cast(pl.Int32)] + salary_exprs)
    long = df.unpivot(
        index="Year",
        on=_EXPERIENCE_COLS,
        variable_name="experience_str",
        value_name="min_salary_m",
    ).with_columns([
        (pl.col("min_salary_m") / 1_000_000),
        pl.col("experience_str")
        .str.replace("10+", "10", literal=True)
        .cast(pl.Int32)
        .alias("experience"),
    ]).drop("experience_str")
    return long


def get_replacement_level_contracts(
    replacement_percent: float = 1.20,
) -> pl.DataFrame:
    """Return 1-year contracts at or below replacement_percent of the NFL minimum salary.

    A player's experience tier is determined by their rookie_season from nflreadr
    players data. Only contracts with a year_signed covered by the minimum salary
    table (2006–2030) are included.

    Args:
        replacement_percent: Ceiling multiplier on the minimum salary (default 1.20 = 120%).

    Returns:
        DataFrame with columns: player, year_signed, years, value, min_salary, experience,
        draft_year, draft_overall. value and min_salary are in millions of dollars.
    """
    contracts = (
        nflreadpy.load_contracts()
        .filter(pl.col("years") == 1)
        .select(["player", "year_signed", "years", "value", "otc_id", "draft_year", "draft_overall"])
    )

    players = (
        nflreadpy.load_players()
        .select([
            pl.col("otc_id").cast(pl.Int32, strict=False).alias("otc_id"),
            pl.col("rookie_season"),
        ])
        .drop_nulls()
    )

    joined = (
        contracts
        .join(players, on="otc_id", how="left")
        .drop_nulls(subset=["rookie_season"])
        .with_columns([
            (pl.col("year_signed") - pl.col("rookie_season"))
            .clip(lower_bound=0, upper_bound=10)
            .cast(pl.Int32)
            .alias("experience"),
        ])
    )

    salary = _load_min_salary_long()

    result = (
        joined
        .join(
            salary,
            left_on=["year_signed", "experience"],
            right_on=["Year", "experience"],
            how="inner",
        )
        .filter(pl.col("value") <= pl.col("min_salary_m") * replacement_percent)
        .select(["player", "year_signed", "years", "value", "min_salary_m", "experience",
                 "draft_year", "draft_overall"])
        .rename({"min_salary_m": "min_salary"})
        .sort(["year_signed", "player"])
    )
    return result


def get_replacement_level_av(
    av_data: pl.LazyFrame,
    replacement_contracts: pl.DataFrame,
) -> pl.DataFrame:
    """Return season-level AV for players on replacement-level contracts.

    Joins the prepared AV LazyFrame to replacement contracts on player name,
    season year, draft year, and draft pick. The four-key join uniquely
    identifies each player, eliminating false matches between players who share
    a name. Undrafted players (null draft_overall) have no Stathead rows and
    are naturally excluded.

    When a player holds multiple replacement-level contracts in the same season,
    the lowest-value contract is retained to represent the floor.

    Args:
        av_data: LazyFrame from prepare_av_data(load_parquets_from_dir(...)).
        replacement_contracts: DataFrame from get_replacement_level_contracts().

    Returns:
        Eager DataFrame with columns: Player, year_signed, value, AV.1.
    """
    contract_pairs = (
        replacement_contracts
        .sort("value")
        .unique(subset=["player", "year_signed"], keep="first")
        .select([
            pl.col("player").alias("Player"),
            pl.col("year_signed").cast(pl.Int64).alias("Season"),
            pl.col("draft_year").cast(pl.Int64).alias("Draft Year"),
            pl.col("draft_overall").cast(pl.Int64).alias("Pick"),
            pl.col("value"),
        ])
    )

    result = (
        av_data
        .join(
            contract_pairs.lazy(),
            on=["Player", "Season", "Draft Year", "Pick"],
            how="inner",
        )
        .select(["Player", pl.col("Season").alias("year_signed"), "value", "AV.1"])
        .collect()
        .sort(["year_signed", "Player"])
    )
    return result
