"""Annual Approximate Value (AV) analysis functions for NFL draft pick evaluation.

All public functions operate over the Stathead annual AV parquet dataset stored in
``data/raw/stathead/annual_av/``. Each parquet file covers one (draft year, season
year) combination. A player drafted in year Y appears in up to four files:
``draft{Y}_season{Y}.parquet`` through ``draft{Y}_season{Y+3}.parquet``.

The key derived metric is ``rookie_contract_av``: the sum of a player's season-level
AV (``AV.1``) across all tracked seasons. This approximates production delivered
during the typical rookie contract window. The name ``career_av`` is reserved for
future full-career analysis that would require additional data sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
from scipy.stats import skewnorm

from src import curve_fitting
from src.curve_fitting import (
    ExpDecayModel,
    IndividualFitResult,
    LogDecayModel,
    StatsFitResult,
)
from src.data_ingest import load_parquets_from_dir


# ---------------------------------------------------------------------------
# Position normalization
# ---------------------------------------------------------------------------

_POSITION_GROUPS: dict[str, str] = {
    "FL":"WR",
    "FB": "RB", "RH":"RB", "LH":"RB",
    "LDE": "DE", "RDE": "DE",
    "NT": "DT", "LDT":"DT", "RDT":"DT",
    "LG": "OG", "RG": "OG", "LT": "OT", "RT": "OT", "T":"OT", "G":"OG", "C":"OC",
    "LCB": "CB", "RCB": "CB",
    "LILB": "LB", "RILB": "LB", "LOLB": "LB", "ROLB": "LB", "LLB": "LB", "ILB":"LB", "OLB":"LB", "RLB":"LB","MLB":"LB",
    "FS": "S", "SS": "S", "DB": "S",
}
"""Maps raw ``Pos`` variants to 12 standard position groups.

Positions absent from this dict are left unchanged (e.g. QB, WR, TE, DE,
DT, LB, CB, K, P stay as-is).
"""

_SPECALIST: list[str] = ['K', 'KR', 'P', 'PR', 'LS']
"""Specialist positions excluded from normalized position-group analysis."""

_GENERALIST: list[str] = ['DL', 'OL']
"""Generalist for a line group positions excluded from normalized position-group analysis because they don't ever play in the first year."""

# ---------------------------------------------------------------------------
# Data preparation and aggregation (public)
# ---------------------------------------------------------------------------


def prepare_av_data(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """Cast raw string columns to analysis-ready types and drop unusable rows.

    The raw parquet files store all values as strings. This function performs
    the minimum type coercion needed for analysis and removes rows where
    ``AV.1`` cannot be interpreted as a number.

    Input columns used (all originally ``String``):
        - ``Pick``: Overall draft pick number → cast to ``Int64``.
        - ``AV.1``: Season-level Approximate Value → cast to ``Float64``.
        - ``Draft Year``: Year the player was drafted → cast to ``Int64``.
        - ``Season``: Season year → cast to ``Int64``.
        - ``Pos``: Player position → whitespace stripped, kept as ``String``.

    All other columns are passed through unchanged.

    Args:
        lazy_frame: LazyFrame containing raw parquet data with string columns.

    Returns:
        LazyFrame with ``Pick`` (Int64), ``AV.1`` (Float64),
        ``Draft Year`` (Int64), ``Season`` (Int64) cast, and ``Pos``
        whitespace-stripped; rows where ``AV.1`` is null after casting
        are dropped.
    """
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
    """Aggregate season-level AV into one ``rookie_contract_av`` value per player.

    Groups by ``(Player, Pick, Draft Year)`` — a three-part key required to
    correctly separate players who share a name and draft year (e.g. two
    players named "Alex Smith" drafted in 2005 at picks 1 and 71).

    Input columns required:
        - ``Player`` (String): Player name.
        - ``Pick`` (Int64): Overall pick number.
        - ``Draft Year`` (Int64): Year the player was drafted.
        - ``Draft Team`` (String): Team that drafted the player.
        - ``Season`` (Int64): Season year.
        - ``AV.1`` (Float64): Season-level Approximate Value.

    Args:
        lazy_frame: LazyFrame already processed through :func:`prepare_av_data`.
        max_seasons_from_draft: Seasons after the draft year to include,
            expressed as a strict upper bound (``Season - Draft Year <
            max_seasons_from_draft``). Default ``4`` retains seasons 0–3
            (the typical four-year rookie contract window).
        min_season_av: Minimum single-season AV required for a season to be
            included in the sum. Seasons with ``AV.1 < min_season_av`` are
            dropped before aggregation. Default ``0`` retains all seasons and
            preserves prior behaviour.

    Returns:
        LazyFrame with one row per unique (Player, Pick, Draft Year)
        combination. Output columns: ``Player`` (String), ``Pick`` (Int64),
        ``Draft Year`` (Int64), ``Draft Team`` (String),
        ``rookie_contract_av`` (Float64).
        Negative values are valid and must not be filtered.
    """
    return (
        lazy_frame
        .filter(pl.col("Season") - pl.col("Draft Year") < max_seasons_from_draft)
        .filter(pl.col("AV.1") >= min_season_av)
        .group_by(["Player", "Pick", "Draft Year", "Draft Team"])
        .agg(pl.col("AV.1").sum().alias("rookie_contract_av"))
    )


def filter_top_percentile_per_pick(
    df: pl.DataFrame,
    av_col: str,
    percentile: float = 0.10,
) -> pl.DataFrame:
    """Keep only the top ``percentile`` fraction of players by AV within each pick.

    Within each unique pick number the rows are ranked descending by ``av_col``
    and the top ``ceil(n * percentile)`` rows are retained (at least 1 per pick).

    Args:
        df: Eager DataFrame with at least ``Pick`` (Int64) and ``av_col`` columns.
        av_col: Name of the AV column to rank on.
        percentile: Fraction of rows to keep per pick. Default ``0.10`` keeps
            the top 10 %.

    Returns:
        Filtered DataFrame with the same schema as ``df``.
    """
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_pick_describe(player_av_df: pl.DataFrame) -> pl.DataFrame:
    """Compute descriptive statistics of ``rookie_contract_av`` grouped by pick.

    Input columns required:
        - ``Pick`` (Int64): Overall pick number.
        - ``rookie_contract_av`` (Float64): Total AV over tracked seasons.

    Each row in the output represents one pick number. The statistics describe
    the distribution of ``rookie_contract_av`` across all players drafted at
    that pick position.

    Note: ``std`` will be ``null`` for picks with only one player in the
    dataset — this is mathematically correct and must not be replaced with 0.

    Args:
        player_av_df: Eager DataFrame with one row per player, output of
            :func:`aggregate_player_av` after ``.collect()``.

    Returns:
        Eager DataFrame sorted by ``Pick`` ascending with columns:
            - ``Pick`` (Int64)
            - ``count`` (UInt32): Number of players at this pick.
            - ``null_count`` (UInt32): Null values in ``rookie_contract_av``
              (always 0 after :func:`prepare_av_data`).
            - ``mean`` (Float64): Mean rookie contract AV.
            - ``std`` (Float64): Standard deviation; null for n=1.
            - ``min`` (Float64): Minimum rookie contract AV.
            - ``25%`` (Float64): 25th percentile.
            - ``50%`` (Float64): Median.
            - ``75%`` (Float64): 75th percentile.
            - ``max`` (Float64): Maximum rookie contract AV.
    """
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
    """Fit a skew-normal distribution to ``rookie_contract_av`` for each pick.

    Uses ``scipy.stats.skewnorm.fit`` (MLE) to estimate the three parameters
    of a skew-normal distribution. Picks with fewer than ``min_samples``
    players are excluded to ensure numerically stable estimates.

    Input columns required:
        - ``Pick`` (Int64): Overall pick number.
        - ``rookie_contract_av`` (Float64): Total AV over tracked seasons.

    Args:
        df: Eager DataFrame with one row per player.
        min_samples: Minimum number of players required to attempt a fit.
            Default 5 is the recommended floor for 3-parameter MLE with
            skewnorm. Picks below this threshold are absent from the output.

    Returns:
        Eager DataFrame sorted by ``Pick`` ascending with columns:
            - ``Pick`` (Int64)
            - ``a`` (Float64): Shape (skewness) parameter. Negative = left-
              skewed, positive = right-skewed.
            - ``loc`` (Float64): Location (mean) parameter.
            - ``scale`` (Float64): Scale (spread) parameter.

        Picks with degenerate distributions (e.g. all identical AV values)
        or fewer than ``min_samples`` observations are excluded silently.
    """
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
    """Yield ``(center_year, window_df)`` for every valid rolling window.

    Args:
        df_all: Full aggregated player-AV DataFrame with a ``Draft Year`` column.
        half: Half-width of the window (``window_length // 2``).
    """
    min_year: int = df_all["Draft Year"].min()
    max_year: int = df_all["Draft Year"].max()
    for center in range(min_year + half, max_year - half + 1):
        window_df = df_all.filter(
            (pl.col("Draft Year") >= center - half)
            & (pl.col("Draft Year") <= center + half)
        )
        yield center, window_df


# ---------------------------------------------------------------------------
# Public functions — pick-based analysis
# ---------------------------------------------------------------------------


def pick_based_stats(
    directory: str | Path,
    max_seasons_from_draft: int = 4,
    draft_year_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """Compute per-pick descriptive statistics across all available draft years.

    Loads all parquet files from ``directory`` lazily, aggregates each
    player's season-level AV into a single ``rookie_contract_av`` value,
    then computes descriptive statistics grouped by overall pick number.

    Input data: parquet files named ``draft{YEAR}_season{YEAR}.parquet`` in
    ``directory``, each row representing one player's season. All columns
    are stored as strings in the raw files.

    Args:
        directory: Path to the directory containing annual AV parquet files
            (e.g. ``data/raw/stathead/annual_av``).
        max_seasons_from_draft: Passed to :func:`aggregate_player_av`.
            Strict upper bound on ``Season - Draft Year``; default ``4``
            retains seasons 0–3.
        draft_year_range: Optional ``(start, end)`` tuple (inclusive) to
            restrict analysis to a specific range of draft years.  If
            ``None`` (default), all draft years are included.

    Returns:
        Eager DataFrame sorted by ``Pick`` ascending with columns:
            - ``Pick`` (Int64): Overall pick number.
            - ``count`` (UInt32): Number of players at this pick across all
              draft years.
            - ``null_count`` (UInt32): Always 0 after data preparation.
            - ``mean`` (Float64): Mean rookie contract AV.
            - ``std`` (Float64): Standard deviation; null for n=1.
            - ``min`` (Float64): Minimum rookie contract AV.
            - ``25%`` (Float64): 25th percentile.
            - ``50%`` (Float64): Median.
            - ``75%`` (Float64): 75th percentile.
            - ``max`` (Float64): Maximum rookie contract AV.
    """
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
    """Fit a skew-normal distribution to ``rookie_contract_av`` per pick.

    Intended to receive the output of :func:`aggregate_player_av` as a
    LazyFrame, which is then collected internally.

    Input LazyFrame columns required:
        - ``Pick`` (Int64): Overall pick number.
        - ``rookie_contract_av`` (Float64): Total AV over tracked seasons.

    Args:
        player_av_data: LazyFrame with one row per player, already processed
            through :func:`prepare_av_data` and :func:`aggregate_player_av`.
        min_samples: Minimum players per pick to attempt a fit. Default 5.

    Returns:
        Eager DataFrame sorted by ``Pick`` ascending with columns:
            - ``Pick`` (Int64)
            - ``a`` (Float64): Skewness shape parameter.
            - ``loc`` (Float64): Location parameter.
            - ``scale`` (Float64): Scale parameter.

        Picks with fewer than ``min_samples`` players are excluded from output.
    """
    df = player_av_data.collect()
    return _fit_skewnorm_on_df(df, min_samples=min_samples)


def lookup_player_av(
    player_name: str,
    player_av_data: pl.LazyFrame | pl.DataFrame,
) -> pl.DataFrame:
    """Return the ``rookie_contract_av`` row(s) for a named player.

    Intended for spot-checking the output of :func:`aggregate_player_av`.
    Matching is case-insensitive and uses substring containment, so partial
    names (e.g. ``"Manning"``) will return all matching players.

    Args:
        player_name: Name (or partial name) to search for.
        player_av_data: LazyFrame or eager DataFrame produced by
            :func:`aggregate_player_av`, with columns ``Player``, ``Pick``,
            ``Draft Year``, ``Draft Team``, ``rookie_contract_av``.

    Returns:
        Eager DataFrame with columns ``Player``, ``Draft Year``, ``Pick``,
        ``Draft Team``, ``rookie_contract_av``, sorted by ``Draft Year``
        ascending.  Empty if no match is found.
    """
    return (
        player_av_data.lazy()
        .filter(pl.col("Player").str.to_lowercase().str.contains(player_name.lower()))
        .select(["Player", "Draft Year", "Pick", "Draft Team", "rookie_contract_av"])
        .sort("Draft Year")
        .collect()
    )


def rolling_window_pick_stats(
    directory: str | Path,
    window_length: int,
    max_seasons_from_draft: int = 4,
) -> dict[int, pl.DataFrame]:
    """Compute per-pick descriptive statistics for each rolling window of draft years.

    Windows are centered on successive draft years. For data spanning 1970–2022
    with ``window_length=11``, the first center year is 1975 (covering
    1970–1980) and the last is 2017 (covering 2012–2022), yielding 43 windows.

    The full aggregated dataset (~12K rows) is collected once and then filtered
    in memory for each window to avoid repeatedly scanning all parquet files.

    Args:
        directory: Path to the directory containing annual AV parquet files.
        window_length: Number of draft years in each window. Must be an odd
            integer to ensure symmetric centering around the center year.
        max_seasons_from_draft: Passed to :func:`aggregate_player_av`.
            Strict upper bound on ``Season - Draft Year``; default ``4``
            retains seasons 0–3.

    Returns:
        Dict mapping ``center_year`` (int) to an eager DataFrame with the
        same schema as :func:`pick_based_stats` output.

    Raises:
        ValueError: If ``window_length`` is even.
    """
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
    """Fit skew-normal distributions per pick for each rolling window of draft years.

    Applies the same windowing logic as :func:`rolling_window_pick_stats` but
    fits a skew-normal distribution to the ``rookie_contract_av`` values for
    each pick within each window. The full aggregated dataset is collected
    once and filtered in memory per window.

    Args:
        directory: Path to the directory containing annual AV parquet files.
        window_length: Number of draft years in each window. Must be odd.
        min_samples: Minimum players per pick required to attempt a fit.
            Default 5. Picks below this threshold are absent from each
            window's output DataFrame.
        max_seasons_from_draft: Passed to :func:`aggregate_player_av`.
            Strict upper bound on ``Season - Draft Year``; default ``4``
            retains seasons 0–3.

    Returns:
        Dict mapping ``center_year`` (int) to an eager DataFrame with columns:
            - ``Pick`` (Int64): Overall pick number.
            - ``a`` (Float64): Skewness shape parameter.
            - ``loc`` (Float64): Location parameter.
            - ``scale`` (Float64): Scale parameter.

        Picks with fewer than ``min_samples`` players in a given window are
        excluded from that window's DataFrame.

    Raises:
        ValueError: If ``window_length`` is even.
    """
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


# ---------------------------------------------------------------------------
# Position career development
# ---------------------------------------------------------------------------


def aggregate_career_av_by_position(
    lazy_frame: pl.LazyFrame,
    normalize: bool,
    rounds: list[int] | None = None,
) -> pl.LazyFrame:
    """Return season-level AV annotated with career year and (optionally) normalized position.

    Works on the season-level data from :func:`prepare_av_data` — one row
    per player per season.

    When ``normalize=True``, compound position codes (e.g. ``"LDE/LOLB"``,
    ``"RB-TE"``) are split on ``"/"`` and ``"-"`` and each component is mapped
    through :data:`_POSITION_GROUPS`
    (e.g. ``"LDE"`` → ``"DE"``, ``"LOLB"`` → ``"LB"``). The player-season row is
    then exploded so the ``AV.1`` value is attributed to **every distinct
    normalized position** in the compound. If both components map to the same
    group (e.g. ``"LDE/RDE"`` → both ``"DE"``), only one row is kept.

    When ``normalize=False``, compound positions are left exactly as recorded and
    no exploding occurs.

    Args:
        lazy_frame: LazyFrame already processed through :func:`prepare_av_data`.
        normalize: If ``True``, split compound positions, map components
            through :data:`_POSITION_GROUPS`, and remove any positions in
            :data:`_SPECALIST` (K, KR, P, PR, LS) and ``_GENERALIST`` (DL, OL).
            If ``False``, keep ``Pos`` as-is with no filtering.
        rounds: Optional list of draft round numbers to include (e.g.
            ``[1]`` for first-round picks only, ``[1, 2]`` for the first two
            rounds). If ``None``, all rounds are included.

    Returns:
        LazyFrame with columns ``Player``, ``Pos``, ``Draft Year``,
        ``years_from_draft`` (Int64), ``AV.1``. Rows where
        ``years_from_draft < 0`` or ``Pos`` is null are dropped.
    """
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

    return lf.select(["Player", "Pos", "Draft Year", "years_from_draft", "AV.1"])


def _compute_group_year_describe(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Compute descriptive statistics of ``AV.1`` grouped by an arbitrary column and career year.

    Args:
        df: Eager DataFrame with one row per player-season containing
            ``group_col``, ``years_from_draft`` (Int64), and ``AV.1`` (Float64).
        group_col: Name of the column to group by alongside ``years_from_draft``
            (e.g. ``"Pos"`` or ``"Round"``).

    Returns:
        Eager DataFrame sorted by ``(group_col, years_from_draft)`` ascending
        with columns: ``group_col``, ``years_from_draft`` (Int64),
        ``count`` (UInt32), ``mean`` (Float64), ``std`` (Float64),
        ``min`` (Float64), ``25%`` (Float64), ``50%`` (Float64),
        ``75%`` (Float64), ``max`` (Float64).
    """
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
    """Compute per-position, per-career-year descriptive statistics of annual AV.

    Loads all parquet files from ``directory`` lazily, annotates each
    player-season with ``years_from_draft``, optionally filters to specific
    draft rounds, optionally normalizes position labels to standard groups,
    then computes descriptive statistics grouped by ``(Pos, years_from_draft)``.

    Args:
        directory: Path to the directory containing annual AV parquet files
            (e.g. ``data/raw/stathead/annual_av``).
        normalize: If ``True`` (default), consolidates raw position variants
            using :data:`_POSITION_GROUPS` (e.g. ``"LDE"`` → ``"DE"``).
            If ``False``, all raw positions are kept as-is.
        rounds: Optional list of draft round numbers to restrict the analysis
            to (e.g. ``[1]`` for first-round picks only, ``[1, 2]`` for the
            first two rounds). If ``None`` (default), all rounds are included.

    Returns:
        Eager DataFrame sorted by ``(Pos, years_from_draft)`` ascending with
        columns: ``Pos`` (String), ``years_from_draft`` (Int64),
        ``count`` (UInt32), ``mean`` (Float64), ``std`` (Float64),
        ``min`` (Float64), ``25%`` (Float64), ``50%`` (Float64),
        ``75%`` (Float64), ``max`` (Float64).
    """
    lf = load_parquets_from_dir(directory, lazy=True)
    lf = prepare_av_data(lf)
    lf = aggregate_career_av_by_position(lf, normalize=normalize, rounds=rounds)
    df = lf.collect()
    return _compute_group_year_describe(df, "Pos")


def round_career_stats(directory: str | Path) -> pl.DataFrame:
    """Compute per-draft-round, per-career-year descriptive statistics of annual AV.

    Loads all parquet files from ``directory`` lazily, annotates each
    player-season with ``years_from_draft``, then computes descriptive
    statistics grouped by ``(Round, years_from_draft)``.

    Args:
        directory: Path to the directory containing annual AV parquet files
            (e.g. ``data/raw/stathead/annual_av``).

    Returns:
        Eager DataFrame sorted by ``(Round, years_from_draft)`` ascending with
        columns: ``Round`` (Int64), ``years_from_draft`` (Int64),
        ``count`` (UInt32), ``mean`` (Float64), ``std`` (Float64),
        ``min`` (Float64), ``25%`` (Float64), ``50%`` (Float64),
        ``75%`` (Float64), ``max`` (Float64).
    """
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


# ---------------------------------------------------------------------------
# Curve fitting — exponential decay
# ---------------------------------------------------------------------------


def exponential_av_fit(
    player_av_data: pl.LazyFrame | pl.DataFrame,
    max_pick: int = 250,
    av_col: str = "rookie_contract_av",
) -> IndividualFitResult:
    """Fit an exponential decay curve to individual player rookie contract AV by pick.

    Uses the model ``f(pick) = a * exp(-b * pick) + c`` against every individual
    player's ``rookie_contract_av`` value.

    Args:
        player_av_data: LazyFrame or eager DataFrame with one row per player.
            Must contain ``Pick`` (Int64) and the column named ``av_col``.
        max_pick: Maximum pick number to include in the fit. Default 250.
        av_col: Name of the AV column to fit. Default ``"rookie_contract_av"``.

    Returns:
        :class:`~curve_fitting.IndividualFitResult` with keys ``popt``,
        ``pcov``, ``perr``, ``x_fit``, ``y_fit``, ``y_upper``, ``y_lower``,
        ``picks``, ``av_values``.

    Raises:
        RuntimeError: If ``curve_fit`` fails to converge.
        ValueError: If fewer than 4 valid data points remain after filtering.
    """
    return curve_fitting.fit_individuals(
        player_av_data, ExpDecayModel, max_pick=max_pick, av_col=av_col
    )


def exponential_av_fit_stat(
    stats_df: pl.DataFrame,
    stat_col: str = "mean",
    max_pick: int = 250,
) -> StatsFitResult:
    """Fit an exponential decay curve to any per-pick descriptive statistic.

    Fits ``f(pick) = a * exp(-b * pick) + c`` against the values in
    ``stat_col``, which may be any column produced by :func:`pick_based_stats`.

    Args:
        stats_df: Output of :func:`pick_based_stats`, one row per pick.
        stat_col: Name of the column to fit. Default ``"mean"``.
        max_pick: Maximum pick number to include. Default 250.

    Returns:
        :class:`~curve_fitting.StatsFitResult` with keys ``popt``, ``pcov``,
        ``perr``, ``x_fit``, ``y_fit``, ``y_upper``, ``y_lower``, ``picks``,
        ``stat_values``, ``iqr_picks``, ``q25``, ``q75``.

    Raises:
        RuntimeError: If ``curve_fit`` fails to converge.
        ValueError: If fewer than 4 valid picks remain after filtering.
    """
    return curve_fitting.fit_stats(
        stats_df, ExpDecayModel, stat_col=stat_col, max_pick=max_pick
    )


def fit_result_to_dataframe(
    fit_result: IndividualFitResult | StatsFitResult,
) -> pl.DataFrame:
    """Convert an exponential fit result to a saveable DataFrame.

    Re-evaluates the fitted curve at every integer pick from 1 through the
    maximum pick used in the fit with the 1-sigma confidence band.

    Args:
        fit_result: Return value of :func:`exponential_av_fit` or
            :func:`exponential_av_fit_stat`.

    Returns:
        Eager DataFrame with columns ``pick`` (Int64), ``y_fit``,
        ``y_upper``, ``y_lower`` (Float64), one row per integer pick.
    """
    return curve_fitting.fit_result_to_dataframe(fit_result, ExpDecayModel)


# ---------------------------------------------------------------------------
# Curve fitting — logarithmic decay
# ---------------------------------------------------------------------------


def logarithmic_av_fit(
    player_av_data: pl.LazyFrame | pl.DataFrame,
    max_pick: int = 250,
    av_col: str = "rookie_contract_av",
) -> IndividualFitResult:
    """Fit a logarithmic decay curve to individual player rookie contract AV by pick.

    Uses the model ``f(pick) = a * ln(pick) + b`` against every individual
    player's AV value.

    Args:
        player_av_data: LazyFrame or eager DataFrame with one row per player.
            Must contain ``Pick`` (Int64) and the column named ``av_col``.
        max_pick: Maximum pick number to include. Default 250.
        av_col: Name of the AV column. Default ``"rookie_contract_av"``.

    Returns:
        :class:`~curve_fitting.IndividualFitResult` with keys ``popt``,
        ``pcov``, ``perr``, ``x_fit``, ``y_fit``, ``y_upper``, ``y_lower``,
        ``picks``, ``av_values``.

    Raises:
        RuntimeError: If ``curve_fit`` fails to converge.
        ValueError: If fewer than 3 valid data points remain after filtering.
    """
    return curve_fitting.fit_individuals(
        player_av_data, LogDecayModel, max_pick=max_pick, av_col=av_col
    )


def logarithmic_av_fit_stat(
    stats_df: pl.DataFrame,
    stat_col: str = "mean",
    max_pick: int = 250,
) -> StatsFitResult:
    """Fit a logarithmic decay curve to a per-pick statistic column.

    Fits ``f(pick) = a * ln(pick) + b`` via nonlinear least squares.
    For a decreasing function ``a < 0`` and ``b`` is the value at pick e ≈ 2.72.

    Args:
        stats_df: Per-pick stats DataFrame (output of :func:`pick_based_stats`).
        stat_col: Column name to fit (e.g. ``"mean"``, ``"50%"``).
        max_pick: Upper pick bound (inclusive) for the fit.

    Returns:
        :class:`~curve_fitting.StatsFitResult` with keys ``popt``, ``pcov``,
        ``perr``, ``x_fit``, ``y_fit``, ``y_upper``, ``y_lower``, ``picks``,
        ``stat_values``, ``iqr_picks``, ``q25``, ``q75``.
    """
    return curve_fitting.fit_stats(
        stats_df, LogDecayModel, stat_col=stat_col, max_pick=max_pick
    )


def logarithmic_fit_result_to_dataframe(
    fit_result: IndividualFitResult | StatsFitResult,
) -> pl.DataFrame:
    """Convert a logarithmic fit result to a saveable DataFrame.

    Re-evaluates ``f(pick) = a * ln(pick) + b`` at every integer pick from 1
    through the maximum pick used in the fit, with a 1-sigma confidence band.

    Args:
        fit_result: Return value of :func:`logarithmic_av_fit` or
            :func:`logarithmic_av_fit_stat`.

    Returns:
        Eager DataFrame with columns ``pick`` (Int64), ``y_fit``,
        ``y_upper``, ``y_lower`` (Float64), one row per integer pick.
    """
    return curve_fitting.fit_result_to_dataframe(fit_result, LogDecayModel)
