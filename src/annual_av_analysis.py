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

import warnings
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

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
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_av_data(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
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


def _aggregate_player_av(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
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
        lazy_frame: LazyFrame already processed through :func:`_prepare_av_data`.

    Returns:
        LazyFrame with one row per unique (Player, Pick, Draft Year)
        combination. Output columns: ``Player`` (String), ``Pick`` (Int64),
        ``Draft Year`` (Int64), ``Draft Team`` (String),
        ``rookie_contract_av`` (Float64).
        Negative values are valid and must not be filtered.
    """
    return (
        lazy_frame.filter(pl.col("Season") - pl.col("Draft Year") <= 3)
        .group_by(["Player", "Pick", "Draft Year", "Draft Team"])
        .agg(pl.col("AV.1").sum().alias("rookie_contract_av"))
    )


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
            :func:`_aggregate_player_av` after ``.collect()``.

    Returns:
        Eager DataFrame sorted by ``Pick`` ascending with columns:
            - ``Pick`` (Int64)
            - ``count`` (UInt32): Number of players at this pick.
            - ``null_count`` (UInt32): Null values in ``rookie_contract_av``
              (always 0 after :func:`_prepare_av_data`).
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


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def pick_based_stats(directory: str | Path) -> pl.DataFrame:
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
    lf = _prepare_av_data(lf)
    lf = _aggregate_player_av(lf)
    df = lf.collect()
    return _compute_pick_describe(df)


def skew_normal_fit(
    player_av_data: pl.LazyFrame,
    min_samples: int = 5,
) -> pl.DataFrame:
    """Fit a skew-normal distribution to ``rookie_contract_av`` per pick.

    Intended to receive the output of :func:`_aggregate_player_av` as a
    LazyFrame, which is then collected internally.

    Input LazyFrame columns required:
        - ``Pick`` (Int64): Overall pick number.
        - ``rookie_contract_av`` (Float64): Total AV over tracked seasons.

    Args:
        player_av_data: LazyFrame with one row per player, already processed
            through :func:`_prepare_av_data` and :func:`_aggregate_player_av`.
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


def rolling_window_pick_stats(
    directory: str | Path,
    window_length: int,
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

    Returns:
        Dict mapping ``center_year`` (int) to an eager DataFrame with the
        same schema as :func:`pick_based_stats` output:
            - ``Pick`` (Int64)
            - ``count`` (UInt32)
            - ``null_count`` (UInt32)
            - ``mean`` (Float64)
            - ``std`` (Float64)
            - ``min`` (Float64)
            - ``25%`` (Float64)
            - ``50%`` (Float64)
            - ``75%`` (Float64)
            - ``max`` (Float64)

    Raises:
        ValueError: If ``window_length`` is even.
    """
    if window_length % 2 == 0:
        raise ValueError(
            f"window_length must be odd for symmetric centering, got {window_length}."
        )

    half = window_length // 2

    lf = load_parquets_from_dir(directory, lazy=True)
    df_all = _aggregate_player_av(_prepare_av_data(lf)).collect()

    min_year: int = df_all["Draft Year"].min()
    max_year: int = df_all["Draft Year"].max()
    first_center = min_year + half
    last_center = max_year - half

    result: dict[int, pl.DataFrame] = {}
    for center in range(first_center, last_center + 1):
        window_df = df_all.filter(
            (pl.col("Draft Year") >= center - half)
            & (pl.col("Draft Year") <= center + half)
        )
        result[center] = _compute_pick_describe(window_df)

    return result


def rolling_window_skew_fit(
    directory: str | Path,
    window_length: int,
    min_samples: int = 5,
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
    df_all = _aggregate_player_av(_prepare_av_data(lf)).collect()

    min_year: int = df_all["Draft Year"].min()
    max_year: int = df_all["Draft Year"].max()
    first_center = min_year + half
    last_center = max_year - half

    result: dict[int, pl.DataFrame] = {}
    for center in range(first_center, last_center + 1):
        window_df = df_all.filter(
            (pl.col("Draft Year") >= center - half)
            & (pl.col("Draft Year") <= center + half)
        )
        result[center] = _fit_skewnorm_on_df(window_df, min_samples=min_samples)

    return result


# ---------------------------------------------------------------------------
# Position career development
# ---------------------------------------------------------------------------


def _aggregate_career_av_by_position(
    lazy_frame: pl.LazyFrame,
    normalize: bool,
    rounds: list[int] | None = None,
) -> pl.LazyFrame:
    """Return season-level AV annotated with career year and (optionally) normalized position.

    Works on the season-level data from :func:`_prepare_av_data` — one row
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
        lazy_frame: LazyFrame already processed through :func:`_prepare_av_data`.
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


def _compute_position_year_describe(df: pl.DataFrame) -> pl.DataFrame:
    """Compute descriptive statistics of ``AV.1`` grouped by position and career year."""
    return _compute_group_year_describe(df, "Pos")


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
    lf = _prepare_av_data(lf)
    lf = _aggregate_career_av_by_position(lf, normalize=normalize, rounds=rounds)
    df = lf.collect()
    return _compute_position_year_describe(df)


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
        _prepare_av_data(lf)
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
# Exponential fit
# ---------------------------------------------------------------------------


class ExponentialFitResult(TypedDict):
    """Return type of :func:`exponential_av_fit`.

    Attributes:
        popt: Fitted parameters ``[a, b, c]`` for ``f(x) = a * exp(-b*x) + c``.
        pcov: 3×3 covariance matrix of ``popt`` from ``curve_fit``.
        perr: 1-sigma uncertainties on each parameter, ``sqrt(diag(pcov))``.
        x_fit: Dense array of pick numbers for plotting the fitted curve.
        y_fit: Fitted AV values at each ``x_fit`` point.
        y_upper: Upper 1-sigma bound at each ``x_fit`` point.
        y_lower: Lower 1-sigma bound at each ``x_fit`` point.
        picks: Individual pick numbers for every player used in the fit
            (repeated — one entry per player, not per unique pick).
        av_values: Individual ``rookie_contract_av`` values for every player
            used in the fit, aligned with ``picks``.
    """

    popt: np.ndarray
    pcov: np.ndarray
    perr: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray
    picks: np.ndarray
    av_values: np.ndarray


def _exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay model: ``f(x) = a * exp(-b * x) + c``."""
    return a * np.exp(-b * x) + c


def exponential_av_fit(
    player_av_data: pl.LazyFrame,
    max_pick: int = 250,
) -> ExponentialFitResult:
    """Fit an exponential decay curve to individual player rookie contract AV by pick.

    Uses ``scipy.optimize.curve_fit`` to fit the model
    ``f(pick) = a * exp(-b * pick) + c`` against every individual player's
    ``rookie_contract_av`` value. Each pick number appears once per player
    drafted at that position (typically ~52 data points per pick across the
    full dataset), giving the fit the full statistical weight of the population
    rather than fitting to per-pick means.

    The 1-sigma confidence band is derived from the covariance matrix via
    error propagation: ``sigma²(x) = J(x) @ pcov @ J(x).T`` where ``J`` is
    the Jacobian of the model with respect to the parameters.

    Input LazyFrame columns required:
        - ``Pick`` (Int64): Overall pick number.
        - ``rookie_contract_av`` (Float64): Total AV over tracked seasons,
          one row per player. Output of :func:`_aggregate_player_av`.

    Args:
        player_av_data: LazyFrame with one row per player, already processed
            through :func:`_prepare_av_data` and :func:`_aggregate_player_av`.
            Typically ~12,000 rows covering all draft years in the dataset.
        max_pick: Maximum pick number to include in the fit. Default 250.
            Players drafted beyond this pick are excluded before fitting.

    Returns:
        :class:`ExponentialFitResult` dict with keys:
            - ``popt`` (ndarray, shape (3,)): Fitted parameters ``[a, b, c]``.
            - ``pcov`` (ndarray, shape (3, 3)): Covariance matrix of ``popt``.
            - ``perr`` (ndarray, shape (3,)): 1-sigma parameter uncertainties.
            - ``x_fit`` (ndarray): 500-point pick axis for smooth curve plotting.
            - ``y_fit`` (ndarray): Fitted AV values at ``x_fit`` points.
            - ``y_upper`` (ndarray): Upper 1-sigma bound at ``x_fit`` points.
            - ``y_lower`` (ndarray): Lower 1-sigma bound at ``x_fit`` points.
            - ``picks`` (ndarray): Individual pick numbers used in the fit
              (one entry per player, repeated across draft years).
            - ``av_values`` (ndarray): Individual ``rookie_contract_av`` values
              aligned with ``picks``.

    Warns:
        UserWarning: If any diagonal element of ``pcov`` exceeds ``1e6`` or
            is infinite, indicating the fit may be over-parameterized or that
            the model is poorly constrained by the available data.

    Raises:
        RuntimeError: If ``scipy.optimize.curve_fit`` fails to converge.
        ValueError: If fewer than 4 valid data points remain after filtering.
    """
    df = (
        player_av_data.filter(pl.col("Pick") <= max_pick)
        .select(["Pick", "rookie_contract_av"])
        .drop_nulls()
        .collect()
        .sort("Pick")
    )

    if len(df) < 4:
        raise ValueError(
            f"Only {len(df)} player records after filtering to max_pick={max_pick}. "
            "Need at least 4 for a 3-parameter exponential fit."
        )

    picks = df["Pick"].to_numpy().astype(float)
    av_values = df["rookie_contract_av"].to_numpy()

    # Initial guess derived from the data range
    p0 = [float(av_values.max()), 0.01, float(av_values.min())]

    try:
        popt, pcov = curve_fit(_exp_decay, picks, av_values, p0=p0, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError(f"Exponential curve fit failed to converge: {exc}") from exc

    perr = np.sqrt(np.diag(pcov))

    if np.any(np.isinf(pcov)) or np.any(perr > 1e6):
        warnings.warn(
            "Exponential fit may be over-parameterized: one or more parameter "
            "uncertainties are extremely large (perr > 1e6 or infinite). "
            "Consider simplifying the model or checking data quality.",
            UserWarning,
            stacklevel=2,
        )

    # Dense x axis for smooth curve
    x_fit = np.linspace(picks.min(), picks.max(), 500)
    y_fit = _exp_decay(x_fit, *popt)

    # 1-sigma band via error propagation: sigma²(x) = J(x) @ pcov @ J(x).T
    a, b, _ = popt
    J = np.column_stack(
        [
            np.exp(-b * x_fit),               # df/da
            -a * x_fit * np.exp(-b * x_fit),  # df/db
            np.ones_like(x_fit),              # df/dc
        ]
    )  # shape (500, 3)
    sigma_sq = np.einsum("ij,jk,ik->i", J, pcov, J)
    sigma = np.sqrt(np.abs(sigma_sq))

    return ExponentialFitResult(
        popt=popt,
        pcov=pcov,
        perr=perr,
        x_fit=x_fit,
        y_fit=y_fit,
        y_upper=y_fit + sigma,
        y_lower=y_fit - sigma,
        picks=picks,
        av_values=av_values,
    )


class ExponentialMeansFitResult(TypedDict):
    """Return type of :func:`exponential_av_fit_means`.

    Attributes:
        popt: Fitted parameters ``[a, b, c]`` for ``f(x) = a * exp(-b*x) + c``.
        pcov: 3×3 covariance matrix of ``popt`` from ``curve_fit``.
        perr: 1-sigma uncertainties on each parameter, ``sqrt(diag(pcov))``.
        x_fit: Dense array of pick numbers for plotting the fitted curve.
        y_fit: Fitted AV values at each ``x_fit`` point.
        y_upper: Upper 1-sigma bound at each ``x_fit`` point.
        y_lower: Lower 1-sigma bound at each ``x_fit`` point.
        picks: Unique pick numbers from the input stats (one per pick position).
        means: Mean ``rookie_contract_av`` per pick from the input stats,
            aligned with ``picks``.
        iqr_picks: Pick numbers where both 25th and 75th percentiles are available.
        q25: 25th percentile AV per pick, aligned with ``iqr_picks``.
        q75: 75th percentile AV per pick, aligned with ``iqr_picks``.
    """

    popt: np.ndarray
    pcov: np.ndarray
    perr: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray
    picks: np.ndarray
    means: np.ndarray
    iqr_picks: np.ndarray
    q25: np.ndarray
    q75: np.ndarray


def exponential_av_fit_means(
    stats_df: pl.DataFrame,
    max_pick: int = 250,
) -> ExponentialMeansFitResult:
    """Fit an exponential decay curve to mean rookie contract AV by pick number.

    Uses ``scipy.optimize.curve_fit`` to fit the model
    ``f(pick) = a * exp(-b * pick) + c`` against the per-pick mean AV values
    from :func:`pick_based_stats`. Each pick contributes exactly one data
    point (its mean), giving every pick equal weight in the fit regardless of
    how many players were drafted there.

    The 1-sigma confidence band is derived from the covariance matrix via
    error propagation: ``sigma²(x) = J(x) @ pcov @ J(x).T`` where ``J`` is
    the Jacobian of the model with respect to the parameters.

    Input DataFrame columns required:
        - ``Pick`` (Int64): Overall pick number.
        - ``mean`` (Float64): Mean ``rookie_contract_av`` per pick. Rows where
          ``mean`` is null are excluded before fitting.

    Args:
        stats_df: Output of :func:`pick_based_stats`, one row per pick with
            descriptive statistics of ``rookie_contract_av``.
        max_pick: Maximum pick number to include in the fit. Default 250.
            Picks beyond this value are excluded before fitting.

    Returns:
        :class:`ExponentialMeansFitResult` dict with keys:
            - ``popt`` (ndarray, shape (3,)): Fitted parameters ``[a, b, c]``.
            - ``pcov`` (ndarray, shape (3, 3)): Covariance matrix of ``popt``.
            - ``perr`` (ndarray, shape (3,)): 1-sigma parameter uncertainties.
            - ``x_fit`` (ndarray): 500-point pick axis for smooth curve plotting.
            - ``y_fit`` (ndarray): Fitted AV values at ``x_fit`` points.
            - ``y_upper`` (ndarray): Upper 1-sigma bound at ``x_fit`` points.
            - ``y_lower`` (ndarray): Lower 1-sigma bound at ``x_fit`` points.
            - ``picks`` (ndarray): Unique pick numbers used in the fit.
            - ``means`` (ndarray): Mean AV per pick aligned with ``picks``.

    Warns:
        UserWarning: If any diagonal element of ``pcov`` exceeds ``1e6`` or
            is infinite, indicating the fit may be over-parameterized.

    Raises:
        RuntimeError: If ``scipy.optimize.curve_fit`` fails to converge.
        ValueError: If fewer than 4 valid picks remain after filtering.
    """
    df = (
        stats_df.filter(pl.col("Pick") <= max_pick)
        .select(["Pick", "mean"])
        .drop_nulls()
        .sort("Pick")
    )

    iqr_df = (
        stats_df.filter(pl.col("Pick") <= max_pick)
        .select(["Pick", "25%", "75%"])
        .drop_nulls()
        .sort("Pick")
    )

    if len(df) < 4:
        raise ValueError(
            f"Only {len(df)} valid picks after filtering to max_pick={max_pick}. "
            "Need at least 4 for a 3-parameter exponential fit."
        )

    picks = df["Pick"].to_numpy().astype(float)
    means = df["mean"].to_numpy()

    p0 = [float(means.max()), 0.01, float(means.min())]

    try:
        popt, pcov = curve_fit(_exp_decay, picks, means, p0=p0, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError(f"Exponential curve fit failed to converge: {exc}") from exc

    perr = np.sqrt(np.diag(pcov))

    if np.any(np.isinf(pcov)) or np.any(perr > 1e6):
        warnings.warn(
            "Exponential fit may be over-parameterized: one or more parameter "
            "uncertainties are extremely large (perr > 1e6 or infinite). "
            "Consider simplifying the model or checking data quality.",
            UserWarning,
            stacklevel=2,
        )

    x_fit = np.linspace(picks.min(), picks.max(), 500)
    y_fit = _exp_decay(x_fit, *popt)

    a, b, _ = popt
    J = np.column_stack(
        [
            np.exp(-b * x_fit),               # df/da
            -a * x_fit * np.exp(-b * x_fit),  # df/db
            np.ones_like(x_fit),              # df/dc
        ]
    )
    sigma_sq = np.einsum("ij,jk,ik->i", J, pcov, J)
    sigma = np.sqrt(np.abs(sigma_sq))

    return ExponentialMeansFitResult(
        popt=popt,
        pcov=pcov,
        perr=perr,
        x_fit=x_fit,
        y_fit=y_fit,
        y_upper=y_fit + sigma,
        y_lower=y_fit - sigma,
        picks=picks,
        means=means,
        iqr_picks=iqr_df["Pick"].to_numpy(),
        q25=iqr_df["25%"].to_numpy(),
        q75=iqr_df["75%"].to_numpy(),
    )
