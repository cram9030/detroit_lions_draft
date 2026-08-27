"""Rookie-contract pace projections.

For a draft class that has **not** yet completed its 4-year rookie contract,
computes the annual AV each player must produce over the remaining contract
years to close the gap to the Expected AV Above Replacement (EAVAR) for
their pick (i.e. to reach ``surplus_av == 0``), then finds the historical
players at the same position whose *actual*, fully-realized career
trajectory most closely matches that required pace profile.

Typical usage
-------------
>>> from src.career_av import position_career_stats
>>> from src.rookie_contract_pace import (
...     build_position_reference_trajectories,
...     compute_pace_requirements,
...     find_closest_career_comps,
... )
>>>
>>> pos_stats_norm = position_career_stats("data/raw/stathead/annual_av", normalize=True)
>>> pace_df = compute_pace_requirements("DET", 2023, pos_stats=pos_stats_norm)
>>> reference_df = build_position_reference_trajectories()
>>> comps = find_closest_career_comps(
...     target_profile=[pace_df["yr0_av"][0], pace_df["yr1_av"][0], pace_df["yr2_av"][0], pace_df["yr3_av"][0]],
...     position=pace_df["Pos"][0],
...     team="DET",
...     reference_df=reference_df,
... )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.annual_av_analysis import prepare_av_data
from src.data_ingest import load_parquets_from_dir
from src.positions import PLAYER_POSITION_OVERRIDES as _DEFAULT_OVERRIDES
from src.positions import canonicalize_positions, normalize_pos
from src.surplus_av import compute_surplus_av, load_team_draft_class

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW_DIR = _PROJECT_ROOT / "data/raw/stathead/annual_av"
_DEFAULT_EAVAR_PATH = _PROJECT_ROOT / "data/processed/expected_av_above_replacement.csv"

ROOKIE_CONTRACT_YEARS: list[int] = [0, 1, 2, 3]


def allocate_required_av(
    diff: float,
    position: str,
    remaining_years: list[int],
    pos_stats: pl.DataFrame,
    year_col: str = "years_from_draft",
    pos_col: str = "Pos",
    value_col: str = "mean",
) -> dict[int, float]:
    """Allocate a target AV difference across remaining rookie-contract years.

    Splits ``diff`` across ``remaining_years`` in proportion to each year's
    share of the position's typical annual AV, so a position whose AV
    normally leans heavily on year 1 (e.g. most positions decline with
    career year) is asked for more of the shortfall early and less late.

    This is intentionally generic — it only needs a target difference, a
    position, the years to allocate over, and a position/year AV table — so
    it can be reused anywhere a total AV target needs to be spread across
    specific career years by position shape (not just rookie-contract pace).

    Args:
        diff: Total additional AV needed across all of ``remaining_years``
            combined (target total AV minus AV already produced). Values
            ``<= 0`` mean the target is already met — every remaining year
            gets ``0.0``.
        position: Normalized position group (e.g. ``"WR"``, ``"DE"``).
        remaining_years: ``years_from_draft`` values not yet observed,
            e.g. ``[2, 3]``.
        pos_stats: Position/career-year AV table — e.g. ``pos_stats_norm``
            from ``scripts/run_analysis.py``
            (``src.career_av.position_career_stats(..., normalize=True)``).
            Must contain ``[pos_col, year_col, value_col]``.
        year_col: Column holding the career year (default ``"years_from_draft"``).
        pos_col: Column holding the position group (default ``"Pos"``).
        value_col: Column holding the typical AV value used to weight the
            allocation (default ``"mean"``).

    Returns:
        ``{year: required_av}`` for every year in ``remaining_years``. When
        the position has no usable (positive) typical AV for any remaining
        year, ``diff`` is split evenly instead.
    """
    if not remaining_years:
        return {}
    if diff <= 0:
        return {year: 0.0 for year in remaining_years}

    weights: dict[int, float] = {}
    for year in remaining_years:
        row = pos_stats.filter((pl.col(pos_col) == position) & (pl.col(year_col) == year))
        val = row[value_col][0] if len(row) > 0 else None
        weights[year] = max(float(val), 0.0) if val is not None else 0.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        equal_share = diff / len(remaining_years)
        return {year: round(equal_share, 3) for year in remaining_years}

    return {
        year: round(diff * (weight / total_weight), 3) for year, weight in weights.items()
    }


def available_rookie_years(year: int, raw_dir: Path) -> list[int]:
    """Return the ``years_from_draft`` values whose season data already exists on disk."""
    available = [0, 1]
    for offset in (2, 3):
        if (raw_dir / f"draft{year}_season{year + offset}.parquet").exists():
            available.append(offset)
    return available


def compute_pace_requirements(
    team: str,
    year: int,
    raw_dir: Path | None = None,
    eavar_path: Path | None = None,
    pos_stats: pl.DataFrame | None = None,
    position_overrides: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Compute the required annual AV pace for each player in an incomplete rookie-contract class.

    For each remaining (not-yet-observed) rookie-contract year, allocates the
    gap between the player's target total (``eavar + replacement_level``)
    and their AV produced so far, using :func:`allocate_required_av` weighted
    by ``pos_stats``.

    Args:
        team: Three-letter team code (e.g. ``"DET"``).
        year: Draft year.
        raw_dir: Directory containing the annual AV parquets. Defaults to
            ``data/raw/stathead/annual_av``.
        eavar_path: Path to ``expected_av_above_replacement.csv``.
        pos_stats: Position/career-year AV table (``pos_stats_norm``). If
            ``None``, computed via
            ``src.career_av.position_career_stats(raw_dir, normalize=True)``.
        position_overrides: Optional per-player position overrides, forwarded
            to :func:`src.surplus_av.load_team_draft_class`.

    Returns:
        DataFrame with one row per player and columns
        ``[Player, Pos, Pick, Round, Draft Year, eavar, replacement_level,
        total_observed_av, target_total_av, required_av_remaining,
        yr0_av..yr3_av, yr0_type..yr3_type]``. ``yr{n}_type`` is
        ``"observed"`` for years already played and ``"required"`` for the
        allocated pace over remaining years.

    Raises:
        ValueError: If the draft class has already completed all 4
            rookie-contract seasons (i.e. there is nothing to project).
    """
    raw_dir = raw_dir or _DEFAULT_RAW_DIR
    eavar_path = eavar_path or _DEFAULT_EAVAR_PATH

    available_years = available_rookie_years(year, raw_dir)
    if len(available_years) >= len(ROOKIE_CONTRACT_YEARS):
        raise ValueError(
            f"{team} {year} draft class has already completed all 4 rookie-contract "
            "seasons — there are no remaining years to project a pace for."
        )
    remaining_years = [y for y in ROOKIE_CONTRACT_YEARS if y not in available_years]

    if pos_stats is None:
        from src.career_av import position_career_stats

        pos_stats = position_career_stats(raw_dir, normalize=True)

    draft_df = load_team_draft_class(team, year, raw_dir, position_overrides=position_overrides)

    wide = draft_df.pivot(
        index=["Player", "Pos", "Pick", "Round", "Draft Year"],
        on="years_from_draft",
        values="AV.1",
    )
    for y in available_years:
        col = str(y)
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))
        else:
            wide = wide.with_columns(pl.col(col).fill_null(0.0))
    wide = wide.rename({str(y): f"obs_yr{y}" for y in available_years})
    wide = wide.with_columns(
        pl.sum_horizontal([pl.col(f"obs_yr{y}") for y in available_years]).alias("total_4yr_av")
    ).sort("Pick")

    scored = compute_surplus_av(wide, eavar_path=eavar_path)

    rows: list[dict] = []
    for row in scored.iter_rows(named=True):
        pos = row["Pos"]
        surplus = row["surplus_av"]
        diff = -surplus if surplus is not None else 0.0
        required = allocate_required_av(diff, pos, remaining_years, pos_stats)

        record = {
            "Player": row["Player"],
            "Pos": pos,
            "Pick": row["Pick"],
            "Round": row["Round"],
            "Draft Year": row["Draft Year"],
            "eavar": row["eavar"],
            "replacement_level": row["replacement_level"],
            "total_observed_av": round(row["total_4yr_av"], 3),
            "target_total_av": round((row["eavar"] or 0.0) + (row["replacement_level"] or 0.0), 3),
            "required_av_remaining": round(diff, 3),
        }
        for y in ROOKIE_CONTRACT_YEARS:
            if y in available_years:
                record[f"yr{y}_av"] = round(row[f"obs_yr{y}"], 3)
                record[f"yr{y}_type"] = "observed"
            else:
                record[f"yr{y}_av"] = required.get(y, 0.0)
                record[f"yr{y}_type"] = "required"
        rows.append(record)

    return pl.DataFrame(rows)


def build_position_reference_trajectories(
    raw_dir: Path | None = None,
    position_overrides: dict[str, str] | None = None,
    max_years: int = 4,
) -> pl.DataFrame:
    """Build one row per player with a fully-realized rookie-contract AV trajectory.

    Restricted to players whose entire rookie-contract window has already
    elapsed in the dataset (``Draft Year + max_years - 1 <= latest season on
    record``), so every row is an actual completed career, never a partial
    one. Positions are normalized and canonicalized the same way as
    :func:`src.surplus_av.load_team_draft_class`.

    Args:
        raw_dir: Directory containing the annual AV parquets. Defaults to
            ``data/raw/stathead/annual_av``.
        position_overrides: Optional per-player position overrides.
        max_years: Number of rookie-contract years to build (default 4).

    Returns:
        DataFrame with columns
        ``[Player, Pos, Pick, Round, Draft Year, Draft Team, yr0..yr{max_years-1}]``.
    """
    raw_dir = raw_dir or _DEFAULT_RAW_DIR
    overrides = {**_DEFAULT_OVERRIDES, **(position_overrides or {})}

    lf = prepare_av_data(load_parquets_from_dir(raw_dir))
    df = (
        lf.with_columns(
            [
                pl.col("Round").cast(pl.Int64, strict=False),
                (pl.col("Season") - pl.col("Draft Year")).alias("years_from_draft"),
            ]
        )
        .filter(pl.col("years_from_draft").is_in(list(range(max_years))))
        .select(
            [
                "Player",
                "Pos",
                "Pick",
                "Round",
                "Draft Year",
                "Draft Team",
                "Season",
                "years_from_draft",
                "AV.1",
            ]
        )
        .collect()
    )
    if df.is_empty():
        return df

    max_season = int(df["Season"].max())
    df = df.filter(pl.col("Draft Year") + (max_years - 1) <= max_season)
    if df.is_empty():
        return df

    df = df.with_columns(
        pl.struct(["Player", "Pos"])
        .map_elements(
            lambda s: normalize_pos(s["Player"], s["Pos"], overrides),
            return_dtype=pl.String,
        )
        .alias("Pos")
    )
    df = canonicalize_positions(df)

    wide = df.group_by(["Player", "Pos", "Pick", "Round", "Draft Year", "Draft Team"]).agg(
        [
            pl.col("AV.1").filter(pl.col("years_from_draft") == y).sum().alias(f"yr{y}")
            for y in range(max_years)
        ]
    )
    for y in range(max_years):
        wide = wide.with_columns(pl.col(f"yr{y}").fill_null(0.0))

    return wide.sort(["Pos", "Draft Year", "Pick"])


def find_closest_career_comps(
    target_profile: list[float],
    position: str,
    team: str,
    reference_df: pl.DataFrame,
    n_neighbors: int = 3,
    close_tolerance: float = 1.0,
    exclude_players: set[str] | None = None,
) -> pl.DataFrame:
    """Find historical players at ``position`` whose actual trajectory best matches ``target_profile``.

    Ranks candidates by Euclidean distance to ``target_profile`` over the
    same career years. Among candidates within ``close_tolerance`` AV of the
    single closest match — i.e. when there are many similarly-close
    profiles — prefers players drafted by ``team``, then the most recently
    drafted, before falling back to the next-nearest candidates by plain
    distance to fill out ``n_neighbors``.

    Args:
        target_profile: Required/observed AV per rookie-contract year,
            e.g. ``[8.0, 10.0, 7.5, 6.2]`` for years 0-3.
        position: Normalized position group to search within.
        team: Team code used to break ties among "close" candidates.
        reference_df: Output of :func:`build_position_reference_trajectories`.
        n_neighbors: Maximum number of comparison players to return.
        close_tolerance: AV distance from the closest match within which a
            candidate counts as part of the "many close profiles" tie-break
            group.
        exclude_players: Player names to exclude (e.g. the target player's
            own draft class, so a still-active pick can't match itself).

    Returns:
        Up to ``n_neighbors`` rows of ``reference_df`` plus a ``distance``
        column, sorted by distance ascending. Empty (with a ``distance``
        column) if no historical players are available at ``position``.
    """
    pos_ref = reference_df.filter(pl.col("Pos") == position)
    if exclude_players:
        pos_ref = pos_ref.filter(~pl.col("Player").is_in(list(exclude_players)))
    if pos_ref.is_empty():
        return pos_ref.with_columns(pl.lit(None).cast(pl.Float64).alias("distance"))

    max_years = len(target_profile)
    year_cols = [f"yr{y}" for y in range(max_years)]
    matrix = pos_ref.select(year_cols).to_numpy().astype(float)
    target = np.array(target_profile, dtype=float)
    dists = np.linalg.norm(matrix - target, axis=1)

    pos_ref = pos_ref.with_columns(pl.Series("distance", dists))
    min_dist = float(dists.min())
    pos_ref = pos_ref.with_columns(
        [
            (pl.col("Draft Team") == team).alias("_same_team"),
            (pl.col("distance") <= min_dist + close_tolerance).alias("_is_close"),
        ]
    )

    close = pos_ref.filter(pl.col("_is_close")).sort(
        ["_same_team", "Draft Year", "distance"], descending=[True, True, False]
    )
    selected = close.head(n_neighbors)
    if len(selected) < n_neighbors:
        rest = (
            pos_ref.filter(~pl.col("_is_close")).sort("distance").head(n_neighbors - len(selected))
        )
        selected = pl.concat([selected, rest])

    return selected.drop(["_same_team", "_is_close"]).sort("distance")
