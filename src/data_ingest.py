"""Data ingestion utilities for the Detroit Lions draft analysis pipeline."""

import json
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Season performance
# ---------------------------------------------------------------------------

def _srs_for_season(games: pl.DataFrame) -> dict[str, float]:
    """Solve the SRS linear system for one season using a normalised schedule matrix."""
    teams = sorted(games["team"].unique().to_list())
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    margin_sum = np.zeros(n)
    A = np.zeros((n, n))

    for row in games.iter_rows(named=True):
        i = idx[row["team"]]
        j = idx.get(row["opponent"])
        if j is not None:
            A[i, j] += 1
            margin_sum[i] += row["pf"] - row["pa"]

    games_played = A.sum(axis=1)
    safe = np.where(games_played == 0, 1.0, games_played)
    mov = margin_sum / safe
    A_norm = A / safe[:, np.newaxis]

    lhs = np.eye(n) - A_norm
    rhs = mov.copy()
    lhs[-1, :] = 1.0
    rhs[-1] = 0.0

    try:
        srs_vals = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        srs_vals = mov

    return {t: round(float(s), 3) for t, s in zip(teams, srs_vals)}


def compute_season_performance() -> pl.DataFrame:
    """Return ``(team, season, win_pct, point_diff, srs)`` from nflreadpy schedules.

    Team codes are nflreadpy abbreviations (e.g. GB, KC).
    """
    sched = nflreadpy.load_schedules()
    reg = (
        sched
        .filter(pl.col("game_type") == "REG")
        .filter(pl.col("home_score").is_not_null())
        .select(["season", "home_team", "home_score", "away_team", "away_score"])
    )

    home = reg.select(
        pl.col("season"),
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
        pl.col("away_team").alias("opponent"),
    )
    away = reg.select(
        pl.col("season"),
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
        pl.col("home_team").alias("opponent"),
    )
    all_games = pl.concat([home, away])

    agg = (
        all_games
        .with_columns(
            (pl.col("pf") > pl.col("pa")).cast(pl.Float64).alias("win"),
            (pl.col("pf") == pl.col("pa")).cast(pl.Float64).alias("tie"),
        )
        .group_by(["season", "team"])
        .agg(
            pl.sum("win").alias("wins"),
            pl.sum("tie").alias("ties"),
            pl.len().alias("games"),
            pl.sum("pf").alias("total_pf"),
            pl.sum("pa").alias("total_pa"),
        )
        .with_columns(
            ((pl.col("wins") + 0.5 * pl.col("ties")) / pl.col("games")).alias("win_pct"),
            (pl.col("total_pf") - pl.col("total_pa")).alias("point_diff"),
        )
    )

    srs_rows: list[dict] = []
    for season in sorted(all_games["season"].unique().to_list()):
        sg = all_games.filter(pl.col("season") == season)
        for team, srs in _srs_for_season(sg).items():
            srs_rows.append({"season": season, "team": team, "srs": srs})

    srs_df = pl.DataFrame(srs_rows)
    return (
        agg
        .join(srs_df, on=["season", "team"], how="left")
        .select(["season", "team", "win_pct", "point_diff", "srs"])
        .sort(["season", "team"])
    )


# ---------------------------------------------------------------------------
# Baked draft data
# ---------------------------------------------------------------------------

_DEFAULT_BAKED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "baked"


def load_baked_data(
    years: list[int],
    baked_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Load per-(team, draft_year) class surplus and obs_yr* totals from baked JSONs.

    ``class_surplus`` is ``None`` for partially-observed years (models are not
    committed to surplus values until all 4 seasons have completed).

    Args:
        years: Draft years to load.
        baked_dir: Directory containing ``draft_{year}.json`` files.
            Defaults to ``data/processed/baked/``.

    Returns:
        DataFrame with columns
        ``[stathead_team, draft_year, class_surplus, obs_yr0, obs_yr1, obs_yr2, obs_yr3]``.
    """
    baked_dir = Path(baked_dir) if baked_dir else _DEFAULT_BAKED_DIR
    rows = []
    for year in years:
        path = baked_dir / f"draft_{year}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for team, tdata in data["teams"].items():
            if tdata.get("fully_observed"):
                surplus = tdata.get("class_summary", {}).get("class_surplus")
                players = tdata.get("players", [])
            else:
                surplus = None
                models = tdata.get("models", {})
                players = next(iter(models.values())).get("players", []) if models else []

            rows.append({
                "stathead_team": team,
                "draft_year": year,
                "class_surplus": surplus,
                "obs_yr0": sum(p.get("obs_yr0") or 0.0 for p in players),
                "obs_yr1": sum(p.get("obs_yr1") or 0.0 for p in players),
                "obs_yr2": sum(p.get("obs_yr2") or 0.0 for p in players),
                "obs_yr3": sum(p.get("obs_yr3") or 0.0 for p in players),
            })

    return pl.DataFrame(rows)
