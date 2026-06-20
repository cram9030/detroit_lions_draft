"""Surplus AV analysis for NFL draft classes.

Computes each player's 4-year AV (observed + projected) and subtracts the
Expected AV Above Replacement (EAVAR) for their pick position to produce
a per-player surplus.  Summing player surpluses yields a class-level score.

Typical usage
-------------
>>> from src.surplus_av import load_team_draft_class, aggregate_observed_av, compute_surplus_av
>>> from src.surplus_av import aggregate_model_av
>>> from src.models.factory import make_career_av_model
>>> from pathlib import Path
>>>
>>> draft_df = load_team_draft_class("DET", 2024)
>>> # If all 4 seasons have been played:
>>> players_df = aggregate_observed_av(draft_df)
>>> # Otherwise, project missing seasons with a model:
>>> model = make_career_av_model("parametric")
>>> model.load(Path("models/parametric"))
>>> players_df = aggregate_model_av(draft_df, model)
>>> results = compute_surplus_av(players_df)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.positions import PLAYER_POSITION_OVERRIDES as _DEFAULT_OVERRIDES
from src.positions import _GENERALIST as _GENERALIST_LIST
from src.positions import canonicalize_positions, normalize_pos

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW_DIR = _PROJECT_ROOT / "data/raw/stathead/annual_av"
_DEFAULT_EAVAR_PATH = _PROJECT_ROOT / "data/processed/expected_av_above_replacement.csv"

_GENERALIST: frozenset[str] = frozenset(_GENERALIST_LIST)

# Keep backward-compatible alias for callers that import _normalize_pos from this module
_normalize_pos = normalize_pos


def load_team_draft_class(
    team: str,
    year: int,
    raw_dir: Path | None = None,
    position_overrides: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Load a team's draft class for a given year with all available observed AV.

    Requires that at least two completed seasons of data exist: the rookie
    season (``draft{year}_season{year}.parquet``) and the second season
    (``draft{year}_season{year+1}.parquet``).  Seasons 3 and 4 are loaded
    when the corresponding parquets are present.

    Args:
        team: Three-letter team code (e.g. ``"DET"``).
        year: Draft year (e.g. ``2024``).
        raw_dir: Directory containing the annual AV parquets.  Defaults to
            ``data/raw/stathead/annual_av`` relative to the project root.

    Returns:
        DataFrame with columns
        ``[Player, Pos, Pick, Draft Year, years_from_draft, AV.1]``.

    Raises:
        ValueError: If fewer than two completed seasons of data are available,
            or if the required seasons exist on disk but contain no finalized
            AV data for the requested team (parquets downloaded before the
            season stats were finalized).
    """
    raw_dir = raw_dir or _DEFAULT_RAW_DIR
    # Merge caller overrides on top of the module-level defaults so callers can
    # extend or override individual entries without losing the baseline corrections.
    _overrides = {**_DEFAULT_OVERRIDES, **(position_overrides or {})}

    # --- 1. File existence check for required seasons ----------------------
    for season in [year, year + 1]:
        path = raw_dir / f"draft{year}_season{season}.parquet"
        if not path.exists():
            raise ValueError(
                f"At least 2 completed seasons are required but this file is missing:\n"
                f"  {path}\n\n"
                "To download, update config/stathead_annual_av.json and run:\n"
                "  python src/stathead_downloader.py --config config/stathead_annual_av.json"
            )

    # --- 2. Load all available seasons -------------------------------------
    optional = [
        raw_dir / f"draft{year}_season{year + 2}.parquet",
        raw_dir / f"draft{year}_season{year + 3}.parquet",
    ]
    paths = [
        raw_dir / f"draft{year}_season{year}.parquet",
        raw_dir / f"draft{year}_season{year + 1}.parquet",
    ] + [p for p in optional if p.exists()]

    frames = [
        pl.read_parquet(p)
        .filter(pl.col("Draft Team") == team)
        .filter(pl.col("Team").str.split(",").list.contains(team))
        for p in paths
    ]
    raw = pl.concat(frames)

    # --- 3. Type cast — AV.1 left nullable (no fill_null) ------------------
    prepared = (
        raw.with_columns(
            [
                pl.col("Pick").cast(pl.Int64, strict=False),
                pl.col("Round").cast(pl.Int64, strict=False),
                pl.col("Season").cast(pl.Int64, strict=False),
                pl.col("Draft Year").cast(pl.Int64, strict=False),
                pl.col("AV.1").cast(pl.Float64, strict=False),
                pl.col("Pos").str.strip_chars(),
            ]
        )
        .with_columns(
            (pl.col("Season") - pl.col("Draft Year")).alias("years_from_draft")
        )
        .filter(pl.col("years_from_draft").is_in([0, 1, 2, 3]))
    )

    # --- 4. Validate required seasons have finalized AV --------------------
    for yfd in [0, 1]:
        has_data = prepared.filter(
            (pl.col("years_from_draft") == yfd) & pl.col("AV.1").is_not_null()
        )
        if has_data.is_empty():
            season = year + yfd
            raise ValueError(
                f"Season {season} parquet exists but {team} has no finalized AV data "
                f"(all AV.1 values are null).\n"
                "Re-download the data:\n"
                "  python src/stathead_downloader.py --config config/stathead_annual_av.json"
            )

    # --- 5. Normalize positions --------------------------------------------
    prepared = prepared.with_columns(
        pl.struct(["Player", "Pos"])
        .map_elements(
            lambda s: _normalize_pos(s["Player"], s["Pos"], _overrides),
            return_dtype=pl.String,
        )
        .alias("Pos")
    )

    # --- 6. Canonicalize: one position per player across all seasons -------
    prepared = canonicalize_positions(prepared)

    # --- 7. Drop rows with no AV (unfinalized / stale season data) ---------
    prepared = prepared.filter(pl.col("AV.1").is_not_null())

    # --- 8. One row per (player, season) after position merging ------------
    prepared = (
        prepared
        .group_by(["Player", "Pos", "Pick", "Round", "Draft Year", "years_from_draft"])
        .agg(pl.sum("AV.1"))
    )

    return prepared.select(
        ["Player", "Pos", "Pick", "Round", "Draft Year", "years_from_draft", "AV.1"]
    )


def project_player_seasons(
    model,
    player: str,
    pos: str,
    obs_av: list[float],
    overrides: dict[str, str] | None = None,
    pick: int | None = None,
) -> tuple[float, float] | None:
    """Return effective (yr2, yr3) AV for a player, projecting any missing seasons.

    Uses observed values directly when available; calls ``model.predict`` only
    for seasons beyond the observed window.

    Args:
        model: A fitted ``CareerAVModel``.
        player: Player name (used for position override lookup).
        pos: Raw position code from the data.
        obs_av: Observed AV values starting from year 0 (length 2, 3, or 4).
        overrides: Optional per-player position overrides.
        pick: Draft pick number passed to ``model.predict()`` for models that
            use it (e.g. KNN).  Ignored by other models.

    Returns:
        ``(yr2_av, yr3_av)`` or ``None`` if the position is unknown to the model
        and fewer than four seasons have been observed.
    """
    if len(obs_av) >= 4:
        return float(obs_av[2]), float(obs_av[3])

    norm_pos = _normalize_pos(player, pos, overrides)
    try:
        result = model.predict(norm_pos, obs_av, pick=pick)
    except ValueError:
        return None

    preds = dict(zip(result["predicted_years"], result["y_pred"]))
    yr2 = float(obs_av[2]) if len(obs_av) >= 3 else float(preds.get(2, 0.0))
    yr3 = float(preds.get(3, 0.0))
    return yr2, yr3


def aggregate_observed_av(draft_class_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate AV over all four observed seasons per player.  No model needed.

    Use when the draft class has been in the league for four full seasons
    (i.e. ``years_from_draft == 3`` is present in ``draft_class_df``).
    Players absent from a season's data contributed 0 AV that year.

    Args:
        draft_class_df: Output of :func:`load_team_draft_class`.

    Returns:
        DataFrame with columns
        ``[Player, Pos, Pick, Draft Year,
           obs_yr0, obs_yr1, obs_yr2, obs_yr3, total_4yr_av]``.
        All ``obs_yr*`` values are ``0.0`` when the player had no recorded AV
        for that season (cut, injured, etc.).
    """
    observed_years = sorted(draft_class_df["years_from_draft"].unique().to_list())

    wide = draft_class_df.pivot(
        index=["Player", "Pos", "Pick", "Round", "Draft Year"],
        on="years_from_draft",
        values="AV.1",
    ).rename({str(yr): f"obs_yr{yr}" for yr in observed_years})

    for col in ("obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))
        else:
            wide = wide.with_columns(pl.col(col).fill_null(0.0))

    return (
        wide
        .with_columns(
            (pl.col("obs_yr0") + pl.col("obs_yr1") + pl.col("obs_yr2") + pl.col("obs_yr3"))
            .round(1)
            .alias("total_4yr_av")
        )
        .sort("Pick")
        .select(["Player", "Pos", "Pick", "Round", "Draft Year", "obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3", "total_4yr_av"])
    )


def aggregate_model_av(
    draft_class_df: pl.DataFrame,
    model,
    position_overrides: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Aggregate observed AV and project missing seasons via a career AV model.

    Use when the draft class has fewer than four completed seasons (i.e.
    ``years_from_draft == 3`` is absent).  Observed values are always used
    as-is; the model fills only the seasons not yet played.

    Args:
        draft_class_df: Output of :func:`load_team_draft_class`.
        model: A fitted ``CareerAVModel``.
        position_overrides: Optional per-player position map (player name →
            normalized position group).  Needed when Stathead records a
            generalist code (``"OL"``, ``"DL"``) that the model does not
            recognise.

    Returns:
        DataFrame with columns
        ``[Player, Pos, Pick, Draft Year,
           obs_yr0, obs_yr1, obs_yr2, obs_yr3,
           proj_yr2, proj_yr3, total_4yr_av, is_projected]``.
        ``obs_yr2`` / ``obs_yr3`` are ``null`` when not yet observed.
        ``proj_yr2`` is the model's yr2 contribution (0.0 when yr2 is observed).
        ``proj_yr3`` is always the model's yr3 prediction (yr3 has not been played).
    """
    _overrides = {**_DEFAULT_OVERRIDES, **(position_overrides or {})}

    observed_years = sorted(draft_class_df["years_from_draft"].unique().to_list())

    wide = draft_class_df.pivot(
        index=["Player", "Pos", "Pick", "Round", "Draft Year"],
        on="years_from_draft",
        values="AV.1",
    ).rename({str(yr): f"obs_yr{yr}" for yr in observed_years})

    for col in ("obs_yr0", "obs_yr1"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))
        else:
            wide = wide.with_columns(pl.col(col).fill_null(0.0))

    for col in ("obs_yr2", "obs_yr3"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    wide = wide.sort("Pick")

    rows = []
    for row in wide.iter_rows(named=True):
        player = row["Player"]
        pos = row["Pos"]
        pick = row["Pick"]
        pick_round = row["Round"]
        draft_year = row["Draft Year"]
        yr0 = row["obs_yr0"]
        yr1 = row["obs_yr1"]
        yr2_obs = row["obs_yr2"]

        obs_av: list[float] = [yr0, yr1]
        if yr2_obs is not None:
            obs_av.append(yr2_obs)

        proj = project_player_seasons(model, player, pos, obs_av, _overrides, pick=int(pick) if pick is not None else None)
        proj_yr2 = proj[0] if proj else 0.0
        proj_yr3 = proj[1] if proj else 0.0

        # proj_yr2 already equals yr2_obs when yr2 is observed (project_player_seasons
        # returns obs_av[2] directly); zero it in the output column so the display
        # reflects only the model's contribution.
        out_proj_yr2 = 0.0 if yr2_obs is not None else proj_yr2
        total = yr0 + yr1 + (yr2_obs if yr2_obs is not None else proj_yr2) + proj_yr3

        rows.append(
            {
                "Player": player,
                "Pos": pos,
                "Pick": pick,
                "Round": pick_round,
                "Draft Year": draft_year,
                "obs_yr0": round(yr0, 1),
                "obs_yr1": round(yr1, 1),
                "obs_yr2": round(yr2_obs, 1) if yr2_obs is not None else None,
                "obs_yr3": None,
                "proj_yr2": round(out_proj_yr2, 1),
                "proj_yr3": round(proj_yr3, 1),
                "total_4yr_av": round(total, 1),
                "is_projected": True,
            }
        )

    return pl.DataFrame(rows)


def compute_surplus_av(
    players_4yr_df: pl.DataFrame,
    eavar_path: Path | None = None,
) -> pl.DataFrame:
    """Compute surplus AV = total_4yr_av - EAVAR for each player's pick.

    Joins ``players_4yr_df`` (output of :func:`aggregate_4yr_av`) against
    ``expected_av_above_replacement.csv`` on pick number.  Players whose
    pick is not in the EAVAR table receive a null ``surplus_av``.

    Args:
        players_4yr_df: Output of :func:`aggregate_4yr_av`.
        eavar_path: Path to ``expected_av_above_replacement.csv``.  Defaults
            to ``data/processed/expected_av_above_replacement.csv``.

    Returns:
        Input DataFrame with columns
        ``[eavar, eavar_upper, eavar_lower, replacement_level, surplus_av]``
        appended.
    """
    eavar_path = eavar_path or _DEFAULT_EAVAR_PATH
    eavar_df = pl.read_csv(eavar_path).rename({"pick": "Pick"})

    # Picks beyond the last entry in the EAVAR table use that last pick's values.
    max_pick = eavar_df["Pick"].max()
    result = (
        players_4yr_df
        .with_columns(pl.col("Pick").clip(upper_bound=max_pick).alias("_pick_key"))
        .join(eavar_df, left_on="_pick_key", right_on="Pick", how="left")
        .drop("_pick_key")
    )

    result = result.with_columns(
        (pl.col("total_4yr_av") - pl.col("replacement_level")).round(1)
        .alias("total_4yr_av_above_replacement")
    )
    result = result.with_columns(
        (pl.col("total_4yr_av_above_replacement") - pl.col("eavar")).round(1)
        .alias("surplus_av")
    )
    return result


def aggregate_drafts(
    team: str,
    years: list[int],
    model_name: str = "linear",
    parametric_curve: str = "quadratic",
    models_dir: Path | None = None,
    raw_dir: Path | None = None,
    position_overrides: dict[str, str] | None = None,
) -> dict:
    """Aggregate surplus AV across multiple draft years for a team.

    For each year, loads the draft class, determines whether it is fully
    observed (all 4 seasons present) or partially observed, computes surplus
    AV, and sums across all successful years.  Years where data is unavailable
    are silently skipped.

    Args:
        team: Stathead team code (e.g. ``"DET"``).
        years: List of draft years to aggregate.
        model_name: Career AV model for partially-observed classes.
        parametric_curve: Curve variant when model_name is ``"parametric"``.
        models_dir: Directory containing trained model artifacts.
        raw_dir: Directory containing annual AV parquets.
        position_overrides: Per-player position overrides.

    Returns:
        Dict with keys:
        - ``total_surplus_av``: sum of class-level surplus_av across years.
        - ``per_year``: ``{year: class_surplus_av}`` for successful years.
        - ``n_years``: number of years with valid data.
        - ``avg_surplus_per_year``: ``total / n_years`` (0.0 when n_years==0).
    """
    from src.models.factory import make_career_av_model

    models_dir = models_dir or _PROJECT_ROOT / "models"
    raw_dir = raw_dir or _DEFAULT_RAW_DIR

    per_year: dict[int, float] = {}

    for year in years:
        try:
            draft_df = load_team_draft_class(team, year, raw_dir, position_overrides=position_overrides)
        except (ValueError, FileNotFoundError):
            continue

        # Determine if year-3 season file exists (fully observed)
        yr3_path = raw_dir / f"draft{year}_season{year + 3}.parquet"
        fully_observed = yr3_path.exists()

        if fully_observed:
            players_df = aggregate_observed_av(draft_df)
        else:
            try:
                if model_name == "parametric":
                    from src.models.factory import make_career_av_model as _make
                    model = _make(model_name, curve=parametric_curve)
                    model_path = models_dir / "parametric" / parametric_curve
                else:
                    model = make_career_av_model(model_name)
                    model_path = models_dir / model_name
                model.load(model_path)
            except Exception:
                continue
            players_df = aggregate_model_av(draft_df, model, position_overrides)

        surplus_df = compute_surplus_av(players_df)
        class_surplus = round(float(surplus_df["surplus_av"].drop_nulls().sum()), 1)
        per_year[year] = class_surplus

    total = sum(per_year.values())
    n = len(per_year)
    return {
        "total_surplus_av": round(total, 1),
        "per_year": per_year,
        "n_years": n,
        "avg_surplus_per_year": round(total / n, 3) if n > 0 else 0.0,
    }
