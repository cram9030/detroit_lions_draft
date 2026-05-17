"""Surplus AV analysis for NFL draft classes.

Computes each player's 4-year AV (observed + projected) and subtracts the
Expected AV Above Replacement (EAVAR) for their pick position to produce
a per-player surplus.  Summing player surpluses yields a class-level score.

Typical usage
-------------
>>> from src.surplus_av import load_team_draft_class, aggregate_4yr_av, compute_surplus_av
>>> from src.models.factory import make_career_av_model
>>> from pathlib import Path
>>>
>>> draft_df = load_team_draft_class("DET", 2024)
>>> model = make_career_av_model("parametric")
>>> model.load(Path("models/parametric"))
>>> players_df = aggregate_4yr_av(draft_df, model)
>>> results = compute_surplus_av(players_df)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.annual_av_analysis import _POSITION_GROUPS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW_DIR = _PROJECT_ROOT / "data/raw/stathead/annual_av"
_DEFAULT_EAVAR_PATH = _PROJECT_ROOT / "data/processed/expected_av_above_replacement.csv"


def _normalize_pos(
    player: str,
    pos: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Return the normalized position group, applying per-player overrides first."""
    if overrides and player in overrides:
        return overrides[player]
    first = pos.replace("-", "/").split("/")[0].strip()
    return _POSITION_GROUPS.get(first, first)


def load_team_draft_class(
    team: str,
    year: int,
    raw_dir: Path | None = None,
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
        ValueError: If fewer than two completed seasons of data are available.
    """
    raw_dir = raw_dir or _DEFAULT_RAW_DIR

    required = [
        raw_dir / f"draft{year}_season{year}.parquet",
        raw_dir / f"draft{year}_season{year + 1}.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise ValueError(
            f"At least 2 completed seasons are required but the following data files "
            f"are missing:\n  " + "\n  ".join(missing)
        )

    optional = [
        raw_dir / f"draft{year}_season{year + 2}.parquet",
        raw_dir / f"draft{year}_season{year + 3}.parquet",
    ]
    paths = required + [p for p in optional if p.exists()]

    frames = []
    for path in paths:
        df = pl.read_parquet(path)
        frames.append(df.filter(pl.col("Draft Team") == team))

    raw = pl.concat(frames)

    prepared = (
        raw.with_columns(
            [
                pl.col("Pick").cast(pl.Int64, strict=False),
                pl.col("Season").cast(pl.Int64, strict=False),
                pl.col("Draft Year").cast(pl.Int64, strict=False),
                pl.col("AV.1").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("Pos").str.strip_chars(),
            ]
        )
        .with_columns(
            (pl.col("Season") - pl.col("Draft Year")).alias("years_from_draft")
        )
        .filter(pl.col("years_from_draft").is_in([0, 1, 2, 3]))
    )

    prepared = prepared.with_columns(
        pl.struct(["Player", "Pos"])
        .map_elements(
            lambda s: _normalize_pos(s["Player"], s["Pos"]),
            return_dtype=pl.String,
        )
        .alias("Pos")
    )

    return prepared.select(
        ["Player", "Pos", "Pick", "Draft Year", "years_from_draft", "AV.1"]
    )


def project_player_seasons(
    model,
    player: str,
    pos: str,
    obs_av: list[float],
    overrides: dict[str, str] | None = None,
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

    Returns:
        ``(yr2_av, yr3_av)`` or ``None`` if the position is unknown to the model
        and fewer than four seasons have been observed.
    """
    if len(obs_av) >= 4:
        return float(obs_av[2]), float(obs_av[3])

    norm_pos = _normalize_pos(player, pos, overrides)
    try:
        result = model.predict(norm_pos, obs_av)
    except ValueError:
        return None

    preds = dict(zip(result["predicted_years"], result["y_pred"]))
    yr2 = float(obs_av[2]) if len(obs_av) >= 3 else float(preds.get(2, 0.0))
    yr3 = float(preds.get(3, 0.0))
    return yr2, yr3


def aggregate_4yr_av(
    draft_class_df: pl.DataFrame,
    model,
) -> pl.DataFrame:
    """Aggregate observed and projected AV over the first four seasons per player.

    Seasons already present in ``draft_class_df`` are used as-is.  Any
    seasons beyond the observed window are projected via ``model``.

    Args:
        draft_class_df: Output of :func:`load_team_draft_class`.
        model: A fitted ``CareerAVModel``.

    Returns:
        DataFrame with columns
        ``[Player, Pos, Pick, Draft Year,
           obs_yr0, obs_yr1, obs_yr2, obs_yr3,
           proj_yr2, proj_yr3, total_4yr_av, is_projected]``.
        ``obs_yr2`` / ``obs_yr3`` are ``null`` when not yet observed.
        ``proj_yr2`` / ``proj_yr3`` reflect model output (0.0 when observed).
    """
    observed_years = sorted(draft_class_df["years_from_draft"].unique().to_list())

    wide = draft_class_df.pivot(
        index=["Player", "Pos", "Pick", "Draft Year"],
        on="years_from_draft",
        values="AV.1",
    ).rename({str(yr): f"obs_yr{yr}" for yr in observed_years})

    # Ensure required columns exist and fill required seasons
    for col in ("obs_yr0", "obs_yr1"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))
        else:
            wide = wide.with_columns(pl.col(col).fill_null(0.0))

    # Optional year columns stay nullable — null means "season not yet played"
    for col in ("obs_yr2", "obs_yr3"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    wide = wide.sort("Pick")

    rows = []
    for row in wide.iter_rows(named=True):
        player = row["Player"]
        pos = row["Pos"]
        pick = row["Pick"]
        draft_year = row["Draft Year"]
        yr0 = row["obs_yr0"]
        yr1 = row["obs_yr1"]
        yr2_obs = row["obs_yr2"]
        yr3_obs = row["obs_yr3"]

        obs_av: list[float] = [yr0, yr1]
        if yr2_obs is not None:
            obs_av.append(yr2_obs)
        if yr3_obs is not None:
            obs_av.append(yr3_obs)

        proj = project_player_seasons(model, player, pos, obs_av)
        if proj is None:
            proj_yr2, proj_yr3 = 0.0, 0.0
        else:
            proj_yr2, proj_yr3 = proj

        # For total: use observed when available, else projected
        eff_yr2 = yr2_obs if yr2_obs is not None else proj_yr2
        eff_yr3 = yr3_obs if yr3_obs is not None else proj_yr3
        # Projected yr2/yr3 are 0 when that year was already observed
        out_proj_yr2 = 0.0 if yr2_obs is not None else proj_yr2
        out_proj_yr3 = 0.0 if yr3_obs is not None else proj_yr3

        total = yr0 + yr1 + eff_yr2 + eff_yr3
        is_projected = len(obs_av) < 4

        rows.append(
            {
                "Player": player,
                "Pos": pos,
                "Pick": pick,
                "Draft Year": draft_year,
                "obs_yr0": round(yr0, 1),
                "obs_yr1": round(yr1, 1),
                "obs_yr2": round(yr2_obs, 1) if yr2_obs is not None else None,
                "obs_yr3": round(yr3_obs, 1) if yr3_obs is not None else None,
                "proj_yr2": round(out_proj_yr2, 1),
                "proj_yr3": round(out_proj_yr3, 1),
                "total_4yr_av": round(total, 1),
                "is_projected": is_projected,
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

    result = players_4yr_df.join(eavar_df, on="Pick", how="left")
    result = result.with_columns(
        (pl.col("total_4yr_av") - pl.col("eavar")).round(1).alias("surplus_av")
    )
    return result
