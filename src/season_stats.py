"""Season counting-stat lookups (games played/started, positional totals) via Pro-Football-Reference.

Enriches player-comparison plots (e.g. rookie-contract pace comps) with real
box-score context for a player's specific season — games played/started plus
the counting stats appropriate to their position.

Pulled directly from each player's PFR page (:func:`src.pfr_downloader.fetch_player_career_stats`)
rather than nflreadpy: nflreadpy's advanced-stats endpoint (the only source it
has for games-started) only covers 2018+ and has no offensive-line rows at
all, which silently produced wrong/missing games-played and games-started
values for exactly those cases. PFR's own per-position season table carries
``games``/``games_started`` for every position and every season, since that's
the same page a human would check by hand.

Player identity (``pfr_player_id``) is resolved through
:func:`src.data_ingest.load_nflreadr_draft_picks` — that crosswalk itself is
fine; only the season-stat source was wrong.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

from src.data_ingest import load_nflreadr_draft_picks
from src.pfr_downloader import fetch_player_career_stats
from src.scraper_utils import build_session, load_cookies

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_COOKIES_PATH = str(_PROJECT_ROOT / "secrets" / "cookies.json")


def _fmt_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _int_or_none(value: str | None) -> int | None:
    """Like ``_fmt_int``, but a genuinely missing value stays ``None`` instead of collapsing to 0.

    Used for G/GS specifically — a real "started 0 games" is meaningfully
    different from "no data for this season" and must not be conflated.
    """
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _fmt_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _format_stat_line(pfr_table: str, row: dict) -> str:
    """Build a short counting-stat summary appropriate to whichever PFR table the season came from.

    Returns ``""`` for tables with no meaningful counting stat to show
    (offensive line rows, which only ever land on ``receiving_and_rushing``
    or ``rushing_and_receiving`` with blank stat columns) rather than
    fabricating one.
    """
    if pfr_table == "passing":
        return (
            f"{_fmt_int(row.get('pass_cmp'))}/{_fmt_int(row.get('pass_att'))} Cmp/Att, "
            f"{_fmt_int(row.get('pass_yds'))} Pass Yds, {_fmt_int(row.get('pass_td'))} Pass TD, "
            f"{_fmt_int(row.get('pass_int'))} INT"
        )
    if pfr_table in ("rushing_and_receiving", "receiving_and_rushing"):
        rush_att = _fmt_int(row.get("rush_att"))
        rec = _fmt_int(row.get("rec"))
        if rush_att == 0 and rec == 0:
            return ""  # offensive line — table has G/GS but no real stat line
        parts = []
        if rush_att:
            parts.append(f"{rush_att} Car, {_fmt_int(row.get('rush_yds'))} Rush Yds, {_fmt_int(row.get('rush_td'))} Rush TD")
        if rec:
            parts.append(f"{rec}/{_fmt_int(row.get('targets'))} Rec/Tgt, {_fmt_int(row.get('rec_yds'))} Rec Yds, {_fmt_int(row.get('rec_td'))} Rec TD")
        return ", ".join(parts)
    if pfr_table == "defense":
        tkl = _fmt_int(row.get("tackles_combined"))
        sacks = _fmt_float(row.get("sacks"))
        interceptions = _fmt_int(row.get("def_int"))
        pd_ = _fmt_int(row.get("pass_defended"))
        tfl = _fmt_int(row.get("tackles_loss"))
        if tkl == 0 and sacks == 0 and interceptions == 0 and pd_ == 0:
            return ""  # offensive lineman whose only PFR table happens to be "defense" (e.g. a fumble recovery)
        if sacks or tfl:
            return f"{tkl} Tkl, {sacks:.1f} Sk, {tfl} TFL"
        return f"{tkl} Tkl, {interceptions} INT, {pd_} PD"
    return ""  # e.g. "games_played" — G/GS only, no positional stat table exists


def load_season_counting_stats(
    players: pl.DataFrame,
    cookies_path: str | None = _DEFAULT_COOKIES_PATH,
) -> pl.DataFrame:
    """Attach games played/started and a positional stat line to each player-season.

    Args:
        players: One row per player-season needed, with columns
            ``[Player, Pos, Pick, Draft Year, season]`` — ``season`` is the
            actual NFL season (``Draft Year + years_from_draft``). Duplicate
            rows are fine.
        cookies_path: Path to a Cookie-Editor export used to fetch
            pro-football-reference.com (same file used by
            ``stathead_downloader.py`` / ``pfr_downloader.py``). Each unique
            player is fetched (and cached to
            ``data/raw/pfr/player_career_stats/{pfr_id}.parquet``) at most
            once ever — a cache hit costs no network request.

    Returns:
        ``players`` plus ``[G, GS, stat_line]``. ``G``/``GS``/``stat_line``
        are ``None``/``None``/``""`` wherever PFR has no season row for that
        player-season (e.g. a season they didn't play, or the player has no
        pfr_player_id / no fetchable page). ``stat_line`` is additionally
        ``""`` for offensive line, which has no meaningful counting stat.
    """
    if players.is_empty():
        return players.with_columns(
            pl.lit(None).cast(pl.Int64).alias("G"),
            pl.lit(None).cast(pl.Int64).alias("GS"),
            pl.lit("").alias("stat_line"),
        )

    draft_years = sorted(players["Draft Year"].unique().to_list())
    ids = (
        load_nflreadr_draft_picks(draft_years)
        .select(["Player", "Draft Year", "Pick", "pfr_player_id"])
        .unique(subset=["Player", "Draft Year", "Pick"])
    )
    joined = players.join(ids, on=["Player", "Draft Year", "Pick"], how="left")

    needed = (
        joined.select(["Player", "Pos", "pfr_player_id"])
        .drop_nulls(subset=["pfr_player_id"])
        .unique(subset=["pfr_player_id"])
    )

    try:
        cookies = load_cookies(cookies_path) if cookies_path else None
    except FileNotFoundError:
        log.warning(
            "No cookies file at %s — PFR player pages require an authenticated session "
            "(same cookies used by stathead_downloader.py). Returning games/stats as unavailable.",
            cookies_path,
        )
        return players.with_columns(
            pl.lit(None).cast(pl.Int64).alias("G"),
            pl.lit(None).cast(pl.Int64).alias("GS"),
            pl.lit("").alias("stat_line"),
        )

    session = build_session(
        cookies=cookies,
        extra_headers={"Referer": "https://www.pro-football-reference.com/"},
        impersonate="chrome124",
    )

    season_rows: list[dict] = []
    for row in needed.iter_rows(named=True):
        career = fetch_player_career_stats(session, row["pfr_player_id"], position=row["Pos"])
        if career is None:
            continue

        # A season with a mid-year trade gets one row per team plus a combined
        # "2TM"/"3TM" summary row, all sharing the same year_id — keep only the
        # combined row (falling back to the lone row otherwise) so a traded
        # player's season doesn't fan out into duplicate join matches.
        by_season: dict[str, dict] = {}
        for season_row in career.to_dict(orient="records"):
            year_id = season_row.get("year_id")
            if not year_id or not str(year_id).isdigit():
                continue  # career-total row or similar non-season row
            team = str(season_row.get("team_name_abbr") or "")
            is_multi_team_summary = bool(re.fullmatch(r"\dTM", team))
            existing = by_season.get(year_id)
            if existing is None or is_multi_team_summary:
                by_season[year_id] = season_row

        for year_id, season_row in by_season.items():
            season_rows.append(
                {
                    "pfr_player_id": row["pfr_player_id"],
                    "season": int(year_id),
                    "G": _int_or_none(season_row.get("games")),
                    "GS": _int_or_none(season_row.get("games_started")),
                    "stat_line": _format_stat_line(season_row.get("_pfr_table"), season_row),
                }
            )

    stats = (
        pl.DataFrame(
            season_rows,
            schema={"pfr_player_id": pl.Utf8, "season": pl.Int64, "G": pl.Int64, "GS": pl.Int64, "stat_line": pl.Utf8},
        )
        if season_rows
        else pl.DataFrame(schema={"pfr_player_id": pl.Utf8, "season": pl.Int64, "G": pl.Int64, "GS": pl.Int64, "stat_line": pl.Utf8})
    )

    result = joined.join(stats, on=["pfr_player_id", "season"], how="left")
    result = result.with_columns(pl.col("stat_line").fill_null(""))
    return result.select([*players.columns, "G", "GS", "stat_line"])
