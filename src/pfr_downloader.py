"""
pfr_downloader.py
=================
Config-driven downloader for Pro-Football-Reference (PFR) HTML tables.

Supports URL templates with {variable} substitution iterated over a team list,
year range, or any custom list defined in a JSON config file. Multiple tables
can be extracted per page by specifying table IDs. PFR's comment-wrapped tables
are handled automatically.

PFR is part of the sports-reference.com subscription family. Export your
browser session cookies with Cookie-Editor and save to secrets/cookies.json
(the same file used by stathead_downloader.py).

Usage:
    python src/pfr_downloader.py --config config/pfr_executives.json
    python src/pfr_downloader.py --config config/pfr_standings.json --csv
    python src/pfr_downloader.py --config config/pfr_executives.json --cookies secrets/cookies.json

Adding a new PFR data source:
    1. Create a new JSON config in config/ (copy an existing one as a template).
    2. Set url_template, iterate, and tables for the new page.
    3. Run the downloader — no code changes required.
"""

import argparse
import io
import json
import logging
import re
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work whether this
# file is run directly (`python src/pfr_downloader.py`) or imported by pytest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from bs4 import BeautifulSoup

from src.scraper_utils import build_session, fetch_page, load_cookies, load_progress, save_progress

PROJECT_ROOT = _PROJECT_ROOT

log = logging.getLogger(__name__)


# =============================================================================
# LOGGING
# =============================================================================

def _setup_logging() -> None:
    log_path = PROJECT_ROOT / "pfr_downloader.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PFR config-driven table downloader")
    p.add_argument(
        "--config",
        required=True,
        help="Path to query config JSON (e.g. config/pfr_executives.json)",
    )
    p.add_argument(
        "--cookies",
        default=str(PROJECT_ROOT / "secrets" / "cookies.json"),
        help="Path to Cookie-Editor export (default: secrets/cookies.json)",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Save as CSV instead of Parquet",
    )
    return p.parse_args()


# =============================================================================
# CONFIG
# =============================================================================

def load_pfr_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_iterations"] = expand_iterations(cfg["iterate"])
    return cfg


# =============================================================================
# URL BUILDING
# =============================================================================

def build_pfr_url(template: str, variables: dict) -> str:
    """Substitute {variable} placeholders in a URL template."""
    return template.format(**variables)


# =============================================================================
# ITERATION EXPANSION
# =============================================================================

def expand_iterations(iterate_cfg: dict) -> list[dict]:
    """
    Expand an iteration config to a flat list of variable dicts.

    Supported formats (one key per config):
      {"team": ["det", "chi", "gb"]}          -> explicit list
      {"year": {"start": 2020, "end": 2022}}  -> inclusive integer range
    """
    if len(iterate_cfg) != 1:
        raise ValueError(
            f"iterate must have exactly one key, got {list(iterate_cfg.keys())!r}. "
            "Use separate config files for independent dimensions."
        )
    key, spec = next(iter(iterate_cfg.items()))
    if isinstance(spec, list):
        return [{key: v} for v in spec]
    if isinstance(spec, dict) and "start" in spec and "end" in spec:
        return [{key: y} for y in range(spec["start"], spec["end"] + 1)]
    raise ValueError(
        f"Invalid iteration spec for key {key!r}: {spec!r}. "
        "Expected a list or {\"start\": N, \"end\": N}."
    )


# =============================================================================
# HTML PARSING
# =============================================================================

def unwrap_pfr_comments(html: str) -> str:
    """
    Remove HTML comment wrappers around tables.

    PFR wraps certain tables in <!-- ... --> to defer rendering. This strips
    the comment delimiters so BeautifulSoup/pandas can see the tables.
    """
    return re.sub(r'<!--\s*(<table[\s\S]*?</table>)\s*-->', r'\1', html)


def parse_pfr_table(
    html: str,
    table_id: str | None = None,
    table_index: int | None = None,
) -> pd.DataFrame | None:
    """
    Extract a single table from PFR HTML.

    Comment-wrapped tables are automatically unwrapped before parsing.
    Selection priority: table_id > table_index > first table found.
    Returns None if the requested table is absent or empty.
    """
    html = unwrap_pfr_comments(html)
    soup = BeautifulSoup(html, "html.parser")

    if table_id is not None:
        table = soup.find("table", id=table_id)
    elif table_index is not None:
        tables = soup.find_all("table")
        if table_index >= len(tables):
            return None
        table = tables[table_index]
    else:
        table = soup.find("table")

    if table is None:
        return None

    try:
        df = pd.read_html(io.StringIO(str(table)), header=0)[0]
    except Exception as exc:
        log.warning("Could not parse table: %s", exc)
        return None

    df = df.reset_index(drop=True)
    if df.empty:
        return None

    return df.astype(str)


# =============================================================================
# ACCESS WALL DETECTION
# =============================================================================

def is_pfr_blocked(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
    signals = [
        "rate limited", "too many requests", "please slow down", "access denied",
        "sign in", "log in", "subscribe",
    ]
    return any(s in text for s in signals) and "<table" not in html.lower()


# =============================================================================
# PER-PLAYER CAREER STATS
# =============================================================================
#
# Unlike the config-driven bulk downloader above (one URL template iterated
# over a static team/year list), this fetches a single player's PFR page —
# the URL is derived from their pfr_id, and the relevant stat table's id
# depends on their position rather than being fixed in a config file.
#
# Every one of PFR's per-position season tables (passing, rushing_and_receiving,
# receiving_and_rushing, defense, kicking, punting, returns) carries `games`/
# `games_started` columns regardless of position — including offensive line,
# which gets no rushing/receiving/defensive stats but still shows up on
# `receiving_and_rushing` with real G/GS. That makes G/GS reliable straight
# from this table for every position, unlike nflreadpy's advanced-stats
# endpoint, which only covers 2018+ and never includes offensive linemen at all.

_DEFAULT_PLAYER_STATS_DIR = PROJECT_ROOT / "data/raw/pfr/player_career_stats"

# Ordered by which table id a position's players actually appear under first;
# a player can appear on more than one (e.g. a QB with a garbage-time tackle
# shows up on "defense" too), so order determines which one wins.
_CAREER_TABLE_PRIORITY: dict[str, list[str]] = {
    "QB": ["passing", "rushing_and_receiving", "receiving_and_rushing"],
    "RB": ["rushing_and_receiving", "receiving_and_rushing"],
    "WR": ["receiving_and_rushing", "rushing_and_receiving"],
    "TE": ["receiving_and_rushing", "rushing_and_receiving"],
    "DE": ["defense"], "DT": ["defense"], "LB": ["defense"], "CB": ["defense"], "S": ["defense"],
    "OT": ["receiving_and_rushing", "rushing_and_receiving", "defense", "games_played"],
    "OG": ["receiving_and_rushing", "rushing_and_receiving", "defense", "games_played"],
    "OC": ["receiving_and_rushing", "rushing_and_receiving", "defense", "games_played"],
    "K": ["kicking"], "P": ["punting"],
}
# Fallback search order when a position is unmapped or its preferred table is absent.
# "games_played" is PFR's bare-bones fallback for a player with literally no
# recorded stat category (a backup lineman who never touched the ball on
# offense/defense/special teams) — last resort, since it has no positional
# stats at all, only games/started.
_ALL_CAREER_TABLE_IDS: list[str] = [
    "passing", "rushing_and_receiving", "receiving_and_rushing", "defense",
    "kicking", "punting", "returns", "games_played",
]

# "games_played" uses different data-stat names ("g"/"gs"/"team") than every
# other career table ("games"/"games_started"/"team_name_abbr") — normalize
# so downstream code can rely on one consistent set of column names.
_GAMES_PLAYED_COLUMN_ALIASES: dict[str, str] = {"g": "games", "gs": "games_started", "team": "team_name_abbr"}


def player_page_url(pfr_id: str) -> str:
    """Build a PFR player page URL from their pfr_id, e.g. ``DeckTa00`` -> .../players/D/DeckTa00.htm."""
    return f"https://www.pro-football-reference.com/players/{pfr_id[0].upper()}/{pfr_id}.htm"


def _parse_table_by_data_stat(html: str, table_id: str) -> pd.DataFrame | None:
    """Extract a PFR table's ``tbody`` rows keyed by each cell's ``data-stat`` attribute.

    PFR's season-stat tables use a two-row header (a grouping row like
    "Receiving"/"Rushing" over the real column labels) — ``pandas.read_html``
    (used by :func:`parse_pfr_table`) takes the first row as the header and
    mangles the real column names into ``Unnamed: N`` / group-name-suffixed
    columns. Each ``<td>``/``<th>``'s ``data-stat`` attribute is a stable
    machine name (``games``, ``games_started``, ``pass_yds``, ...) independent
    of the visible header layout, so read from that instead.
    """
    html = unwrap_pfr_comments(html)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=table_id)
    if table is None or table.find("tbody") is None:
        return None

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        classes = tr.get("class") or []
        if "thead" in classes:  # mid-table repeated header row, not a data row
            continue
        row = {
            c.get("data-stat"): c.get_text(strip=True)
            for c in tr.find_all(["th", "td"])
            if c.get("data-stat")
        }
        if row:
            rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def parse_player_career_stats(html: str, position: str | None = None) -> pd.DataFrame | None:
    """Extract the season-by-season games/positional-stat table from a PFR player page.

    Tries table ids in the order appropriate for ``position`` first (see
    ``_CAREER_TABLE_PRIORITY``), then falls back to every known table id, so
    an unmapped or unexpected position still finds whatever table exists.
    Returns ``None`` if the player has no recognized career-stats table.
    """
    order = list(_CAREER_TABLE_PRIORITY.get(position or "", []))
    order += [t for t in _ALL_CAREER_TABLE_IDS if t not in order]

    for table_id in order:
        df = _parse_table_by_data_stat(html, table_id)
        if df is not None:
            if table_id == "games_played":
                df = df.rename(columns=_GAMES_PLAYED_COLUMN_ALIASES)
            df["_pfr_table"] = table_id
            return df
    return None


def fetch_player_career_stats(
    session,
    pfr_id: str,
    position: str | None = None,
    cache_dir: Path | None = None,
    sleep_sec: float = 3.0,
    max_retries: int = 3,
    retry_backoff: float = 10.0,
) -> pd.DataFrame | None:
    """Fetch one player's season-by-season PFR career-stats table, caching to Parquet.

    A cache hit costs no network request at all. On a miss, fetches the
    player's page, extracts the table, and writes the cache before returning
    — so repeated calls (e.g. the same historical comp appearing across
    multiple pace-comparison plots) only ever hit the network once per player.

    Returns:
        DataFrame with one row per season plus a ``_pfr_table`` column
        naming which PFR table it came from, or ``None`` if the page
        couldn't be fetched or had no recognized stats table.
    """
    cache_dir = cache_dir or _DEFAULT_PLAYER_STATS_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pfr_id}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = player_page_url(pfr_id)
    html = fetch_page(session, url, max_retries=max_retries, retry_backoff=retry_backoff)
    if html is None:
        log.warning("No response fetching player page for %s", pfr_id)
        return None
    if is_pfr_blocked(html):
        log.error(
            "BLOCKED RESPONSE fetching %s — cookies may have expired or you are rate-limited.", pfr_id
        )
        return None

    df = parse_player_career_stats(html, position=position)
    time.sleep(sleep_sec)
    if df is None:
        log.warning("No recognized career-stats table found for %s", pfr_id)
        return None

    df.to_parquet(cache_path, index=False)
    return df


# =============================================================================
# OUTPUT PATHS
# =============================================================================

def make_pfr_output_path(
    output_dir: Path,
    variables: dict,
    table_suffix: str | None = None,
    use_csv: bool = False,
) -> Path:
    """
    Build an output file path from iteration variables and an optional table suffix.

    Examples:
        {"team": "det"}, "executives"  ->  output_dir/det_executives.csv
        {"year": 2025},  "afc"         ->  output_dir/2025_afc.csv
        {"team": "det"}, None          ->  output_dir/det.csv
    """
    iter_key = "_".join(str(v) for v in variables.values())
    stem = f"{iter_key}_{table_suffix}" if table_suffix else iter_key
    ext = ".csv" if use_csv else ".parquet"
    return output_dir / f"{stem}{ext}"


def _progress_key(variables: dict, table_suffix: str | None) -> str:
    iter_key = "_".join(str(v) for v in variables.values())
    return f"{iter_key}_{table_suffix}" if table_suffix else iter_key


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    _setup_logging()
    args = parse_args()

    log.info("=" * 60)
    log.info("PFR downloader starting")
    log.info("Config  : %s", args.config)
    log.info("Cookies : %s", args.cookies)
    log.info("Format  : %s", "CSV" if args.csv else "Parquet")
    log.info("=" * 60)

    cfg = load_pfr_config(args.config)
    output_dir = PROJECT_ROOT / cfg.get("output_dir", "data/raw/pfr")
    output_dir.mkdir(parents=True, exist_ok=True)

    cookies = load_cookies(args.cookies)
    session = build_session(
        cookies=cookies,
        extra_headers={"Referer": "https://www.pro-football-reference.com/"},
        impersonate="chrome124",
    )
    completed = load_progress(output_dir)

    iterations = cfg["_iterations"]
    tables_cfg = cfg.get("tables", [])
    sleep_sec = cfg["sleep_between_requests"]
    max_retries = cfg["max_retries"]
    retry_backoff = cfg["retry_backoff"]

    total = len(iterations)
    saved = skipped = errors = 0

    for i, variables in enumerate(iterations, 1):
        url = build_pfr_url(cfg["url_template"], variables)
        iter_label = ", ".join(f"{k}={v}" for k, v in variables.items())
        log.info("[%d/%d] %s  →  %s", i, total, iter_label, url)

        # Check if all tables for this iteration are already done
        pending_tables = [
            t for t in tables_cfg
            if _progress_key(variables, t.get("output_suffix")) not in completed
        ]
        if not pending_tables:
            log.info("  Skipping (all tables cached): %s", iter_label)
            skipped += len(tables_cfg) or 1
            continue

        html = fetch_page(session, url, max_retries=max_retries, retry_backoff=retry_backoff)
        if html is None:
            log.error("  No response; skipping %s.", iter_label)
            errors += 1
            time.sleep(sleep_sec)
            continue

        if is_pfr_blocked(html):
            log.error(
                "  BLOCKED RESPONSE DETECTED — cookies may have expired or you are rate-limited.\n"
                "  Re-export cookies with Cookie-Editor and replace secrets/cookies.json,\n"
                "  or wait a few minutes before retrying."
            )
            return

        for table_cfg in tables_cfg:
            table_id = table_cfg.get("id")
            table_index = table_cfg.get("index")
            suffix = table_cfg.get("output_suffix")
            key = _progress_key(variables, suffix)

            if key in completed:
                log.info("  Skipping table %r (cached)", suffix or table_id)
                skipped += 1
                continue

            df = parse_pfr_table(html, table_id=table_id, table_index=table_index)
            if df is None or df.empty:
                log.warning("  Table %r not found or empty for %s.", table_id or table_index, iter_label)
                continue

            out_path = make_pfr_output_path(output_dir, variables, suffix, use_csv=args.csv)
            if args.csv:
                df.to_csv(out_path, index=False)
            else:
                df.to_parquet(out_path, index=False)

            log.info("  Saved %d rows → %s", len(df), out_path.name)
            saved += 1
            completed.add(key)
            save_progress(output_dir, completed)

        time.sleep(sleep_sec)

    log.info("=" * 60)
    log.info("Done.")
    log.info("  Files saved     : %d", saved)
    log.info("  Skipped (cache) : %d", skipped)
    log.info("  Errors          : %d", errors)
    log.info("  Output folder   : %s", output_dir.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    run()
