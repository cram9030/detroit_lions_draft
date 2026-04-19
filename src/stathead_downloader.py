"""
stathead_downloader.py
======================
Bulk downloader for Stathead Football Player Season Finder.

Iterates over all configured draft-year / season-year combinations,
paginates through 200-row pages, parses the HTML table, and writes
one Parquet file per combination under data/raw/stathead/.

Usage:
    python src/stathead_downloader.py
    python src/stathead_downloader.py --config config/stathead_annual_av.json
    python src/stathead_downloader.py --cookies secrets/cookies.json
"""

import argparse
import io
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = (
    "https://www.sports-reference.com"
    "/stathead/football/player-season-finder.cgi"
)

log = logging.getLogger(__name__)


# =============================================================================
# LOGGING
# =============================================================================

def _setup_logging() -> None:
    log_path = PROJECT_ROOT / "stathead_downloader.log"
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
    p = argparse.ArgumentParser(description="Stathead bulk downloader")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "stathead_annual_av.json"),
        help="Path to query config JSON (default: config/stathead_annual_av.json)",
    )
    p.add_argument(
        "--cookies",
        default=str(PROJECT_ROOT / "secrets" / "cookies.json"),
        help="Path to Cookie-Editor export (default: secrets/cookies.json)",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Save as CSV instead of Parquet (useful for troubleshooting)",
    )
    return p.parse_args()


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    # Expand draft_year_start/end shorthand into individual (year, year) ranges
    if "draft_year_start" in cfg and "draft_year_end" in cfg:
        cfg["draft_year_ranges"] = [
            (y, y) for y in range(cfg["draft_year_start"], cfg["draft_year_end"] + 1)
        ]
    else:
        cfg["draft_year_ranges"] = [tuple(r) for r in cfg.get("draft_year_ranges", [])]

    return cfg


# =============================================================================
# COOKIES & SESSION
# =============================================================================

def load_cookies(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Cookie file not found: {p.resolve()}\n"
            "Export cookies with Cookie-Editor and save to secrets/cookies.json"
        )
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        cookies = {c["name"]: c["value"] for c in raw if "name" in c and "value" in c}
    elif isinstance(raw, dict):
        cookies = raw
    else:
        raise ValueError("Unrecognised cookie file format.")
    if not cookies:
        raise ValueError("No cookies found — double-check the export.")
    log.info("Loaded %d cookies from %s", len(cookies), path)
    return cookies


def build_session(cookies: dict) -> requests.Session:
    session = requests.Session()
    session.cookies.update(cookies)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.sports-reference.com/",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


# =============================================================================
# URL BUILDER
# =============================================================================

def build_url(
    cfg: dict,
    draft_year_min: int,
    draft_year_max: int,
    year_min: int,
    year_max: int,
    offset: int,
) -> str:
    params = dict(cfg["fixed_params"])
    params.update({
        "draft_year_min": str(draft_year_min),
        "draft_year_max": str(draft_year_max),
        "year_min":       str(year_min),
        "year_max":       str(year_max),
        "offset":         str(offset),
    })
    req = requests.Request("GET", BASE_URL, params=params)
    return req.prepare().url


# =============================================================================
# HTML PARSING
# =============================================================================

def parse_table(html: str) -> pd.DataFrame | None:
    soup = BeautifulSoup(html, "html.parser")

    table = (
        soup.find("table", id="stats")
        or soup.find("table", {"class": re.compile(r"stats_table", re.I)})
        or soup.find("table", {"class": re.compile(r"sortable", re.I)})
    )
    if table is None:
        for t in soup.find_all("table"):
            if len(t.find_all("tr")) > 2:
                table = t
                break

    if table is None:
        return None

    try:
        # io.StringIO prevents pandas from treating the HTML string as a file path
        df_pd = pd.read_html(io.StringIO(str(table)), header=0)[0]
    except Exception as exc:
        log.warning("Could not parse table: %s", exc)
        return None

    if "Rk" in df_pd.columns:
        mask = df_pd["Rk"].astype(str).str.strip()
        df_pd = df_pd[~mask.isin(["Rk", ""])]
        df_pd = df_pd.dropna(subset=["Rk"])

    df_pd = df_pd.reset_index(drop=True)
    if df_pd.empty:
        return None

    # Cast everything to str for consistent schema across pages
    return df_pd.astype(str)


def detect_total_results(html: str) -> int | None:
    for pattern in [
        r"of\s+([\d,]+)\s+results?",
        r"Showing\s+\d+\s+to\s+\d+\s+of\s+([\d,]+)",
        r"([\d,]+)\s+results?\s+found",
    ]:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def is_login_wall(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
    signals = ["sign in", "log in", "subscribe", "stathead membership"]
    return any(s in text for s in signals) and "rk" not in text


# =============================================================================
# HTTP
# =============================================================================

def fetch_page(session: requests.Session, url: str, cfg: dict) -> str | None:
    backoff = cfg["retry_backoff"]
    for attempt in range(1, cfg["max_retries"] + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                wait = backoff * attempt
                log.warning("Rate limited (429). Waiting %.0fs before retry %d…", wait, attempt)
                time.sleep(wait)
            else:
                log.warning("HTTP %d on attempt %d: %s", resp.status_code, attempt, url)
                time.sleep(backoff)
        except requests.RequestException as exc:
            log.warning("Request error on attempt %d: %s", attempt, exc)
            time.sleep(backoff)
        backoff *= 2
    log.error("Giving up after %d attempts: %s", cfg["max_retries"], url)
    return None


# =============================================================================
# OUTPUT PATHS
# =============================================================================

def make_output_path(
    output_dir: Path,
    draft_year_min: int,
    draft_year_max: int,
    season_year_min: int,
    season_year_max: int,
    use_csv: bool = False,
) -> Path:
    draft_str = (
        f"draft{draft_year_min}"
        if draft_year_min == draft_year_max
        else f"draft{draft_year_min}-{draft_year_max}"
    )
    season_str = (
        f"season{season_year_min}"
        if season_year_min == season_year_max
        else f"season{season_year_min}-{season_year_max}"
    )
    ext = ".csv" if use_csv else ".parquet"
    return output_dir / f"{draft_str}_{season_str}{ext}"


def combo_key(dy_min: int, dy_max: int, sy_min: int, sy_max: int) -> str:
    return f"draft{dy_min}-{dy_max}_season{sy_min}-{sy_max}"


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

def load_progress(output_dir: Path) -> set:
    p = output_dir / ".progress.json"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(output_dir: Path, completed: set) -> None:
    p = output_dir / ".progress.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(sorted(completed), f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    _setup_logging()
    args = parse_args()

    log.info("=" * 60)
    log.info("Stathead bulk downloader starting")
    log.info("Config  : %s", args.config)
    log.info("Cookies : %s", args.cookies)
    log.info("Format  : %s", "CSV (troubleshooting)" if args.csv else "Parquet")
    log.info("=" * 60)

    cfg = load_config(args.config)
    output_dir = PROJECT_ROOT / cfg.get("output_dir", "data/raw/stathead/annual_av")
    output_dir.mkdir(parents=True, exist_ok=True)

    cookies = load_cookies(args.cookies)
    session = build_session(cookies)
    completed = load_progress(output_dir)

    draft_ranges       = cfg["draft_year_ranges"]
    seasons_after      = cfg.get("seasons_after_draft")
    global_season_years = cfg.get("season_years", [])
    page_size          = cfg["page_size"]
    sleep_sec          = cfg["sleep_between_requests"]

    def season_years_for(dy_min: int) -> list[int]:
        if seasons_after is not None:
            return list(range(dy_min, dy_min + seasons_after))
        return global_season_years

    # Pre-compute total combinations for progress logging
    total_combinations = sum(len(season_years_for(dy_min)) for dy_min, _ in draft_ranges)
    log.info(
        "Draft ranges: %d | Mode: %s | Combinations: %d",
        len(draft_ranges),
        f"{seasons_after} seasons per draft year" if seasons_after else "fixed season list",
        total_combinations,
    )

    combo_num    = 0
    saved_files  = 0
    skipped      = 0
    empty_combos = 0

    for (dy_min, dy_max) in draft_ranges:
        for season_year in season_years_for(dy_min):
            combo_num += 1
            sy_min = sy_max = season_year
            key = combo_key(dy_min, dy_max, sy_min, sy_max)

            log.info(
                "[%d/%d] Draft %s–%s | Season %s–%s",
                combo_num, total_combinations, dy_min, dy_max, sy_min, sy_max,
            )

            if key in completed:
                log.info("  Skipping (already done): %s", key)
                skipped += 1
                continue

            out_path = make_output_path(output_dir, dy_min, dy_max, sy_min, sy_max, args.csv)
            pages: list[pd.DataFrame] = []
            total_results = None
            count_checked = False
            offset = 0

            while True:
                url = build_url(cfg, dy_min, dy_max, sy_min, sy_max, offset)
                log.info("  Fetching offset=%-5d  %s", offset, url)
                html = fetch_page(session, url, cfg)

                if html is None:
                    log.error("  No response; stopping pagination.")
                    break

                if is_login_wall(html):
                    log.error(
                        "  LOGIN WALL DETECTED — cookies may have expired.\n"
                        "  Re-export and replace secrets/cookies.json."
                    )
                    return

                if not count_checked:
                    count_checked = True
                    total_results = detect_total_results(html)
                    if total_results is not None:
                        n_pages = -(-total_results // page_size)
                        log.info("  Total results: %d (%d page(s))", total_results, n_pages)
                    else:
                        log.info("  No result count in page; will paginate until empty.")

                df = parse_table(html)

                if df is None or df.empty:
                    if offset == 0:
                        log.info("  No data for this combination.")
                        empty_combos += 1
                    else:
                        log.info("  Empty at offset %d — end of results.", offset)
                    break

                log.info("  Got %d rows at offset %d", len(df), offset)
                pages.append(df)

                offset += page_size
                if total_results is not None and offset >= total_results:
                    break

                time.sleep(sleep_sec)

            if pages:
                combined = pd.concat(pages, ignore_index=True)
                if args.csv:
                    combined.to_csv(out_path, index=False)
                else:
                    combined.to_parquet(out_path, index=False)
                log.info("  Saved %d rows → %s", len(combined), out_path.name)
                saved_files += 1
                completed.add(key)
                save_progress(output_dir, completed)

            time.sleep(sleep_sec)

    log.info("=" * 60)
    log.info("Done.")
    log.info("  Parquet files saved : %d", saved_files)
    log.info("  Skipped (cached)    : %d", skipped)
    log.info("  Empty combos        : %d", empty_combos)
    log.info("  Output folder       : %s", output_dir.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    run()
