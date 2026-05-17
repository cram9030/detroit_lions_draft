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
