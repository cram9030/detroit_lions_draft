"""
scraper_utils.py
================
Shared HTTP, retry, and progress-tracking utilities used by all downloader
modules (stathead_downloader, pfr_downloader, etc.).
"""

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def build_session(
    cookies: dict | None = None,
    extra_headers: dict | None = None,
) -> requests.Session:
    session = requests.Session()
    if cookies:
        session.cookies.update(cookies)
    headers = dict(_DEFAULT_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    session.headers.update(headers)
    return session


def fetch_page(
    session: requests.Session,
    url: str,
    max_retries: int = 3,
    retry_backoff: float = 10.0,
) -> str | None:
    backoff = retry_backoff
    for attempt in range(1, max_retries + 1):
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
    log.error("Giving up after %d attempts: %s", max_retries, url)
    return None


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
