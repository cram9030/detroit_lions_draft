"""General Manager assessment: executives lookup, overlap detection, aggregation."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from src.surplus_av import aggregate_drafts
from src.trade_value import aggregate_trade_value

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EXECUTIVES_DIR = _PROJECT_ROOT / "data" / "raw" / "pfr" / "executives"
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"
_DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "stathead" / "annual_av"
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

ALL_PFR_CODES: list[str] = sorted([
    "atl", "buf", "car", "chi", "cin", "cle", "clt", "crd",
    "dal", "den", "det", "gnb", "htx", "jax", "kan", "mia",
    "min", "nor", "nwe", "nyg", "nyj", "oti", "phi", "pit",
    "rai", "ram", "rav", "sdg", "sea", "sfo", "tam", "was",
])

# Map Stathead team codes → nflreadpy team_abbr values (for logo/color lookup)
STATHEAD_TO_NFLREADPY: dict[str, str] = {
    "ATL": "ATL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "CLE": "CLE", "IND": "IND", "ARI": "ARI",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GNB": "GB",
    "HOU": "HOU", "JAX": "JAX", "KAN": "KC",  "MIA": "MIA",
    "MIN": "MIN", "NOR": "NO",  "NWE": "NE",  "NYG": "NYG",
    "NYJ": "NYJ", "TEN": "TEN", "PHI": "PHI", "PIT": "PIT",
    "OAK": "LV",  "LVR": "LV",  "BAL": "BAL", "SDG": "LAC",
    "LAC": "LAC", "SEA": "SEA", "SFO": "SF",  "STL": "LA",
    "LAR": "LA",  "TAM": "TB",  "WAS": "WAS",
}

_PFR_TO_STATHEAD_STATIC: dict[str, str] = {
    "atl": "ATL", "buf": "BUF", "car": "CAR", "chi": "CHI",
    "cin": "CIN", "cle": "CLE", "clt": "IND", "crd": "ARI",
    "dal": "DAL", "den": "DEN", "det": "DET", "gnb": "GNB",
    "htx": "HOU", "jax": "JAX", "kan": "KAN", "mia": "MIA",
    "min": "MIN", "nor": "NOR", "nwe": "NWE", "nyg": "NYG",
    "nyj": "NYJ", "oti": "TEN", "phi": "PHI", "pit": "PIT",
    "rav": "BAL", "sea": "SEA", "sfo": "SFO", "tam": "TAM", "was": "WAS",
}

# Regex to extract M/D/YY or M/D/YYYY dates from Notes strings
_DATE_RE = re.compile(r"\b(\d{1,2})/\d{1,2}/\d{2,4}\b")


def pfr_to_stathead(pfr_code: str, year: int) -> str:
    """Translate a PFR franchise code to its Stathead team code for a given year."""
    if pfr_code == "rai":
        return "OAK" if year <= 2019 else "LVR"
    if pfr_code == "ram":
        return "STL" if year <= 2015 else "LAR"
    if pfr_code == "sdg":
        return "SDG" if year <= 2016 else "LAC"
    return _PFR_TO_STATHEAD_STATIC[pfr_code]


def load_executives(pfr_code: str, executives_dir: Path | None = None) -> pl.DataFrame:
    """Load the executives parquet for the given PFR team code.

    Args:
        pfr_code: Lowercase PFR team code (e.g. ``"det"``).
        executives_dir: Directory containing ``{pfr_code}_executives.parquet`` files.

    Returns:
        DataFrame with columns ``[Person, Teams, From, To, Titles, Notes]``.

    Raises:
        FileNotFoundError: If the parquet for the team does not exist.
    """
    executives_dir = executives_dir or _DEFAULT_EXECUTIVES_DIR
    path = Path(executives_dir) / f"{pfr_code}_executives.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Executives parquet not found: {path}\n"
            "Run: python src/pfr_downloader.py --config config/pfr_executives.json"
        )
    return pl.read_parquet(path)


def _is_qualifying_gm(title: str) -> bool:
    """Return True if the title represents a true (non-interim, non-assistant) GM."""
    if "General Manager" not in title:
        return False
    if "Interim" in title:
        return False
    if "Assistant" in title:
        return False
    return True


def _parse_month_from_notes(notes: str | None) -> int | None:
    """Extract the first month number from a Notes string like 'Fired on 11/5/15'."""
    if not notes:
        return None
    m = _DATE_RE.search(notes)
    if m:
        return int(m.group(1))
    return None


def _was_gm_in_april(row: dict, year: int) -> bool:
    """Decide if an executive row covers April of the given year.

    Uses the Notes field date when available; falls back to tenure boundaries.
    """
    from_yr = int(row["From"])
    to_yr = int(row["To"])
    notes = row.get("Notes") or ""
    month = _parse_month_from_notes(notes)

    if month is not None:
        # GM fired this year: was April GM only if fired in May or later
        if any(kw in notes for kw in ("Fired", "fired", "Dismissed", "dismissed")):
            if to_yr == year:
                return month >= 5
            return True  # fired in a different year, still here in April
        # GM hired this year: was April GM only if started by end of April
        if any(kw in notes for kw in ("Named", "named", "Hired", "hired", "Appointed", "appointed")):
            if from_yr == year:
                return month <= 4
            return True

    # No useful date in Notes: use tenure boundaries
    # If their tenure ENDS this year, they likely oversaw the April draft (fired post-season)
    # If their tenure STARTS this year (and we can't parse month), assume they came after the draft
    if from_yr < year:
        return True
    if to_yr > year:
        return True
    # Both from_yr == year == to_yr → single-year stint; assume full year
    return True


def fetch_gm(
    pfr_code: str,
    year: int,
    executives_dir: Path | None = None,
) -> dict | None:
    """Return GM info for a franchise and year, or None if no qualifying GM exists.

    Excludes Interim and Assistant General Managers.  When two non-interim GMs
    both cover the same year (transition year), the April criteria selects the
    one who held the role during April.

    Args:
        pfr_code: Lowercase PFR team code (e.g. ``"det"``).
        year: Draft year to look up.
        executives_dir: Directory containing executives parquets.

    Returns:
        ``{"person": str, "from_year": int, "to_year": int}`` or ``None``.
    """
    df = load_executives(pfr_code, executives_dir)
    candidates = [
        row for row in df.iter_rows(named=True)
        if _is_qualifying_gm(row["Titles"])
        and int(row["From"]) <= year <= int(row["To"])
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        r = candidates[0]
        return {"person": r["Person"], "from_year": int(r["From"]), "to_year": int(r["To"])}

    # Multiple qualifying GMs in the same year — apply April criteria
    april_candidates = [r for r in candidates if _was_gm_in_april(r, year)]

    if len(april_candidates) == 1:
        r = april_candidates[0]
        return {"person": r["Person"], "from_year": int(r["From"]), "to_year": int(r["To"])}

    if len(april_candidates) == 0:
        return None

    # Still tied — prefer the incumbent (whose tenure started before this year)
    incumbents = [r for r in april_candidates if int(r["From"]) < year]
    chosen = incumbents[0] if incumbents else april_candidates[0]
    return {"person": chosen["Person"], "from_year": int(chosen["From"]), "to_year": int(chosen["To"])}


def get_gm_year_range(
    pfr_code: str,
    year: int,
    executives_dir: Path | None = None,
) -> tuple[str, int, int] | None:
    """Return ``(gm_name, from_year, to_year)`` for the GM of pfr_code in year.

    Returns ``None`` if no qualifying GM is found.
    """
    result = fetch_gm(pfr_code, year, executives_dir)
    if result is None:
        return None
    return result["person"], result["from_year"], result["to_year"]


def find_overlapping_gms(
    overlap_from: int,
    overlap_to: int,
    executives_dir: Path | None = None,
    years_cap: int | None = None,
) -> pl.DataFrame:
    """Find all non-interim GMs from every franchise whose tenure overlaps [overlap_from, overlap_to].

    Args:
        overlap_from: Start year of the target GM's tenure window.
        overlap_to: End year of the target GM's tenure window.
        executives_dir: Directory containing executives parquets.
        years_cap: Optional cap on the effective overlap_to (e.g. last year with data).

    Returns:
        DataFrame with columns
        ``[pfr_code, stathead_code, gm_name, gm_from, gm_to, overlap_from, overlap_to]``.
    """
    executives_dir = Path(executives_dir) if executives_dir else _DEFAULT_EXECUTIVES_DIR
    effective_to = min(overlap_to, years_cap) if years_cap else overlap_to

    rows: list[dict] = []
    for pfr_code in ALL_PFR_CODES:
        path = executives_dir / f"{pfr_code}_executives.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)

        for row in df.iter_rows(named=True):
            if not _is_qualifying_gm(row["Titles"]):
                continue
            gm_from = int(row["From"])
            gm_to = int(row["To"])
            # Overlap condition: gm_from <= effective_to AND gm_to >= overlap_from
            if gm_from > effective_to or gm_to < overlap_from:
                continue
            # Compute intersection years
            eff_overlap_from = max(gm_from, overlap_from)
            eff_overlap_to = min(gm_to, effective_to)
            # Use the most representative year for stathead code (use overlap midpoint)
            mid_year = (eff_overlap_from + eff_overlap_to) // 2
            stathead_code = pfr_to_stathead(pfr_code, mid_year)
            rows.append({
                "pfr_code": pfr_code,
                "stathead_code": stathead_code,
                "gm_name": row["Person"],
                "gm_from": gm_from,
                "gm_to": gm_to,
                "overlap_from": eff_overlap_from,
                "overlap_to": eff_overlap_to,
            })

    if not rows:
        return pl.DataFrame(schema={
            "pfr_code": pl.String,
            "stathead_code": pl.String,
            "gm_name": pl.String,
            "gm_from": pl.Int64,
            "gm_to": pl.Int64,
            "overlap_from": pl.Int64,
            "overlap_to": pl.Int64,
        })

    return pl.DataFrame(rows)


def get_surplus_value(
    gm_records_df: pl.DataFrame,
    model_name: str = "linear",
    parametric_curve: str = "quadratic",
    models_dir: Path | None = None,
    raw_dir: Path | None = None,
    position_overrides: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Add surplus AV aggregation columns to gm_records_df.

    Calls ``aggregate_drafts`` for each row's (stathead_code, overlap_from..overlap_to).

    Args:
        gm_records_df: Output of :func:`find_overlapping_gms`.
        model_name: Career AV model for partially-observed classes.
        parametric_curve: Curve variant when model_name is ``"parametric"``.
        models_dir: Directory containing trained model artifacts.
        raw_dir: Directory containing annual AV parquets.
        position_overrides: Per-player position overrides.

    Returns:
        Input DataFrame with ``[total_surplus_av, n_draft_years, avg_surplus_per_year]`` appended.
    """
    models_dir = models_dir or _DEFAULT_MODELS_DIR
    raw_dir = raw_dir or _DEFAULT_RAW_DIR

    total_surplus_list: list[float] = []
    n_draft_years_list: list[int] = []
    avg_surplus_list: list[float] = []

    for row in gm_records_df.iter_rows(named=True):
        years = list(range(int(row["overlap_from"]), int(row["overlap_to"]) + 1))
        agg = aggregate_drafts(
            team=row["stathead_code"],
            years=years,
            model_name=model_name,
            parametric_curve=parametric_curve,
            models_dir=models_dir,
            raw_dir=raw_dir,
            position_overrides=position_overrides,
        )
        total_surplus_list.append(agg["total_surplus_av"])
        n_draft_years_list.append(agg["n_years"])
        avg_surplus_list.append(agg["avg_surplus_per_year"])

    return gm_records_df.with_columns([
        pl.Series("total_surplus_av", total_surplus_list, dtype=pl.Float64),
        pl.Series("n_draft_years", n_draft_years_list, dtype=pl.Int64),
        pl.Series("avg_surplus_per_year", avg_surplus_list, dtype=pl.Float64),
    ])


def get_trade_value(
    gm_records_df: pl.DataFrame,
    data_dir: Path | None = None,
    chart_name: str = "fitz_spiel",
) -> pl.DataFrame:
    """Add trade value aggregation columns to gm_records_df.

    Calls ``aggregate_trade_value`` for each row's (stathead_code, overlap_from..overlap_to).

    Args:
        gm_records_df: Output of :func:`find_overlapping_gms` (or after get_surplus_value).
        data_dir: Directory containing trade chart CSVs.
        chart_name: Column prefix in analyze_draft_trades output (default: ``"fitz_spiel"``).

    Returns:
        Input DataFrame with ``[total_trade_value, n_trade_years, avg_trade_per_year]`` appended.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR

    total_trade_list: list[float] = []
    n_trade_years_list: list[int] = []
    avg_trade_list: list[float] = []

    for row in gm_records_df.iter_rows(named=True):
        years = list(range(int(row["overlap_from"]), int(row["overlap_to"]) + 1))
        agg = aggregate_trade_value(
            team=row["stathead_code"],
            years=years,
            data_dir=data_dir,
            chart_name=chart_name,
        )
        total_trade_list.append(agg["total_trade_value"])
        n_trade_years_list.append(agg["n_years"])
        avg_trade_list.append(agg["avg_trade_per_year"])

    return gm_records_df.with_columns([
        pl.Series("total_trade_value", total_trade_list, dtype=pl.Float64),
        pl.Series("n_trade_years", n_trade_years_list, dtype=pl.Int64),
        pl.Series("avg_trade_per_year", avg_trade_list, dtype=pl.Float64),
    ])
