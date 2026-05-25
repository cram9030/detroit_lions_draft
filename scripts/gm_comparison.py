"""GM Draft Performance Quadrant — compare GMs across a shared time window.

Given a target team and year, finds that team's General Manager, determines their
PFR-recorded tenure, locates every other franchise's GM whose tenure overlaps,
aggregates draft surplus AV and trade value for each GM, and produces a quadrant
scatter chart with team logos.

Usage
-----
    python scripts/gm_comparison.py                          # DET 2024 defaults
    python scripts/gm_comparison.py --team DET --year 2024
    python scripts/gm_comparison.py --team KAN --year 2022 --model knn
    python scripts/gm_comparison.py --year 2022 --chart rich_hill

Notes
-----
- Only draft years with ≥ 2 completed seasons contribute to surplus AV.
- Trade value uses the Fitzgerald-Spielberger chart by default.
- Cincinnati Bengals are omitted: PFR records their de-facto GM (Duke Tobin)
  as 'Director of Player Personnel', not 'General Manager'.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gm_assessment import (
    filter_gm_records,
    find_overlapping_gms,
    get_gm_year_range,
    get_surplus_value,
    get_trade_value,
)
from src.plot_av import plot_gm_quadrant

_MODELS_DIR = PROJECT_ROOT / "models"
_EXECUTIVES_DIR = PROJECT_ROOT / "data" / "raw" / "pfr" / "executives"
_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stathead" / "annual_av"
_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _pfr_code_for_team(team: str) -> str:
    """Convert a Stathead team code to its PFR code (lowercase)."""
    _STATHEAD_TO_PFR: dict[str, str] = {
        "ATL": "atl", "BUF": "buf", "CAR": "car", "CHI": "chi",
        "CIN": "cin", "CLE": "cle", "IND": "clt", "ARI": "crd",
        "DAL": "dal", "DEN": "den", "DET": "det", "GNB": "gnb",
        "HOU": "htx", "JAX": "jax", "KAN": "kan", "MIA": "mia",
        "MIN": "min", "NOR": "nor", "NWE": "nwe", "NYG": "nyg",
        "NYJ": "nyj", "TEN": "oti", "PHI": "phi", "PIT": "pit",
        "OAK": "rai", "LVR": "rai", "BAL": "rav", "SDG": "sdg",
        "LAC": "sdg", "SEA": "sea", "SFO": "sfo", "STL": "ram",
        "LAR": "ram", "TAM": "tam", "WAS": "was",
    }
    return _STATHEAD_TO_PFR.get(team.upper(), team.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="GM Draft Performance Quadrant")
    parser.add_argument("--team", default="DET", help="Stathead team code (default: DET)")
    parser.add_argument("--year", type=int, default=2024, help="Draft year (default: 2024)")
    parser.add_argument(
        "--model", default="linear",
        choices=["parametric", "knn", "linear", "ridge"],
        help="Career AV model for projected draft classes (default: linear)",
    )
    parser.add_argument(
        "--parametric-curve", default="quadratic",
        help="Parametric curve variant when --model=parametric (default: quadratic)",
    )
    parser.add_argument(
        "--chart", default="eaar",
        help="Trade chart column prefix for aggregation (default: eaar — EAVAR units)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/gm_comparison",
        help="Directory to save HTML output (default: outputs/gm_comparison)",
    )
    parser.add_argument(
        "--data-end-year", type=int, default=2024,
        help="Last draft year with ≥2 completed seasons (default: 2024)",
    )
    args = parser.parse_args()

    team = args.team.upper()
    year = args.year
    pfr_code = _pfr_code_for_team(team)

    # 1. Find target GM and their year range
    print(f"Looking up GM for {team} in {year}...")
    gm_info = get_gm_year_range(pfr_code, year, _EXECUTIVES_DIR)
    if gm_info is None:
        print(f"No qualifying General Manager found for {team} in {year}. Exiting.")
        sys.exit(1)

    gm_name, gm_from, gm_to = gm_info
    print(f"  Found: {gm_name} ({gm_from}–{gm_to})")

    # 2. Find all overlapping GMs (each assessed over their full career within data bounds)
    print(f"\nFinding GMs overlapping {gm_from}–{gm_to} (data window capped at {args.data_end_year})...")
    gm_df = find_overlapping_gms(
        gm_from, gm_to, _EXECUTIVES_DIR, years_cap=args.data_end_year
    )
    print(f"  {len(gm_df)} GM entries across {gm_df['pfr_code'].n_unique()} franchises")

    # 3. Aggregate surplus AV for each GM
    print("\nAggregating draft surplus AV...")
    gm_df = get_surplus_value(
        gm_df,
        model_name=args.model,
        parametric_curve=args.parametric_curve,
        models_dir=_MODELS_DIR,
        raw_dir=_RAW_DIR,
    )

    # 4. Exclude GMs with no qualifying draft data (< 2 completed seasons)
    before = len(gm_df)
    gm_df = filter_gm_records(gm_df)
    excluded = before - len(gm_df)
    if excluded:
        print(f"  Excluded {excluded} GM(s) with no qualifying draft data")

    # 5. Aggregate trade value for each GM
    print("Aggregating trade values (EAVAR units)...")
    gm_df = get_trade_value(gm_df, data_dir=_DATA_DIR, chart_name=args.chart)

    # 6. Print summary table
    print(f"\n{'GM':<30} {'Team':<6} {'Assessed':>12} {'Surplus/Yr':>10} {'EAVAR/Yr':>10}")
    print("-" * 74)
    for row in gm_df.sort("avg_surplus_per_year", descending=True).iter_rows(named=True):
        assessed_label = f"{row['overlap_from']}–{row['overlap_to']}"
        print(
            f"{row['gm_name']:<30} {row['stathead_code']:<6} {assessed_label:>12}"
            f" {row['avg_surplus_per_year']:>10.2f} {row['avg_trade_per_year']:>10.2f}"
        )

    # 6. Build and save the quadrant plot
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{team}_{year}_gm_quadrant.html"

    chart_title = (
        f"GM Draft Performance: {gm_name} ({team}, {gm_from}–{gm_to}) vs Peers"
    )
    print(f"\nGenerating quadrant plot → {output_path}")
    fig = plot_gm_quadrant(
        gm_df,
        title=chart_title,
        highlight_team=team,
        export_path=output_path,
        export_format="html",
    )
    print("Done.")


if __name__ == "__main__":
    main()
