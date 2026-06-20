"""All-GM Draft Performance — survey every NFL GM over a date range.

Given a start and end year, finds every GM whose tenure overlaps that window
across all 32 NFL franchises, aggregates draft surplus AV and trade value for
each GM, and produces three charts: a quadrant scatter, a draft surplus bar
chart, and a trade value bar chart.

Usage
-----
    python scripts/all_gm_performance.py                        # 2001–2024 defaults
    python scripts/all_gm_performance.py --from-year 2010 --to-year 2024
    python scripts/all_gm_performance.py --model knn --chart rich_hill
    python scripts/all_gm_performance.py --from-year 2015 --to-year 2024 --model linear

Notes
-----
- Only draft years with ≥ 2 completed seasons contribute to surplus AV.
- Trade value uses the EAAR chart by default.
- Cincinnati Bengals are omitted: PFR records their de-facto GM (Duke Tobin)
  as 'Director of Player Personnel', not 'General Manager'.
- --from-year controls which GMs are included (overlap window); --data-start-year
  controls the floor of each GM's assessed period (limited by Stathead data
  availability, default 2010).
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
    get_surplus_value,
    get_trade_value,
)
from src.plot_av import (
    plot_gm_quadrant,
    plot_gm_surplus,
    plot_gm_trade,
)

_MODELS_DIR = PROJECT_ROOT / "models"
_EXECUTIVES_DIR = PROJECT_ROOT / "data" / "raw" / "pfr" / "executives"
_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stathead" / "annual_av"
_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-GM Draft Performance — survey every NFL GM over a date range"
    )
    parser.add_argument(
        "--from-year", type=int, default=2001,
        help="Start of the overlap window (default: 2001)",
    )
    parser.add_argument(
        "--to-year", type=int, default=2024,
        help="End of the overlap window (default: 2024)",
    )
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
        "--output-dir", default="outputs/gm_performance",
        help="Directory to save HTML output (default: outputs/gm_performance)",
    )
    parser.add_argument(
        "--data-end-year", type=int, default=2024,
        help="Last draft year with ≥2 completed seasons (default: 2024)",
    )
    parser.add_argument(
        "--data-start-year", type=int, default=None,
        help="Floor year for each GM's assessed window (default: same as --from-year)",
    )
    args = parser.parse_args()

    from_year = args.from_year
    to_year = args.to_year
    data_start_year = args.data_start_year if args.data_start_year is not None else from_year

    # 1. Find all GMs whose tenure overlaps the requested window
    print(f"Finding all GMs overlapping {from_year}–{to_year} "
          f"(data window: {data_start_year}–{args.data_end_year})...")
    gm_df = find_overlapping_gms(
        from_year,
        to_year,
        _EXECUTIVES_DIR,
        years_cap=args.data_end_year,
        data_start_year=data_start_year,
    )
    print(f"  {len(gm_df)} GM entries across {gm_df['pfr_code'].n_unique()} franchises")

    # 2. Aggregate surplus AV for each GM
    print("\nAggregating draft surplus AV...")
    gm_df = get_surplus_value(
        gm_df,
        model_name=args.model,
        parametric_curve=args.parametric_curve,
        models_dir=_MODELS_DIR,
        raw_dir=_RAW_DIR,
    )

    # 3. Exclude GMs with no qualifying draft data (< 2 completed seasons)
    before = len(gm_df)
    gm_df = filter_gm_records(gm_df)
    excluded = before - len(gm_df)
    if excluded:
        print(f"  Excluded {excluded} GM(s) with no qualifying draft data")

    # 4. Aggregate trade value for each GM
    print("Aggregating trade values (EAVAR units)...")
    gm_df = get_trade_value(gm_df, data_dir=_DATA_DIR, chart_name=args.chart)

    # 5. Print summary table
    print(f"\n{'GM':<30} {'Team':<6} {'Assessed':>12} {'Surplus/Yr':>10} {'EAVAR/Yr':>10}")
    print("-" * 74)
    for row in gm_df.sort("avg_surplus_per_year", descending=True).iter_rows(named=True):
        assessed_label = f"{row['overlap_from']}–{row['overlap_to']}"
        print(
            f"{row['gm_name']:<30} {row['stathead_code']:<6} {assessed_label:>12}"
            f" {row['avg_surplus_per_year']:>10.2f} {row['avg_trade_per_year']:>10.2f}"
        )

    # 6. Build and save plots
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    range_label = f"{from_year}_{to_year}"
    chart_title = f"GM Draft Performance: All GMs, {from_year}–{to_year}"
    trade_chart_title = f"GM Trade Performance: All GMs, {from_year}–{to_year}"

    quadrant_path = output_dir / f"{range_label}_gm_quadrant.html"
    surplus_path = output_dir / f"{range_label}_gm_draft_bar.html"
    trade_path = output_dir / f"{range_label}_gm_trade_bar.html"

    print(f"\nGenerating quadrant plot → {quadrant_path}")
    plot_gm_quadrant(
        gm_df,
        title=chart_title,
        export_path=quadrant_path,
        export_format="html",
    )

    plot_gm_surplus(
        gm_df,
        title=chart_title,
        export_path=surplus_path,
        export_format="html",
    )

    plot_gm_trade(
        gm_df,
        title=trade_chart_title,
        export_path=trade_path,
        export_format="html",
    )

    print("Done.")


if __name__ == "__main__":
    main()
