"""Compute and plot Expected Approximate Value Above Replacement (EAVAR).

EAVAR measures how much more value a drafted player is expected to produce
over their rookie contract than a readily available replacement-level player.

Methodology
-----------
1. Fit an exponential decay curve to individual player rookie contract AV
   (sum of AV.1 over the 4-year rookie contract window) by draft pick.
2. Compute the replacement level as the mean season AV for players on
   1-year contracts at or below ``replacement_percent`` × NFL minimum salary,
   scaled to 4 seasons to match the rookie contract window.
3. EAVAR(pick) = f(pick) − replacement_level

Usage
-----
    python scripts/expected_av_above_replacement.py
    python scripts/expected_av_above_replacement.py --replacement-percent 1.10
    python scripts/expected_av_above_replacement.py --skip-plots
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.annual_av_analysis import (
    aggregate_player_av,
    exponential_av_fit,
    fit_result_to_dataframe,
    prepare_av_data,
)
from src.contracts import get_replacement_level_av, get_replacement_level_contracts
from src.data_ingest import load_csv, load_parquets_from_dir
from src.data_output import save_data
from src.plot_av import plot_eavar, plot_normalized_pick_value_comparison

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expected AV Above Replacement pipeline.")
    parser.add_argument(
        "--replacement-percent",
        type=float,
        default=1.20,
        help="Ceiling multiplier on the NFL minimum salary for replacement level (default: 1.20).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip all Plotly figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fit exponential decay to individual player rookie contract AV
    # ------------------------------------------------------------------
    print("Loading data and fitting exponential decay to rookie contract AV...")
    player_av_lf = aggregate_player_av(prepare_av_data(load_parquets_from_dir(RAW_DIR)))
    fit_result = exponential_av_fit(player_av_lf, max_pick=250)
    a, b, c = fit_result["popt"]
    print(f"  f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")

    # ------------------------------------------------------------------
    # 2. Compute replacement level baseline
    # ------------------------------------------------------------------
    print(f"Computing replacement level ({args.replacement_percent:.0%} of NFL minimum)...")
    av_lf = prepare_av_data(load_parquets_from_dir(RAW_DIR))
    contracts_df = get_replacement_level_contracts(args.replacement_percent)
    av_df = get_replacement_level_av(av_lf, contracts_df)
    mean_season_av = float(av_df["AV.1"].mean())
    replacement_level = mean_season_av * 4  # scale to 4-year rookie contract window
    print(f"  Replacement-level player-seasons: {len(av_df)}")
    print(f"  Mean season AV (replacement): {mean_season_av:.3f}")
    print(f"  Replacement level (×4 seasons): {replacement_level:.3f}")

    # ------------------------------------------------------------------
    # 3. Build and save EAVAR DataFrame
    # ------------------------------------------------------------------
    print("Computing EAVAR by pick...")
    fit_df = fit_result_to_dataframe(fit_result)
    eavar_df = fit_df.with_columns([
        (pl.col("y_fit") - replacement_level).alias("eavar"),
        (pl.col("y_upper") - replacement_level).alias("eavar_upper"),
        (pl.col("y_lower") - replacement_level).alias("eavar_lower"),
        pl.lit(replacement_level).alias("replacement_level"),
    ]).select(["pick", "eavar", "eavar_upper", "eavar_lower", "replacement_level"])

    save_data(eavar_df, PROCESSED_DIR / "expected_av_above_replacement.csv", format="csv")
    print(f"  Saved expected_av_above_replacement.csv ({len(eavar_df)} picks)")
    print(f"  Pick 1 EAVAR:   {eavar_df.filter(pl.col('pick') == 1)['eavar'][0]:.3f}")
    print(f"  Pick 32 EAVAR:  {eavar_df.filter(pl.col('pick') == 32)['eavar'][0]:.3f}")
    print(f"  Pick 250 EAVAR: {eavar_df.filter(pl.col('pick') == 250)['eavar'][0]:.3f}")

    if args.skip_plots:
        print("Skipping plots (--skip-plots).")
        print("\nAnalysis complete.")
        print(f"  Processed data: {PROCESSED_DIR}")
        return

    # ------------------------------------------------------------------
    # 4. EAVAR curve plot
    # ------------------------------------------------------------------
    print("Generating EAVAR plot...")
    plot_eavar(
        fit_result,
        replacement_level=replacement_level,
        title=f"Expected AV Above Replacement by Draft Pick (1970–2022, repl.={replacement_level:.1f})",
        export_path=FIGURES_DIR / "expected_av_above_replacement.html",
        export_format="html",
    )
    print("  Saved expected_av_above_replacement.html")

    # ------------------------------------------------------------------
    # 5. EAVAR vs trade charts normalized comparison
    # ------------------------------------------------------------------
    print("Generating EAVAR vs trade charts comparison...")

    # Normalize EAVAR to pick 1 = 1.0 for fair comparison
    eavar_pick1 = float(eavar_df.filter(pl.col("pick") == 1)["eavar"][0])
    eavar_trade_df = (
        eavar_df
        .select(["pick", "eavar"])
        .rename({"pick": "Pick", "eavar": "Value"})
        .with_columns([
            pl.col("Pick").cast(pl.Int64),
            (pl.col("Value") / eavar_pick1).alias("Value"),
        ])
    )

    trade_charts = {
        "Jimmy Johnson": (
            load_csv(PROCESSED_DIR / "jimmy_johnson_trade_chart.csv")
            .unique(subset=["Pick"], keep="first")
            .sort("Pick")
            .select(["Pick", "Value"])
            .with_columns(pl.col("Pick").cast(pl.Int64))
        ),
        "Fitzgerald-Spielberger": (
            load_csv(PROCESSED_DIR / "fitzgerald_spielberger_trade_chart.csv")
            .select(["Pick", "Value"])
            .with_columns(pl.col("Pick").cast(pl.Int64))
        ),
        "PFF WAR": (
            load_csv(PROCESSED_DIR / "pff_war_draft_chart.csv")
            .rename({"PFF_WAR_Normalized": "Value"})
            .select(["Pick", "Value"])
            .with_columns(pl.col("Pick").cast(pl.Int64))
        ),
        "5-Year AV": (
            load_csv(PROCESSED_DIR / "5_year_av_chart.csv")
            .rename({"Pk": "Pick", "FP Val": "Value"})
            .select(["Pick", "Value"])
            .with_columns(pl.col("Pick").cast(pl.Int64))
        ),
        "Rich Hill": (
            load_csv(PROCESSED_DIR / "Rich-Hill.csv")
            .rename({"pick": "Pick", "value": "Value"})
            .select(["Pick", "Value"])
            .with_columns(pl.col("Pick").cast(pl.Int64))
        ),
    }

    plot_normalized_pick_value_comparison(
        fits={},
        trade_charts={"EAVAR": eavar_trade_df, **trade_charts},
        title=f"Pick Value Comparison — EAVAR vs Trade Charts (repl.={replacement_level:.1f})",
        export_path=FIGURES_DIR / "eavar_vs_trade_charts.html",
        export_format="html",
    )
    print("  Saved eavar_vs_trade_charts.html")

    print("\nAnalysis complete.")
    print(f"  Processed data: {PROCESSED_DIR}")
    print(f"  Figures:        {FIGURES_DIR}")


if __name__ == "__main__":
    main()
