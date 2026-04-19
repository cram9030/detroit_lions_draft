"""Top-level script for the annual AV analysis pipeline.

Usage
-----
Run all analyses (default)::

    python run_analysis.py

Skip specific steps::

    python run_analysis.py --skip-skew
    python run_analysis.py --skip-rolling
    python run_analysis.py --skip-exp-fit
    python run_analysis.py --skip-plots
    python run_analysis.py --skip-skew --skip-rolling

Available ``--skip-*`` flags
----------------------------
--skip-skew       Skip full-dataset and rolling-window skew-normal fits.
--skip-rolling    Skip all rolling-window analyses (stats + skew + animated plot).
--skip-exp-fit    Skip exponential decay fit and its plot.
--skip-plots      Skip all Plotly figure generation.
"""

import argparse
from pathlib import Path

import polars as pl

from src.annual_av_analysis import (
    _aggregate_player_av,
    _prepare_av_data,
    exponential_av_fit,
    exponential_av_fit_means,
    pick_based_stats,
    rolling_window_pick_stats,
    rolling_window_skew_fit,
    skew_normal_fit,
)
from src.data_ingest import load_parquets_from_dir
from src.data_output import save_data
from src.plot_av import (
    plot_animated_rolling_window,
    plot_exponential_fit,
    plot_exponential_fit_means,
    plot_pick_av,
)

RAW_DIR = Path("data/raw/stathead/annual_av")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("outputs/figures")
WINDOW_LENGTH = 11  # odd integer — covers center ± 5 draft years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annual AV analysis pipeline.")
    parser.add_argument("--skip-skew", action="store_true",
                        help="Skip full-dataset and rolling-window skew-normal fits.")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Skip all rolling-window analyses and the animated plot.")
    parser.add_argument("--skip-exp-fit", action="store_true",
                        help="Skip exponential decay curve fit and its plot.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip all Plotly figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build the per-player LazyFrame once — reused by skew fit and exp fit.
    # LazyFrames are free to create; collection only happens when needed inside
    # each function.
    player_av_lf = _aggregate_player_av(_prepare_av_data(load_parquets_from_dir(RAW_DIR)))

    # ------------------------------------------------------------------
    # 1. Full-dataset per-pick stats
    # ------------------------------------------------------------------
    print("Computing full-dataset pick stats...")
    stats_df = pick_based_stats(RAW_DIR)
    save_data(stats_df, PROCESSED_DIR / "pick_stats.csv", format="csv")
    print(f"  Saved pick_stats.csv ({len(stats_df)} picks)")

    # ------------------------------------------------------------------
    # 2. Full-dataset skew-normal fit
    # ------------------------------------------------------------------
    if not args.skip_skew:
        print("Fitting skew-normal distributions (full dataset)...")
        skew_df = skew_normal_fit(player_av_lf)
        save_data(skew_df, PROCESSED_DIR / "skew_params.csv", format="csv")
        print(f"  Saved skew_params.csv ({len(skew_df)} picks fitted)")
    else:
        print("Skipping skew-normal fit (--skip-skew).")

    # ------------------------------------------------------------------
    # 3. Rolling-window pick stats
    # ------------------------------------------------------------------
    rolling_stats: dict | None = None
    if not args.skip_rolling:
        print(f"Computing rolling-window pick stats (window={WINDOW_LENGTH})...")
        rolling_stats = rolling_window_pick_stats(RAW_DIR, window_length=WINDOW_LENGTH)
        center_years = sorted(rolling_stats.keys())
        print(f"  Windows: {center_years[0]}–{center_years[-1]} ({len(center_years)} frames)")

        rolling_long = pl.concat(
            [df.with_columns(pl.lit(yr).alias("center_year")) for yr, df in rolling_stats.items()]
        )
        save_data(rolling_long, PROCESSED_DIR / "rolling_pick_stats.parquet", format="parquet")
        print(f"  Saved rolling_pick_stats.parquet ({len(rolling_long)} rows)")

        # ------------------------------------------------------------------
        # 4. Rolling-window skew-normal fit
        # ------------------------------------------------------------------
        if not args.skip_skew:
            print(f"Fitting rolling-window skew-normal distributions (window={WINDOW_LENGTH})...")
            rolling_skew = rolling_window_skew_fit(RAW_DIR, window_length=WINDOW_LENGTH)
            rolling_skew_long = pl.concat(
                [df.with_columns(pl.lit(yr).alias("center_year")) for yr, df in rolling_skew.items()]
            )
            save_data(
                rolling_skew_long, PROCESSED_DIR / "rolling_skew_params.parquet", format="parquet"
            )
            print(f"  Saved rolling_skew_params.parquet ({len(rolling_skew_long)} rows)")
        else:
            print("Skipping rolling skew-normal fit (--skip-skew).")
    else:
        print("Skipping rolling-window analyses (--skip-rolling).")

    # ------------------------------------------------------------------
    # 5. Static pick AV plot
    # ------------------------------------------------------------------
    if not args.skip_plots:
        print("Generating static pick AV plot...")
        plot_pick_av(
            stats_df,
            title="Rookie Contract AV by Draft Pick (1970–2022)",
            export_path=FIGURES_DIR / "pick_av_static.html",
            export_format="html",
        )
        print("  Saved pick_av_static.html")

        # ------------------------------------------------------------------
        # 6. Animated rolling-window plot
        # ------------------------------------------------------------------
        if rolling_stats is not None:
            print("Generating animated rolling-window plot...")
            plot_animated_rolling_window(
                rolling_stats,
                export_path=FIGURES_DIR / "pick_av_animated.html",
            )
            print("  Saved pick_av_animated.html")
        else:
            print("Skipping animated plot — rolling stats not computed (--skip-rolling).")
    else:
        print("Skipping all plots (--skip-plots).")

    # ------------------------------------------------------------------
    # 7. Exponential fit
    # ------------------------------------------------------------------
    if not args.skip_exp_fit:
        print("Fitting exponential decay curve to per-player rookie contract AV...")
        fit_result = exponential_av_fit(player_av_lf, max_pick=250)
        a, b, c = fit_result["popt"]
        print(f"  f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")
        print(f"  Parameter uncertainties (1σ): {fit_result['perr']}")

        # ------------------------------------------------------------------
        # 8. Exponential fit plot
        # ------------------------------------------------------------------
        if not args.skip_plots:
            print("Generating exponential fit plot (individual players)...")
            plot_exponential_fit(
                fit_result,
                title="Rookie Contract AV — Exponential Decay Fit by Pick (1970–2022)",
                export_path=FIGURES_DIR / "pick_av_exp_fit.html",
                export_format="html",
            )
            print("  Saved pick_av_exp_fit.html")

        # ------------------------------------------------------------------
        # 9. Exponential fit on per-pick means
        # ------------------------------------------------------------------
        print("Fitting exponential decay curve to per-pick mean AV...")
        means_fit_result = exponential_av_fit_means(stats_df, max_pick=250)
        a, b, c = means_fit_result["popt"]
        print(f"  f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")
        print(f"  Parameter uncertainties (1σ): {means_fit_result['perr']}")

        if not args.skip_plots:
            print("Generating exponential fit plot (per-pick means)...")
            plot_exponential_fit_means(
                means_fit_result,
                title="Rookie Contract AV — Exponential Decay Fit on Means by Pick (1970–2022)",
                export_path=FIGURES_DIR / "pick_av_exp_fit_means.html",
                export_format="html",
            )
            print("  Saved pick_av_exp_fit_means.html")
    else:
        print("Skipping exponential fit (--skip-exp-fit).")

    print("\nAnalysis complete.")
    print(f"  Processed data: {PROCESSED_DIR}")
    print(f"  Figures:        {FIGURES_DIR}")


if __name__ == "__main__":
    main()
