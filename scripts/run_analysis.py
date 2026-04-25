"""Top-level script for the annual AV analysis pipeline.

Usage
-----
Run all analyses (default)::

    python scripts/run_analysis.py

Skip specific steps::

    python scripts/run_analysis.py --skip-skew
    python scripts/run_analysis.py --skip-rolling
    python scripts/run_analysis.py --skip-exp-fit
    python scripts/run_analysis.py --skip-plots
    python scripts/run_analysis.py --skip-skew --skip-rolling

Available ``--skip-*`` flags
----------------------------
--skip-skew        Skip full-dataset and rolling-window skew-normal fits.
--skip-rolling     Skip all rolling-window analyses (stats + skew + animated plot).
--skip-exp-fit     Skip exponential decay fit and its plot.
--skip-plots       Skip all Plotly figure generation.
--skip-position    Skip position-based career AV analysis and plots.
--skip-comparison  Skip multi-metric normalized pick value comparison.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.annual_av_analysis import (
    _aggregate_player_av,
    _filter_top_percentile_per_pick,
    _prepare_av_data,
    exponential_av_fit,
    exponential_av_fit_means,
    fit_result_to_dataframe,
    pick_based_stats,
    position_career_stats,
    rolling_window_pick_stats,
    rolling_window_skew_fit,
    skew_normal_fit,
)
from src.data_ingest import load_csv, load_nflreadr_draft_picks, load_parquets_from_dir
from src.data_output import save_data
from src.plot_av import (
    plot_animated_rolling_window,
    plot_exponential_fit,
    plot_exponential_fit_means,
    plot_normalized_pick_value_comparison,
    plot_pick_av,
    plot_position_career_av,
)

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
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
    parser.add_argument("--skip-position", action="store_true",
                        help="Skip position-based career AV analysis and plots.")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip multi-metric normalized pick value comparison.")
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
            print("Fitting exponential decay curves for each rolling window...")
            rolling_fit = {
                yr: exponential_av_fit_means(df)
                for yr, df in rolling_stats.items()
            }
            print("Generating animated rolling-window plot...")
            plot_animated_rolling_window(
                rolling_fit,
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
        save_data(
            fit_result_to_dataframe(fit_result),
            PROCESSED_DIR / "exp_fit_rookie_contract_av.csv",
            format="csv",
        )
        print("  Saved exp_fit_rookie_contract_av.csv")

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

        save_data(
            fit_result_to_dataframe(means_fit_result),
            PROCESSED_DIR / "exp_fit_rookie_contract_av_means.csv",
            format="csv",
        )
        print("  Saved exp_fit_rookie_contract_av_means.csv")

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

    # ------------------------------------------------------------------
    # 10. Position career AV analysis
    # ------------------------------------------------------------------
    if not args.skip_position:
        print("Computing position career stats (normalized)...")
        pos_stats_norm = position_career_stats(RAW_DIR, normalize=True)
        save_data(
            pos_stats_norm,
            PROCESSED_DIR / "position_career_stats_normalized.csv",
            format="csv",
        )
        print(f"  Saved position_career_stats_normalized.csv ({len(pos_stats_norm)} rows)")

        print("Computing position career stats (raw positions)...")
        pos_stats_raw = position_career_stats(RAW_DIR, normalize=False)
        save_data(
            pos_stats_raw,
            PROCESSED_DIR / "position_career_stats_raw.csv",
            format="csv",
        )
        print(f"  Saved position_career_stats_raw.csv ({len(pos_stats_raw)} rows)")

        print("Computing position career stats (normalized, round 1)...")
        pos_stats_r1 = position_career_stats(RAW_DIR, normalize=True, rounds=[1])
        save_data(
            pos_stats_r1,
            PROCESSED_DIR / "position_career_stats_normalized_r1.csv",
            format="csv",
        )
        print(f"  Saved position_career_stats_normalized_r1.csv ({len(pos_stats_r1)} rows)")

        if not args.skip_plots:
            print("Generating position career AV plot (normalized)...")
            plot_position_career_av(
                pos_stats_norm,
                title="Annual AV Development by Position — Normalized Groups (1970–2025)",
                export_path=FIGURES_DIR / "position_career_av_normalized.html",
                export_format="html",
            )
            print("  Saved position_career_av_normalized.html")

            print("Generating position career AV plot (all positions)...")
            plot_position_career_av(
                pos_stats_raw,
                title="Annual AV Development by Position — All Positions (1970–2025)",
                export_path=FIGURES_DIR / "position_career_av_raw.html",
                export_format="html",
            )
            print("  Saved position_career_av_raw.html")

            print("Generating position career AV plot (normalized, round 1)...")
            plot_position_career_av(
                pos_stats_r1,
                title="Annual AV Development by Position — Round 1 Picks (1970–2025)",
                export_path=FIGURES_DIR / "position_career_av_normalized_r1.html",
                export_format="html",
            )
            print("  Saved position_career_av_normalized_r1.html")
    else:
        print("Skipping position career analysis (--skip-position).")

    # ------------------------------------------------------------------
    # 11. Multi-metric normalized pick value comparison
    # ------------------------------------------------------------------
    if not args.skip_comparison:
        print("Loading nflreadpy draft pick data (dr_av)...")
        nflreadr_df = load_nflreadr_draft_picks().drop_nulls(subset=["dr_av"])
        print(f"  Loaded {len(nflreadr_df)} picks with dr_av "
              f"({nflreadr_df['Draft Year'].min()}–{nflreadr_df['Draft Year'].max()})")

        # Ensure rookie contract fit is available even if --skip-exp-fit was set
        if args.skip_exp_fit:
            print("Computing rookie contract AV fit for comparison...")
            fit_result = exponential_av_fit(player_av_lf, max_pick=250)

        print("Computing rookie contract AV top-10% fit...")
        rc_df = player_av_lf.collect()
        rc_top10_df = _filter_top_percentile_per_pick(rc_df, "rookie_contract_av")
        fit_rc_top10 = exponential_av_fit(rc_top10_df, max_pick=250, av_col="rookie_contract_av")
        save_data(
            fit_result_to_dataframe(fit_rc_top10),
            PROCESSED_DIR / "exp_fit_rookie_contract_av_top10.csv",
            format="csv",
        )
        print("  Saved exp_fit_rookie_contract_av_top10.csv")

        print("Computing dr_av fit...")
        fit_dr = exponential_av_fit(nflreadr_df, max_pick=250, av_col="dr_av")
        a, b, c = fit_dr["popt"]
        print(f"  f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")
        save_data(
            fit_result_to_dataframe(fit_dr),
            PROCESSED_DIR / "exp_fit_dr_av.csv",
            format="csv",
        )
        print("  Saved exp_fit_dr_av.csv")

        print("Computing dr_av top-10% fit...")
        dr_top10_df = _filter_top_percentile_per_pick(nflreadr_df, "dr_av")
        fit_dr_top10 = exponential_av_fit(dr_top10_df, max_pick=250, av_col="dr_av")
        a, b, c = fit_dr_top10["popt"]
        print(f"  f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")
        save_data(
            fit_result_to_dataframe(fit_dr_top10),
            PROCESSED_DIR / "exp_fit_dr_av_top10.csv",
            format="csv",
        )
        print("  Saved exp_fit_dr_av_top10.csv")

        if not args.skip_plots:
            print("Generating normalized pick value comparison plot...")
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
                fits={
                    "Rookie Contract AV": fit_result,
                    "Rookie Contract AV (Top 10%)": fit_rc_top10,
                    "Draft AV": fit_dr,
                    "Draft AV (Top 10%)": fit_dr_top10,
                },
                trade_charts=trade_charts,
                title="Pick Value Comparison — Normalized to Pick 1 (1970–2022)",
                export_path=FIGURES_DIR / "pick_value_comparison_normalized.html",
                export_format="html",
            )
            print("  Saved pick_value_comparison_normalized.html")
    else:
        print("Skipping multi-metric comparison (--skip-comparison).")

    print("\nAnalysis complete.")
    print(f"  Processed data: {PROCESSED_DIR}")
    print(f"  Figures:        {FIGURES_DIR}")


if __name__ == "__main__":
    main()
