"""Standalone script for generating skew-normal distribution visualizations.

Loads pre-processed skew parameters and raw player AV data, then creates up
to three Plotly figures saved as HTML files.

Usage
-----
Default (3D surface + 2D pick animation)::

    python scripts/plot_skew.py

Also create the 3D animated surface by center year::

    python scripts/plot_skew.py --with-time-animation

Skip individual plots::

    python scripts/plot_skew.py --skip-3d
    python scripts/plot_skew.py --skip-pick-animation

Restrict all plots to a pick range::

    python scripts/plot_skew.py --max-pick 64

Override data and output directories::

    python scripts/plot_skew.py --data-dir data/processed --raw-dir data/raw/stathead/annual_av
    python scripts/plot_skew.py --output-dir outputs/custom

Available ``--skip-*`` / ``--with-*`` flags
--------------------------------------------
--skip-3d              Skip the static 3D surface plot.
--skip-pick-animation  Skip the 2D animation cycling through pick numbers.
--with-time-animation  Generate the 3D animated surface by center year
                       (requires rolling_skew_params.parquet).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.annual_av_analysis import aggregate_player_av, prepare_av_data
from src.data_ingest import load_csv, load_parquets_from_dir
from src.plot_av import (
    plot_animated_skew_by_pick,
    plot_animated_skew_by_year,
    plot_skew_3d,
)

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate skew-normal distribution visualizations for NFL draft AV analysis."
    )
    parser.add_argument(
        "--with-time-animation",
        action="store_true",
        help=(
            "Generate the 3D animated surface cycling through rolling-window center years. "
            "Requires rolling_skew_params.parquet in the processed data directory."
        ),
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip the static 3D surface plot.",
    )
    parser.add_argument(
        "--skip-pick-animation",
        action="store_true",
        help="Skip the 2D animation cycling through pick numbers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output HTML files. Defaults to outputs/figures/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing processed CSV/Parquet files. Defaults to data/processed/.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing raw Stathead annual AV parquet files. "
            "Defaults to data/raw/stathead/annual_av/."
        ),
    )
    parser.add_argument(
        "--max-pick",
        type=int,
        default=None,
        help="Restrict all plots to picks at or below this number.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir or RAW_DIR
    data_dir = args.data_dir or PROCESSED_DIR
    figures_dir = args.output_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading skew parameters...")
    skew_params_df = load_csv(data_dir / "skew_params.csv")
    print(f"  Loaded skew_params.csv ({len(skew_params_df)} picks)")

    print("Loading player AV data (raw parquets)...")
    player_av_df = aggregate_player_av(
        prepare_av_data(load_parquets_from_dir(raw_dir))
    ).collect()
    print(f"  Loaded {len(player_av_df)} player records")

    rolling_skew_df: pl.DataFrame | None = None
    if args.with_time_animation:
        rolling_path = data_dir / "rolling_skew_params.parquet"
        if not rolling_path.exists():
            print(
                "ERROR: rolling_skew_params.parquet not found. "
                "Run 'python scripts/run_analysis.py' first to generate it."
            )
            sys.exit(1)
        print("Loading rolling skew parameters...")
        rolling_skew_df = pl.read_parquet(rolling_path)
        n_years = rolling_skew_df["center_year"].n_unique()
        print(
            f"  Loaded rolling_skew_params.parquet "
            f"({len(rolling_skew_df)} rows, {n_years} center years)"
        )

    # ------------------------------------------------------------------
    # Plot 1: Static 3D surface
    # ------------------------------------------------------------------
    if not args.skip_3d:
        print("Generating static 3D skew-normal surface...")
        fig1 = plot_skew_3d(
            skew_params_df=skew_params_df,
            player_av_df=player_av_df,
            title="Skew-Normal AV Distributions by Draft Pick (1970–2022)",
            max_pick=args.max_pick,
        )
        out_path = figures_dir / "skew_3d_surface.html"
        fig1.write_html(str(out_path))
        print(f"  Saved skew_3d_surface.html")
    else:
        print("Skipping static 3D surface (--skip-3d).")

    # ------------------------------------------------------------------
    # Plot 2: 2D animated by pick
    # ------------------------------------------------------------------
    if not args.skip_pick_animation:
        print("Generating 2D animated skew-normal plot by pick...")
        fig2 = plot_animated_skew_by_pick(
            skew_params_df=skew_params_df,
            player_av_df=player_av_df,
            title="Skew-Normal AV Distribution by Pick Number (1970–2022)",
            max_pick=args.max_pick,
            export_path=figures_dir / "skew_animated_by_pick.html",
        )
        print(f"  Saved skew_animated_by_pick.html")
    else:
        print("Skipping pick animation (--skip-pick-animation).")

    # ------------------------------------------------------------------
    # Plot 3: 3D animated by center year (optional)
    # ------------------------------------------------------------------
    if args.with_time_animation and rolling_skew_df is not None:
        print("Generating 3D animated skew-normal surface by center year...")
        fig3 = plot_animated_skew_by_year(
            rolling_skew_df=rolling_skew_df,
            player_av_df=player_av_df,
            title="Skew-Normal AV Distributions by Pick — Rolling Window",
            max_pick=args.max_pick,
            export_path=figures_dir / "skew_animated_by_year.html",
        )
        print(f"  Saved skew_animated_by_year.html")

    print("\nDone.")
    print(f"  Figures: {figures_dir}")


if __name__ == "__main__":
    main()
