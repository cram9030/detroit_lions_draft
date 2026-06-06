"""Correlate draft class metrics with season performance.

Lagged Draft Class Correlation: for each lag k=0..3, how does a draft class
from year Y correlate with season Y+k win %, point differential, and SRS?
Answers: "Does drafting well today predict winning k years from now?"

Composite Active Roster Draft Value: for each season Y, the composite draft
contribution is obs_yr0(class Y) + obs_yr1(class Y-1) + obs_yr2(class Y-2) +
obs_yr3(class Y-3). How does total active roster draft value correlate with
that season's performance?

Usage:
    python scripts/draft_performance_correlation.py
    python scripts/draft_performance_correlation.py --min-year 2010
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingest import compute_season_performance, load_baked_data
from src.gm_assessment import STATHEAD_TO_NFLREADPY
from src.plot_av import (
    plot_draft_composite_scatter,
    plot_draft_lag_heatmap,
    plot_draft_lag_scatter,
)
from src.trade_value import aggregate_trade_value

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

PERF_METRICS = ["win_pct", "point_diff", "srs"]
PERF_LABELS = {"win_pct": "Win %", "point_diff": "Point Differential", "srs": "SRS"}
DRAFT_METRICS = ["class_surplus", "trade_av", "net_av"]
DRAFT_LABELS = {"class_surplus": "Surplus AV", "trade_av": "Trade AV", "net_av": "Net AV"}


# ---------------------------------------------------------------------------
# Trade value computation
# ---------------------------------------------------------------------------

def compute_trade_values(stathead_teams: list[str], draft_years: list[int]) -> pl.DataFrame:
    """Return net EAVAR trade value per (stathead_team, draft_year) via aggregate_trade_value."""
    rows = []
    for sh_team in stathead_teams:
        nfl_team = STATHEAD_TO_NFLREADPY.get(sh_team, sh_team)
        agg = aggregate_trade_value(nfl_team, draft_years)
        for year, val in agg["per_year"].items():
            rows.append({"stathead_team": sh_team, "draft_year": year, "trade_av": val})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lagged Draft Class Correlation
# ---------------------------------------------------------------------------

def approach3_lagged_correlation(
    draft_df: pl.DataFrame,
    season_perf: pl.DataFrame,
) -> dict:
    """Compute Pearson r and Spearman ρ for each (draft_metric, lag, perf_metric).

    Returns a nested dict: results[dm][lag][pm] = (pearson_r, spearman_r, n).
    """
    nfl_map = STATHEAD_TO_NFLREADPY
    df = draft_df.with_columns(
        pl.col("stathead_team")
        .map_elements(lambda t: nfl_map.get(t, t), return_dtype=pl.String)
        .alias("team")
    )

    results: dict[str, dict[int, dict[str, tuple]]] = {
        dm: {lag: {} for lag in range(4)} for dm in DRAFT_METRICS
    }

    for lag in range(4):
        lag_df = (
            df.with_columns((pl.col("draft_year") + lag).alias("season"))
            .join(season_perf.select(["season", "team"] + PERF_METRICS), on=["season", "team"], how="inner")
        )
        for dm in DRAFT_METRICS:
            for pm in PERF_METRICS:
                pair = lag_df.select([dm, pm]).drop_nulls()
                if len(pair) < 10:
                    continue
                x = pair[dm].to_numpy()
                y = pair[pm].to_numpy()
                pr, _ = stats.pearsonr(x, y)
                sr, _ = stats.spearmanr(x, y)
                results[dm][lag][pm] = (round(pr, 3), round(sr, 3), len(pair))

    return results


def print_lag_correlations(results: dict) -> None:
    col_w = 14
    header = f"{'Metric':<16} {'Lag':<5}" + "".join(
        f"  {PERF_LABELS[pm]:<{col_w}}" for pm in PERF_METRICS
    )

    for stat_name, stat_idx in [("Pearson r", 0), ("Spearman ρ", 1)]:
        print(f"\n=== Lagged Draft Class Correlation — {stat_name} ===\n")
        print(header)
        print("-" * len(header))
        for dm in DRAFT_METRICS:
            for lag in range(4):
                vals = []
                n = 0
                for pm in PERF_METRICS:
                    entry = results[dm][lag].get(pm)
                    if entry:
                        vals.append(f"{entry[stat_idx]:>{col_w}.3f}")
                        n = entry[2]
                    else:
                        vals.append(f"{'N/A':>{col_w}}")
                row = f"{DRAFT_LABELS[dm]:<16} {lag:<5}" + "  ".join(vals) + f"  n={n}"
                print(row)
            print()


# ---------------------------------------------------------------------------
# Composite Active Roster Draft Value
# ---------------------------------------------------------------------------

def build_approach2_composite(baked_df: pl.DataFrame, season_perf: pl.DataFrame) -> pl.DataFrame:
    """For each (team, season), compute composite = yr0(Y) + yr1(Y-1) + yr2(Y-2) + yr3(Y-3).

    Requires that obs data for all 4 draft classes is present in baked_df.
    """
    nfl_map = STATHEAD_TO_NFLREADPY
    df = baked_df.with_columns(
        pl.col("stathead_team")
        .map_elements(lambda t: nfl_map.get(t, t), return_dtype=pl.String)
        .alias("team")
    )

    yr0 = df.select(["team", "draft_year", "obs_yr0"]).rename({"draft_year": "season", "obs_yr0": "c_yr0"})
    yr1 = df.select(["team", "draft_year", "obs_yr1"]).with_columns((pl.col("draft_year") + 1).alias("season")).drop("draft_year").rename({"obs_yr1": "c_yr1"})
    yr2 = df.select(["team", "draft_year", "obs_yr2"]).with_columns((pl.col("draft_year") + 2).alias("season")).drop("draft_year").rename({"obs_yr2": "c_yr2"})
    yr3 = df.select(["team", "draft_year", "obs_yr3"]).with_columns((pl.col("draft_year") + 3).alias("season")).drop("draft_year").rename({"obs_yr3": "c_yr3"})

    composite = (
        yr0
        .join(yr1, on=["team", "season"], how="inner")
        .join(yr2, on=["team", "season"], how="inner")
        .join(yr3, on=["team", "season"], how="inner")
        .with_columns(
            (pl.col("c_yr0") + pl.col("c_yr1") + pl.col("c_yr2") + pl.col("c_yr3")).alias("composite_av")
        )
        .join(season_perf.select(["season", "team"] + PERF_METRICS), on=["season", "team"], how="inner")
    )
    return composite


def approach2_correlation(composite: pl.DataFrame) -> dict:
    """Return Pearson r and Spearman ρ for composite_av vs each perf metric."""
    results = {}
    for pm in PERF_METRICS:
        pair = composite.select(["composite_av", pm]).drop_nulls()
        if len(pair) < 10:
            continue
        x = pair["composite_av"].to_numpy()
        y = pair[pm].to_numpy()
        pr, _ = stats.pearsonr(x, y)
        sr, _ = stats.spearmanr(x, y)
        results[pm] = (round(pr, 3), round(sr, 3), len(pair))
    return results


def print_approach2_correlations(results: dict) -> None:
    print("\n=== Composite Active Roster Draft Value — Correlation with Season Performance ===\n")
    print(f"{'Metric':<20} {'Pearson r':>12} {'Spearman ρ':>12} {'n':>6}")
    print("-" * 54)
    for pm in PERF_METRICS:
        entry = results.get(pm)
        if entry:
            print(f"{PERF_LABELS[pm]:<20} {entry[0]:>12.3f} {entry[1]:>12.3f} {entry[2]:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Earliest draft class year to include (default: 2000)",
    )
    args = parser.parse_args()

    observed_years = list(range(args.min_year, 2023))  # fully-observed classes
    all_baked_years = list(range(args.min_year, 2025))  # includes 2023, 2024 for obs_yr*

    print("Loading season performance from nflreadpy schedules...")
    season_perf = compute_season_performance()

    print("Loading baked draft data...")
    baked_df = load_baked_data(all_baked_years)

    stathead_teams = sorted(baked_df["stathead_team"].unique().to_list())

    print(f"Computing trade values for {len(stathead_teams)} teams × {len(observed_years)} years...")
    trade_df = compute_trade_values(stathead_teams, observed_years)

    # Build lagged correlation dataset (fully observed only, with trade AV)
    observed_df = (
        baked_df
        .filter(pl.col("class_surplus").is_not_null())
        .join(trade_df, on=["stathead_team", "draft_year"], how="left")
        .with_columns(
            (pl.col("class_surplus") + pl.col("trade_av").fill_null(0.0)).alias("net_av")
        )
    )

    print(f"\nLagged draft class data: {len(observed_df)} team-draft-class rows")

    # --- Lagged Draft Class Correlation ---
    lag_results = approach3_lagged_correlation(observed_df, season_perf)
    print_lag_correlations(lag_results)

    out = FIGURES_DIR / "draft_lag_correlation_heatmap.html"
    plot_draft_lag_heatmap(lag_results, DRAFT_METRICS, DRAFT_LABELS, PERF_METRICS, PERF_LABELS, export_path=out)
    print(f"\nSaved: {out}")

    for dm in DRAFT_METRICS:
        for pm in PERF_METRICS:
            dm_slug = dm.replace("_", "-")
            pm_slug = pm.replace("_", "-")
            out = FIGURES_DIR / f"draft_lag_{dm_slug}_vs_{pm_slug}.html"
            plot_draft_lag_scatter(observed_df, season_perf, lag_results, dm, pm, DRAFT_LABELS, PERF_LABELS, export_path=out)
            print(f"Saved: {out}")

    # --- Composite Active Roster Draft Value ---
    print("\nBuilding composite active roster draft value...")
    composite = build_approach2_composite(baked_df, season_perf)
    print(f"Composite active roster data: {len(composite)} team-season rows (seasons {composite['season'].min()}–{composite['season'].max()})")

    a2_results = approach2_correlation(composite)
    print_approach2_correlations(a2_results)

    out = FIGURES_DIR / "draft_approach2_composite_scatter.html"
    plot_draft_composite_scatter(composite, a2_results, PERF_METRICS, PERF_LABELS, export_path=out)
    print(f"Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
