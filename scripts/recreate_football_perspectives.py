"""Recreate Chase Stewart's Football Perspectives NFL draft AV value chart.

Source: https://www.footballperspective.com/creating-a-nfl-draft-value-chart-part-i/

Stewart graded each player drafted 1980–2007 on their first five seasons of AV,
then subtracted 3.6 from each draft slot to convert total AV to marginal value
(the last slot is worth ~3.7 AV, so pick value above zero baseline is AV - 3.6).
An exponential decay curve was fitted to those marginal per-pick means.

This script:
1. Recomputes per-pick statistics for the same cohort (1980–2007, first 5 seasons).
2. Applies the 3.6-point marginal adjustment to all AV statistics.
3. Fits three exponential decay models and compares them:
     a. Individual player AV (one point per player — maximum statistical weight)
     b. Per-pick mean AV  (replicates Stewart's published method)
     c. Per-pick median AV (robust to outlier star players)
4. Compares all three fits to data/processed/5_year_av_chart.csv via least squares.
5. Saves the fitted curves and an interactive Plotly comparison figure.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annual_av_analysis import (
    _aggregate_player_av,
    _prepare_av_data,
    exponential_av_fit,
    exponential_av_fit_stat,
    fit_result_to_dataframe,
    pick_based_stats,
)
from src.data_ingest import load_parquets_from_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"

DRAFT_YEAR_START = 1980
DRAFT_YEAR_END = 2007
MAX_SEASONS_FROM_DRAFT = 5  # seasons 0–4 (< 5) → first five career years
MARGINAL_ADJUSTMENT = 3.6   # subtract to convert total AV → marginal value
MAX_PICK = 224              # published FP chart covers picks 1–224

# Colors follow the notebook palette
_COLOR_INDIVIDUAL = "#59a14f"   # green  — individual player fit
_COLOR_MEAN       = "#4e79a7"   # blue   — per-pick mean fit
_COLOR_MEDIAN     = "#f28e2b"   # orange — per-pick median fit
_COLOR_FP_REF     = "#e15759"   # red    — Football Perspectives reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _least_squares(fit_df: pl.DataFrame, fp_vals: np.ndarray) -> dict[str, float]:
    """Return SS_res, R², and RMSE comparing fit_df['y_fit'] to fp_vals."""
    y_fit = fit_df["y_fit"].to_numpy()
    residuals = y_fit - fp_vals
    ss_res = float((residuals**2).sum())
    ss_tot = float(((fp_vals - fp_vals.mean()) ** 2).sum())
    return {
        "ss_res": ss_res,
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt((residuals**2).mean())),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Compute per-pick stats and player-level data
    #         (1980–2007, first 5 seasons)
    # ------------------------------------------------------------------
    print(f"Loading AV data (draft years {DRAFT_YEAR_START}–{DRAFT_YEAR_END}, "
          f"first {MAX_SEASONS_FROM_DRAFT} seasons)...")

    stats_df = pick_based_stats(
        RAW_DIR,
        max_seasons_from_draft=MAX_SEASONS_FROM_DRAFT,
        draft_year_range=(DRAFT_YEAR_START, DRAFT_YEAR_END),
    )
    print(f"  {len(stats_df)} pick positions, "
          f"{int(stats_df['count'].sum())} player-pick observations")

    # Player-level data — needed for the individual-player exponential fit
    player_av_lf = _aggregate_player_av(
        _prepare_av_data(load_parquets_from_dir(RAW_DIR, lazy=True)).filter(
            (pl.col("Draft Year") >= DRAFT_YEAR_START)
            & (pl.col("Draft Year") <= DRAFT_YEAR_END)
        ),
        max_seasons_from_draft=MAX_SEASONS_FROM_DRAFT,
        min_season_av=2,
    )

    # ------------------------------------------------------------------
    # Step 2: Apply marginal value adjustment (−3.6) to all AV statistics
    # ------------------------------------------------------------------
    av_stat_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    stats_adjusted = stats_df.with_columns(
        [(pl.col(c) - MARGINAL_ADJUSTMENT).alias(c) for c in av_stat_cols]
    )

    player_av_adjusted = player_av_lf.with_columns(
        (pl.col("rookie_contract_av") - MARGINAL_ADJUSTMENT).alias("rookie_contract_av")
    )

    # ------------------------------------------------------------------
    # Step 3: Fit exponential decay — three methods
    # ------------------------------------------------------------------
    print("Fitting exponential decay curves...")

    print("  (a) Individual player AV...")
    fit_individual = exponential_av_fit(player_av_adjusted, max_pick=MAX_PICK)
    a, b, c = fit_individual["popt"]
    print(f"      f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")

    print("  (b) Per-pick mean AV...")
    fit_mean = exponential_av_fit_stat(stats_adjusted, stat_col="mean", max_pick=MAX_PICK)
    a, b, c = fit_mean["popt"]
    print(f"      f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")

    print("  (c) Per-pick median AV...")
    fit_median = exponential_av_fit_stat(stats_adjusted, stat_col="50%", max_pick=MAX_PICK)
    a, b, c = fit_median["popt"]
    print(f"      f(pick) = {a:.3f} * exp(-{b:.5f} * pick) + {c:.3f}")

    df_individual = fit_result_to_dataframe(fit_individual)
    df_mean       = fit_result_to_dataframe(fit_mean)
    df_median     = fit_result_to_dataframe(fit_median)

    # ------------------------------------------------------------------
    # Step 4: Load reference and compute least-squares comparison
    # ------------------------------------------------------------------
    fp_chart = (
        pl.read_csv(
            PROCESSED_DIR / "5_year_av_chart.csv",
            schema_overrides={"JJ Val": pl.Float64},
        )
        .rename({"Pk": "pick", "FP Val": "fp_val"})
        .select(["pick", "fp_val"])
    )

    fp_vals = (
        df_mean.join(fp_chart, on="pick", how="inner")["fp_val"].to_numpy()
    )

    metrics = {
        "Individual fit": _least_squares(df_individual.join(fp_chart, on="pick", how="inner"), fp_vals),
        "Mean fit":       _least_squares(df_mean.join(fp_chart, on="pick", how="inner"), fp_vals),
        "Median fit":     _least_squares(df_median.join(fp_chart, on="pick", how="inner"), fp_vals),
    }

    print(f"\nLeast-Squares Comparison vs. Football Perspectives ({len(fp_vals)} picks):")
    print(f"  {'Method':<20} {'R²':>7}  {'RMSE':>7}  {'SS_res':>10}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*10}")
    for name, m in metrics.items():
        print(f"  {name:<20} {m['r2']:>7.4f}  {m['rmse']:>7.4f}  {m['ss_res']:>10.2f}")

    # ------------------------------------------------------------------
    # Step 5: Save fitted curves
    # ------------------------------------------------------------------
    df_individual.write_csv(PROCESSED_DIR / "exp_fit_5yr_av_marginal_individual.csv")
    df_mean.write_csv(PROCESSED_DIR / "exp_fit_5yr_av_marginal_mean.csv")
    df_median.write_csv(PROCESSED_DIR / "exp_fit_5yr_av_marginal_median.csv")
    print(f"\nSaved fitted curves → {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/exp_fit_5yr_av_marginal_*.csv")

    # ------------------------------------------------------------------
    # Step 6: Build Plotly comparison figure
    # ------------------------------------------------------------------
    print("Building comparison figure...")

    adj_means = stats_adjusted.filter(pl.col("Pick") <= MAX_PICK).sort("Pick")

    def _band_xy(df: pl.DataFrame) -> tuple[list, list]:
        x = df["pick"].to_numpy()
        return (
            np.concatenate([x, x[::-1]]).tolist(),
            np.concatenate([df["y_upper"].to_numpy(), df["y_lower"].to_numpy()[::-1]]).tolist(),
        )

    fig = go.Figure()

    # 1-sigma bands (behind the curves, very transparent)
    for df, color, label in [
        (df_individual, _COLOR_INDIVIDUAL, "Individual"),
        (df_mean,       _COLOR_MEAN,       "Mean"),
        (df_median,     _COLOR_MEDIAN,     "Median"),
    ]:
        bx, by = _band_xy(df)
        r, g, b_val = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b_val},0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=label,
        ))

    # Raw adjusted per-pick means (show underlying data scatter)
    fig.add_trace(go.Scatter(
        x=adj_means["Pick"].to_list(),
        y=adj_means["mean"].to_list(),
        mode="markers",
        marker=dict(color="#aec7e8", size=4, opacity=0.6),
        name="Adjusted per-pick mean (1980–2007)",
    ))

    # Three fit curves
    r2_ind = metrics["Individual fit"]["r2"]
    r2_mn  = metrics["Mean fit"]["r2"]
    r2_med = metrics["Median fit"]["r2"]

    fig.add_trace(go.Scatter(
        x=df_individual["pick"].to_list(), y=df_individual["y_fit"].to_list(),
        mode="lines", line=dict(color=_COLOR_INDIVIDUAL, width=2),
        name=f"Individual player fit (R²={r2_ind:.3f})",
        legendgroup="Individual",
    ))
    fig.add_trace(go.Scatter(
        x=df_mean["pick"].to_list(), y=df_mean["y_fit"].to_list(),
        mode="lines", line=dict(color=_COLOR_MEAN, width=2),
        name=f"Mean fit (R²={r2_mn:.3f})",
        legendgroup="Mean",
    ))
    fig.add_trace(go.Scatter(
        x=df_median["pick"].to_list(), y=df_median["y_fit"].to_list(),
        mode="lines", line=dict(color=_COLOR_MEDIAN, width=2),
        name=f"Median fit (R²={r2_med:.3f})",
        legendgroup="Median",
    ))

    # Football Perspectives reference curve
    fig.add_trace(go.Scatter(
        x=fp_chart["pick"].to_list(), y=fp_chart["fp_val"].to_list(),
        mode="lines", line=dict(color=_COLOR_FP_REF, width=2, dash="dash"),
        name="Football Perspectives (Stewart 2013)",
    ))

    fig.update_layout(
        title=(
            "Football Perspectives Draft Value Chart — Three-Method Recreation<br>"
            f"<sup>1980–2007 drafts · 5-year AV · −{MARGINAL_ADJUSTMENT} marginal adjustment · "
            f"R²: individual={r2_ind:.3f}, mean={r2_mn:.3f}, median={r2_med:.3f}</sup>"
        ),
        xaxis=dict(title="Overall Pick", range=[1, MAX_PICK]),
        yaxis=dict(title="Marginal 5-Year AV"),
        template="plotly_white",
        legend=dict(groupclick="toggleitem"),
    )

    out_html = FIGURES_DIR / "fp_av_recreation.html"
    fig.write_html(str(out_html))
    print(f"Saved figure       → {out_html.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
