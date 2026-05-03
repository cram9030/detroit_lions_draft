"""Recreate Chase Stewart's Football Perspectives NFL draft AV value chart.

Source: https://www.footballperspective.com/creating-a-nfl-draft-value-chart-part-i/

Stewart graded each player drafted 1980–2007 on their first five seasons of AV
(excluding seasons with AV < 2), fitted a logarithmic decay curve to the
individual player data, then normalized so that pick 224 = 0.1.

This script:
1. Aggregates per-player 5-year AV for the 1980–2007 cohort (min season AV = 2).
2. Fits a logarithmic decay model: f(pick) = a·ln(pick) + b.
3. Normalizes the curve so pick 224 = 0.1.
4. Compares to data/processed/5_year_av_chart.csv via least squares.
5. Saves the fitted curve and an interactive Plotly figure.
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
    logarithmic_av_fit,
    logarithmic_fit_result_to_dataframe,
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
MIN_SEASON_AV = 2           # exclude seasons below this threshold (Stewart's rule)
MAX_PICK = 224              # published FP chart covers picks 1–224

_COLOR_FIT = "#76b7b2"   # teal  — individual player logarithmic fit
_COLOR_REF = "#e15759"   # red   — Football Perspectives reference


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
    # Step 1: Aggregate per-player 5-year AV (1980–2007, min season AV = 2)
    # ------------------------------------------------------------------
    print(f"Loading AV data (draft years {DRAFT_YEAR_START}–{DRAFT_YEAR_END}, "
          f"first {MAX_SEASONS_FROM_DRAFT} seasons, min season AV = {MIN_SEASON_AV})...")

    player_av_lf = _aggregate_player_av(
        _prepare_av_data(load_parquets_from_dir(RAW_DIR, lazy=True)).filter(
            (pl.col("Draft Year") >= DRAFT_YEAR_START)
            & (pl.col("Draft Year") <= DRAFT_YEAR_END)
        ),
        max_seasons_from_draft=MAX_SEASONS_FROM_DRAFT,
        min_season_av=MIN_SEASON_AV,
    )

    # ------------------------------------------------------------------
    # Step 2: Fit logarithmic decay to individual player AV
    # ------------------------------------------------------------------
    print("Fitting logarithmic decay curve...")
    fit = logarithmic_av_fit(player_av_lf.filter(pl.col("Pick")<=224), max_pick=MAX_PICK)
    a, b = fit["popt"]
    print(f"  f(pick) = {a:.3f} * ln(pick) + {b:.3f}")

    df_fit = logarithmic_fit_result_to_dataframe(fit)

    # ------------------------------------------------------------------
    # Step 3: Normalize so pick 224 = 0.1
    # ------------------------------------------------------------------
    shift = df_fit.filter(pl.col("pick") == MAX_PICK)["y_fit"][0] - 0.1
    df_fit = df_fit.with_columns(
        [(pl.col(c) - shift).alias(c) for c in ("y_fit", "y_upper", "y_lower")]
    )

    # ------------------------------------------------------------------
    # Step 4: Compare to Football Perspectives reference
    # ------------------------------------------------------------------
    fp_chart = (
        pl.read_csv(
            PROCESSED_DIR / "5_year_av_chart.csv",
            schema_overrides={"JJ Val": pl.Float64},
        )
        .rename({"Pk": "pick", "FP Val": "fp_val"})
        .select(["pick", "fp_val"])
    )

    fp_vals = df_fit.join(fp_chart, on="pick", how="inner")["fp_val"].to_numpy()
    m = _least_squares(df_fit.join(fp_chart, on="pick", how="inner"), fp_vals)

    print(f"\nLeast-Squares vs. Football Perspectives ({len(fp_vals)} picks):")
    print(f"  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  SS_res={m['ss_res']:.2f}")

    # ------------------------------------------------------------------
    # Step 5: Save fitted curve
    # ------------------------------------------------------------------
    df_fit.write_csv(PROCESSED_DIR / "fp_av_recreation_log_fit.csv")
    print(f"\nSaved fitted curve → {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/fp_av_recreation_log_fit.csv")

    # ------------------------------------------------------------------
    # Step 6: Build Plotly figure
    # ------------------------------------------------------------------
    print("Building figure...")

    x = df_fit["pick"].to_numpy()
    band_x = np.concatenate([x, x[::-1]]).tolist()
    band_y = np.concatenate([
        df_fit["y_upper"].to_numpy(),
        df_fit["y_lower"].to_numpy()[::-1],
    ]).tolist()

    r, g, b_val = int(_COLOR_FIT[1:3], 16), int(_COLOR_FIT[3:5], 16), int(_COLOR_FIT[5:7], 16)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself",
        fillcolor=f"rgba({r},{g},{b_val},0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_fit["pick"].to_list(), y=df_fit["y_fit"].to_list(),
        mode="lines", line=dict(color=_COLOR_FIT, width=2),
        name=f"Individual log fit (R²={m['r2']:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=fp_chart["pick"].to_list(), y=fp_chart["fp_val"].to_list(),
        mode="lines", line=dict(color=_COLOR_REF, width=2, dash="dash"),
        name="Football Perspectives (Stewart 2013)",
    ))

    fig.update_layout(
        title=(
            "Football Perspectives Draft Value Chart — Recreation<br>"
            f"<sup>1980–{DRAFT_YEAR_END} drafts · 5-year AV · min season AV = {MIN_SEASON_AV} · "
            f"normalized pick {MAX_PICK} = 0.1 · R²={m['r2']:.3f}</sup>"
        ),
        xaxis=dict(title="Overall Pick", range=[1, MAX_PICK]),
        yaxis=dict(title="Normalized Pick Value"),
        template="plotly_white",
    )

    out_html = FIGURES_DIR / "fp_av_recreation.html"
    fig.write_html(str(out_html))
    print(f"Saved figure       → {out_html.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
