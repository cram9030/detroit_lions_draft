"""Detroit Lions 2024 draft class — projected 4-year AV vs pick expectation.

This script loads observed AV for the Lions' 2024 draft picks (years 0 and 1),
projects years 2–3 using the trained ParametricCurveModel, KNNTrajectoryModel,
and RidgeRegressionModel, and compares each model's projected 4-year total to
the historical expectation derived from pick position.

Prerequisites
-------------
1. **2024 draft data** — the raw store must contain:
       data/raw/stathead/annual_av/draft2024_season2024.parquet
       data/raw/stathead/annual_av/draft2024_season2025.parquet

   To download, update ``config/stathead_annual_av.json``::

       "draft_year_start": 2024,
       "draft_year_end":   2024

   then run::

       python -m src.stathead_downloader --config config/stathead_annual_av.json

2. **Trained models** — ``models/parametric/params.json``,
   ``models/knn/_config.joblib``, and ``models/ridge/_config.joblib`` must exist::

       python scripts/train_models.py --model parametric
       python scripts/train_models.py --model knn
       python scripts/train_models.py --model ridge

Outputs
-------
- Console: per-player comparison table + class-level summary
- ``outputs/figures/lions_2024_player_comparison.html``
- ``outputs/figures/lions_2024_class_comparison.html``
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import plotly.graph_objects as go

from src.annual_av_analysis import exponential_av_fit_stat
from src.models.factory import make_career_av_model
from src.surplus_av import _normalize_pos, load_team_draft_class, project_player_seasons

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

_DRAFT_YEAR = 2024

# Generalist codes (OL, DL) aren't resolvable without knowing the specific position.
# Override per-player where the actual position is known.
_PLAYER_POSITION_OVERRIDES: dict[str, str] = {
    "Giovanni Manu": "OT",
    "Mekhi Wingo": "DT",
    "Christian Mahogany": "OG",
}


def _check_prerequisites() -> None:
    params_path = MODELS_DIR / "parametric" / "gamma" / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Parametric model not found at {params_path}\n"
            "Train it first:\n"
            "  python scripts/train_models.py --model parametric --curve gamma"
        )

    knn_path = MODELS_DIR / "knn" / "_config.joblib"
    if not knn_path.exists():
        raise FileNotFoundError(
            f"KNN model not found at {knn_path}\n"
            "Train it first:\n"
            "  python scripts/train_models.py --model knn"
        )

    ridge_path = MODELS_DIR / "ridge" / "_config.joblib"
    if not ridge_path.exists():
        raise FileNotFoundError(
            f"Ridge model not found at {ridge_path}\n"
            "Train it first:\n"
            "  python scripts/train_models.py --model ridge"
        )




def _build_pick_expectation(max_pick: int = 260) -> dict[int, float]:
    """Return expected 4-year cumulative AV per pick number from the exp-fit model."""
    pick_stats_path = PROCESSED_DIR / "pick_stats.csv"
    if not pick_stats_path.exists():
        raise FileNotFoundError(
            f"pick_stats.csv not found at {pick_stats_path}\n"
            "Run the analysis pipeline first:\n"
            "  python scripts/run_analysis.py"
        )

    stats_df = pl.read_csv(pick_stats_path)
    fit = exponential_av_fit_stat(stats_df, stat_col="mean", max_pick=max_pick)
    import numpy as np
    a, b, c = fit["popt"]
    picks = stats_df.filter(pl.col("Pick") <= max_pick)["Pick"].to_list()
    return {int(p): float(max(0.0, a * np.exp(-b * p) + c)) for p in picks}



def main() -> None:
    _check_prerequisites()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Lions 2024 observed AV...")
    obs_df = load_team_draft_class("DET", _DRAFT_YEAR)

    # Pivot to one row per player with yr0 and yr1 AV
    wide = (
        obs_df
        .pivot(index=["Player", "Pos", "Pick"], on="years_from_draft", values="AV.1")
        .rename({"0": "obs_yr0", "1": "obs_yr1"})
        .with_columns([
            pl.col("obs_yr0").fill_null(0.0),
            pl.col("obs_yr1").fill_null(0.0),
        ])
        .sort("Pick")
    )

    print(f"  {len(wide)} Lions 2024 picks with observed data")

    print("Loading models...")
    _models = {}
    for _name in ("parametric", "knn", "ridge"):
        _m = make_career_av_model(_name)
        load_path = MODELS_DIR / "parametric" / "gamma" if _name == "parametric" else MODELS_DIR / _name
        _m.load(load_path)
        _models[_name] = _m
    parametric_model = _models["parametric"]
    knn_model = _models["knn"]
    ridge_model = _models["ridge"]

    print("Building pick expectation baseline...")
    pick_exp = _build_pick_expectation()

    # Build comparison table
    rows = []
    for row in wide.iter_rows(named=True):
        player = row["Player"]
        pos = row["Pos"]
        pick = row["Pick"]
        obs0 = row["obs_yr0"]
        obs1 = row["obs_yr1"]
        obs_av = [obs0, obs1]

        pick_int = int(pick) if pick is not None else None

        # Parametric projection
        param_proj = project_player_seasons(parametric_model, player, pos, obs_av, _PLAYER_POSITION_OVERRIDES, pick=pick_int)
        if param_proj is None:
            param_yr2, param_yr3 = 0.0, 0.0
            param_note = f"(pos '{_normalize_pos(player, pos, _PLAYER_POSITION_OVERRIDES)}' not in parametric model)"
        else:
            param_yr2, param_yr3 = param_proj
            param_note = ""

        # KNN projection
        knn_proj = project_player_seasons(knn_model, player, pos, obs_av, _PLAYER_POSITION_OVERRIDES, pick=pick_int)
        if knn_proj is None:
            knn_yr2, knn_yr3 = 0.0, 0.0
            knn_note = f"(pos '{_normalize_pos(player, pos, _PLAYER_POSITION_OVERRIDES)}' not in KNN model)"
        else:
            knn_yr2, knn_yr3 = knn_proj
            knn_note = ""

        # Ridge projection
        ridge_proj = project_player_seasons(ridge_model, player, pos, obs_av, _PLAYER_POSITION_OVERRIDES, pick=pick_int)
        if ridge_proj is None:
            ridge_yr2, ridge_yr3 = 0.0, 0.0
            ridge_note = f"(pos '{_normalize_pos(player, pos, _PLAYER_POSITION_OVERRIDES)}' not in ridge model)"
        else:
            ridge_yr2, ridge_yr3 = ridge_proj
            ridge_note = ""

        param_4yr = obs0 + obs1 + param_yr2 + param_yr3
        knn_4yr = obs0 + obs1 + knn_yr2 + knn_yr3
        ridge_4yr = obs0 + obs1 + ridge_yr2 + ridge_yr3
        exp_4yr = pick_exp.get(int(pick), 0.0) if pick is not None else 0.0

        rows.append({
            "Player": player,
            "Pos": pos,
            "Pick": pick,
            "Obs yr0": round(obs0, 1),
            "Obs yr1": round(obs1, 1),
            "Param yr2": round(param_yr2, 1),
            "Param yr3": round(param_yr3, 1),
            "Cum Param 4yr": round(param_4yr, 1),
            "KNN yr2": round(knn_yr2, 1),
            "KNN yr3": round(knn_yr3, 1),
            "Cum KNN 4yr": round(knn_4yr, 1),
            "Ridge yr2": round(ridge_yr2, 1),
            "Ridge yr3": round(ridge_yr3, 1),
            "Cum Ridge 4yr": round(ridge_4yr, 1),
            "Exp 4yr": round(exp_4yr, 1),
            "Delta (param)": round(param_4yr - exp_4yr, 1),
            "Delta (knn)": round(knn_4yr - exp_4yr, 1),
            "Delta (ridge)": round(ridge_4yr - exp_4yr, 1),
            "_param_note": param_note,
            "_knn_note": knn_note,
            "_ridge_note": ridge_note,
        })

    results_df = pl.DataFrame(rows)

    print("\n" + "=" * 110)
    print("Detroit Lions 2024 Draft Class — Parametric vs KNN vs Ridge vs Pick Expectation (4-Year AV)")
    print("=" * 110)
    display_cols = [
        "Player", "Pos", "Pick",
        "Obs yr0", "Obs yr1",
        "Param yr2", "Param yr3", "Cum Param 4yr",
        "KNN yr2", "KNN yr3", "Cum KNN 4yr",
        "Ridge yr2", "Ridge yr3", "Cum Ridge 4yr",
        "Exp 4yr", "Delta (param)", "Delta (knn)", "Delta (ridge)",
    ]
    with pl.Config(tbl_width_chars=250, tbl_cols=-1):
        print(results_df.select(display_cols))

    for row in results_df.iter_rows(named=True):
        if row["_param_note"]:
            print(f"  Note — {row['Player']} (parametric): {row['_param_note']}")
        if row["_knn_note"]:
            print(f"  Note — {row['Player']} (KNN): {row['_knn_note']}")
        if row["_ridge_note"]:
            print(f"  Note — {row['Player']} (ridge): {row['_ridge_note']}")

    # Class summary
    param_total = results_df["Cum Param 4yr"].sum()
    knn_total = results_df["Cum KNN 4yr"].sum()
    ridge_total = results_df["Cum Ridge 4yr"].sum()
    class_exp = results_df["Exp 4yr"].sum()

    def _pct(val: float) -> str:
        if class_exp > 0:
            return f"{100 * (val - class_exp) / class_exp:+.1f}%"
        return "n/a"

    print(f"\nClass totals:")
    print(f"  Parametric 4yr AV : {param_total:.1f}  (Δ = {param_total - class_exp:+.1f}, {_pct(param_total)})")
    print(f"  KNN 4yr AV        : {knn_total:.1f}  (Δ = {knn_total - class_exp:+.1f}, {_pct(knn_total)})")
    print(f"  Ridge 4yr AV      : {ridge_total:.1f}  (Δ = {ridge_total - class_exp:+.1f}, {_pct(ridge_total)})")
    print(f"  Expected by pick  : {class_exp:.1f}")

    # ------------------------------------------------------------------
    # Player comparison chart
    # ------------------------------------------------------------------
    players = results_df["Player"].to_list()
    param_vals = results_df["Cum Param 4yr"].to_list()
    knn_vals = results_df["Cum KNN 4yr"].to_list()
    ridge_vals = results_df["Cum Ridge 4yr"].to_list()
    exp_vals = results_df["Exp 4yr"].to_list()

    fig_players = go.Figure()
    fig_players.add_bar(name="Parametric 4yr AV", x=players, y=param_vals, marker_color="#1f77b4")
    fig_players.add_bar(name="KNN 4yr AV", x=players, y=knn_vals, marker_color="#2ca02c")
    fig_players.add_bar(name="Ridge 4yr AV", x=players, y=ridge_vals, marker_color="#9467bd")
    fig_players.add_bar(name="Expected by Pick", x=players, y=exp_vals, marker_color="#ff7f0e")
    fig_players.update_layout(
        title="Lions 2024 Draft — Parametric vs KNN vs Ridge vs Pick Expectation (per player)",
        xaxis_title="Player",
        yaxis_title="4-Year AV",
        barmode="group",
        xaxis_tickangle=-35,
        template="plotly_white",
    )
    fig_players.write_html(str(FIGURES_DIR / "lions_2024_player_comparison.html"))
    print("\n  Saved lions_2024_player_comparison.html")

    # ------------------------------------------------------------------
    # Class-level bar chart
    # ------------------------------------------------------------------
    fig_class = go.Figure()
    fig_class.add_bar(
        x=["Parametric", "KNN", "Ridge", "Expected by Pick"],
        y=[param_total, knn_total, ridge_total, class_exp],
        marker_color=["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"],
        text=[f"{param_total:.1f}", f"{knn_total:.1f}", f"{ridge_total:.1f}", f"{class_exp:.1f}"],
        textposition="outside",
    )
    fig_class.update_layout(
        title=(
            f"Lions 2024 Draft Class — Total 4yr AV  "
            f"(Param Δ = {param_total - class_exp:+.1f}, "
            f"KNN Δ = {knn_total - class_exp:+.1f}, "
            f"Ridge Δ = {ridge_total - class_exp:+.1f})"
        ),
        yaxis_title="Total 4-Year AV",
        template="plotly_white",
    )
    fig_class.write_html(str(FIGURES_DIR / "lions_2024_class_comparison.html"))
    print("  Saved lions_2024_class_comparison.html")


if __name__ == "__main__":
    main()
