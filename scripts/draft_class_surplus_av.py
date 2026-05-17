"""Draft class surplus AV analysis.

Loads a team's draft class, aggregates observed + projected 4-year AV, and
computes each player's surplus relative to the Expected AV Above Replacement
(EAVAR) for their pick.  Outputs a console table and two HTML charts.

Prerequisites
-------------
1. **Draft data** — the raw store must contain at least two completed seasons::

       data/raw/stathead/annual_av/draft{YEAR}_season{YEAR}.parquet
       data/raw/stathead/annual_av/draft{YEAR}_season{YEAR+1}.parquet

2. **Trained model** — e.g. for ``--model parametric``::

       python scripts/train_models.py --model parametric

3. **Processed EAVAR** — ``data/processed/expected_av_above_replacement.csv``::

       python scripts/run_analysis.py

Usage
-----
    python scripts/draft_class_surplus_av.py --team DET --year 2024
    python scripts/draft_class_surplus_av.py --team DET --year 2024 --model knn

Outputs
-------
- Console: per-player table + class summary
- ``outputs/figures/{TEAM}_{YEAR}_player_surplus_av.html``
- ``outputs/figures/{TEAM}_{YEAR}_class_surplus_av.html``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import polars as pl

from src.models.factory import make_career_av_model
from src.surplus_av import aggregate_4yr_av, compute_surplus_av, load_team_draft_class

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draft class surplus AV analysis")
    p.add_argument("--team", required=True, help="Three-letter team code, e.g. DET")
    p.add_argument("--year", required=True, type=int, help="Draft year, e.g. 2024")
    p.add_argument(
        "--model",
        default="parametric",
        choices=["parametric", "knn", "ridge"],
        help="Career AV model to use for projections (default: parametric)",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing trained model artefacts (default: models/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write HTML charts (default: outputs/figures/)",
    )
    return p.parse_args()


def _check_model(model_name: str, models_dir: Path) -> None:
    """Raise FileNotFoundError with a helpful message if the model is missing."""
    paths = {
        "parametric": models_dir / "parametric" / "params.json",
        "knn": models_dir / "knn" / "_config.joblib",
        "ridge": models_dir / "ridge" / "_config.joblib",
    }
    path = paths[model_name]
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found at {path}\n"
            f"Train it first:\n"
            f"  python scripts/train_models.py --model {model_name}"
        )


def main() -> None:
    args = parse_args()
    team = args.team.upper()
    year = args.year
    model_name = args.model
    models_dir = args.models_dir or MODELS_DIR
    output_dir = args.output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _check_model(model_name, models_dir)

    print(f"Loading {team} {year} draft class...")
    draft_df = load_team_draft_class(team, year)
    n_players = draft_df["Player"].n_unique()
    print(f"  {n_players} players found")

    print(f"Loading {model_name} model...")
    model = make_career_av_model(model_name)
    model.load(models_dir / model_name)

    print("Aggregating 4-year AV...")
    players_df = aggregate_4yr_av(draft_df, model)

    print("Computing surplus AV vs EAVAR...")
    results = compute_surplus_av(players_df)

    # ------------------------------------------------------------------
    # Console table
    # ------------------------------------------------------------------
    display_cols = [
        "Player", "Pos", "Pick",
        "obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3",
        "proj_yr2", "proj_yr3",
        "total_4yr_av", "eavar", "surplus_av", "is_projected",
    ]
    print()
    print("=" * 120)
    print(f"{team} {year} Draft Class — 4-Year AV vs EAVAR (model: {model_name})")
    print("=" * 120)
    with pl.Config(tbl_width_chars=250, tbl_cols=-1):
        print(results.select([c for c in display_cols if c in results.columns]))

    # Class totals
    valid = results.filter(pl.col("surplus_av").is_not_null())
    class_surplus = valid["surplus_av"].sum()
    class_total_av = valid["total_4yr_av"].sum()
    class_eavar = valid["eavar"].sum()
    n_projected = results["is_projected"].sum()

    print(f"\nClass totals ({len(valid)} players with EAVAR data, {n_projected} projected):")
    print(f"  Total 4yr AV  : {class_total_av:.1f}")
    print(f"  Total EAVAR   : {class_eavar:.1f}")
    print(f"  Class surplus : {class_surplus:+.1f}")

    # ------------------------------------------------------------------
    # Per-player chart
    # ------------------------------------------------------------------
    players = results["Player"].to_list()
    total_av_vals = results["total_4yr_av"].to_list()
    eavar_vals = [v if v is not None else 0.0 for v in results["eavar"].to_list()]
    surplus_vals = [v if v is not None else 0.0 for v in results["surplus_av"].to_list()]

    fig_players = go.Figure()
    fig_players.add_bar(
        name="Total 4yr AV",
        x=players,
        y=total_av_vals,
        marker_color="#1f77b4",
    )
    fig_players.add_bar(
        name="EAVAR (pick expectation)",
        x=players,
        y=eavar_vals,
        marker_color="#ff7f0e",
    )
    fig_players.update_layout(
        title=f"{team} {year} Draft — 4yr AV vs EAVAR per player (model: {model_name})",
        xaxis_title="Player",
        yaxis_title="4-Year AV",
        barmode="group",
        xaxis_tickangle=-35,
        height=520,
    )
    player_chart = output_dir / f"{team}_{year}_player_surplus_av.html"
    fig_players.write_html(str(player_chart))
    print(f"\n  Saved {player_chart.name}")

    # ------------------------------------------------------------------
    # Per-player surplus chart
    # ------------------------------------------------------------------
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in surplus_vals]
    fig_surplus = go.Figure(
        go.Bar(
            x=players,
            y=surplus_vals,
            marker_color=colors,
            text=[f"{v:+.1f}" for v in surplus_vals],
            textposition="outside",
        )
    )
    fig_surplus.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    fig_surplus.update_layout(
        title=(
            f"{team} {year} Draft — Surplus AV per player "
            f"(class total: {class_surplus:+.1f}, model: {model_name})"
        ),
        xaxis_title="Player",
        yaxis_title="Surplus AV (total 4yr AV − EAVAR)",
        xaxis_tickangle=-35,
        height=520,
    )
    class_chart = output_dir / f"{team}_{year}_class_surplus_av.html"
    fig_surplus.write_html(str(class_chart))
    print(f"  Saved {class_chart.name}")


if __name__ == "__main__":
    main()
