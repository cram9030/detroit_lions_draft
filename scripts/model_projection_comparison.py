"""Per-player projection comparison for any team/draft class.

For each player in the specified draft class, this script generates career
projections using 1, 2, and 3 years of observed AV as model input (up to however
many seasons are available), then plots actual vs. projected AV as line plots per
player.

Usage
-----
    python scripts/model_projection_comparison.py --year 2022 [--team DET] [--model {parametric,knn,ridge,all}]

Defaults to team ``DET`` and all three models on each plot.

Prerequisites
-------------
Trained model artifacts must exist:
    models/parametric/params.json
    models/knn/_config.joblib
    models/ridge/_config.joblib

Train with:
    python scripts/train_models.py --model parametric
    python scripts/train_models.py --model knn
    python scripts/train_models.py --model ridge

Outputs
-------
    outputs/figures/{team}_{year}_{player_slug}_projection.html  (one per player)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import plotly.graph_objects as go

from src.models.factory import make_career_av_model
from src.surplus_av import _normalize_pos, load_team_draft_class

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"

_YEARS = [0, 1, 2, 3]

_MODEL_COLORS = {
    "parametric": "#1f77b4",
    "knn":        "#2ca02c",
    "ridge":      "#9467bd",
}
_WINDOW_DASH = {1: "dash", 2: "dot", 3: "dashdot"}
_WINDOW_LABEL = {1: "1yr input", 2: "2yr input", 3: "3yr input"}

_PLAYER_POSITION_OVERRIDES: dict[str, str] = {}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _check_prerequisites(model_names: list[str]) -> None:
    checks = {
        "parametric": MODELS_DIR / "parametric" / "params.json",
        "knn":        MODELS_DIR / "knn" / "_config.joblib",
        "ridge":      MODELS_DIR / "ridge" / "_config.joblib",
    }
    for name in model_names:
        path = checks[name]
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model not found at {path}\n"
                f"Train it first:\n  python scripts/train_models.py --model {name}"
            )


def _build_trajectory(
    obs_av: list[float],
    n_input: int,
    predicted_years: list[int],
    y_pred: list[float],
    y_upper: list[float],
    y_lower: list[float],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Build a 4-element trajectory combining observed window + model predictions.

    Observed years (< n_input) use actual values; predicted years use model output.
    Years the model doesn't cover are None (rendered as gaps in Plotly lines).
    """
    pred_map  = dict(zip(predicted_years, y_pred))
    upper_map = dict(zip(predicted_years, y_upper))
    lower_map = dict(zip(predicted_years, y_lower))

    y, y_up, y_lo = [], [], []
    for yr in _YEARS:
        if yr < n_input:
            y.append(obs_av[yr])
            y_up.append(None)
            y_lo.append(None)
        elif yr in pred_map:
            y.append(pred_map[yr])
            y_up.append(upper_map[yr])
            y_lo.append(lower_map[yr])
        else:
            y.append(None)
            y_up.append(None)
            y_lo.append(None)
    return y, y_up, y_lo


def _make_player_figure(
    player: str,
    pos: str,
    pick: int,
    year: int,
    team: str,
    actual: list[float],
    projections: dict[str, dict[int, tuple[list, list, list]]],
) -> go.Figure:
    """Build per-player line plot: actual AV vs. model projections at 1/2/3yr input.

    projections: {model_name: {n_input: (y, y_upper, y_lower)}}
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=_YEARS,
        y=actual,
        mode="lines+markers",
        name="Actual AV",
        line=dict(color="#222222", width=3),
        marker=dict(size=9),
    ))

    for model_name, window_data in projections.items():
        color = _MODEL_COLORS[model_name]
        rgba_band = _hex_to_rgba(color, 0.12)

        for n_input, (y, y_up, y_lo) in sorted(window_data.items()):
            label = f"{model_name} ({_WINDOW_LABEL[n_input]})"

            fig.add_trace(go.Scatter(
                x=_YEARS,
                y=y,
                mode="lines+markers",
                name=label,
                line=dict(color=color, dash=_WINDOW_DASH[n_input], width=2),
                marker=dict(size=5),
            ))

            # Shaded confidence band over the predicted portion only
            pred_x  = [yr for yr in _YEARS if y_up[yr] is not None]
            pred_up = [y_up[yr] for yr in _YEARS if y_up[yr] is not None]
            pred_lo = [y_lo[yr] for yr in _YEARS if y_lo[yr] is not None]

            if pred_x:
                fig.add_trace(go.Scatter(
                    x=pred_x, y=pred_lo,
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=pred_x, y=pred_up,
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=rgba_band,
                    showlegend=False, hoverinfo="skip",
                ))

    fig.update_layout(
        title=f"{player} (Pick {pick}, {pos}) — {year} {team} Draft | Actual vs. Projections",
        xaxis=dict(
            title="Year from Draft",
            tickvals=_YEARS,
            ticktext=["Yr 0", "Yr 1", "Yr 2", "Yr 3"],
        ),
        yaxis_title="Annual AV",
        legend=dict(orientation="v", x=1.02, xanchor="left"),
        height=480,
        width=860,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft class — per-player projection comparison plots"
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Draft year, e.g. 2022",
    )
    parser.add_argument(
        "--team",
        default="DET",
        help="Three-letter team code (default: DET)",
    )
    parser.add_argument(
        "--model",
        choices=["parametric", "knn", "ridge", "all"],
        default="all",
        help="Which model(s) to include (default: all)",
    )
    args = parser.parse_args()

    model_names = (
        ["parametric", "knn", "ridge"] if args.model == "all" else [args.model]
    )

    _check_prerequisites(model_names)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.team} {args.year} observed AV...")
    obs_df = load_team_draft_class(args.team, args.year)

    available_years = sorted(obs_df["years_from_draft"].unique().to_list())
    n_obs_global = len(available_years)

    wide = (
        obs_df
        .pivot(index=["Player", "Pos", "Pick"], on="years_from_draft", values="AV.1")
        .rename({str(yr): f"obs_yr{yr}" for yr in available_years})
    )
    # Fill null for available seasons (0 AV that year) and add None columns for
    # seasons not yet in the data (future years — should not appear as 0 in plots).
    fill_exprs = [pl.col(f"obs_yr{yr}").fill_null(0.0) for yr in available_years]
    null_exprs = [
        pl.lit(None).cast(pl.Float64).alias(f"obs_yr{yr}")
        for yr in range(4) if yr not in available_years
    ]
    wide = wide.with_columns(fill_exprs + null_exprs).sort("Pick")
    print(f"  {len(wide)} {args.team} {args.year} picks loaded ({n_obs_global} seasons available)")

    print(f"Loading model(s): {model_names}")
    models = {}
    for name in model_names:
        m = make_career_av_model(name)
        m.load(MODELS_DIR / name)
        models[name] = m

    saved = []
    for row in wide.iter_rows(named=True):
        player = row["Player"]
        pos    = row["Pos"]
        pick   = row["Pick"]
        actual = [row["obs_yr0"], row["obs_yr1"], row["obs_yr2"], row["obs_yr3"]]
        norm_pos = _normalize_pos(player, pos, _PLAYER_POSITION_OVERRIDES)

        print(f"\n  {player} ({pos}, pick {pick})")

        projections: dict[str, dict[int, tuple]] = {}

        for model_name, model in models.items():
            if norm_pos is None:
                print(f"    [{model_name}] skipped — position '{pos}' not resolvable")
                continue

            window_data: dict[int, tuple] = {}
            for n_input in range(1, min(n_obs_global, 3) + 1):
                obs_input = actual[:n_input]
                try:
                    result = model.predict(norm_pos, obs_input)
                except Exception as exc:
                    print(f"    [{model_name} {n_input}yr] predict failed: {exc}")
                    continue
                traj = _build_trajectory(
                    actual, n_input,
                    result["predicted_years"],
                    result["y_pred"],
                    result["y_upper"],
                    result["y_lower"],
                )
                window_data[n_input] = traj
            projections[model_name] = window_data

        fig = _make_player_figure(player, pos, pick, args.year, args.team, actual, projections)
        slug = player.lower().replace(" ", "_")
        out_path = FIGURES_DIR / f"{args.team.lower()}_{args.year}_{slug}_projection.html"
        fig.write_html(str(out_path))
        print(f"    Saved {out_path.name}")
        saved.append(out_path.name)

    print(f"\nDone — {len(saved)} plots written to {FIGURES_DIR}/")
    for name in saved:
        print(f"  {name}")


if __name__ == "__main__":
    main()
