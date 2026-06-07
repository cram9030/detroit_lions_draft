"""Population-mean career AV curves for each parametric model variant.

Plots the default (unscaled, population-level) curve that each trained
parametric model has fitted per position. Each output file shows all
requested positions on one plot, colored by the Viridis palette.
One HTML is produced per curve variant.

Usage
-----
    python scripts/plot_parametric_curves.py
    python scripts/plot_parametric_curves.py --curves gamma exp_decay --positions QB WR RB
    python scripts/plot_parametric_curves.py --no-bands
    python scripts/plot_parametric_curves.py --output-dir outputs/figures

Defaults
--------
    --curves      auto-discovered from models/parametric/ subdirectories
    --positions   all (every position present in the loaded models)
    --output-dir  outputs/figures

Outputs
-------
    {output_dir}/parametric_curves_{curve}.html  (one per curve variant)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from src.models.parametric import ParametricCurveModel
from src.models.utils import discover_parametric_curves
from src.positions import _POSITION_ORDER

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def _rgb_str_to_rgba(rgb_str: str, alpha: float) -> str:
    """Convert plotly 'rgb(r,g,b)' string to 'rgba(r,g,b,alpha)'."""
    nums = re.findall(r"[\d.]+", rgb_str)
    r, g, b = int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))
    return f"rgba({r},{g},{b},{alpha})"


def _check_prerequisites(curves: list[str]) -> None:
    missing = [c for c in curves if not (MODELS_DIR / "parametric" / c / "params.json").exists()]
    if missing:
        cmds = "\n".join(
            f"  python scripts/train_models.py --model parametric --curve {c}"
            for c in missing
        )
        raise FileNotFoundError(
            f"Trained model(s) not found for: {missing}\nTrain them first:\n{cmds}"
        )


def _get_default_curve(
    model: ParametricCurveModel, pos: str
) -> tuple[list[float], list[float], list[float]]:
    """Evaluate the unscaled population-mean curve and its ±1σ bands for pos."""
    model_fn, _, _ = model._curve_descriptor
    popt = np.array(model._params[pos]["popt"])
    pcov = np.array(model._params[pos]["pcov"])

    t = np.arange(model.max_years, dtype=float) + 1e-6
    y_pred = np.maximum(model_fn(t, *popt), 0.0)
    perr = np.sqrt(np.maximum(np.diag(pcov), 0))
    y_upper = np.maximum(model_fn(t, *(popt + perr)), y_pred)
    y_lower = np.maximum(np.minimum(model_fn(t, *(popt - perr)), y_pred), 0.0)

    return y_pred.tolist(), y_upper.tolist(), y_lower.tolist()


def _make_curve_figure(
    curve_name: str,
    positions_data: dict[str, tuple[list[float], list[float], list[float]]],
    max_years: int,
    show_bands: bool,
) -> go.Figure:
    """One figure per curve variant: all positions overlaid with Viridis colors."""
    fig = go.Figure()
    years = list(range(max_years))

    positions = list(positions_data.keys())
    n = len(positions)
    viridis = pc.sample_colorscale("Viridis", [i / max(n - 1, 1) for i in range(n)])

    for pos, color in zip(positions, viridis):
        y_pred, y_upper, y_lower = positions_data[pos]

        if show_bands:
            rgba_band = _rgb_str_to_rgba(color, 0.12)
            fig.add_trace(go.Scatter(
                x=years, y=y_lower,
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=years, y=y_upper,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=rgba_band,
                showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=years, y=y_pred,
            mode="lines+markers",
            name=pos,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=f"{curve_name} — Population-Mean Career AV by Position",
        xaxis=dict(
            title="Year from Draft",
            tickvals=years,
            ticktext=[f"Yr {y}" for y in years],
        ),
        yaxis_title="Expected AV (population mean)",
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, xanchor="left"),
    )
    return fig


def main() -> None:
    available = discover_parametric_curves(MODELS_DIR)

    parser = argparse.ArgumentParser(
        description="Plot population-mean career AV curves per parametric variant (all positions on one plot)"
    )
    parser.add_argument(
        "--curves",
        nargs="+",
        default=available if available else ["gamma"],
        metavar="CURVE",
        help=f"Curve variant(s) to plot (default: all discovered — {available or ['gamma']})",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["all"],
        metavar="POS",
        help="Position group(s) to plot, or 'all' (default: all)",
    )
    parser.add_argument(
        "--no-bands",
        action="store_true",
        dest="no_bands",
        help="Suppress ±1σ confidence band shading",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_DIR),
        dest="output_dir",
        help=f"Directory for output HTML files (default: {FIGURES_DIR})",
    )
    args = parser.parse_args()

    _check_prerequisites(args.curves)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading parametric models: {args.curves}")
    models: dict[str, ParametricCurveModel] = {}
    for curve in args.curves:
        m = ParametricCurveModel(curve_name=curve)
        m.load(MODELS_DIR / "parametric" / curve)
        models[curve] = m
        print(f"  {curve}: {sorted(m._params)} ({len(m._params)} positions)")

    all_positions: set[str] = set()
    for m in models.values():
        all_positions.update(m._params.keys())

    if args.positions == ["all"]:
        positions = [p for p in _POSITION_ORDER if p in all_positions]
        positions += sorted(all_positions - set(_POSITION_ORDER))
    else:
        requested = [p.upper() for p in args.positions]
        unknown = [p for p in requested if p not in all_positions]
        if unknown:
            print(f"Warning: positions not found in any model and will be skipped: {unknown}")
        positions = [p for p in requested if p in all_positions]

    if not positions:
        print("No positions to plot. Exiting.")
        return

    max_years = max(m.max_years for m in models.values())
    show_bands = not args.no_bands
    print(f"\nPlotting {len(positions)} position(s): {positions}")
    print(f"Confidence bands: {'on' if show_bands else 'off'}")

    saved = []
    for curve_name, model in models.items():
        positions_data: dict[str, tuple] = {}
        for pos in positions:
            if pos not in model._params:
                print(f"  {curve_name}: skipping {pos} (position not in model)")
                continue
            positions_data[pos] = _get_default_curve(model, pos)

        if not positions_data:
            print(f"  {curve_name}: no positions available, skipping")
            continue

        fig = _make_curve_figure(curve_name, positions_data, max_years, show_bands)
        out_path = output_dir / f"parametric_curves_{curve_name}.html"
        fig.write_html(str(out_path))
        print(f"  Saved {out_path.name}")
        saved.append(out_path.name)

    print(f"\nDone — {len(saved)} plot(s) written to {output_dir}/")
    for name in saved:
        print(f"  {name}")


if __name__ == "__main__":
    main()
