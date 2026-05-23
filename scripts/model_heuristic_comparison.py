"""Compare model val_mae_by_position against a naive last-observed-year heuristic.

The heuristic: given the first 3 observed seasons (years 0–2), predict all
future seasons (years 3–9) as the value of year 2 (the last known AV).

Outputs a markdown table ready to paste into a document.

Usage
-----
    python scripts/model_heuristic_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.annual_av_analysis import aggregate_career_av_by_position, prepare_av_data
from src.data_ingest import load_parquets_from_dir

RAW_DIR = REPO_ROOT / "data" / "raw" / "stathead" / "annual_av"
MODELS_DIR = REPO_ROOT / "models"

_VAL_YEARS = [1975, 1985, 1995, 2005, 2015]
_N_OBS = 3
_MAX_YEARS = 10


def _load_val_df() -> pl.DataFrame:
    raw_lf = load_parquets_from_dir(RAW_DIR)
    prepared_lf = prepare_av_data(raw_lf)
    career_lf = aggregate_career_av_by_position(prepared_lf, normalize=True, rounds=None)
    return career_lf.filter(pl.col("Draft Year").is_in(_VAL_YEARS)).collect()


def _compute_heuristic_mae(val_df: pl.DataFrame) -> dict[str, float]:
    """Last-observed-year heuristic: predict years 3–9 as actual[2]."""
    mae_by_pos: dict[str, list[float]] = {}

    players = (
        val_df
        .group_by(["Player", "Pos"])
        .agg(pl.col("years_from_draft").n_unique().alias("n_years"))
        .filter(pl.col("n_years") >= _N_OBS)
    )

    for row in players.iter_rows(named=True):
        player, pos = row["Player"], row["Pos"]
        player_df = (
            val_df
            .filter((pl.col("Player") == player) & (pl.col("Pos") == pos))
            .sort("years_from_draft")
        )
        actual_dict = dict(zip(
            player_df["years_from_draft"].to_list(),
            player_df["AV.1"].to_list(),
        ))
        actual = [actual_dict.get(yr, 0.0) for yr in range(_MAX_YEARS)]

        last_observed = actual[_N_OBS - 1]
        for pred_yr in range(_N_OBS, _MAX_YEARS):
            err = abs(actual[pred_yr] - last_observed)
            mae_by_pos.setdefault(pos, []).append(err)

    return {pos: sum(errs) / len(errs) for pos, errs in mae_by_pos.items()}


def _load_model_metadata() -> list[tuple[str, dict[str, float]]]:
    """Return (label, val_mae_by_position) pairs for all found metadata files."""
    results = []

    for name in ["knn", "ridge"]:
        path = MODELS_DIR / name / "metadata.json"
        if path.exists():
            meta = json.loads(path.read_text())
            results.append((name, meta["val_mae_by_position"]))

    parametric_dir = MODELS_DIR / "parametric"
    if parametric_dir.exists():
        for curve_dir in sorted(parametric_dir.iterdir()):
            if curve_dir.is_dir():
                path = curve_dir / "metadata.json"
                if path.exists():
                    meta = json.loads(path.read_text())
                    results.append((f"parametric/{curve_dir.name}", meta["val_mae_by_position"]))

    return results


def _overall(mae_by_pos: dict[str, float]) -> float:
    return sum(mae_by_pos.values()) / len(mae_by_pos)


def _build_markdown_table(
    models: list[tuple[str, dict[str, float]]],
    heuristic_mae: dict[str, float],
) -> str:
    all_positions = sorted(
        {pos for _, mae in models for pos in mae} | set(heuristic_mae)
    )

    col_labels = [label for label, _ in models] + ["Heuristic"]
    headers = ["Position"] + col_labels

    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for pos in all_positions:
        cells = [pos]
        for _, mae_dict in models:
            val = mae_dict.get(pos)
            cells.append(f"{val:.4f}" if val is not None else "N/A")
        cells.append(f"{heuristic_mae.get(pos, float('nan')):.4f}")
        rows.append("| " + " | ".join(cells) + " |")

    overall_cells = ["**Overall**"]
    for _, mae_dict in models:
        overall_cells.append(f"**{_overall(mae_dict):.4f}**")
    overall_cells.append(f"**{_overall(heuristic_mae):.4f}**")
    rows.append("| " + " | ".join(overall_cells) + " |")

    return "\n".join(rows)


def main() -> None:
    print("Loading validation data...", file=sys.stderr)
    val_df = _load_val_df()

    print("Computing heuristic MAE...", file=sys.stderr)
    heuristic_mae = _compute_heuristic_mae(val_df)

    print("Loading model metadata...", file=sys.stderr)
    models = _load_model_metadata()
    if not models:
        print("No model metadata found.", file=sys.stderr)
        sys.exit(1)

    table = _build_markdown_table(models, heuristic_mae)
    print(table)


if __name__ == "__main__":
    main()
