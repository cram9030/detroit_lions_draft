"""Shared utilities for career AV model management."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def check_model_exists(model_name: str, models_dir: Path, curve_name: str | None = None) -> None:
    """Raise FileNotFoundError with a helpful train command if the model is missing.

    For parametric models, checks for ``models/parametric/{curve}/params.json``
    when curve_name is provided, or scans for any available curve otherwise.
    For other models (knn, ridge, linear), checks for ``models/{name}/_config.joblib``.
    """
    if model_name == "parametric":
        if curve_name:
            path = models_dir / "parametric" / curve_name / "params.json"
            train_cmd = f"  python scripts/train_models.py --model {model_name} --curve {curve_name}"
            if not path.exists():
                raise FileNotFoundError(
                    f"parametric/{curve_name} model not found at {path}\n"
                    f"Train it first:\n"
                    f"{train_cmd}"
                )
        else:
            curves = discover_parametric_curves(models_dir)
            if not curves:
                raise FileNotFoundError(
                    f"No parametric model variants found in {models_dir / 'parametric'}\n"
                    f"Train one first:\n"
                    f"  python scripts/train_models.py --model parametric"
                )
        return
    path = models_dir / model_name / "_config.joblib"
    train_cmd = f"  python scripts/train_models.py --model {model_name}"
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found at {path}\n"
            f"Train it first:\n"
            f"{train_cmd}"
        )


def discover_parametric_curves(models_dir: Path) -> list[str]:
    """Return sorted list of trained parametric curve variants.

    Scans models/parametric/ for sub-directories that contain a params.json.
    Returns [] if no variants exist or the directory is absent.
    """
    param_dir = models_dir / "parametric"
    if not param_dir.is_dir():
        return []
    return sorted(
        d.name for d in param_dir.iterdir()
        if d.is_dir() and (d / "params.json").exists()
    )


def build_training_matrix(
    trajectory_df: pl.DataFrame,
    max_years: int,
) -> tuple[np.ndarray, list[str]]:
    """Build a player × year AV matrix from a trajectory DataFrame.

    Pivots trajectory_df by years_from_draft, fills nulls with 0, and pads
    to max_years columns.

    Args:
        trajectory_df: DataFrame with columns [Player, years_from_draft, AV.1]
            already filtered to a single position.
        max_years: Number of career years to model. Output matrix has this many columns.

    Returns:
        Tuple of (matrix, player_names) where matrix has shape
        (n_players, max_years) and player_names is sorted alphabetically.
    """
    pivoted = (
        trajectory_df
        .sort(["Player", "years_from_draft"])
        .pivot(index="Player", on="years_from_draft", values="AV.1", aggregate_function="sum")
        .sort("Player")
        .fill_null(0)
    )
    year_cols = [str(y) for y in range(max_years)]
    available = [c for c in year_cols if c in pivoted.columns]
    matrix = pivoted.select(available).to_numpy().astype(float)
    if matrix.shape[1] < max_years:
        pad = np.zeros((matrix.shape[0], max_years - matrix.shape[1]))
        matrix = np.hstack([matrix, pad])
    player_names = pivoted["Player"].to_list()
    return matrix, player_names
