"""RidgeRegressionModel — per-position multi-output Ridge regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV

from src.models.protocol import PredictionResult

_MAX_YEARS = 10
_ALPHAS = [0.1, 1.0, 10.0, 100.0]


class RidgeRegressionModel:
    """Predicts future AV seasons from observed early-career seasons via Ridge regression.

    A single multi-output RidgeCV model is trained per position.  Features are
    the observed AV years zero-padded to _MAX_YEARS - 1 columns; targets are
    AV years X through _MAX_YEARS - 1.  At inference, only the first n_obs
    features are filled; the rest remain zero.
    """

    def __init__(self) -> None:
        # {pos: {"model": RidgeCV, "residual_std": np.ndarray, "n_train": int}}
        self._models: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Train one multi-output RidgeCV per position."""
        for pos in trajectory_df["Pos"].unique().to_list():
            sub = trajectory_df.filter(pl.col("Pos") == pos)

            complete_players = (
                sub.group_by("Player")
                .agg(pl.col("years_from_draft").n_unique().alias("n_years"))
                .filter(pl.col("n_years") == _MAX_YEARS)
                ["Player"]
            )
            sub_complete = sub.filter(pl.col("Player").is_in(complete_players.to_list()))

            if len(sub_complete) < 5:
                continue

            pivoted = (
                sub_complete
                .sort(["Player", "years_from_draft"])
                .pivot(index="Player", on="years_from_draft", values="AV.1")
                .sort("Player")
            )
            year_cols = [str(y) for y in range(_MAX_YEARS)]
            available = [c for c in year_cols if c in pivoted.columns]
            matrix = pivoted.select(available).to_numpy().astype(float)
            if matrix.shape[1] < _MAX_YEARS:
                pad = np.zeros((matrix.shape[0], _MAX_YEARS - matrix.shape[1]))
                matrix = np.hstack([matrix, pad])

            # Features: all years zero-padded to MAX_YEARS - 1; targets: years 1..(MAX_YEARS-1)
            n_feature_cols = _MAX_YEARS - 1
            X = matrix[:, :n_feature_cols]
            Y = matrix[:, 1:]  # years 1 through MAX_YEARS-1

            model = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
            model.fit(X, Y)

            residuals = Y - model.predict(X)
            residual_std = residuals.std(axis=0)

            self._models[pos] = {
                "model": model,
                "residual_std": residual_std,
            }

    def predict(self, position: str, observed_av: list[float]) -> PredictionResult:
        if position not in self._models:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._models)}")

        entry = self._models[position]
        model: RidgeCV = entry["model"]
        residual_std: np.ndarray = entry["residual_std"]

        n_obs = len(observed_av)

        # Build feature vector: observed years then zeros
        n_feature_cols = _MAX_YEARS - 1
        x = np.zeros((1, n_feature_cols))
        x[0, :n_obs] = observed_av

        y_full = model.predict(x)[0]  # length MAX_YEARS - 1

        # y_full[i] is the prediction for year i+1; we want years n_obs..(MAX_YEARS-1)
        y_pred = y_full[n_obs - 1:]  # years n_obs through MAX_YEARS-1
        sigma = residual_std[n_obs - 1:]

        y_pred = np.maximum(y_pred, 0.0)
        y_upper = np.maximum(y_pred + sigma, y_pred)
        y_lower = np.maximum(y_pred - sigma, 0.0)

        return PredictionResult(
            position=position,
            observed_years=list(range(n_obs)),
            observed_av=list(observed_av),
            predicted_years=list(range(n_obs, _MAX_YEARS)),
            y_pred=y_pred.tolist(),
            y_upper=y_upper.tolist(),
            y_lower=y_lower.tolist(),
        )

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        for pos, entry in self._models.items():
            joblib.dump(entry, model_dir / f"{pos}.joblib")

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self._models = {}
        for path in sorted(model_dir.glob("*.joblib")):
            pos = path.stem
            self._models[pos] = joblib.load(path)
