"""LinearRegressionModel — per-position multi-output OLS regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from src.models.protocol import PredictionResult

_N_INPUTS = [1, 2, 3]


class LinearRegressionModel:
    """Predicts future AV seasons from observed early-career seasons via OLS regression.

    One LinearRegression sub-model is trained per (position, input-window) pair
    for input windows of 1, 2, and 3 seasons.  At inference, predict() selects
    the sub-model whose input size matches len(observed_av), so predicted_years
    always starts immediately after the observed window — matching the adaptive
    behaviour of KNN and Parametric models.
    """

    def __init__(self, max_years: int = 10) -> None:
        self.max_years = max_years
        # {pos: {n_input: {"model": LinearRegression, "residual_std": np.ndarray}}}
        self._models: dict[str, dict[int, dict[str, Any]]] = {}

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Train one LinearRegression sub-model per (position, n_input) pair."""
        for pos in trajectory_df["Pos"].unique().to_list():
            sub = trajectory_df.filter(pl.col("Pos") == pos)
            sub = sub.filter(pl.col("years_from_draft") < self.max_years)

            if sub.select("Player").n_unique() < 5:
                continue

            pivoted = (
                sub
                .sort(["Player", "years_from_draft"])
                .pivot(index="Player", on="years_from_draft", values="AV.1", aggregate_function="sum")
                .sort("Player")
                .fill_null(0)
            )
            year_cols = [str(y) for y in range(self.max_years)]
            available = [c for c in year_cols if c in pivoted.columns]
            matrix = pivoted.select(available).to_numpy().astype(float)
            if matrix.shape[1] < self.max_years:
                pad = np.zeros((matrix.shape[0], self.max_years - matrix.shape[1]))
                matrix = np.hstack([matrix, pad])

            pos_models: dict[int, dict[str, Any]] = {}
            for n_input in _N_INPUTS:
                X = matrix[:, :n_input]
                Y = matrix[:, n_input:]

                model = LinearRegression(fit_intercept=True)
                model.fit(X, Y)

                residuals = Y - model.predict(X)
                pos_models[n_input] = {
                    "model": model,
                    "residual_std": residuals.std(axis=0),
                }

            self._models[pos] = pos_models

    def predict(self, position: str, observed_av: list[float], pick: int | None = None) -> PredictionResult:
        if position not in self._models:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._models)}")

        pos_models = self._models[position]
        n_obs = len(observed_av)

        # Select the largest trained input window that fits the observed data;
        # fall back to the smallest if n_obs is less than every trained window.
        candidates = [k for k in pos_models if k <= n_obs]
        n_input = max(candidates) if candidates else min(pos_models)

        entry = pos_models[n_input]
        model: LinearRegression = entry["model"]
        residual_std: np.ndarray = entry["residual_std"]

        x = np.zeros((1, n_input))
        x[0, :min(n_obs, n_input)] = observed_av[:n_input]

        y_full = model.predict(x)[0]  # length = max_years - n_input

        y_pred = np.maximum(y_full, 0.0)
        y_upper = np.maximum(y_pred + residual_std, y_pred)
        y_lower = np.maximum(y_pred - residual_std, 0.0)

        return PredictionResult(
            position=position,
            observed_years=list(range(n_obs)),
            observed_av=list(observed_av),
            predicted_years=list(range(n_input, self.max_years)),
            y_pred=y_pred.tolist(),
            y_upper=y_upper.tolist(),
            y_lower=y_lower.tolist(),
        )

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        joblib.dump(
            {"max_years": self.max_years, "models": self._models},
            model_dir / "_config.joblib",
        )

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        data = joblib.load(model_dir / "_config.joblib")
        self.max_years = data["max_years"]
        self._models = data["models"]
