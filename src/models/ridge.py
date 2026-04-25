"""RidgeRegressionModel — per-position multi-output Ridge regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV

from src.models.protocol import PredictionResult

_ALPHAS = [0.1, 1.0, 10.0, 100.0]


class RidgeRegressionModel:
    """Predicts future AV seasons from observed early-career seasons via Ridge regression.

    A single multi-output RidgeCV model is trained per position.  Features are
    the observed AV years zero-padded to max_years - 1 columns; targets are
    AV years X through max_years - 1.  At inference, only the first n_obs
    features are filled; the rest remain zero.
    """

    def __init__(self, max_years: int = 10, n_input: int = 2) -> None:
        self.max_years = max_years
        self.n_input = n_input  # number of early-career seasons used as features
        # {pos: {"model": RidgeCV, "residual_std": np.ndarray, "n_input": int}}
        self._models: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Train one multi-output RidgeCV per position."""
        for pos in trajectory_df["Pos"].unique().to_list():
            sub = trajectory_df.filter(pl.col("Pos") == pos)

            # Truncate to the first max_years seasons so long-career players
            # contribute only the same window we're trying to predict.
            sub = sub.filter(pl.col("years_from_draft") < self.max_years)

            complete_players = (
                sub.group_by("Player")
                .agg(pl.col("years_from_draft").n_unique().alias("n_years"))
                .filter(pl.col("n_years") >= self.max_years)
                ["Player"]
            )
            sub_complete = sub.filter(pl.col("Player").is_in(complete_players.to_list()))

            if len(sub_complete) < 5:
                continue

            pivoted = (
                sub_complete
                .sort(["Player", "years_from_draft"])
                .pivot(index="Player", on="years_from_draft", values="AV.1", aggregate_function="sum")
                .sort("Player")
            )
            year_cols = [str(y) for y in range(self.max_years)]
            available = [c for c in year_cols if c in pivoted.columns]
            matrix = pivoted.select(available).to_numpy().astype(float)
            if matrix.shape[1] < self.max_years:
                pad = np.zeros((matrix.shape[0], self.max_years - matrix.shape[1]))
                matrix = np.hstack([matrix, pad])

            # Features: only the first n_input observed seasons; targets: years n_input..max_years-1.
            # Using only early-career seasons as features prevents leakage where the model
            # would learn "year k feature ≈ year k target" and then predict 0 at inference
            # when the future-year features are zero-padded.
            n_feature_cols = self.n_input
            X = matrix[:, :n_feature_cols]
            Y = matrix[:, n_feature_cols:]  # years n_input through max_years-1

            model = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
            model.fit(X, Y)

            residuals = Y - model.predict(X)
            residual_std = residuals.std(axis=0)

            self._models[pos] = {
                "model": model,
                "residual_std": residual_std,
                "n_input": n_feature_cols,
            }

    def predict(self, position: str, observed_av: list[float]) -> PredictionResult:
        if position not in self._models:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._models)}")

        entry = self._models[position]
        model: RidgeCV = entry["model"]
        residual_std: np.ndarray = entry["residual_std"]
        n_input: int = entry.get("n_input", self.n_input)

        n_obs = len(observed_av)

        # Feature vector contains only the first n_input observed seasons.
        x = np.zeros((1, n_input))
        x[0, :min(n_obs, n_input)] = observed_av[:n_input]

        # y_full[i] is the prediction for year n_input + i
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
            {"max_years": self.max_years, "n_input": self.n_input, "models": self._models},
            model_dir / "_config.joblib",
        )

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        config_path = model_dir / "_config.joblib"
        if config_path.exists():
            data = joblib.load(config_path)
            self.max_years = data["max_years"]
            self.n_input = data.get("n_input", self.n_input)
            self._models = data["models"]
        else:
            # legacy format: one .joblib per position
            self._models = {}
            for path in sorted(model_dir.glob("*.joblib")):
                pos = path.stem
                self._models[pos] = joblib.load(path)
