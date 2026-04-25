"""KNNTrajectoryModel — nearest-neighbour trajectory matching."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.models.protocol import PredictionResult

_MAX_YEARS = 10


class KNNTrajectoryModel:
    """Finds the K most similar historical trajectories and averages their future AV.

    Similarity is computed on the observed seasons only, so it works regardless
    of how many years have been observed.
    """

    def __init__(self, n_neighbors: int = 10) -> None:
        self.n_neighbors = n_neighbors
        # {pos: np.ndarray shape (n_players, MAX_YEARS)}
        self._reference: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Build per-position trajectory reference matrices from complete trajectories."""
        for pos in trajectory_df["Pos"].unique().to_list():
            sub = trajectory_df.filter(pl.col("Pos") == pos)

            # Keep only players with all _MAX_YEARS seasons recorded
            complete_players = (
                sub.group_by("Player")
                .agg(pl.col("years_from_draft").n_unique().alias("n_years"))
                .filter(pl.col("n_years") == _MAX_YEARS)
                ["Player"]
            )
            sub_complete = sub.filter(pl.col("Player").is_in(complete_players.to_list()))

            if len(sub_complete) == 0:
                continue

            pivoted = (
                sub_complete
                .sort(["Player", "years_from_draft"])
                .pivot(index="Player", on="years_from_draft", values="AV.1")
                .sort("Player")
            )
            # Drop the Player column; columns are year indices (0..9)
            year_cols = [str(y) for y in range(_MAX_YEARS)]
            available = [c for c in year_cols if c in pivoted.columns]
            matrix = pivoted.select(available).to_numpy().astype(float)
            # Pad missing year columns with zeros
            if matrix.shape[1] < _MAX_YEARS:
                pad = np.zeros((matrix.shape[0], _MAX_YEARS - matrix.shape[1]))
                matrix = np.hstack([matrix, pad])
            self._reference[pos] = matrix

    def predict(self, position: str, observed_av: list[float]) -> PredictionResult:
        if position not in self._reference:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._reference)}")

        matrix = self._reference[position]
        n_obs = len(observed_av)
        obs_arr = np.array(observed_av, dtype=float)

        # Distance on observed dimensions only
        ref_obs = matrix[:, :n_obs]
        dists = np.linalg.norm(ref_obs - obs_arr, axis=1)

        k = min(self.n_neighbors, len(dists))
        nn_idx = np.argsort(dists)[:k]
        nn_future = matrix[nn_idx, n_obs:]

        weights = 1.0 / (dists[nn_idx] + 1e-9)
        weights /= weights.sum()

        y_pred = (nn_future * weights[:, None]).sum(axis=0)
        y_std = nn_future.std(axis=0)

        return PredictionResult(
            position=position,
            observed_years=list(range(n_obs)),
            observed_av=list(observed_av),
            predicted_years=list(range(n_obs, _MAX_YEARS)),
            y_pred=y_pred.tolist(),
            y_upper=(y_pred + y_std).tolist(),
            y_lower=np.maximum(y_pred - y_std, 0.0).tolist(),
        )

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        for pos, matrix in self._reference.items():
            joblib.dump(matrix, model_dir / f"{pos}.joblib")

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self._reference = {}
        for path in sorted(model_dir.glob("*.joblib")):
            pos = path.stem
            self._reference[pos] = joblib.load(path)
