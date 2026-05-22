"""KNNTrajectoryModel — nearest-neighbour trajectory matching."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.models.protocol import PredictionResult


class KNNTrajectoryModel:
    """Finds the K most similar historical trajectories and averages their future AV.

    Similarity is computed on the observed seasons only, so it works regardless
    of how many years have been observed.

    When the training data includes a Pick column, distance is computed in a
    joint (normalized_pick, normalized_av) feature space controlled by
    ``pick_weight``.  If Pick is absent from training data, or if ``pick`` is
    not supplied to ``predict()``, the model falls back to AV-only distances.
    """

    def __init__(self, n_neighbors: int = 10, max_years: int = 10, pick_weight: float = 1.0) -> None:
        self.n_neighbors = n_neighbors
        self.max_years = max_years
        self.pick_weight = pick_weight
        # {pos: np.ndarray shape (n_players, max_years)} — raw AV for output
        self._reference: dict[str, np.ndarray] = {}
        # {pos: np.ndarray shape (n_players,)} — normalized pick per reference player
        self._picks: dict[str, np.ndarray] = {}
        # {pos: float} — 95th-pct AV used to normalize AV for distance computation
        self._av_scale: dict[str, float] = {}
        # (global_min_pick, global_max_pick) for pick normalization
        self._pick_range: tuple[float, float] = (1.0, 260.0)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Build per-position trajectory reference matrices from complete trajectories."""
        has_pick = "Pick" in trajectory_df.columns

        if has_pick:
            all_picks = trajectory_df.select("Pick").drop_nulls()["Pick"].to_numpy().astype(float)
            if len(all_picks) > 0:
                self._pick_range = (float(all_picks.min()), float(all_picks.max()))

        for pos in trajectory_df["Pos"].unique().to_list():
            sub = trajectory_df.filter(pl.col("Pos") == pos)

            # Truncate to the first max_years seasons so long-career players
            # contribute only the same window we're trying to predict.
            sub = sub.filter(pl.col("years_from_draft") < self.max_years)

            if len(sub) == 0:
                continue

            # Extract per-player pick BEFORE the pivot loses the column.
            if has_pick:
                player_picks = (
                    sub
                    .group_by("Player")
                    .agg(pl.col("Pick").first().alias("Pick"))
                    .sort("Player")
                )

            pivoted = (
                sub
                .sort(["Player", "years_from_draft"])
                .pivot(index="Player", on="years_from_draft", values="AV.1", aggregate_function="sum")
                .sort("Player")
                .fill_null(0)
            )
            # Drop the Player column; columns are year indices (0..max_years-1)
            year_cols = [str(y) for y in range(self.max_years)]
            available = [c for c in year_cols if c in pivoted.columns]
            matrix = pivoted.select(available).to_numpy().astype(float)
            # Pad missing year columns with zeros
            if matrix.shape[1] < self.max_years:
                pad = np.zeros((matrix.shape[0], self.max_years - matrix.shape[1]))
                matrix = np.hstack([matrix, pad])
            self._reference[pos] = matrix

            if has_pick:
                # Align picks to the sorted player order in pivoted
                picks_arr = (
                    pivoted.select("Player")
                    .join(player_picks, on="Player", how="left")
                    ["Pick"].to_numpy().astype(float)
                )
                p_min, p_max = self._pick_range
                self._picks[pos] = (picks_arr - p_min) / (p_max - p_min + 1e-9)

                # 95th-pct non-zero AV as the normalization scale for this position
                nonzero = matrix[matrix > 0]
                self._av_scale[pos] = float(np.percentile(nonzero, 95)) if len(nonzero) > 0 else 1.0

    def predict(
        self, position: str, observed_av: list[float], pick: int | None = None
    ) -> PredictionResult:
        if position not in self._reference:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._reference)}")

        matrix = self._reference[position]
        n_obs = len(observed_av)
        obs_arr = np.array(observed_av, dtype=float)

        if pick is not None and position in self._picks:
            # Pick-aware distance in normalized (pick, av) space
            p_min, p_max = self._pick_range
            pick_norm = (float(pick) - p_min) / (p_max - p_min + 1e-9)
            av_s = self._av_scale.get(position, 1.0) + 1e-9

            ref_obs_norm = matrix[:, :n_obs] / av_s
            obs_norm = obs_arr / av_s
            dist_av = np.linalg.norm(ref_obs_norm - obs_norm, axis=1)
            dist_pick = np.abs(self._picks[position] - pick_norm)
            dists = np.sqrt(dist_av ** 2 + (self.pick_weight * dist_pick) ** 2)
        else:
            # AV-only distance (backward compatible; also used when model trained without Pick)
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
            predicted_years=list(range(n_obs, self.max_years)),
            y_pred=y_pred.tolist(),
            y_upper=(y_pred + y_std).tolist(),
            y_lower=np.maximum(y_pred - y_std, 0.0).tolist(),
        )

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        joblib.dump(
            {
                "max_years": self.max_years,
                "n_neighbors": self.n_neighbors,
                "pick_weight": self.pick_weight,
                "reference": self._reference,
                "picks": self._picks,
                "av_scale": self._av_scale,
                "pick_range": self._pick_range,
            },
            model_dir / "_config.joblib",
        )

    def load(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        config_path = model_dir / "_config.joblib"
        if config_path.exists():
            data = joblib.load(config_path)
            self.max_years = data["max_years"]
            self.n_neighbors = data.get("n_neighbors", self.n_neighbors)
            self.pick_weight = data.get("pick_weight", 1.0)
            self._reference = data["reference"]
            self._picks = data.get("picks", {})
            self._av_scale = data.get("av_scale", {})
            self._pick_range = data.get("pick_range", (1.0, 260.0))
        else:
            # legacy format: one .joblib per position
            self._reference = {}
            for path in sorted(model_dir.glob("*.joblib")):
                pos = path.stem
                self._reference[pos] = joblib.load(path)
