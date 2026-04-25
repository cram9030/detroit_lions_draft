"""ParametricCurveModel — Gamma-shaped population curve with individual scaling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from src.models.protocol import PredictionResult

_MAX_YEARS = 10


def _gamma_curve(t: np.ndarray, a: float, alpha: float, b: float, c: float) -> np.ndarray:
    """f(t) = a * t^alpha * exp(-b*t) + c"""
    return a * np.power(t, alpha) * np.exp(-b * t) + c


class ParametricCurveModel:
    """Fits a Gamma-shaped curve to the population mean AV trajectory per position.

    At inference time the curve shape is held fixed and a single scale factor is
    computed from the player's observed seasons to personalise the projection.
    """

    def __init__(self) -> None:
        # {pos: {"popt": list[float], "pcov": list[list[float]]}}
        self._params: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Fit one Gamma curve per position to the population mean AV."""
        means = (
            trajectory_df
            .group_by(["Pos", "years_from_draft"])
            .agg(pl.col("AV.1").mean().alias("mean_av"))
            .sort(["Pos", "years_from_draft"])
        )

        for pos in means["Pos"].unique().to_list():
            sub = means.filter(pl.col("Pos") == pos).sort("years_from_draft")
            t = sub["years_from_draft"].to_numpy().astype(float)
            y = sub["mean_av"].to_numpy().astype(float)

            # Shift t by a small epsilon so t^alpha is defined at t=0 for non-integer alpha
            t_fit = t + 1e-6

            try:
                popt, pcov = curve_fit(
                    _gamma_curve,
                    t_fit,
                    y,
                    p0=[y.max(), 1.0, 0.3, y.min()],
                    bounds=([0, 0.1, 0.01, -1], [50, 5, 5, 10]),
                    maxfev=10_000,
                )
            except RuntimeError:
                # Fall back to a simple exponential-like shape
                popt = np.array([y.max(), 1.0, 0.3, max(0.0, y.min())])
                pcov = np.eye(4) * 1.0

            self._params[pos] = {
                "popt": popt.tolist(),
                "pcov": pcov.tolist(),
            }

    def predict(self, position: str, observed_av: list[float]) -> PredictionResult:
        if position not in self._params:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._params)}")

        popt = np.array(self._params[position]["popt"])
        pcov = np.array(self._params[position]["pcov"])

        n_obs = len(observed_av)
        obs_t = np.arange(n_obs, dtype=float) + 1e-6
        pop_at_obs = _gamma_curve(obs_t, *popt)

        # Scale factor: least-squares scale of individual vs population shape
        safe_pop = np.where(np.abs(pop_at_obs) > 1e-9, pop_at_obs, 1e-9)
        scale = float(np.mean(np.array(observed_av) / safe_pop))
        scale = max(0.0, scale)

        pred_t = np.arange(n_obs, _MAX_YEARS, dtype=float) + 1e-6
        y_pred = scale * _gamma_curve(pred_t, *popt)

        # Uncertainty via Jacobian propagation: σ_f = sqrt(J @ pcov @ J^T)
        perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        y_upper = scale * _gamma_curve(pred_t, *(popt + perr))
        y_lower = scale * _gamma_curve(pred_t, *(popt - perr))

        # Enforce ordering so lower ≤ pred ≤ upper pointwise
        y_upper = np.maximum(y_upper, y_pred)
        y_lower = np.minimum(y_lower, y_pred)
        y_lower = np.maximum(y_lower, 0.0)

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
        path = Path(model_dir) / "params.json"
        path.write_text(json.dumps(self._params, indent=2))

    def load(self, model_dir: str | Path) -> None:
        path = Path(model_dir) / "params.json"
        self._params = json.loads(path.read_text())
