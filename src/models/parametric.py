"""ParametricCurveModel — population curve with pluggable curve shape and individual scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from src.curve_fitting import (
    ExpDecayModel,
    GammaCurveModel,
    LogDecayModel,
    ModelDescriptor,
    QuadraticModel,
    CubicModel,
    QuarticModel,
)
from src.models.protocol import PredictionResult

_GAMMA_BOUNDS = ([0, 0.1, 0.01, -1], [50, 5, 5, 10])

# Registry maps curve_name → (ModelDescriptor, bounds | None).
# Curves that need constrained fitting supply explicit bounds; others use None
# which lets scipy.optimize.curve_fit default to unconstrained.
_CURVE_REGISTRY: dict[str, tuple[ModelDescriptor, Any]] = {
    "gamma": (GammaCurveModel, _GAMMA_BOUNDS),
    "exp_decay": (ExpDecayModel, None),
    "log_decay": (LogDecayModel, None),
    "quadratic": (QuadraticModel, None),
    "cubic": (CubicModel, None),
    "quartic": (QuarticModel, None),
}


class ParametricCurveModel:
    """Fits a population-mean trajectory curve per position, then scales to individuals.

    The curve shape is pluggable via ``curve_name`` (looked up in ``_CURVE_REGISTRY``)
    or a custom ``curve`` descriptor together with an explicit ``curve_name`` string
    (used for serialisation).

    At inference the fitted curve is held fixed and a single scale factor is
    computed from the player's observed seasons to personalise the projection.
    """

    def __init__(
        self,
        max_years: int = 10,
        curve_name: str = "gamma",
        curve: ModelDescriptor | None = None,
        bounds: Any = None,
    ) -> None:
        self.max_years = max_years
        self.curve_name = curve_name

        if curve is not None:
            # Caller supplied a descriptor directly (custom or registry curve).
            self._curve_descriptor: ModelDescriptor = curve
            self._bounds = bounds
        elif curve_name in _CURVE_REGISTRY:
            self._curve_descriptor, self._bounds = _CURVE_REGISTRY[curve_name]
            if bounds is not None:
                self._bounds = bounds  # explicit bounds override registry default
        else:
            raise ValueError(
                f"Unknown curve '{curve_name}'. Known: {sorted(_CURVE_REGISTRY)}. "
                "Pass a custom 'curve' descriptor together with 'curve_name' to use "
                "a non-registry curve."
            )

        # {pos: {"popt": list[float], "pcov": list[list[float]]}}
        self._params: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        """Fit one curve per position to the population mean AV."""
        model_fn, _, p0_fn = self._curve_descriptor

        sub = trajectory_df.filter(pl.col("years_from_draft") < self.max_years)

        # Aggregate multi-team seasons so each (Player, Pos, year) is unique.
        sub_agg = (
            sub.group_by(["Player", "Pos", "years_from_draft"])
            .agg(pl.col("AV.1").sum())
        )

        # Expand every player to all max_years slots; missing seasons get AV.1 = 0
        # so short-career players contribute zeros to late-year means rather than
        # being excluded (survivorship bias).
        players = sub.select(["Player", "Pos"]).unique()
        year_df = pl.DataFrame({"years_from_draft": list(range(self.max_years))})
        full_df = (
            players.join(year_df, how="cross")
            .join(sub_agg, on=["Player", "Pos", "years_from_draft"], how="left")
            .with_columns(pl.col("AV.1").fill_null(0.0))
        )

        means = (
            full_df
            .group_by(["Pos", "years_from_draft"])
            .agg(pl.col("AV.1").mean().alias("mean_av"))
            .sort(["Pos", "years_from_draft"])
        )

        for pos in means["Pos"].unique().to_list():
            sub = means.filter(pl.col("Pos") == pos).sort("years_from_draft")
            t = sub["years_from_draft"].to_numpy().astype(float)
            y = sub["mean_av"].to_numpy().astype(float)

            # Small epsilon keeps t^alpha and log(t) defined at t=0.
            t_fit = t + 1e-6

            fit_kwargs: dict[str, Any] = {"maxfev": 10_000}
            if self._bounds is not None:
                fit_kwargs["bounds"] = self._bounds

            try:
                popt, pcov = curve_fit(
                    model_fn,
                    t_fit,
                    y,
                    p0=p0_fn(y),
                    **fit_kwargs,
                )
            except RuntimeError:
                n_params = model_fn.__code__.co_argcount - 1
                popt = np.array(p0_fn(y) if len(p0_fn(y)) == n_params else [float(y.max())] + [0.0] * (n_params - 1))
                pcov = np.eye(len(popt)) * 1.0

            self._params[pos] = {
                "popt": popt.tolist(),
                "pcov": pcov.tolist(),
            }

    def predict(self, position: str, observed_av: list[float], pick: int | None = None) -> PredictionResult:
        if position not in self._params:
            raise ValueError(f"Unknown position '{position}'. Known: {sorted(self._params)}")

        model_fn, _, _ = self._curve_descriptor
        popt = np.array(self._params[position]["popt"])
        pcov = np.array(self._params[position]["pcov"])

        n_obs = len(observed_av)
        obs_t = np.arange(n_obs, dtype=float) + 1e-6
        pop_at_obs = model_fn(obs_t, *popt)

        # Scale factor: least-squares scale of individual vs population shape.
        safe_pop = np.where(np.abs(pop_at_obs) > 1e-9, pop_at_obs, 1e-9)
        scale = float(np.mean(np.array(observed_av) / safe_pop))
        scale = max(0.0, scale)

        pred_t = np.arange(n_obs, self.max_years, dtype=float) + 1e-6
        y_pred = scale * model_fn(pred_t, *popt)

        # Uncertainty via ±1σ parameter perturbation.
        perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        y_upper = scale * model_fn(pred_t, *(popt + perr))
        y_lower = scale * model_fn(pred_t, *(popt - perr))

        # Enforce ordering so lower ≤ pred ≤ upper pointwise.
        y_upper = np.maximum(y_upper, y_pred)
        y_lower = np.minimum(y_lower, y_pred)
        y_lower = np.maximum(y_lower, 0.0)

        return PredictionResult(
            position=position,
            observed_years=list(range(n_obs)),
            observed_av=list(observed_av),
            predicted_years=list(range(n_obs, self.max_years)),
            y_pred=y_pred.tolist(),
            y_upper=y_upper.tolist(),
            y_lower=y_lower.tolist(),
        )

    def save(self, model_dir: str | Path) -> None:
        path = Path(model_dir) / "params.json"
        path.write_text(json.dumps(
            {
                "max_years": self.max_years,
                "curve_name": self.curve_name,
                "positions": self._params,
            },
            indent=2,
        ))

    def load(self, model_dir: str | Path) -> None:
        path = Path(model_dir) / "params.json"
        data = json.loads(path.read_text())
        if "max_years" in data:
            self.max_years = data["max_years"]
            self._params = data["positions"]
            curve_name = data.get("curve_name", "gamma")
        else:
            raise ValueError(
                f"Models need to be regenerated."
            )

        self.curve_name = curve_name
        if curve_name in _CURVE_REGISTRY:
            self._curve_descriptor, self._bounds = _CURVE_REGISTRY[curve_name]
        else:
            raise ValueError(
                f"Unknown curve '{curve_name}' in saved model. "
                f"Known curves: {sorted(_CURVE_REGISTRY)}"
            )
