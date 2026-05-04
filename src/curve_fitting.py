"""Generic curve-fitting utilities for NFL draft pick AV analysis.

Provides two pre-built model descriptors (:data:`ExpDecayModel` and
:data:`LogDecayModel`) and a shared fitting engine (:func:`fit_individuals`,
:func:`fit_stats`) that both share.  The functions in
``annual_av_analysis`` are thin wrappers around this module.

Model descriptor format: ``(model_fn, jacobian_fn, p0_fn)``
    - ``model_fn(x, *params) -> ndarray``
    - ``jacobian_fn(x, params) -> ndarray  # shape (len(x), n_params)``
    - ``p0_fn(values) -> list[float]``  # data-driven initial guess
"""

from __future__ import annotations

import warnings
from typing import Callable, TypedDict

import numpy as np
import polars as pl
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class IndividualFitResult(TypedDict):
    """Result of :func:`fit_individuals` — one row per player in the input.

    Attributes:
        popt: Fitted model parameters.
        pcov: Covariance matrix of ``popt``.
        perr: 1-sigma parameter uncertainties, ``sqrt(diag(pcov))``.
        x_fit: 500-point pick axis for smooth curve plotting.
        y_fit: Fitted values at ``x_fit``.
        y_upper: Upper 1-sigma bound at ``x_fit``.
        y_lower: Lower 1-sigma bound at ``x_fit``.
        picks: Individual pick numbers used in the fit (one per player).
        av_values: Individual AV values aligned with ``picks``.
    """

    popt: np.ndarray
    pcov: np.ndarray
    perr: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray
    picks: np.ndarray
    av_values: np.ndarray


class StatsFitResult(TypedDict):
    """Result of :func:`fit_stats` — one row per unique pick in the input.

    Attributes:
        popt: Fitted model parameters.
        pcov: Covariance matrix of ``popt``.
        perr: 1-sigma parameter uncertainties, ``sqrt(diag(pcov))``.
        x_fit: 500-point pick axis for smooth curve plotting.
        y_fit: Fitted values at ``x_fit``.
        y_upper: Upper 1-sigma bound at ``x_fit``.
        y_lower: Lower 1-sigma bound at ``x_fit``.
        picks: Unique pick numbers used in the fit.
        stat_values: Per-pick statistic values aligned with ``picks``.
        iqr_picks: Pick numbers where both 25th and 75th percentiles are
            available, for IQR context.
        q25: 25th-percentile AV per pick, aligned with ``iqr_picks``.
        q75: 75th-percentile AV per pick, aligned with ``iqr_picks``.
    """

    popt: np.ndarray
    pcov: np.ndarray
    perr: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray
    picks: np.ndarray
    stat_values: np.ndarray
    iqr_picks: np.ndarray
    q25: np.ndarray
    q75: np.ndarray


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay: ``f(x) = a * exp(-b * x) + c``."""
    return a * np.exp(-b * x) + c


def log_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logarithmic decay: ``f(x) = a * ln(x) + b``."""
    return a * np.log(x) + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Quadratic polynomial: ``f(x) = a * x² + b * x + c``."""
    return a * x**2 + b * x + c


def cubic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Cubic polynomial: ``f(x) = a * x³ + b * x² + c * x + d``."""
    return a * x**3 + b * x**2 + c * x + d


def quartic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Quartic polynomial: ``f(x) = a * x⁴ + b * x³ + c * x² + d * x + e``."""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def _exp_jacobian(x: np.ndarray, popt: np.ndarray) -> np.ndarray:
    a, b, _ = popt
    return np.column_stack([
        np.exp(-b * x),
        -a * x * np.exp(-b * x),
        np.ones_like(x),
    ])


def _log_jacobian(x: np.ndarray, popt: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.column_stack([np.log(x), np.ones_like(x)])


def _quadratic_jacobian(x: np.ndarray, popt: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.column_stack([x**2, x, np.ones_like(x)])


def _cubic_jacobian(x: np.ndarray, popt: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.column_stack([x**3, x**2, x, np.ones_like(x)])


def _quartic_jacobian(x: np.ndarray, popt: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.column_stack([x**4, x**3, x**2, x, np.ones_like(x)])


# ---------------------------------------------------------------------------
# Pre-built model descriptors  (model_fn, jacobian_fn, p0_fn)
# ---------------------------------------------------------------------------

ModelDescriptor = tuple[Callable, Callable, Callable]

ExpDecayModel: ModelDescriptor = (
    exp_decay,
    _exp_jacobian,
    lambda v: [float(v.max()), 0.01, float(v.min())],
)
"""Three-parameter exponential decay: ``f(pick) = a * exp(-b*pick) + c``."""

LogDecayModel: ModelDescriptor = (
    log_decay,
    _log_jacobian,
    lambda _: [-5.0, 20.0],
)
"""Two-parameter logarithmic decay: ``f(pick) = a * ln(pick) + b``."""

QuadraticModel: ModelDescriptor = (
    quadratic,
    _quadratic_jacobian,
    lambda v: [0.0, -(float(v.max()) - float(v.min())) / 250, float(v.max())],
)
"""Three-parameter quadratic polynomial: ``f(pick) = a * pick² + b * pick + c``."""

CubicModel: ModelDescriptor = (
    cubic,
    _cubic_jacobian,
    lambda v: [0.0, 0.0, -(float(v.max()) - float(v.min())) / 250, float(v.max())],
)
"""Four-parameter cubic polynomial: ``f(pick) = a * pick³ + b * pick² + c * pick + d``."""

QuarticModel: ModelDescriptor = (
    quartic,
    _quartic_jacobian,
    lambda v: [0.0, 0.0, 0.0, -(float(v.max()) - float(v.min())) / 250, float(v.max())],
)
"""Five-parameter quartic polynomial: ``f(pick) = a * pick⁴ + b * pick³ + c * pick² + d * pick + e``."""


# ---------------------------------------------------------------------------
# Shared fitting engine
# ---------------------------------------------------------------------------


def _fit_and_band(
    picks: np.ndarray,
    values: np.ndarray,
    model_fn: Callable,
    jacobian_fn: Callable,
    p0: list[float],
) -> dict:
    """Fit ``model_fn`` to ``(picks, values)`` and compute the 1-sigma band.

    Returns a plain dict with keys ``popt``, ``pcov``, ``perr``,
    ``x_fit``, ``y_fit``, ``y_upper``, ``y_lower``, ``picks``.
    """
    try:
        popt, pcov, infodict, mesg, ier = curve_fit(model_fn, picks, values, p0=p0, maxfev=10000, full_output=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Curve fit failed to converge: {exc}") from exc

    perr = np.sqrt(np.diag(pcov))

    if np.any(np.isinf(pcov)) or np.any(perr > 1e6):
        warnings.warn(
            "Curve fit may be over-parameterized: parameter uncertainties are "
            "extremely large (perr > 1e6 or infinite).",
            UserWarning,
            stacklevel=4,
        )

    x_fit = np.linspace(picks.min(), picks.max(), 500)
    y_fit = model_fn(x_fit, *popt)
    J = jacobian_fn(x_fit, popt)
    sigma = np.sqrt(np.abs(np.einsum("ij,jk,ik->i", J, pcov, J)))

    return dict(
        popt=popt,
        pcov=pcov,
        perr=perr,
        x_fit=x_fit,
        y_fit=y_fit,
        y_upper=y_fit + sigma,
        y_lower=y_fit - sigma,
        picks=picks,
        infodict=infodict,
    )


# ---------------------------------------------------------------------------
# Public fit functions
# ---------------------------------------------------------------------------


def fit_individuals(
    player_data: pl.LazyFrame | pl.DataFrame,
    model: ModelDescriptor,
    max_pick: int = 250,
    av_col: str = "rookie_contract_av",
) -> IndividualFitResult:
    """Fit a curve to individual player AV values by pick number.

    Each player contributes one data point at their pick position, giving the
    full statistical weight of the population to the fit.

    Args:
        player_data: LazyFrame or DataFrame with one row per player.
            Must contain ``Pick`` (Int64) and the column named ``av_col``.
        model: Model descriptor tuple ``(model_fn, jacobian_fn, p0_fn)``.
            Use :data:`ExpDecayModel` or :data:`LogDecayModel`.
        max_pick: Maximum pick to include. Default 250.
        av_col: Name of the AV column. Default ``"rookie_contract_av"``.

    Returns:
        :class:`IndividualFitResult` with fit parameters, confidence band,
        and the raw ``picks`` / ``av_values`` arrays used in the fit.

    Raises:
        RuntimeError: If ``curve_fit`` fails to converge.
        ValueError: If fewer than ``n_params + 1`` data points remain after
            filtering.
    """
    model_fn, jacobian_fn, p0_fn = model

    df = (
        player_data.lazy()
        .filter(pl.col("Pick") <= max_pick)
        .select(["Pick", av_col])
        .drop_nulls()
        .collect()
        .sort("Pick")
    )

    n_params = model_fn.__code__.co_argcount - 1  # subtract x
    if len(df) < n_params + 1:
        raise ValueError(
            f"Only {len(df)} player records after filtering to max_pick={max_pick}. "
            f"Need at least {n_params + 1} for a {n_params}-parameter fit."
        )

    picks = df["Pick"].to_numpy().astype(float)
    av_values = df[av_col].to_numpy()

    base = _fit_and_band(picks, av_values, model_fn, jacobian_fn, p0_fn(av_values))
    return IndividualFitResult(**base, av_values=av_values)


def fit_stats(
    stats_df: pl.DataFrame,
    model: ModelDescriptor,
    stat_col: str = "mean",
    max_pick: int = 250,
) -> StatsFitResult:
    """Fit a curve to a per-pick descriptive statistic column.

    Each unique pick contributes one data point (the value of ``stat_col``
    at that pick), giving every pick equal weight in the fit.

    Args:
        stats_df: Per-pick stats DataFrame (output of ``pick_based_stats``).
            Must contain ``Pick``, ``stat_col``, ``25%``, and ``75%``.
        model: Model descriptor tuple. Use :data:`ExpDecayModel` or
            :data:`LogDecayModel`.
        stat_col: Column to fit. Default ``"mean"``.
        max_pick: Maximum pick to include. Default 250.

    Returns:
        :class:`StatsFitResult` with fit parameters, confidence band,
        ``stat_values``, and IQR context arrays.

    Raises:
        RuntimeError: If ``curve_fit`` fails to converge.
        ValueError: If fewer than ``n_params + 1`` valid picks remain.
    """
    model_fn, jacobian_fn, p0_fn = model

    df = (
        stats_df.filter(pl.col("Pick") <= max_pick)
        .select(["Pick", stat_col])
        .drop_nulls()
        .sort("Pick")
    )
    iqr_df = (
        stats_df.filter(pl.col("Pick") <= max_pick)
        .select(["Pick", "25%", "75%"])
        .drop_nulls()
        .sort("Pick")
    )

    n_params = model_fn.__code__.co_argcount - 1
    if len(df) < n_params + 1:
        raise ValueError(
            f"Only {len(df)} valid picks after filtering to max_pick={max_pick}. "
            f"Need at least {n_params + 1} for a {n_params}-parameter fit."
        )

    picks = df["Pick"].to_numpy().astype(float)
    stat_values = df[stat_col].to_numpy()

    base = _fit_and_band(picks, stat_values, model_fn, jacobian_fn, p0_fn(stat_values))
    return StatsFitResult(
        **base,
        stat_values=stat_values,
        iqr_picks=iqr_df["Pick"].to_numpy(),
        q25=iqr_df["25%"].to_numpy(),
        q75=iqr_df["75%"].to_numpy(),
    )


def fit_result_to_dataframe(
    fit_result: IndividualFitResult | StatsFitResult,
    model: ModelDescriptor,
) -> pl.DataFrame:
    """Re-evaluate a fit result at every integer pick and return a DataFrame.

    Args:
        fit_result: Return value of :func:`fit_individuals` or :func:`fit_stats`.
        model: The same model descriptor used to produce ``fit_result``.

    Returns:
        DataFrame with columns ``pick`` (Int64), ``y_fit``, ``y_upper``,
        ``y_lower`` (Float64), one row per integer pick from 1 to max_pick.
    """
    model_fn, jacobian_fn, _ = model
    popt = fit_result["popt"]
    pcov = fit_result["pcov"]
    max_pick = int(round(float(fit_result["x_fit"][-1])))
    picks = np.arange(1, max_pick + 1, dtype=float)

    y_fit = model_fn(picks, *popt)
    J = jacobian_fn(picks, popt)
    sigma = np.sqrt(np.abs(np.einsum("ij,jk,ik->i", J, pcov, J)))

    return pl.DataFrame({
        "pick": picks.astype(int).tolist(),
        "y_fit": y_fit.tolist(),
        "y_upper": (y_fit + sigma).tolist(),
        "y_lower": (y_fit - sigma).tolist(),
    })
