"""Curve-fitting wrappers for AV-by-pick analysis."""

from __future__ import annotations

import polars as pl

from src import curve_fitting
from src.curve_fitting import (
    ExpDecayModel,
    IndividualFitResult,
    LogDecayModel,
    StatsFitResult,
)


def exponential_av_fit(
    player_av_data: pl.LazyFrame | pl.DataFrame,
    max_pick: int = 250,
    av_col: str = "rookie_contract_av",
) -> IndividualFitResult:
    """Fit an exponential decay curve to individual player rookie contract AV by pick."""
    return curve_fitting.fit_individuals(
        player_av_data, ExpDecayModel, max_pick=max_pick, av_col=av_col
    )


def exponential_av_fit_stat(
    stats_df: pl.DataFrame,
    stat_col: str = "mean",
    max_pick: int = 250,
) -> StatsFitResult:
    """Fit an exponential decay curve to any per-pick descriptive statistic."""
    return curve_fitting.fit_stats(
        stats_df, ExpDecayModel, stat_col=stat_col, max_pick=max_pick
    )


def fit_result_to_dataframe(
    fit_result: IndividualFitResult | StatsFitResult,
) -> pl.DataFrame:
    """Convert an exponential fit result to a saveable DataFrame."""
    return curve_fitting.fit_result_to_dataframe(fit_result, ExpDecayModel)


def logarithmic_av_fit(
    player_av_data: pl.LazyFrame | pl.DataFrame,
    max_pick: int = 250,
    av_col: str = "rookie_contract_av",
) -> IndividualFitResult:
    """Fit a logarithmic decay curve to individual player rookie contract AV by pick."""
    return curve_fitting.fit_individuals(
        player_av_data, LogDecayModel, max_pick=max_pick, av_col=av_col
    )


def logarithmic_av_fit_stat(
    stats_df: pl.DataFrame,
    stat_col: str = "mean",
    max_pick: int = 250,
) -> StatsFitResult:
    """Fit a logarithmic decay curve to a per-pick statistic column."""
    return curve_fitting.fit_stats(
        stats_df, LogDecayModel, stat_col=stat_col, max_pick=max_pick
    )


def logarithmic_fit_result_to_dataframe(
    fit_result: IndividualFitResult | StatsFitResult,
) -> pl.DataFrame:
    """Convert a logarithmic fit result to a saveable DataFrame."""
    return curve_fitting.fit_result_to_dataframe(fit_result, LogDecayModel)
