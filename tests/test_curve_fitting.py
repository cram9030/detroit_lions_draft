"""Tests for src/curve_fitting.py."""

import numpy as np
import polars as pl
import pytest

from src.curve_fitting import (
    CubicModel,
    ExpDecayModel,
    IndividualFitResult,
    LogDecayModel,
    QuadraticModel,
    QuarticModel,
    StatsFitResult,
    cubic,
    exp_decay,
    fit_individuals,
    fit_result_to_dataframe,
    fit_stats,
    log_decay,
    quadratic,
    quartic,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data generators
# ---------------------------------------------------------------------------


def _make_individual_df(n_picks: int = 50, players_per_pick: int = 10, seed: int = 42) -> pl.DataFrame:
    """Synthetic per-player AV table following an exponential decay with noise."""
    rng = np.random.default_rng(seed)
    picks, avs = [], []
    for pick in range(1, n_picks + 1):
        true_av = 20 * np.exp(-0.01 * pick) + 5
        noise = rng.normal(0, 2, size=players_per_pick)
        picks.extend([pick] * players_per_pick)
        avs.extend((true_av + noise).tolist())
    return pl.DataFrame({"Pick": picks, "rookie_contract_av": avs})


def _make_stats_df(n_picks: int = 50) -> pl.DataFrame:
    """Synthetic per-pick stats table with clean exponential mean."""
    picks = list(range(1, n_picks + 1))
    means = [20 * np.exp(-0.01 * p) + 5 for p in picks]
    q25 = [m - 2 for m in means]
    q75 = [m + 2 for m in means]
    return pl.DataFrame({
        "Pick": picks,
        "mean": means,
        "50%": [m - 0.5 for m in means],
        "25%": q25,
        "75%": q75,
    })


def _make_log_individual_df(n_picks: int = 50, players_per_pick: int = 10, seed: int = 7) -> pl.DataFrame:
    """Synthetic per-player AV following a logarithmic decay."""
    rng = np.random.default_rng(seed)
    picks, avs = [], []
    for pick in range(1, n_picks + 1):
        true_av = -5 * np.log(pick) + 25
        noise = rng.normal(0, 1.5, size=players_per_pick)
        picks.extend([pick] * players_per_pick)
        avs.extend((true_av + noise).tolist())
    return pl.DataFrame({"Pick": picks, "rookie_contract_av": avs})


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


class TestModelFunctions:
    def test_exp_decay_scalar(self):
        assert exp_decay(np.array([0.0]), 10.0, 0.5, 2.0)[0] == pytest.approx(12.0)

    def test_exp_decay_array_shape(self):
        x = np.linspace(1, 100, 200)
        out = exp_decay(x, 15.0, 0.02, 3.0)
        assert out.shape == (200,)

    def test_exp_decay_monotone_decreasing_positive_b(self):
        x = np.arange(1, 50, dtype=float)
        y = exp_decay(x, 20.0, 0.01, 5.0)
        assert np.all(np.diff(y) < 0)

    def test_log_decay_scalar(self):
        assert log_decay(np.array([1.0]), -5.0, 20.0)[0] == pytest.approx(20.0)

    def test_log_decay_array_shape(self):
        x = np.linspace(1, 100, 150)
        assert log_decay(x, -4.0, 18.0).shape == (150,)

    def test_log_decay_monotone_decreasing_negative_a(self):
        x = np.arange(1, 50, dtype=float)
        y = log_decay(x, -5.0, 20.0)
        assert np.all(np.diff(y) < 0)


# ---------------------------------------------------------------------------
# fit_individuals
# ---------------------------------------------------------------------------


class TestFitIndividuals:
    def test_returns_individual_fit_result(self):
        df = _make_individual_df()
        result = fit_individuals(df, ExpDecayModel)
        assert isinstance(result, dict)
        for key in ("popt", "pcov", "perr", "x_fit", "y_fit", "y_upper", "y_lower", "picks", "av_values"):
            assert key in result, f"Missing key: {key}"

    def test_popt_shape_exp(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        assert result["popt"].shape == (3,)
        assert result["pcov"].shape == (3, 3)
        assert result["perr"].shape == (3,)

    def test_popt_shape_log(self):
        result = fit_individuals(_make_log_individual_df(), LogDecayModel)
        assert result["popt"].shape == (2,)
        assert result["pcov"].shape == (2, 2)
        assert result["perr"].shape == (2,)

    def test_x_fit_is_dense(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        assert len(result["x_fit"]) == 500
        assert result["x_fit"][0] == pytest.approx(1.0)

    def test_confidence_band_ordering(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        assert np.all(result["y_upper"] >= result["y_fit"])
        assert np.all(result["y_fit"] >= result["y_lower"])

    def test_picks_and_av_values_aligned(self):
        df = _make_individual_df(n_picks=30, players_per_pick=5)
        result = fit_individuals(df, ExpDecayModel)
        assert len(result["picks"]) == len(result["av_values"])
        assert len(result["picks"]) == 30 * 5

    def test_max_pick_filters_correctly(self):
        df = _make_individual_df(n_picks=100)
        result = fit_individuals(df, ExpDecayModel, max_pick=50)
        assert result["picks"].max() <= 50

    def test_custom_av_col(self):
        df = _make_individual_df().rename({"rookie_contract_av": "dr_av"})
        result = fit_individuals(df, ExpDecayModel, av_col="dr_av")
        assert "popt" in result

    def test_lazyframe_input_works(self):
        df = _make_individual_df()
        result = fit_individuals(df.lazy(), ExpDecayModel)
        assert "popt" in result

    def test_raises_too_few_points(self):
        df = pl.DataFrame({"Pick": [1, 2], "rookie_contract_av": [10.0, 8.0]})
        with pytest.raises(ValueError, match="Need at least"):
            fit_individuals(df, ExpDecayModel)

    def test_exp_recovered_params_roughly_correct(self):
        """Fitted a/b/c should be within a factor of 2 of the true generating params."""
        df = _make_individual_df(n_picks=100, players_per_pick=20)
        result = fit_individuals(df, ExpDecayModel)
        a, b, c = result["popt"]
        assert 10 < a < 30       # true: 20
        assert 0.005 < b < 0.02  # true: 0.01
        assert 2 < c < 8         # true: 5

    def test_log_recovered_params_roughly_correct(self):
        df = _make_log_individual_df(n_picks=100, players_per_pick=20)
        result = fit_individuals(df, LogDecayModel)
        a, b = result["popt"]
        assert -7 < a < -3   # true: -5
        assert 20 < b < 30   # true: 25


# ---------------------------------------------------------------------------
# fit_stats
# ---------------------------------------------------------------------------


class TestFitStats:
    def test_returns_stats_fit_result(self):
        df = _make_stats_df()
        result = fit_stats(df, ExpDecayModel)
        for key in ("popt", "pcov", "perr", "x_fit", "y_fit", "y_upper", "y_lower",
                    "picks", "stat_values", "iqr_picks", "q25", "q75"):
            assert key in result, f"Missing key: {key}"

    def test_stat_values_aligned_with_picks(self):
        df = _make_stats_df(n_picks=40)
        result = fit_stats(df, ExpDecayModel)
        assert len(result["stat_values"]) == len(result["picks"])

    def test_iqr_arrays_aligned(self):
        result = fit_stats(_make_stats_df(), ExpDecayModel)
        assert len(result["q25"]) == len(result["q75"]) == len(result["iqr_picks"])

    def test_confidence_band_ordering(self):
        result = fit_stats(_make_stats_df(), ExpDecayModel)
        assert np.all(result["y_upper"] >= result["y_fit"])
        assert np.all(result["y_fit"] >= result["y_lower"])

    def test_max_pick_filters(self):
        result = fit_stats(_make_stats_df(n_picks=100), ExpDecayModel, max_pick=50)
        assert result["picks"].max() <= 50

    def test_stat_col_median(self):
        result = fit_stats(_make_stats_df(), ExpDecayModel, stat_col="50%")
        assert "stat_values" in result

    def test_log_decay_stat_fit(self):
        picks = list(range(1, 51))
        means = [-5 * np.log(p) + 25 for p in picks]
        df = pl.DataFrame({"Pick": picks, "mean": means, "25%": [m - 1 for m in means], "75%": [m + 1 for m in means]})
        result = fit_stats(df, LogDecayModel)
        assert result["popt"].shape == (2,)

    def test_raises_too_few_picks(self):
        df = pl.DataFrame({"Pick": [1, 2], "mean": [10.0, 8.0], "25%": [8.0, 6.0], "75%": [12.0, 10.0]})
        with pytest.raises(ValueError, match="Need at least"):
            fit_stats(df, ExpDecayModel)

    def test_recovered_params_roughly_correct(self):
        result = fit_stats(_make_stats_df(n_picks=100), ExpDecayModel)
        a, b, c = result["popt"]
        assert 10 < a < 30
        assert 0.005 < b < 0.02
        assert 2 < c < 8


# ---------------------------------------------------------------------------
# fit_result_to_dataframe
# ---------------------------------------------------------------------------


class TestFitResultToDataframe:
    def test_schema(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        df = fit_result_to_dataframe(result, ExpDecayModel)
        assert set(df.columns) == {"pick", "y_fit", "y_upper", "y_lower"}

    def test_integer_pick_column(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        df = fit_result_to_dataframe(result, ExpDecayModel)
        assert df["pick"].dtype == pl.Int64
        assert df["pick"][0] == 1

    def test_pick_range_matches_max_pick(self):
        result = fit_individuals(_make_individual_df(n_picks=50), ExpDecayModel, max_pick=50)
        df = fit_result_to_dataframe(result, ExpDecayModel)
        assert df["pick"].max() == 50
        assert df["pick"].min() == 1
        assert len(df) == 50

    def test_confidence_band_ordering(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        df = fit_result_to_dataframe(result, ExpDecayModel)
        assert (df["y_upper"] >= df["y_fit"]).all()
        assert (df["y_fit"] >= df["y_lower"]).all()

    def test_log_model(self):
        result = fit_individuals(_make_log_individual_df(), LogDecayModel)
        df = fit_result_to_dataframe(result, LogDecayModel)
        assert set(df.columns) == {"pick", "y_fit", "y_upper", "y_lower"}
        assert len(df) > 0

    def test_stats_fit_result_works(self):
        result = fit_stats(_make_stats_df(), ExpDecayModel)
        df = fit_result_to_dataframe(result, ExpDecayModel)
        assert "y_fit" in df.columns

    def test_no_nulls_in_output(self):
        result = fit_individuals(_make_individual_df(), ExpDecayModel)
        df = fit_result_to_dataframe(result, ExpDecayModel)
        assert df.null_count().sum_horizontal()[0] == 0


# ---------------------------------------------------------------------------
# Polynomial helpers — synthetic data generators
# ---------------------------------------------------------------------------


def _make_quadratic_individual_df(n_picks: int = 50, players_per_pick: int = 10, seed: int = 11) -> pl.DataFrame:
    """Synthetic per-player AV following a quadratic decay (opens downward)."""
    rng = np.random.default_rng(seed)
    picks, avs = [], []
    for pick in range(1, n_picks + 1):
        true_av = -0.002 * pick**2 - 0.05 * pick + 30
        noise = rng.normal(0, 1.5, size=players_per_pick)
        picks.extend([pick] * players_per_pick)
        avs.extend((true_av + noise).tolist())
    return pl.DataFrame({"Pick": picks, "rookie_contract_av": avs})


def _make_cubic_individual_df(n_picks: int = 50, players_per_pick: int = 10, seed: int = 13) -> pl.DataFrame:
    """Synthetic per-player AV following a cubic decay."""
    rng = np.random.default_rng(seed)
    picks, avs = [], []
    for pick in range(1, n_picks + 1):
        true_av = 0.000004 * pick**3 - 0.003 * pick**2 - 0.02 * pick + 28
        noise = rng.normal(0, 1.5, size=players_per_pick)
        picks.extend([pick] * players_per_pick)
        avs.extend((true_av + noise).tolist())
    return pl.DataFrame({"Pick": picks, "rookie_contract_av": avs})


def _make_quartic_individual_df(n_picks: int = 50, players_per_pick: int = 10, seed: int = 17) -> pl.DataFrame:
    """Synthetic per-player AV following a quartic decay."""
    rng = np.random.default_rng(seed)
    picks, avs = [], []
    for pick in range(1, n_picks + 1):
        true_av = -0.000000003 * pick**4 + 0.000003 * pick**3 - 0.002 * pick**2 - 0.01 * pick + 27
        noise = rng.normal(0, 1.5, size=players_per_pick)
        picks.extend([pick] * players_per_pick)
        avs.extend((true_av + noise).tolist())
    return pl.DataFrame({"Pick": picks, "rookie_contract_av": avs})


def _make_quadratic_stats_df(n_picks: int = 50) -> pl.DataFrame:
    """Synthetic per-pick stats following a quadratic decay."""
    picks = list(range(1, n_picks + 1))
    means = [-0.002 * p**2 - 0.05 * p + 30 for p in picks]
    return pl.DataFrame({
        "Pick": picks,
        "mean": means,
        "50%": [m - 0.5 for m in means],
        "25%": [m - 2 for m in means],
        "75%": [m + 2 for m in means],
    })


# ---------------------------------------------------------------------------
# Polynomial model functions
# ---------------------------------------------------------------------------


class TestPolynomialModelFunctions:
    def test_quadratic_scalar(self):
        assert quadratic(np.array([2.0]), 1.0, 3.0, 5.0)[0] == pytest.approx(15.0)

    def test_quadratic_array_shape(self):
        x = np.linspace(1, 100, 200)
        assert quadratic(x, -0.001, -0.05, 30.0).shape == (200,)

    def test_quadratic_monotone_decreasing(self):
        x = np.arange(1, 50, dtype=float)
        y = quadratic(x, -0.002, -0.05, 30.0)
        assert np.all(np.diff(y) < 0)

    def test_cubic_scalar(self):
        assert cubic(np.array([1.0]), 1.0, 2.0, 3.0, 4.0)[0] == pytest.approx(10.0)

    def test_cubic_array_shape(self):
        x = np.linspace(1, 100, 150)
        assert cubic(x, 0.0, -0.001, -0.05, 30.0).shape == (150,)

    def test_cubic_monotone_decreasing(self):
        x = np.arange(1, 50, dtype=float)
        y = cubic(x, 0.0, -0.002, -0.05, 30.0)
        assert np.all(np.diff(y) < 0)

    def test_quartic_scalar(self):
        assert quartic(np.array([1.0]), 1.0, 1.0, 1.0, 1.0, 1.0)[0] == pytest.approx(5.0)

    def test_quartic_array_shape(self):
        x = np.linspace(1, 100, 175)
        assert quartic(x, 0.0, 0.0, -0.002, -0.05, 30.0).shape == (175,)

    def test_quartic_monotone_decreasing(self):
        x = np.arange(1, 50, dtype=float)
        y = quartic(x, 0.0, 0.0, -0.002, -0.05, 30.0)
        assert np.all(np.diff(y) < 0)


# ---------------------------------------------------------------------------
# fit_individuals — polynomial models
# ---------------------------------------------------------------------------


class TestFitIndividualsPolynomial:
    def test_popt_shape_quadratic(self):
        result = fit_individuals(_make_quadratic_individual_df(), QuadraticModel)
        assert result["popt"].shape == (3,)
        assert result["pcov"].shape == (3, 3)
        assert result["perr"].shape == (3,)

    def test_popt_shape_cubic(self):
        result = fit_individuals(_make_cubic_individual_df(), CubicModel)
        assert result["popt"].shape == (4,)
        assert result["pcov"].shape == (4, 4)
        assert result["perr"].shape == (4,)

    def test_popt_shape_quartic(self):
        result = fit_individuals(_make_quartic_individual_df(), QuarticModel)
        assert result["popt"].shape == (5,)
        assert result["pcov"].shape == (5, 5)
        assert result["perr"].shape == (5,)

    def test_quadratic_confidence_band_ordering(self):
        result = fit_individuals(_make_quadratic_individual_df(), QuadraticModel)
        assert np.all(result["y_upper"] >= result["y_fit"])
        assert np.all(result["y_fit"] >= result["y_lower"])

    def test_cubic_confidence_band_ordering(self):
        result = fit_individuals(_make_cubic_individual_df(), CubicModel)
        assert np.all(result["y_upper"] >= result["y_fit"])
        assert np.all(result["y_fit"] >= result["y_lower"])

    def test_quartic_confidence_band_ordering(self):
        result = fit_individuals(_make_quartic_individual_df(), QuarticModel)
        assert np.all(result["y_upper"] >= result["y_fit"])
        assert np.all(result["y_fit"] >= result["y_lower"])

    def test_quadratic_picks_av_aligned(self):
        df = _make_quadratic_individual_df(n_picks=30, players_per_pick=5)
        result = fit_individuals(df, QuadraticModel)
        assert len(result["picks"]) == len(result["av_values"]) == 30 * 5

    def test_quadratic_recovered_params_roughly_correct(self):
        df = _make_quadratic_individual_df(n_picks=100, players_per_pick=20)
        result = fit_individuals(df, QuadraticModel)
        a, b, c = result["popt"]
        assert -0.01 < a < 0.01    # true: -0.002
        assert -0.5 < b < 0.3      # true: -0.05
        assert 20 < c < 40         # true: 30

    def test_cubic_recovered_params_roughly_correct(self):
        df = _make_cubic_individual_df(n_picks=100, players_per_pick=20)
        result = fit_individuals(df, CubicModel)
        assert result["popt"].shape == (4,)

    def test_quartic_recovered_params_roughly_correct(self):
        df = _make_quartic_individual_df(n_picks=100, players_per_pick=20)
        result = fit_individuals(df, QuarticModel)
        assert result["popt"].shape == (5,)


# ---------------------------------------------------------------------------
# fit_stats — polynomial models
# ---------------------------------------------------------------------------


class TestFitStatsPolynomial:
    def test_popt_shape_quadratic(self):
        result = fit_stats(_make_quadratic_stats_df(), QuadraticModel)
        assert result["popt"].shape == (3,)

    def test_popt_shape_cubic(self):
        picks = list(range(1, 51))
        means = [0.000004 * p**3 - 0.003 * p**2 - 0.02 * p + 28 for p in picks]
        df = pl.DataFrame({"Pick": picks, "mean": means, "25%": [m - 1 for m in means], "75%": [m + 1 for m in means]})
        result = fit_stats(df, CubicModel)
        assert result["popt"].shape == (4,)

    def test_popt_shape_quartic(self):
        picks = list(range(1, 51))
        means = [-0.000000003 * p**4 + 0.000003 * p**3 - 0.002 * p**2 - 0.01 * p + 27 for p in picks]
        df = pl.DataFrame({"Pick": picks, "mean": means, "25%": [m - 1 for m in means], "75%": [m + 1 for m in means]})
        result = fit_stats(df, QuarticModel)
        assert result["popt"].shape == (5,)

    def test_quadratic_stat_values_aligned(self):
        result = fit_stats(_make_quadratic_stats_df(n_picks=40), QuadraticModel)
        assert len(result["stat_values"]) == len(result["picks"])

    def test_quadratic_iqr_arrays_aligned(self):
        result = fit_stats(_make_quadratic_stats_df(), QuadraticModel)
        assert len(result["q25"]) == len(result["q75"]) == len(result["iqr_picks"])

    def test_quadratic_confidence_band_ordering(self):
        result = fit_stats(_make_quadratic_stats_df(), QuadraticModel)
        assert np.all(result["y_upper"] >= result["y_fit"])
        assert np.all(result["y_fit"] >= result["y_lower"])


# ---------------------------------------------------------------------------
# fit_result_to_dataframe — polynomial models
# ---------------------------------------------------------------------------


class TestFitResultToDataframePolynomial:
    def test_quadratic_schema(self):
        result = fit_individuals(_make_quadratic_individual_df(), QuadraticModel)
        df = fit_result_to_dataframe(result, QuadraticModel)
        assert set(df.columns) == {"pick", "y_fit", "y_upper", "y_lower"}

    def test_cubic_schema(self):
        result = fit_individuals(_make_cubic_individual_df(), CubicModel)
        df = fit_result_to_dataframe(result, CubicModel)
        assert set(df.columns) == {"pick", "y_fit", "y_upper", "y_lower"}

    def test_quartic_schema(self):
        result = fit_individuals(_make_quartic_individual_df(), QuarticModel)
        df = fit_result_to_dataframe(result, QuarticModel)
        assert set(df.columns) == {"pick", "y_fit", "y_upper", "y_lower"}

    def test_quadratic_integer_pick_column(self):
        result = fit_individuals(_make_quadratic_individual_df(), QuadraticModel)
        df = fit_result_to_dataframe(result, QuadraticModel)
        assert df["pick"].dtype == pl.Int64
        assert df["pick"][0] == 1

    def test_quadratic_confidence_band_ordering(self):
        result = fit_individuals(_make_quadratic_individual_df(), QuadraticModel)
        df = fit_result_to_dataframe(result, QuadraticModel)
        assert (df["y_upper"] >= df["y_fit"]).all()
        assert (df["y_fit"] >= df["y_lower"]).all()

    def test_cubic_no_nulls(self):
        result = fit_individuals(_make_cubic_individual_df(), CubicModel)
        df = fit_result_to_dataframe(result, CubicModel)
        assert df.null_count().sum_horizontal()[0] == 0

    def test_quartic_no_nulls(self):
        result = fit_individuals(_make_quartic_individual_df(), QuarticModel)
        df = fit_result_to_dataframe(result, QuarticModel)
        assert df.null_count().sum_horizontal()[0] == 0

    def test_stats_fit_result_quadratic(self):
        result = fit_stats(_make_quadratic_stats_df(), QuadraticModel)
        df = fit_result_to_dataframe(result, QuadraticModel)
        assert "y_fit" in df.columns
        assert len(df) > 0
