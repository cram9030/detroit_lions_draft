"""Unit tests for RidgeRegressionModel."""

from pathlib import Path

import polars as pl
import pytest

from src.models.ridge import RidgeRegressionModel

FIXTURES = Path(__file__).parent / "fixtures"
TRAJECTORY_PATH = FIXTURES / "mock_knn_trajectories.parquet"


@pytest.fixture()
def trajectory_df() -> pl.DataFrame:
    return pl.read_parquet(TRAJECTORY_PATH)


@pytest.fixture()
def fitted_model(trajectory_df) -> RidgeRegressionModel:
    model = RidgeRegressionModel()
    model.fit(trajectory_df)
    return model


def test_fit_trains_submodels_per_position(fitted_model):
    assert set(fitted_model._models.keys()) == {"QB", "RB"}
    for pos in fitted_model._models:
        assert set(fitted_model._models[pos].keys()) == {1, 2, 3}


def test_predict_returns_correct_shape(fitted_model):
    observed = [3.0, 4.0]
    result = fitted_model.predict("QB", observed)
    expected_years = 10 - len(observed)
    assert len(result["predicted_years"]) == expected_years
    assert len(result["y_pred"]) == expected_years
    assert len(result["y_upper"]) == expected_years
    assert len(result["y_lower"]) == expected_years


def test_predict_1yr_input_starts_at_year1(fitted_model):
    result = fitted_model.predict("QB", [5.0])
    assert result["predicted_years"][0] == 1
    assert len(result["predicted_years"]) == 9


def test_predict_3yr_input_starts_at_year3(fitted_model):
    result = fitted_model.predict("QB", [5.0, 4.0, 3.0])
    assert result["predicted_years"][0] == 3
    assert len(result["predicted_years"]) == 7


def test_predict_adaptive_window_selects_correct_submodel(fitted_model):
    for n_obs in [1, 2, 3]:
        obs = [5.0] * n_obs
        result = fitted_model.predict("QB", obs)
        assert result["predicted_years"][0] == n_obs, (
            f"With {n_obs} observed years, predicted_years should start at {n_obs}, "
            f"got {result['predicted_years'][0]}"
        )
        assert len(result["predicted_years"]) == 10 - n_obs


def test_predict_monotone_uncertainty(fitted_model):
    result = fitted_model.predict("QB", [3.0, 4.0])
    for lo, mid, hi in zip(result["y_lower"], result["y_pred"], result["y_upper"]):
        assert hi >= mid >= lo


def test_save_load_roundtrip(fitted_model, tmp_path):
    fitted_model.save(tmp_path)
    assert (tmp_path / "_config.joblib").exists()

    fresh = RidgeRegressionModel()
    fresh.load(tmp_path)

    r1 = fitted_model.predict("QB", [3.0, 4.0])
    r2 = fresh.predict("QB", [3.0, 4.0])
    assert r1["y_pred"] == r2["y_pred"]


def test_predict_unknown_position_raises(fitted_model):
    with pytest.raises(ValueError, match="Unknown position"):
        fitted_model.predict("XX", [3.0, 4.0])


def test_fit_includes_short_career_players():
    """Players with fewer than max_years seasons are zero-padded, not excluded."""
    rows = [
        {"Player": f"P{i}", "Pos": "QB", "Draft Year": 2020,
         "years_from_draft": yr, "AV.1": float(yr + 1)}
        for i in range(5) for yr in range(3)
    ]
    df = pl.DataFrame(rows)
    model = RidgeRegressionModel(max_years=10)
    model.fit(df)
    assert "QB" in model._models
    assert set(model._models["QB"].keys()) == {1, 2, 3}
