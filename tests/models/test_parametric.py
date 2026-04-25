"""Unit tests for ParametricCurveModel."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.parametric import ParametricCurveModel

FIXTURES = Path(__file__).parent / "fixtures"
TRAJECTORY_PATH = FIXTURES / "mock_knn_trajectories.parquet"


@pytest.fixture()
def trajectory_df() -> pl.DataFrame:
    return pl.read_parquet(TRAJECTORY_PATH)


@pytest.fixture()
def fitted_model(trajectory_df) -> ParametricCurveModel:
    model = ParametricCurveModel()
    model.fit(trajectory_df)
    return model


def test_fit_populates_params(fitted_model):
    assert len(fitted_model._params) == 2
    assert "QB" in fitted_model._params
    assert "RB" in fitted_model._params


def test_predict_returns_correct_shape(fitted_model):
    observed = [3.0, 4.0]
    result = fitted_model.predict("QB", observed)
    expected_years = 10 - len(observed)
    assert len(result["predicted_years"]) == expected_years
    assert len(result["y_pred"]) == expected_years
    assert len(result["y_upper"]) == expected_years
    assert len(result["y_lower"]) == expected_years
    assert result["observed_years"] == list(range(len(observed)))
    assert result["observed_av"] == observed
    assert result["position"] == "QB"


def test_predict_monotone_uncertainty(fitted_model):
    result = fitted_model.predict("QB", [3.0, 4.0])
    for lo, mid, hi in zip(result["y_lower"], result["y_pred"], result["y_upper"]):
        assert hi >= mid >= lo


def test_save_load_roundtrip(fitted_model, tmp_path):
    fitted_model.save(tmp_path)
    assert (tmp_path / "params.json").exists()

    fresh = ParametricCurveModel()
    fresh.load(tmp_path)

    r1 = fitted_model.predict("QB", [3.0, 4.0])
    r2 = fresh.predict("QB", [3.0, 4.0])
    assert r1["y_pred"] == r2["y_pred"]


def test_predict_unknown_position_raises(fitted_model):
    with pytest.raises(ValueError, match="Unknown position"):
        fitted_model.predict("XX", [3.0, 4.0])
