"""Unit tests for ParametricCurveModel."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.curve_fitting import ExpDecayModel, GammaCurveModel
from src.models.parametric import ParametricCurveModel

FIXTURES = Path(__file__).parent / "fixtures"
TRAJECTORY_PATH = FIXTURES / "mock_knn_trajectories.parquet"

_GAMMA_BOUNDS = ([0, 0.1, 0.01, -1], [50, 5, 5, 10])

# Parameterize tests over two curves so both code paths are exercised.
_CURVE_CASES = [
    pytest.param("gamma", GammaCurveModel, _GAMMA_BOUNDS, id="gamma"),
    pytest.param("exp_decay", ExpDecayModel, None, id="exp_decay"),
]


@pytest.fixture()
def trajectory_df() -> pl.DataFrame:
    return pl.read_parquet(TRAJECTORY_PATH)


@pytest.fixture()
def fitted_model(trajectory_df) -> ParametricCurveModel:
    model = ParametricCurveModel()
    model.fit(trajectory_df)
    return model


# ---------------------------------------------------------------------------
# Parameterized tests — run for each curve type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("curve_name,curve,bounds", _CURVE_CASES)
def test_fit_populates_params(trajectory_df, curve_name, curve, bounds):
    model = ParametricCurveModel(curve_name=curve_name, curve=curve, bounds=bounds)
    model.fit(trajectory_df)
    assert len(model._params) == 2
    assert "QB" in model._params
    assert "RB" in model._params


@pytest.mark.parametrize("curve_name,curve,bounds", _CURVE_CASES)
def test_predict_returns_correct_shape(trajectory_df, curve_name, curve, bounds):
    model = ParametricCurveModel(curve_name=curve_name, curve=curve, bounds=bounds)
    model.fit(trajectory_df)
    observed = [3.0, 4.0]
    result = model.predict("QB", observed)
    expected_years = 10 - len(observed)
    assert len(result["predicted_years"]) == expected_years
    assert len(result["y_pred"]) == expected_years
    assert len(result["y_upper"]) == expected_years
    assert len(result["y_lower"]) == expected_years
    assert result["observed_years"] == list(range(len(observed)))
    assert result["observed_av"] == observed
    assert result["position"] == "QB"


@pytest.mark.parametrize("curve_name,curve,bounds", _CURVE_CASES)
def test_predict_monotone_uncertainty(trajectory_df, curve_name, curve, bounds):
    model = ParametricCurveModel(curve_name=curve_name, curve=curve, bounds=bounds)
    model.fit(trajectory_df)
    result = model.predict("QB", [3.0, 4.0])
    for lo, mid, hi in zip(result["y_lower"], result["y_pred"], result["y_upper"]):
        assert hi >= mid >= lo


@pytest.mark.parametrize("curve_name,curve,bounds", _CURVE_CASES)
def test_save_load_roundtrip(trajectory_df, curve_name, curve, bounds, tmp_path):
    model = ParametricCurveModel(curve_name=curve_name, curve=curve, bounds=bounds)
    model.fit(trajectory_df)
    model.save(tmp_path)
    assert (tmp_path / "params.json").exists()

    fresh = ParametricCurveModel()
    fresh.load(tmp_path)

    assert fresh.curve_name == curve_name
    r1 = model.predict("QB", [3.0, 4.0])
    r2 = fresh.predict("QB", [3.0, 4.0])
    assert r1["y_pred"] == r2["y_pred"]


# ---------------------------------------------------------------------------
# Non-parameterized tests
# ---------------------------------------------------------------------------


def test_predict_unknown_position_raises(fitted_model):
    with pytest.raises(ValueError, match="Unknown position"):
        fitted_model.predict("XX", [3.0, 4.0])


def test_default_curve_is_gamma(fitted_model):
    assert fitted_model.curve_name == "gamma"


def test_curve_name_stored_in_json(fitted_model, tmp_path):
    fitted_model.save(tmp_path)
    data = json.loads((tmp_path / "params.json").read_text())
    assert data["curve_name"] == "gamma"


def test_load_preserves_curve_name(trajectory_df, tmp_path):
    model = ParametricCurveModel(curve_name="exp_decay", curve=ExpDecayModel)
    model.fit(trajectory_df)
    model.save(tmp_path)

    fresh = ParametricCurveModel()
    fresh.load(tmp_path)
    assert fresh.curve_name == "exp_decay"


def test_unknown_curve_name_raises():
    with pytest.raises(ValueError, match="Unknown curve"):
        ParametricCurveModel(curve_name="nonexistent")


def test_fit_uses_zero_padded_means():
    """Short-career players must contribute zeros to late-year means."""
    # 5 players retire after 2 seasons (AV=10); 5 play all 10 seasons (AV=10).
    # Correct year-9 mean = (5*0 + 5*10) / 10 = 5.0.
    # Without the fix, year-9 mean = 10.0 (only long-career players counted).
    rows = []
    for i in range(5):
        for yr in range(2):
            rows.append({"Player": f"Short{i}", "Pos": "QB", "Draft Year": 2015,
                         "years_from_draft": yr, "AV.1": 10.0})
    for i in range(5):
        for yr in range(10):
            rows.append({"Player": f"Long{i}", "Pos": "QB", "Draft Year": 2010,
                         "years_from_draft": yr, "AV.1": 10.0})
    df = pl.DataFrame(rows)
    model = ParametricCurveModel(max_years=10)
    model.fit(df)

    # Scale ≈ 1.0 (observed 10s match population 10s at year 0-1).
    # y_pred[7] = prediction for year 9. With correct means the curve is pulled
    # toward 5 at year 9; without the fix it would stay near 10.
    result = model.predict("QB", [10.0, 10.0])
    assert result["y_pred"][7] < 9.0
