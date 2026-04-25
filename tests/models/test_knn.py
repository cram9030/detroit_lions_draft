"""Unit tests for KNNTrajectoryModel."""

from pathlib import Path

import polars as pl
import pytest

from src.models.knn import KNNTrajectoryModel

FIXTURES = Path(__file__).parent / "fixtures"
TRAJECTORY_PATH = FIXTURES / "mock_knn_trajectories.parquet"


@pytest.fixture()
def trajectory_df() -> pl.DataFrame:
    return pl.read_parquet(TRAJECTORY_PATH)


@pytest.fixture()
def fitted_model(trajectory_df) -> KNNTrajectoryModel:
    model = KNNTrajectoryModel(n_neighbors=3)
    model.fit(trajectory_df)
    return model


def test_fit_builds_reference_matrix(fitted_model):
    assert len(fitted_model._reference) == 2
    assert "QB" in fitted_model._reference
    assert "RB" in fitted_model._reference
    for pos, matrix in fitted_model._reference.items():
        assert matrix.shape[1] == 10


def test_predict_returns_correct_shape(fitted_model):
    observed = [2.0, 3.5, 4.0]
    result = fitted_model.predict("QB", observed)
    expected_years = 10 - len(observed)
    assert len(result["predicted_years"]) == expected_years
    assert len(result["y_pred"]) == expected_years
    assert len(result["y_upper"]) == expected_years
    assert len(result["y_lower"]) == expected_years


def test_save_load_roundtrip(fitted_model, tmp_path):
    fitted_model.save(tmp_path)
    assert (tmp_path / "_config.joblib").exists()

    fresh = KNNTrajectoryModel(n_neighbors=99)  # different default to confirm it gets overwritten
    fresh.load(tmp_path)

    assert fresh.n_neighbors == 3
    r1 = fitted_model.predict("QB", [2.0, 3.5])
    r2 = fresh.predict("QB", [2.0, 3.5])
    assert r1["y_pred"] == r2["y_pred"]


def test_predict_unknown_position_raises(fitted_model):
    with pytest.raises(ValueError, match="Unknown position"):
        fitted_model.predict("XX", [2.0, 3.5])


def test_fit_handles_duplicate_player_year_rows(tmp_path):
    """fit() must not crash when a player-year appears more than once (e.g. multi-team seasons)."""
    base_df = pl.read_parquet(TRAJECTORY_PATH)
    # Duplicate every row so each player has two AV.1 entries per year
    df_with_dupes = pl.concat([base_df, base_df])

    model = KNNTrajectoryModel(n_neighbors=3)
    model.fit(df_with_dupes)  # should not raise

    assert "QB" in model._reference
    result = model.predict("QB", [2.0, 3.5])
    assert len(result["y_pred"]) == 8
