"""Unit tests for KNNTrajectoryModel."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.knn import KNNTrajectoryModel

FIXTURES = Path(__file__).parent / "fixtures"
TRAJECTORY_PATH = FIXTURES / "mock_knn_trajectories.parquet"


def _make_pick_df(n_players: int = 10, picks: list[int] | None = None) -> pl.DataFrame:
    """Build a minimal trajectory DataFrame with a Pick column."""
    if picks is None:
        picks = list(range(1, n_players + 1))
    rows = [
        {
            "Player": f"P{i}",
            "Pos": "QB",
            "Draft Year": 2020,
            "Pick": picks[i],
            "years_from_draft": yr,
            "AV.1": float((n_players - picks[i]) / n_players * 8 + yr),
        }
        for i in range(n_players)
        for yr in range(4)
    ]
    return pl.DataFrame(rows)


@pytest.fixture()
def trajectory_df() -> pl.DataFrame:
    return pl.read_parquet(TRAJECTORY_PATH)


@pytest.fixture()
def fitted_model(trajectory_df) -> KNNTrajectoryModel:
    model = KNNTrajectoryModel(n_neighbors=3)
    model.fit(trajectory_df)
    return model


@pytest.fixture()
def pick_df() -> pl.DataFrame:
    return _make_pick_df()


@pytest.fixture()
def fitted_pick_model(pick_df) -> KNNTrajectoryModel:
    model = KNNTrajectoryModel(n_neighbors=3)
    model.fit(pick_df)
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


def test_fit_includes_short_career_players():
    """Players with fewer than max_years seasons are zero-padded, not excluded."""
    rows = [
        {"Player": f"P{i}", "Pos": "QB", "Draft Year": 2020,
         "years_from_draft": yr, "AV.1": float(yr + 1)}
        for i in range(5) for yr in range(3)
    ]
    df = pl.DataFrame(rows)
    model = KNNTrajectoryModel(n_neighbors=3, max_years=10)
    model.fit(df)
    assert "QB" in model._reference
    assert model._reference["QB"].shape == (5, 10)
    assert (model._reference["QB"][:, 3:] == 0).all()


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


# ---------------------------------------------------------------------------
# Pick-aware tests
# ---------------------------------------------------------------------------

def test_fit_stores_picks_when_pick_column_present(fitted_pick_model):
    """fit() populates _picks and _av_scale when trajectory_df has a Pick column."""
    assert "QB" in fitted_pick_model._picks
    assert "QB" in fitted_pick_model._av_scale
    picks = fitted_pick_model._picks["QB"]
    assert picks.shape == (10,)
    # Normalized picks lie in [0, 1]
    assert float(picks.min()) >= 0.0
    assert float(picks.max()) <= 1.0


def test_fit_no_pick_column_leaves_picks_empty():
    """fit() without Pick column keeps _picks empty (AV-only mode)."""
    df = _make_pick_df().drop("Pick")
    model = KNNTrajectoryModel(n_neighbors=3)
    model.fit(df)
    assert model._picks == {}
    assert model._av_scale == {}


def test_predict_with_pick_differs_from_without(fitted_pick_model):
    """Same observed AV but different pick number produces different predictions."""
    obs = [5.0, 4.0]
    result_early = fitted_pick_model.predict("QB", obs, pick=1)
    result_late  = fitted_pick_model.predict("QB", obs, pick=200)
    assert result_early["y_pred"] != result_late["y_pred"]


def test_predict_without_pick_falls_back_to_av_only(fitted_pick_model):
    """Calling predict() with pick=None uses AV-only distance even when pick data is stored."""
    obs = [3.0, 2.0]
    result_with_none = fitted_pick_model.predict("QB", obs, pick=None)
    result_no_arg    = fitted_pick_model.predict("QB", obs)
    assert result_with_none["y_pred"] == result_no_arg["y_pred"]


def test_save_load_preserves_pick_data(fitted_pick_model, tmp_path):
    """save/load round-trip preserves pick arrays and normalization params."""
    fitted_pick_model.save(tmp_path)
    fresh = KNNTrajectoryModel()
    fresh.load(tmp_path)

    assert "QB" in fresh._picks
    np.testing.assert_array_almost_equal(
        fresh._picks["QB"], fitted_pick_model._picks["QB"]
    )
    assert fresh._av_scale["QB"] == pytest.approx(fitted_pick_model._av_scale["QB"])
    assert fresh._pick_range == fitted_pick_model._pick_range
    assert fresh.pick_weight == fitted_pick_model.pick_weight

    obs = [4.0, 3.0]
    r1 = fitted_pick_model.predict("QB", obs, pick=10)
    r2 = fresh.predict("QB", obs, pick=10)
    assert r1["y_pred"] == r2["y_pred"]
