"""Unit tests for src/surplus_av.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.surplus_av import (
    _normalize_pos,
    aggregate_4yr_av,
    compute_surplus_av,
    load_team_draft_class,
    project_player_seasons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_draft_parquet(
    raw_dir: Path,
    draft_year: int,
    season: int,
    rows: list[dict[str, str]],
) -> None:
    """Write a minimal raw-schema parquet to *raw_dir* for the given season."""
    base = {
        "Rk": "1",
        "AV": "0",
        "Round": "1",
        "College": "Test U",
        "Age": "22",
        "Team": "DET",
        "G": "16",
        "GS": "16",
    }
    full_rows = [{**base, **r} for r in rows]
    df = pl.DataFrame(full_rows).with_columns(pl.all().cast(pl.String))
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(raw_dir / f"draft{draft_year}_season{season}.parquet")


def _make_eavar_csv(path: Path) -> None:
    """Write a minimal EAVAR CSV with picks 1–10."""
    rows = [
        {"pick": str(p), "eavar": str(20.0 - p), "eavar_upper": str(21.0 - p),
         "eavar_lower": str(19.0 - p), "replacement_level": "4.5"}
        for p in range(1, 11)
    ]
    df = pl.DataFrame(rows)
    df.write_csv(path)


class MockModel:
    """Stub CareerAVModel that returns fixed predictions for known positions."""

    _KNOWN = {"QB", "WR", "RB", "DE", "DT", "OT", "OG", "OC", "CB", "LB", "S", "TE"}

    def fit(self, trajectory_df: pl.DataFrame) -> None:
        pass

    def predict(self, position: str, observed_av: list[float]) -> dict[str, Any]:
        if position not in self._KNOWN:
            raise ValueError(f"Unknown position: {position}")
        n_obs = len(observed_av)
        predicted_years = list(range(n_obs, 4))  # predict remaining years up to yr3
        return {
            "position": position,
            "observed_years": list(range(n_obs)),
            "observed_av": observed_av,
            "predicted_years": predicted_years,
            "y_pred": [3.0] * len(predicted_years),
            "y_upper": [4.0] * len(predicted_years),
            "y_lower": [2.0] * len(predicted_years),
        }

    def save(self, model_dir: Any) -> None:
        pass

    def load(self, model_dir: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# _normalize_pos
# ---------------------------------------------------------------------------


class TestNormalizePos:
    def test_lde_maps_to_de(self):
        assert _normalize_pos("Player A", "LDE") == "DE"

    def test_rcb_maps_to_cb(self):
        assert _normalize_pos("Player A", "RCB") == "CB"

    def test_unknown_pos_unchanged(self):
        assert _normalize_pos("Player A", "QB") == "QB"

    def test_compound_pos_uses_first_segment(self):
        assert _normalize_pos("Player A", "LDE/LOLB") == "DE"

    def test_per_player_override_wins(self):
        overrides = {"Player A": "OT"}
        assert _normalize_pos("Player A", "OL", overrides) == "OT"

    def test_override_does_not_affect_other_players(self):
        overrides = {"Player A": "OT"}
        assert _normalize_pos("Player B", "RG", overrides) == "OG"


# ---------------------------------------------------------------------------
# load_team_draft_class
# ---------------------------------------------------------------------------


class TestLoadTeamDraftClass:
    def test_happy_path_returns_correct_schema(self, tmp_path):
        yr0_rows = [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "DET", "Pick": "50",
             "Draft Year": "2022", "Season": "2022", "AV.1": "3", "Pos": "LDE"},
            {"Player": "Carol", "Draft Team": "GB", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "QB"},
        ]
        yr1_rows = [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "10", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "DET", "Pick": "50",
             "Draft Year": "2022", "Season": "2023", "AV.1": "4", "Pos": "LDE"},
        ]
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, yr0_rows)
        _make_draft_parquet(raw_dir, 2022, 2023, yr1_rows)

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)

        assert set(df.columns) == {"Player", "Pos", "Pick", "Draft Year", "years_from_draft", "AV.1"}
        assert df["Player"].to_list().count("Carol") == 0  # GB player excluded
        assert set(df["Player"].unique().to_list()) == {"Alice", "Bob"}

    def test_filters_to_requested_team(self, tmp_path):
        rows = [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "GB", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "QB"},
        ]
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, rows)
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "9", "Pos": "WR"},
        ])

        df = load_team_draft_class("GB", 2022, raw_dir=raw_dir)
        assert df["Player"].to_list() == ["Bob"]

    def test_raises_when_season1_missing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
        ])
        with pytest.raises(ValueError, match="At least 2 completed seasons"):
            load_team_draft_class("DET", 2022, raw_dir=raw_dir)

    def test_raises_when_both_seasons_missing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="At least 2 completed seasons"):
            load_team_draft_class("DET", 2022, raw_dir=raw_dir)

    def test_position_normalized(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "LDE"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "9", "Pos": "RDE"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert all(pos == "DE" for pos in df["Pos"].to_list())

    def test_loads_optional_year2_when_present(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "9", "Pos": "WR"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2024, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2024", "AV.1": "11", "Pos": "WR"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert 2 in df["years_from_draft"].to_list()

    def test_years_from_draft_correct(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "9", "Pos": "WR"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert sorted(df["years_from_draft"].to_list()) == [0, 1]


# ---------------------------------------------------------------------------
# project_player_seasons
# ---------------------------------------------------------------------------


class TestProjectPlayerSeasons:
    def test_known_position_two_seasons_returns_tuple(self):
        model = MockModel()
        result = project_player_seasons(model, "Alice", "WR", [8.0, 10.0])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_known_position_two_seasons_uses_model(self):
        model = MockModel()
        yr2, yr3 = project_player_seasons(model, "Alice", "WR", [8.0, 10.0])
        # MockModel returns 3.0 for each predicted year
        assert yr2 == pytest.approx(3.0)
        assert yr3 == pytest.approx(3.0)

    def test_known_position_three_seasons_uses_obs_yr2(self):
        model = MockModel()
        yr2, yr3 = project_player_seasons(model, "Alice", "WR", [8.0, 10.0, 7.0])
        assert yr2 == pytest.approx(7.0)  # observed
        assert yr3 == pytest.approx(3.0)  # model

    def test_four_seasons_no_model_call(self):
        class FailModel(MockModel):
            def predict(self, position, observed_av):
                raise AssertionError("predict should not be called with 4 seasons")

        model = FailModel()
        yr2, yr3 = project_player_seasons(model, "Alice", "WR", [8.0, 10.0, 7.0, 5.0])
        assert yr2 == pytest.approx(7.0)
        assert yr3 == pytest.approx(5.0)

    def test_unknown_position_returns_none(self):
        model = MockModel()
        result = project_player_seasons(model, "Alice", "UNKNOWN_POS", [8.0, 10.0])
        assert result is None

    def test_position_normalized_before_model_call(self):
        called_with = {}

        class TrackingModel(MockModel):
            def predict(self, position, observed_av):
                called_with["pos"] = position
                return super().predict(position, observed_av)

        model = TrackingModel()
        project_player_seasons(model, "Alice", "LDE", [8.0, 10.0])
        assert called_with["pos"] == "DE"

    def test_override_applied(self):
        called_with = {}

        class TrackingModel(MockModel):
            def predict(self, position, observed_av):
                called_with["pos"] = position
                return super().predict(position, observed_av)

        model = TrackingModel()
        project_player_seasons(model, "Alice", "OL", [8.0, 10.0], overrides={"Alice": "OT"})
        assert called_with["pos"] == "OT"


# ---------------------------------------------------------------------------
# aggregate_4yr_av
# ---------------------------------------------------------------------------


def _make_draft_df(players: list[dict]) -> pl.DataFrame:
    """Build a long-format draft class DataFrame for testing."""
    rows = []
    for p in players:
        for yr, av in enumerate(p["av"]):
            rows.append(
                {
                    "Player": p["name"],
                    "Pos": p["pos"],
                    "Pick": p["pick"],
                    "Draft Year": 2022,
                    "years_from_draft": yr,
                    "AV.1": float(av),
                }
            )
    return pl.DataFrame(rows)


class TestAggregate4yrAv:
    def test_two_seasons_projects_yr2_yr3(self):
        df = _make_draft_df([{"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0]}])
        result = aggregate_4yr_av(df, MockModel())

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        # MockModel returns 3.0 per projected year
        assert row["proj_yr2"] == pytest.approx(3.0)
        assert row["proj_yr3"] == pytest.approx(3.0)
        assert row["total_4yr_av"] == pytest.approx(8.0 + 10.0 + 3.0 + 3.0)
        assert row["is_projected"] is True

    def test_four_seasons_no_projection(self):
        df = _make_draft_df([
            {"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0, 7.0, 5.0]}
        ])
        result = aggregate_4yr_av(df, MockModel())

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        assert row["total_4yr_av"] == pytest.approx(8.0 + 10.0 + 7.0 + 5.0)
        assert row["is_projected"] is False
        assert row["proj_yr2"] == pytest.approx(0.0)
        assert row["proj_yr3"] == pytest.approx(0.0)

    def test_three_seasons_projects_only_yr3(self):
        df = _make_draft_df([
            {"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0, 7.0]}
        ])
        result = aggregate_4yr_av(df, MockModel())

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        assert row["obs_yr2"] == pytest.approx(7.0)
        assert row["proj_yr2"] == pytest.approx(0.0)   # observed, not projected
        assert row["proj_yr3"] == pytest.approx(3.0)   # model
        assert row["total_4yr_av"] == pytest.approx(8.0 + 10.0 + 7.0 + 3.0)
        assert row["is_projected"] is True

    def test_unknown_position_proj_defaults_to_zero(self):
        df = _make_draft_df([
            {"name": "Alice", "pos": "UNKNOWN_POS", "pick": 5, "av": [8.0, 10.0]}
        ])
        result = aggregate_4yr_av(df, MockModel())

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        assert row["proj_yr2"] == pytest.approx(0.0)
        assert row["proj_yr3"] == pytest.approx(0.0)
        assert row["total_4yr_av"] == pytest.approx(8.0 + 10.0)

    def test_sorted_by_pick(self):
        df = _make_draft_df([
            {"name": "Bob", "pos": "WR", "pick": 50, "av": [3.0, 4.0]},
            {"name": "Alice", "pos": "QB", "pick": 5, "av": [8.0, 10.0]},
        ])
        result = aggregate_4yr_av(df, MockModel())
        assert result["Pick"].to_list() == [5, 50]

    def test_output_schema(self):
        df = _make_draft_df([{"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0]}])
        result = aggregate_4yr_av(df, MockModel())
        required = {
            "Player", "Pos", "Pick", "Draft Year",
            "obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3",
            "proj_yr2", "proj_yr3", "total_4yr_av", "is_projected",
        }
        assert required.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# compute_surplus_av
# ---------------------------------------------------------------------------


class TestComputeSurplusAv:
    def test_surplus_equals_total_minus_eavar(self, tmp_path):
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)

        players_df = pl.DataFrame(
            [
                {
                    "Player": "Alice", "Pos": "WR", "Pick": 5, "Draft Year": 2022,
                    "obs_yr0": 8.0, "obs_yr1": 10.0, "obs_yr2": None,
                    "obs_yr3": None, "proj_yr2": 3.0, "proj_yr3": 3.0,
                    "total_4yr_av": 24.0, "is_projected": True,
                }
            ]
        )

        result = compute_surplus_av(players_df, eavar_path=eavar_path)
        # EAVAR for pick 5 = 20.0 - 5 = 15.0
        assert "surplus_av" in result.columns
        expected = 24.0 - 15.0
        assert result["surplus_av"][0] == pytest.approx(expected, abs=0.05)

    def test_pick_not_in_eavar_gives_null(self, tmp_path):
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)  # only picks 1–10

        players_df = pl.DataFrame(
            [
                {
                    "Player": "Bob", "Pos": "OT", "Pick": 999, "Draft Year": 2022,
                    "obs_yr0": 5.0, "obs_yr1": 5.0, "obs_yr2": None,
                    "obs_yr3": None, "proj_yr2": 3.0, "proj_yr3": 3.0,
                    "total_4yr_av": 16.0, "is_projected": True,
                }
            ]
        )

        result = compute_surplus_av(players_df, eavar_path=eavar_path)
        assert result["surplus_av"][0] is None

    def test_eavar_columns_added(self, tmp_path):
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)

        players_df = pl.DataFrame(
            [
                {
                    "Player": "Alice", "Pos": "WR", "Pick": 1, "Draft Year": 2022,
                    "obs_yr0": 10.0, "obs_yr1": 12.0, "obs_yr2": None,
                    "obs_yr3": None, "proj_yr2": 4.0, "proj_yr3": 4.0,
                    "total_4yr_av": 30.0, "is_projected": True,
                }
            ]
        )

        result = compute_surplus_av(players_df, eavar_path=eavar_path)
        for col in ("eavar", "eavar_upper", "eavar_lower", "replacement_level"):
            assert col in result.columns

    def test_multiple_players_correct_join(self, tmp_path):
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)

        players_df = pl.DataFrame(
            [
                {
                    "Player": "Alice", "Pos": "WR", "Pick": 1, "Draft Year": 2022,
                    "obs_yr0": 10.0, "obs_yr1": 10.0, "obs_yr2": None,
                    "obs_yr3": None, "proj_yr2": 3.0, "proj_yr3": 3.0,
                    "total_4yr_av": 26.0, "is_projected": True,
                },
                {
                    "Player": "Bob", "Pos": "DE", "Pick": 3, "Draft Year": 2022,
                    "obs_yr0": 5.0, "obs_yr1": 6.0, "obs_yr2": None,
                    "obs_yr3": None, "proj_yr2": 3.0, "proj_yr3": 3.0,
                    "total_4yr_av": 17.0, "is_projected": True,
                },
            ]
        )

        result = compute_surplus_av(players_df, eavar_path=eavar_path)
        alice_row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        bob_row = result.filter(pl.col("Player") == "Bob").row(0, named=True)
        # eavar for pick 1 = 19.0, pick 3 = 17.0
        assert alice_row["surplus_av"] == pytest.approx(26.0 - 19.0, abs=0.05)
        assert bob_row["surplus_av"] == pytest.approx(17.0 - 17.0, abs=0.05)
