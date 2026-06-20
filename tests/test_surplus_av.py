"""Unit tests for src/surplus_av.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.positions import canonicalize_positions
from src.surplus_av import (
    _normalize_pos,
    aggregate_model_av,
    aggregate_observed_av,
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

    def predict(self, position: str, observed_av: list[float], pick: int | None = None) -> dict[str, Any]:
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

        assert set(df.columns) == {"Player", "Pos", "Pick", "Round", "Draft Year", "years_from_draft", "AV.1"}
        assert df["Player"].to_list().count("Carol") == 0  # GB player excluded
        assert set(df["Player"].unique().to_list()) == {"Alice", "Bob"}

    def test_filters_to_requested_team(self, tmp_path):
        rows = [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "GB", "Team": "GB", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "QB"},
        ]
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, rows)
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "9", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "GB", "Team": "GB", "Pick": "10",
             "Draft Year": "2022", "Season": "2023", "AV.1": "4", "Pos": "QB"},
        ])

        df = load_team_draft_class("GB", 2022, raw_dir=raw_dir)
        assert df["Player"].unique().to_list() == ["Bob"]

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

    def test_player_on_different_team_excluded_from_draft_class(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "J. Houston", "Draft Team": "DET", "Team": "DET",
             "Pick": "6", "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "LB"},
            {"Player": "Alice", "Draft Team": "DET", "Team": "DET",
             "Pick": "10", "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
        ])
        # yr1: Houston cut by DET, signs with DAL — AV earned for DAL must not count
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "J. Houston", "Draft Team": "DET", "Team": "DAL",
             "Pick": "6", "Draft Year": "2022", "Season": "2023", "AV.1": "1", "Pos": "LB"},
            {"Player": "Alice", "Draft Team": "DET", "Team": "DET",
             "Pick": "10", "Draft Year": "2022", "Season": "2023", "AV.1": "9", "Pos": "WR"},
        ])
        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        yr1_players = df.filter(pl.col("years_from_draft") == 1)["Player"].to_list()
        assert "J. Houston" not in yr1_players

    def test_mid_season_trade_away_from_draft_team_included(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Team": "DET",
             "Pick": "5", "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
        ])
        # yr1: traded mid-season (DET,DAL) — row should be kept because DET is present
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Team": "DET,DAL",
             "Pick": "5", "Draft Year": "2022", "Season": "2023", "AV.1": "6", "Pos": "WR"},
        ])
        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        yr1 = df.filter(pl.col("years_from_draft") == 1)
        assert not yr1.is_empty()
        assert yr1["AV.1"][0] == pytest.approx(6.0)


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
            def predict(self, position, observed_av, pick=None):
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
            def predict(self, position, observed_av, pick=None):
                called_with["pos"] = position
                return super().predict(position, observed_av)

        model = TrackingModel()
        project_player_seasons(model, "Alice", "LDE", [8.0, 10.0])
        assert called_with["pos"] == "DE"

    def test_override_applied(self):
        called_with = {}

        class TrackingModel(MockModel):
            def predict(self, position, observed_av, pick=None):
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
                    "Round": p.get("round", 1),
                    "Draft Year": 2022,
                    "years_from_draft": yr,
                    "AV.1": float(av),
                }
            )
    return pl.DataFrame(rows)


class TestAggregateObservedAv:
    def test_four_seasons_correct_totals(self):
        df = _make_draft_df([
            {"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0, 7.0, 5.0]}
        ])
        result = aggregate_observed_av(df)

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        assert row["obs_yr0"] == pytest.approx(8.0)
        assert row["obs_yr1"] == pytest.approx(10.0)
        assert row["obs_yr2"] == pytest.approx(7.0)
        assert row["obs_yr3"] == pytest.approx(5.0)
        assert row["total_4yr_av"] == pytest.approx(30.0)

    def test_null_yr2_yr3_treated_as_zero(self):
        """Player with no yr2/yr3 data (cut after yr1) contributes 0 for those seasons."""
        df = _make_draft_df([
            {"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0, 7.0, 5.0]},
            {"name": "Bob", "pos": "QB", "pick": 10, "av": [3.0, 2.0]},
        ])
        result = aggregate_observed_av(df)

        bob = result.filter(pl.col("Player") == "Bob").row(0, named=True)
        assert bob["obs_yr2"] == pytest.approx(0.0)
        assert bob["obs_yr3"] == pytest.approx(0.0)
        assert bob["total_4yr_av"] == pytest.approx(5.0)

    def test_sorted_by_pick(self):
        df = _make_draft_df([
            {"name": "Bob", "pos": "WR", "pick": 50, "av": [3.0, 4.0, 2.0, 1.0]},
            {"name": "Alice", "pos": "QB", "pick": 5, "av": [8.0, 10.0, 7.0, 5.0]},
        ])
        result = aggregate_observed_av(df)
        assert result["Pick"].to_list() == [5, 50]

    def test_output_schema(self):
        df = _make_draft_df([
            {"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0, 7.0, 5.0]}
        ])
        result = aggregate_observed_av(df)
        expected = {"Player", "Pos", "Pick", "Round", "Draft Year", "obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3", "total_4yr_av"}
        assert set(result.columns) == expected
        for col in ("proj_yr2", "proj_yr3", "is_projected"):
            assert col not in result.columns


class TestAggregateModelAv:
    def test_two_seasons_projects_yr2_yr3(self):
        df = _make_draft_df([{"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0]}])
        result = aggregate_model_av(df, MockModel())

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        # MockModel returns 3.0 per projected year
        assert row["proj_yr2"] == pytest.approx(3.0)
        assert row["proj_yr3"] == pytest.approx(3.0)
        assert row["total_4yr_av"] == pytest.approx(8.0 + 10.0 + 3.0 + 3.0)
        assert row["is_projected"] is True

    def test_three_seasons_projects_only_yr3(self):
        df = _make_draft_df([
            {"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0, 7.0]}
        ])
        result = aggregate_model_av(df, MockModel())

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
        result = aggregate_model_av(df, MockModel())

        row = result.filter(pl.col("Player") == "Alice").row(0, named=True)
        assert row["proj_yr2"] == pytest.approx(0.0)
        assert row["proj_yr3"] == pytest.approx(0.0)
        assert row["total_4yr_av"] == pytest.approx(8.0 + 10.0)

    def test_sorted_by_pick(self):
        df = _make_draft_df([
            {"name": "Bob", "pos": "WR", "pick": 50, "av": [3.0, 4.0]},
            {"name": "Alice", "pos": "QB", "pick": 5, "av": [8.0, 10.0]},
        ])
        result = aggregate_model_av(df, MockModel())
        assert result["Pick"].to_list() == [5, 50]

    def test_output_schema(self):
        df = _make_draft_df([{"name": "Alice", "pos": "WR", "pick": 5, "av": [8.0, 10.0]}])
        result = aggregate_model_av(df, MockModel())
        required = {
            "Player", "Pos", "Pick", "Round", "Draft Year",
            "obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3",
            "proj_yr2", "proj_yr3", "total_4yr_av", "is_projected",
        }
        assert required.issubset(set(result.columns))

    def test_generalist_position_without_override_projects_zero(self):
        """Baseline: OL player (unknown to model) gets 0.0 projections without overrides."""
        df = _make_draft_df([{"name": "Bob", "pos": "OL", "pick": 210, "av": [1.0, 5.0]}])
        result = aggregate_model_av(df, MockModel())
        row = result.filter(pl.col("Player") == "Bob").row(0, named=True)
        assert row["proj_yr2"] == pytest.approx(0.0)
        assert row["proj_yr3"] == pytest.approx(0.0)

    def test_generalist_position_with_override_uses_model(self):
        """With position_overrides, an OL→OG player receives model projections."""
        df = _make_draft_df([{"name": "Bob", "pos": "OL", "pick": 210, "av": [1.0, 5.0]}])
        result = aggregate_model_av(df, MockModel(), position_overrides={"Bob": "OG"})
        row = result.filter(pl.col("Player") == "Bob").row(0, named=True)
        assert row["proj_yr2"] == pytest.approx(3.0)
        assert row["proj_yr3"] == pytest.approx(3.0)
        assert row["total_4yr_av"] == pytest.approx(1.0 + 5.0 + 3.0 + 3.0)


# ---------------------------------------------------------------------------
# compute_surplus_av
# ---------------------------------------------------------------------------


class TestComputeSurplusAv:
    def test_surplus_equals_av_above_replacement_minus_eavar(self, tmp_path):
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
        # eavar for pick 5 = 20.0 - 5 = 15.0; replacement_level = 4.5
        assert "surplus_av" in result.columns
        assert result["total_4yr_av_above_replacement"][0] == pytest.approx(24.0 - 4.5, abs=0.05)
        assert result["surplus_av"][0] == pytest.approx((24.0 - 4.5) - 15.0, abs=0.05)

    def test_pick_beyond_eavar_uses_last_pick(self, tmp_path):
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
        # pick 999 caps to pick 10; eavar for pick 10 = 20.0 - 10 = 10.0
        assert result["surplus_av"][0] is not None
        assert result["surplus_av"][0] == pytest.approx((16.0 - 4.5) - 10.0, abs=0.05)

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
        for col in ("eavar", "eavar_upper", "eavar_lower", "replacement_level", "total_4yr_av_above_replacement"):
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
        # eavar for pick 1 = 19.0, pick 3 = 17.0; replacement_level = 4.5
        assert alice_row["surplus_av"] == pytest.approx((26.0 - 4.5) - 19.0, abs=0.05)
        assert bob_row["surplus_av"] == pytest.approx((17.0 - 4.5) - 17.0, abs=0.05)


# ---------------------------------------------------------------------------
# Bug-fix tests — position canonicalization and null AV handling
# ---------------------------------------------------------------------------


class TestPositionCanonicalization:
    def test_specific_position_beats_generalist(self, tmp_path):
        """Player drafted as LDE (→ DE) then listed as DL (generalist) → canonical DE."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Hutch", "Draft Team": "DET", "Pick": "2",
             "Draft Year": "2022", "Season": "2022", "AV.1": "14", "Pos": "LDE"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Hutch", "Draft Team": "DET", "Pick": "2",
             "Draft Year": "2022", "Season": "2023", "AV.1": "10", "Pos": "DL"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert df.filter(pl.col("Player") == "Hutch")["Pos"].unique().to_list() == ["DE"]

    def test_most_common_specific_position_wins(self, tmp_path):
        """Player listed LB in yr0+yr1, DE in yr2 → LB wins (2 vs 1)."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Houston", "Draft Team": "DET", "Pick": "45",
             "Draft Year": "2022", "Season": "2022", "AV.1": "2", "Pos": "LB"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Houston", "Draft Team": "DET", "Pick": "45",
             "Draft Year": "2022", "Season": "2023", "AV.1": "3", "Pos": "LB"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2024, [
            {"Player": "Houston", "Draft Team": "DET", "Pick": "45",
             "Draft Year": "2022", "Season": "2024", "AV.1": "5", "Pos": "DE"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert df.filter(pl.col("Player") == "Houston")["Pos"].unique().to_list() == ["LB"]

    def test_yr0_position_breaks_tie(self, tmp_path):
        """Player listed as CB in yr0, S in yr1 (tied 1-1) → yr0 wins → CB."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Dave", "Draft Team": "DET", "Pick": "30",
             "Draft Year": "2022", "Season": "2022", "AV.1": "6", "Pos": "CB"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Dave", "Draft Team": "DET", "Pick": "30",
             "Draft Year": "2022", "Season": "2023", "AV.1": "5", "Pos": "S"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert df.filter(pl.col("Player") == "Dave")["Pos"].unique().to_list() == ["CB"]

    def test_only_generalist_positions_falls_back_to_yr0(self, tmp_path):
        """Player listed as OL every season → falls back to yr0 normalized position (OL)."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Lineman", "Draft Team": "DET", "Pick": "20",
             "Draft Year": "2022", "Season": "2022", "AV.1": "7", "Pos": "OL"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Lineman", "Draft Team": "DET", "Pick": "20",
             "Draft Year": "2022", "Season": "2023", "AV.1": "8", "Pos": "OL"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        assert df.filter(pl.col("Player") == "Lineman")["Pos"].unique().to_list() == ["OL"]

    def test_single_row_per_player_per_season(self, tmp_path):
        """After canonicalization, each player has at most one row per years_from_draft."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Hutch", "Draft Team": "DET", "Pick": "2",
             "Draft Year": "2022", "Season": "2022", "AV.1": "14", "Pos": "LDE"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Hutch", "Draft Team": "DET", "Pick": "2",
             "Draft Year": "2022", "Season": "2023", "AV.1": "10", "Pos": "DL"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2024, [
            {"Player": "Hutch", "Draft Team": "DET", "Pick": "2",
             "Draft Year": "2022", "Season": "2024", "AV.1": "11", "Pos": "DE"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        hutch_rows = df.filter(pl.col("Player") == "Hutch")
        # No duplicate (player, years_from_draft) combinations
        assert hutch_rows.shape[0] == hutch_rows.n_unique(subset=["Player", "years_from_draft"])


class TestCanonicalizePositionsBug:
    """Regression tests for the canonicalize_positions inner-join bug.

    Before the fix, canonicalize_positions built its lookup table exclusively
    from year-0 rows and used an inner join, silently dropping any player
    absent from the yr0 parquet (e.g. a player who earned 0 AV in their
    rookie year and was therefore omitted by Stathead).
    """

    def _make_df(self, rows: list[dict]) -> pl.DataFrame:
        schema = {
            "Player": pl.String,
            "Pos": pl.String,
            "Pick": pl.Int64,
            "Draft Year": pl.Int64,
            "years_from_draft": pl.Int64,
            "AV.1": pl.Float64,
        }
        return pl.DataFrame(rows, schema=schema)

    def test_player_with_no_yr0_row_is_not_dropped(self):
        """Bug: inner join on yr0_pos silently dropped players with no yr0 data."""
        df = self._make_df([
            # Anchor player present in yr0 — ensures yr0_pos is non-empty
            {"Player": "Anchor", "Pos": "QB", "Pick": 10, "Draft Year": 2022,
             "years_from_draft": 0, "AV.1": 5.0},
            # Manu-like player: only a yr1 row, no yr0 row at all
            {"Player": "Manu-Like", "Pos": "OT", "Pick": 126, "Draft Year": 2022,
             "years_from_draft": 1, "AV.1": 1.0},
        ])
        result = canonicalize_positions(df)
        assert "Manu-Like" in result["Player"].to_list(), (
            "canonicalize_positions dropped a player with no yr0 row (inner-join bug)"
        )

    def test_player_with_no_yr0_and_generalist_pos_is_not_dropped(self):
        """Bug: generalist Pos (OL) also excluded from pos_counts, compounding the inner-join drop."""
        df = self._make_df([
            {"Player": "Anchor", "Pos": "QB", "Pick": 10, "Draft Year": 2022,
             "years_from_draft": 0, "AV.1": 5.0},
            # OL is generalist — excluded from pos_counts AND absent from yr0_pos
            {"Player": "Manu-Like", "Pos": "OL", "Pick": 126, "Draft Year": 2022,
             "years_from_draft": 1, "AV.1": 1.0},
        ])
        result = canonicalize_positions(df)
        assert "Manu-Like" in result["Player"].to_list(), (
            "canonicalize_positions dropped a no-yr0 generalist player (double-exclusion bug)"
        )
        # Falls back to the generalist code when no specific position is available
        manu_pos = result.filter(pl.col("Player") == "Manu-Like")["Pos"][0]
        assert manu_pos == "OL"


class TestNullAvHandling:
    def test_null_av_rows_excluded(self, tmp_path):
        """Players with null AV.1 in a season have no row for that season."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "DET", "Pick": "50",
             "Draft Year": "2022", "Season": "2022", "AV.1": "3", "Pos": "DE"},
        ])
        # yr1 parquet: Alice has no AV (empty string → null), Bob has real data
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "DET", "Pick": "50",
             "Draft Year": "2022", "Season": "2023", "AV.1": "4", "Pos": "DE"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        # Alice's yr1 null row should be absent; she only has yr0
        alice_yfds = df.filter(pl.col("Player") == "Alice")["years_from_draft"].to_list()
        assert alice_yfds == [0]
        # Bob has both seasons
        bob_yfds = sorted(df.filter(pl.col("Player") == "Bob")["years_from_draft"].to_list())
        assert bob_yfds == [0, 1]

    def test_raises_when_required_season_has_no_av_data(self, tmp_path):
        """ValueError raised when yr1 parquet exists but ALL team rows have null AV.1."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "8", "Pos": "WR"},
        ])
        # yr1: entire DET contingent has null AV (stale parquet)
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "", "Pos": "WR"},
        ])

        with pytest.raises(ValueError, match="no finalized AV data"):
            load_team_draft_class("DET", 2022, raw_dir=raw_dir)

    def test_null_av_not_converted_to_zero(self, tmp_path):
        """A null AV.1 season must not produce a 0.0 row — it should be absent entirely."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2022", "AV.1": "10", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "6", "Pos": "QB"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            # Alice has null AV in yr1; Bob has real data
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2022", "Season": "2023", "AV.1": "", "Pos": "WR"},
            {"Player": "Bob", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2023", "AV.1": "7", "Pos": "QB"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        alice_rows = df.filter(pl.col("Player") == "Alice")
        # Should have yr0 only — no yr1 row with AV.1 == 0.0
        assert 0.0 not in alice_rows["AV.1"].to_list()
        assert 1 not in alice_rows["years_from_draft"].to_list()


class TestMissingYr0Player:
    """Players absent from yr0 (0 AV in rookie season) must not be dropped."""

    def test_player_absent_from_yr0_is_retained(self, tmp_path):
        """Player with 0 AV in yr0 (absent from yr0 parquet) appears in output via yr1 row."""
        raw_dir = tmp_path / "raw"
        # Anchor player — ensures yr0 validation passes
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Anchor", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "QB"},
        ])
        # Manu-like player: totally absent from yr0 parquet, only appears in yr1
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Anchor", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2023", "AV.1": "6", "Pos": "QB"},
            {"Player": "Rookie", "Draft Team": "DET", "Pick": "126",
             "Draft Year": "2022", "Season": "2023", "AV.1": "1", "Pos": "OT"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        rookie_rows = df.filter(pl.col("Player") == "Rookie")
        assert rookie_rows.shape[0] == 1
        assert rookie_rows["years_from_draft"][0] == 1
        assert rookie_rows["AV.1"][0] == pytest.approx(1.0)

    def test_generalist_only_no_yr0_uses_generalist_pos(self, tmp_path):
        """Player absent from yr0, only has generalist Pos in yr1 — retained with generalist pos."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Anchor", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "QB"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Anchor", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2023", "AV.1": "6", "Pos": "QB"},
            {"Player": "Lineman", "Draft Team": "DET", "Pick": "126",
             "Draft Year": "2022", "Season": "2023", "AV.1": "1", "Pos": "OL"},
        ])

        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir)
        lineman_rows = df.filter(pl.col("Player") == "Lineman")
        assert lineman_rows.shape[0] == 1
        assert lineman_rows["Pos"][0] == "OL"

    def test_position_override_applied_at_load_time(self, tmp_path):
        """position_overrides resolve generalist codes before canonicalization."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2022, 2022, [
            {"Player": "Anchor", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2022", "AV.1": "5", "Pos": "QB"},
        ])
        _make_draft_parquet(raw_dir, 2022, 2023, [
            {"Player": "Anchor", "Draft Team": "DET", "Pick": "10",
             "Draft Year": "2022", "Season": "2023", "AV.1": "6", "Pos": "QB"},
            {"Player": "Giovanni Manu", "Draft Team": "DET", "Pick": "126",
             "Draft Year": "2022", "Season": "2023", "AV.1": "1", "Pos": "OL"},
        ])

        overrides = {"Giovanni Manu": "OT"}
        df = load_team_draft_class("DET", 2022, raw_dir=raw_dir, position_overrides=overrides)
        manu_rows = df.filter(pl.col("Player") == "Giovanni Manu")
        assert manu_rows.shape[0] == 1
        assert manu_rows["Pos"][0] == "OT"
