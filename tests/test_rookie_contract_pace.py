"""Unit tests for src/rookie_contract_pace.py."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.rookie_contract_pace import (
    allocate_required_av,
    available_rookie_years,
    build_position_reference_trajectories,
    compute_pace_requirements,
    find_closest_career_comps,
    find_next_year_comp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_draft_parquet(raw_dir: Path, draft_year: int, season: int, rows: list[dict[str, str]]) -> None:
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
    rows = [
        {
            "pick": str(p),
            "eavar": str(20.0 - p),
            "eavar_upper": str(21.0 - p),
            "eavar_lower": str(19.0 - p),
            "replacement_level": "4.5",
        }
        for p in range(1, 11)
    ]
    pl.DataFrame(rows).write_csv(path)


def _pos_stats(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# allocate_required_av
# ---------------------------------------------------------------------------


class TestAllocateRequiredAv:
    def test_no_remaining_years_returns_empty(self):
        pos_stats = _pos_stats([{"Pos": "WR", "years_from_draft": 2, "mean": 5.0}])
        assert allocate_required_av(10.0, "WR", [], pos_stats) == {}

    def test_diff_zero_or_negative_returns_zeros(self):
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 2, "mean": 5.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 3.0},
        ])
        assert allocate_required_av(0.0, "WR", [2, 3], pos_stats) == {2: 0.0, 3: 0.0}
        assert allocate_required_av(-5.0, "WR", [2, 3], pos_stats) == {2: 0.0, 3: 0.0}

    def test_proportional_split_by_position_mean(self):
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 2, "mean": 6.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 4.0},
        ])
        result = allocate_required_av(20.0, "WR", [2, 3], pos_stats)
        # 6:4 ratio of 20 -> 12, 8
        assert result[2] == pytest.approx(12.0)
        assert result[3] == pytest.approx(8.0)
        assert result[2] + result[3] == pytest.approx(20.0)

    def test_single_remaining_year_gets_full_diff(self):
        pos_stats = _pos_stats([{"Pos": "DE", "years_from_draft": 3, "mean": 9.0}])
        result = allocate_required_av(15.0, "DE", [3], pos_stats)
        assert result == {3: 15.0}

    def test_missing_position_falls_back_to_equal_split(self):
        pos_stats = _pos_stats([{"Pos": "WR", "years_from_draft": 2, "mean": 6.0}])
        result = allocate_required_av(10.0, "UNKNOWN", [2, 3], pos_stats)
        assert result == {2: 5.0, 3: 5.0}

    def test_zero_weight_years_fall_back_to_equal_split(self):
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 2, "mean": 0.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 0.0},
        ])
        result = allocate_required_av(10.0, "WR", [2, 3], pos_stats)
        assert result == {2: 5.0, 3: 5.0}

    def test_negative_position_mean_treated_as_zero_weight(self):
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 2, "mean": -1.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 8.0},
        ])
        result = allocate_required_av(8.0, "WR", [2, 3], pos_stats)
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(8.0)

    def test_custom_column_names(self):
        pos_stats = pl.DataFrame(
            [{"position": "WR", "yr": 2, "avg_av": 6.0}, {"position": "WR", "yr": 3, "avg_av": 4.0}]
        )
        result = allocate_required_av(
            10.0, "WR", [2, 3], pos_stats, year_col="yr", pos_col="position", value_col="avg_av"
        )
        assert result[2] == pytest.approx(6.0)
        assert result[3] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# available_rookie_years
# ---------------------------------------------------------------------------


class TestAvailableRookieYears:
    def test_only_required_files_present(self, tmp_path):
        (tmp_path / "draft2023_season2023.parquet").touch()
        (tmp_path / "draft2023_season2024.parquet").touch()
        assert available_rookie_years(2023, tmp_path) == [0, 1]

    def test_all_four_seasons_present(self, tmp_path):
        for s in (2023, 2024, 2025, 2026):
            (tmp_path / f"draft2023_season{s}.parquet").touch()
        assert available_rookie_years(2023, tmp_path) == [0, 1, 2, 3]

    def test_year2_present_year3_missing(self, tmp_path):
        for s in (2023, 2024, 2025):
            (tmp_path / f"draft2023_season{s}.parquet").touch()
        assert available_rookie_years(2023, tmp_path) == [0, 1, 2]

    def test_no_files_at_all_returns_empty(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        assert available_rookie_years(2025, tmp_path) == []

    def test_year0_missing_year1_present_is_honest(self, tmp_path):
        """Year 0 must actually exist too — it is not assumed present."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "draft2023_season2024.parquet").touch()
        assert available_rookie_years(2023, tmp_path) == [1]

    def test_only_year0_present(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "draft2023_season2023.parquet").touch()
        assert available_rookie_years(2023, tmp_path) == [0]


# ---------------------------------------------------------------------------
# compute_pace_requirements
# ---------------------------------------------------------------------------


class TestComputePaceRequirements:
    def _setup_two_season_class(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2023, 2023, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2023", "Season": "2023", "AV.1": "8", "Pos": "WR"},
        ])
        _make_draft_parquet(raw_dir, 2023, 2024, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2023", "Season": "2024", "AV.1": "6", "Pos": "WR"},
        ])
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 2, "mean": 6.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 4.0},
        ])
        return raw_dir, eavar_path, pos_stats

    def test_raises_when_rookie_contract_complete(self, tmp_path):
        raw_dir = tmp_path / "raw"
        for s in (2020, 2021, 2022, 2023):
            (raw_dir).mkdir(parents=True, exist_ok=True)
            (raw_dir / f"draft2020_season{s}.parquet").touch()
        with pytest.raises(ValueError, match="already completed"):
            compute_pace_requirements("DET", 2020, raw_dir=raw_dir)

    def test_required_av_and_split_match_position_shape(self, tmp_path):
        raw_dir, eavar_path, pos_stats = self._setup_two_season_class(tmp_path)
        pace_df = compute_pace_requirements(
            "DET", 2023, raw_dir=raw_dir, eavar_path=eavar_path, pos_stats=pos_stats
        )
        row = pace_df.filter(pl.col("Player") == "Alice").row(0, named=True)

        # pick 5 -> eavar = 20 - 5 = 15.0; replacement_level = 4.5 -> target_total_av = 19.5
        assert row["target_total_av"] == pytest.approx(19.5)
        assert row["total_observed_av"] == pytest.approx(14.0)  # 8 + 6
        diff = 19.5 - 14.0
        assert row["required_av_remaining"] == pytest.approx(diff, abs=0.05)

        # yr0/yr1 observed, yr2/yr3 required, split 6:4 per pos_stats
        assert row["yr0_type"] == "observed"
        assert row["yr1_type"] == "observed"
        assert row["yr2_type"] == "required"
        assert row["yr3_type"] == "required"
        assert row["yr0_av"] == pytest.approx(8.0)
        assert row["yr1_av"] == pytest.approx(6.0)
        assert row["yr2_av"] + row["yr3_av"] == pytest.approx(diff, abs=0.05)
        assert row["yr2_av"] / row["yr3_av"] == pytest.approx(6.0 / 4.0, abs=0.02)

    def test_single_observed_season_does_not_require_two(self, tmp_path):
        """Only year0 exists on disk — this must not raise, unlike load_team_draft_class."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2025, 2025, [
            {"Player": "Rookie", "Draft Team": "DET", "Pick": "8",
             "Draft Year": "2025", "Season": "2025", "AV.1": "6", "Pos": "WR"},
        ])
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 1, "mean": 8.0},
            {"Pos": "WR", "years_from_draft": 2, "mean": 6.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 4.0},
        ])
        pace_df = compute_pace_requirements(
            "DET", 2025, raw_dir=raw_dir, eavar_path=eavar_path, pos_stats=pos_stats
        )
        row = pace_df.filter(pl.col("Player") == "Rookie").row(0, named=True)
        assert row["yr0_type"] == "observed"
        assert row["yr0_av"] == pytest.approx(6.0)
        assert row["yr1_type"] == "required"
        assert row["yr2_type"] == "required"
        assert row["yr3_type"] == "required"
        assert row["total_observed_av"] == pytest.approx(6.0)

    def test_zero_observed_seasons_falls_back_to_nflreadr(self, tmp_path, mocker):
        """No Stathead season file exists at all — bootstrap the roster from nflreadpy."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)  # empty — no parquet files
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)
        pos_stats = _pos_stats([
            {"Pos": "S", "years_from_draft": y, "mean": 8.0 - y} for y in range(4)
        ])

        fallback_df = pl.DataFrame({
            "Draft Year": [2025],
            "Pick": [8],
            "Player": ["Freshly Drafted"],
            "dr_av": [None],
            "team": ["DET"],
            "round": [1],
            "position": ["SAF"],  # nflreadpy's safety code — must normalize to "S"
        })
        mocker.patch("src.data_ingest.load_nflreadr_draft_picks", return_value=fallback_df)

        pace_df = compute_pace_requirements(
            "DET", 2025, raw_dir=raw_dir, eavar_path=eavar_path, pos_stats=pos_stats
        )
        row = pace_df.filter(pl.col("Player") == "Freshly Drafted").row(0, named=True)
        assert row["Pos"] == "S"
        assert row["total_observed_av"] == pytest.approx(0.0)
        for y in range(4):
            assert row[f"yr{y}_type"] == "required"
        assert sum(row[f"yr{y}_av"] for y in range(4)) == pytest.approx(row["required_av_remaining"], abs=0.05)

    def test_pick_missing_from_stathead_added_with_zero_av(self, tmp_path, mocker):
        """Stathead omits picks with 0 AV in every observed season entirely (see
        zero-av-players) — such a pick must still appear in the pace table
        rather than silently vanishing."""
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2025, 2025, [
            {"Player": "Alice", "Draft Team": "DET", "Pick": "5",
             "Draft Year": "2025", "Season": "2025", "AV.1": "8", "Pos": "WR"},
        ])
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)
        pos_stats = _pos_stats(
            [{"Pos": "WR", "years_from_draft": y, "mean": 8.0 - y} for y in range(4)]
            + [{"Pos": "S", "years_from_draft": y, "mean": 8.0 - y} for y in range(4)]
        )

        roster_df = pl.DataFrame({
            "Draft Year": [2025, 2025],
            "Pick": [5, 9],
            "Player": ["Alice", "Ghost"],
            "dr_av": [8.0, None],
            "team": ["DET", "DET"],
            "round": [1, 1],
            "position": ["WR", "SAF"],
        })
        mocker.patch("src.data_ingest.load_nflreadr_draft_picks", return_value=roster_df)

        pace_df = compute_pace_requirements(
            "DET", 2025, raw_dir=raw_dir, eavar_path=eavar_path, pos_stats=pos_stats
        )
        players = pace_df["Player"].to_list()
        assert "Ghost" in players
        row = pace_df.filter(pl.col("Player") == "Ghost").row(0, named=True)
        assert row["Pos"] == "S"
        assert row["yr0_type"] == "observed"
        assert row["yr0_av"] == pytest.approx(0.0)
        assert row["total_observed_av"] == pytest.approx(0.0)

    def test_already_exceeded_target_gets_zero_required(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2023, 2023, [
            {"Player": "Star", "Draft Team": "DET", "Pick": "250",
             "Draft Year": "2023", "Season": "2023", "AV.1": "20", "Pos": "WR"},
        ])
        _make_draft_parquet(raw_dir, 2023, 2024, [
            {"Player": "Star", "Draft Team": "DET", "Pick": "250",
             "Draft Year": "2023", "Season": "2024", "AV.1": "20", "Pos": "WR"},
        ])
        eavar_path = tmp_path / "eavar.csv"
        _make_eavar_csv(eavar_path)
        pos_stats = _pos_stats([
            {"Pos": "WR", "years_from_draft": 2, "mean": 6.0},
            {"Pos": "WR", "years_from_draft": 3, "mean": 4.0},
        ])
        pace_df = compute_pace_requirements(
            "DET", 2023, raw_dir=raw_dir, eavar_path=eavar_path, pos_stats=pos_stats
        )
        row = pace_df.filter(pl.col("Player") == "Star").row(0, named=True)
        assert row["required_av_remaining"] <= 0
        assert row["yr2_av"] == pytest.approx(0.0)
        assert row["yr3_av"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_position_reference_trajectories
# ---------------------------------------------------------------------------


class TestBuildPositionReferenceTrajectories:
    def test_only_fully_realized_careers_included(self, tmp_path):
        raw_dir = tmp_path / "raw"
        # Old player: drafted 2015, all 4 seasons exist -> included
        for i, season in enumerate([2015, 2016, 2017, 2018]):
            _make_draft_parquet(raw_dir, 2015, season, [
                {"Player": "OldTimer", "Draft Team": "DET", "Pick": "10",
                 "Draft Year": "2015", "Season": str(season), "AV.1": str(5 + i), "Pos": "WR"},
            ])
        # Recent player: drafted 2023, only 2 seasons -> excluded (career incomplete)
        _make_draft_parquet(raw_dir, 2023, 2023, [
            {"Player": "Rookie", "Draft Team": "GB", "Pick": "20",
             "Draft Year": "2023", "Season": "2023", "AV.1": "3", "Pos": "WR"},
        ])
        _make_draft_parquet(raw_dir, 2023, 2024, [
            {"Player": "Rookie", "Draft Team": "GB", "Pick": "20",
             "Draft Year": "2023", "Season": "2024", "AV.1": "4", "Pos": "WR"},
        ])

        ref_df = build_position_reference_trajectories(raw_dir=raw_dir)
        players = ref_df["Player"].to_list()
        assert "OldTimer" in players
        assert "Rookie" not in players

        row = ref_df.filter(pl.col("Player") == "OldTimer").row(0, named=True)
        assert row["yr0"] == pytest.approx(5.0)
        assert row["yr1"] == pytest.approx(6.0)
        assert row["yr2"] == pytest.approx(7.0)
        assert row["yr3"] == pytest.approx(8.0)
        assert row["Draft Team"] == "DET"

    def test_position_normalized(self, tmp_path):
        raw_dir = tmp_path / "raw"
        for season in (2015, 2016, 2017, 2018):
            _make_draft_parquet(raw_dir, 2015, season, [
                {"Player": "Edge", "Draft Team": "DET", "Pick": "10",
                 "Draft Year": "2015", "Season": str(season), "AV.1": "5", "Pos": "LDE"},
            ])
        ref_df = build_position_reference_trajectories(raw_dir=raw_dir)
        assert ref_df.filter(pl.col("Player") == "Edge")["Pos"][0] == "DE"

    def test_empty_directory_returns_empty_frame(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_draft_parquet(raw_dir, 2023, 2023, [
            {"Player": "Rookie", "Draft Team": "GB", "Pick": "20",
             "Draft Year": "2023", "Season": "2023", "AV.1": "3", "Pos": "WR"},
        ])
        ref_df = build_position_reference_trajectories(raw_dir=raw_dir)
        assert ref_df.is_empty()

    def test_min_draft_year_excludes_older_players(self, tmp_path):
        raw_dir = tmp_path / "raw"
        for season in (2000, 2001, 2002, 2003):
            _make_draft_parquet(raw_dir, 2000, season, [
                {"Player": "Ancient", "Draft Team": "DET", "Pick": "10",
                 "Draft Year": "2000", "Season": str(season), "AV.1": "5", "Pos": "WR"},
            ])
        for season in (2015, 2016, 2017, 2018):
            _make_draft_parquet(raw_dir, 2015, season, [
                {"Player": "Recent", "Draft Team": "DET", "Pick": "12",
                 "Draft Year": "2015", "Season": str(season), "AV.1": "5", "Pos": "WR"},
            ])
        ref_df = build_position_reference_trajectories(raw_dir=raw_dir, min_draft_year=2010)
        players = ref_df["Player"].to_list()
        assert "Recent" in players
        assert "Ancient" not in players


# ---------------------------------------------------------------------------
# find_closest_career_comps
# ---------------------------------------------------------------------------


def _reference_df() -> pl.DataFrame:
    return pl.DataFrame([
        # Exact match, different team, older
        {"Player": "ExactMatch", "Pos": "WR", "Pick": 10, "Round": 1, "Draft Year": 2010,
         "Draft Team": "GB", "yr0": 8.0, "yr1": 10.0, "yr2": 7.0, "yr3": 5.0},
        # Very close, same team as query, older
        {"Player": "CloseSameTeamOld", "Pos": "WR", "Pick": 12, "Round": 1, "Draft Year": 2008,
         "Draft Team": "DET", "yr0": 8.0, "yr1": 10.0, "yr2": 7.0, "yr3": 5.4},
        # Very close, same team as query, more recent
        {"Player": "CloseSameTeamRecent", "Pos": "WR", "Pick": 15, "Round": 1, "Draft Year": 2018,
         "Draft Team": "DET", "yr0": 8.0, "yr1": 10.0, "yr2": 7.2, "yr3": 5.0},
        # Very close, different team
        {"Player": "CloseOtherTeam", "Pos": "WR", "Pick": 20, "Round": 1, "Draft Year": 2020,
         "Draft Team": "KC", "yr0": 8.0, "yr1": 10.0, "yr2": 7.0, "yr3": 5.3},
        # Far away
        {"Player": "FarAway", "Pos": "WR", "Pick": 200, "Round": 6, "Draft Year": 2012,
         "Draft Team": "NYJ", "yr0": 1.0, "yr1": 1.0, "yr2": 1.0, "yr3": 1.0},
        # Different position entirely
        {"Player": "WrongPos", "Pos": "DE", "Pick": 10, "Round": 1, "Draft Year": 2015,
         "Draft Team": "DET", "yr0": 8.0, "yr1": 10.0, "yr2": 7.0, "yr3": 5.0},
    ])


class TestFindClosestCareerComps:
    def test_returns_n_neighbors_sorted_by_distance(self):
        ref_df = _reference_df()
        result = find_closest_career_comps([8.0, 10.0, 7.0, 5.0], "WR", "LAR", ref_df, n_neighbors=3)
        assert len(result) == 3
        dists = result["distance"].to_list()
        assert dists == sorted(dists)
        assert "FarAway" not in result["Player"].to_list()
        assert "WrongPos" not in result["Player"].to_list()

    def test_close_group_prefers_same_team_then_recency(self):
        ref_df = _reference_df()
        # Query team DET; several candidates within close_tolerance of the exact match (distance 0).
        result = find_closest_career_comps(
            [8.0, 10.0, 7.0, 5.0], "WR", "DET", ref_df, n_neighbors=1, close_tolerance=1.0
        )
        assert result["Player"].to_list() == ["CloseSameTeamRecent"]

    def test_fills_remaining_slots_from_farther_candidates(self):
        ref_df = _reference_df()
        result = find_closest_career_comps(
            [8.0, 10.0, 7.0, 5.0], "WR", "DET", ref_df, n_neighbors=5, close_tolerance=0.01
        )
        # Only ExactMatch is within the tight tolerance; the rest fill in by distance.
        assert len(result) == 5
        assert result["Player"].to_list()[0] == "ExactMatch"
        assert "FarAway" in result["Player"].to_list()

    def test_excludes_players(self):
        ref_df = _reference_df()
        result = find_closest_career_comps(
            [8.0, 10.0, 7.0, 5.0], "WR", "DET", ref_df, n_neighbors=3,
            exclude_players={"CloseSameTeamRecent", "CloseSameTeamOld"},
        )
        players = result["Player"].to_list()
        assert "CloseSameTeamRecent" not in players
        assert "CloseSameTeamOld" not in players

    def test_no_players_at_position_returns_empty(self):
        ref_df = _reference_df()
        result = find_closest_career_comps([1.0, 1.0, 1.0, 1.0], "QB", "DET", ref_df)
        assert result.is_empty()
        assert "distance" in result.columns

    def test_fewer_candidates_than_n_neighbors(self):
        ref_df = _reference_df().filter(pl.col("Player").is_in(["ExactMatch", "FarAway"]))
        result = find_closest_career_comps([8.0, 10.0, 7.0, 5.0], "WR", "DET", ref_df, n_neighbors=3)
        assert len(result) == 2


class TestFindNextYearComp:
    def test_matches_on_single_year_only(self):
        ref_df = _reference_df()
        # FarAway is very different everywhere else, but its yr0 (1.0) exactly matches
        # here — proving the match is on this one year, not the whole profile.
        result = find_next_year_comp(1.0, 0, "WR", "LAR", ref_df, close_tolerance=0.01)
        assert len(result) == 1
        assert result["Player"][0] == "FarAway"
        assert result["distance"][0] == pytest.approx(0.0)

    def test_prefers_same_team_then_recency_among_close(self):
        ref_df = _reference_df()
        # yr3 target 5.0: ExactMatch=5.0, CloseSameTeamOld=5.4, CloseSameTeamRecent=5.0(!), CloseOtherTeam=5.3
        # Within close_tolerance=1.0 of best (0.0), same-team candidates present -> most recent DET wins.
        result = find_next_year_comp(5.0, 3, "WR", "DET", ref_df, close_tolerance=1.0)
        assert len(result) == 1
        assert result["Player"][0] == "CloseSameTeamRecent"

    def test_returns_single_row(self):
        ref_df = _reference_df()
        result = find_next_year_comp(1.0, 0, "WR", "DET", ref_df)
        assert len(result) == 1

    def test_no_players_at_position_returns_empty(self):
        ref_df = _reference_df()
        result = find_next_year_comp(5.0, 2, "QB", "DET", ref_df)
        assert result.is_empty()
        assert "distance" in result.columns

    def test_excludes_players(self):
        ref_df = _reference_df()
        result = find_next_year_comp(
            5.0, 3, "WR", "DET", ref_df, close_tolerance=1.0,
            exclude_players={"CloseSameTeamRecent"},
        )
        assert result["Player"][0] != "CloseSameTeamRecent"
