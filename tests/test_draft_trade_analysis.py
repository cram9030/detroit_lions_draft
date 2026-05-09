"""Tests for analyze_draft_trades in src/trade_value.py."""

from datetime import date

import polars as pl
import pytest

from src.trade_value import analyze_draft_trades

_EXPECTED_COLUMNS = [
    "trade_id",
    "team_traded_with",
    "picks_received",
    "picks_gave",
    "fitz_spiel_value",
    "fitz_spiel_picks",
    "jj_value",
    "jj_picks",
    "pff_value",
    "pff_picks",
    "rich_hill_value",
    "rich_hill_picks",
    "eaar_value",
    "eaar_picks",
]


def _make_ec_trades() -> pl.DataFrame:
    """2021 Eagles-Cowboys draft day trade (real trade_id=1558).

    PHI received pick 10 (DeVonta Smith) from DAL.
    PHI gave picks 12 (Micah Parsons) and 84 (Chauncey Golston) to DAL.
    JJ chart: pick 10 = 1300, pick 12 = 1200, pick 84 = 170.
    PHI net JJ = 1300 - (1200 + 170) = -70.0
    """
    return pl.DataFrame({
        "trade_id": [1558.0, 1558.0, 1558.0],
        "season": [2021, 2021, 2021],
        "trade_date": [date(2021, 4, 29), date(2021, 4, 29), date(2021, 4, 29)],
        "gave": ["DAL", "PHI", "PHI"],
        "received": ["PHI", "DAL", "DAL"],
        "pick_season": [2021.0, 2021.0, 2021.0],
        "pick_round": [1.0, 1.0, 3.0],
        "pick_number": [10.0, 12.0, 84.0],
        "conditional": [0.0, 0.0, 0.0],
        "pfr_id": ["SmitDe07", "ParsMi00", "GolsCh00"],
        "pfr_name": ["DeVonta Smith", "Micah Parsons", "Chauncey Golston"],
    })


def _make_players_all_2021() -> pl.DataFrame:
    return pl.DataFrame({
        "pfr_id": ["SmitDe07", "ParsMi00", "GolsCh00"],
        "draft_year": [2021, 2021, 2021],
    })


class TestAnalyzeDraftTrades:

    def test_returns_dataframe(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        assert isinstance(analyze_draft_trades("PHI", 2021), pl.DataFrame)

    def test_one_row_returned(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        assert len(analyze_draft_trades("PHI", 2021)) == 1

    def test_team_traded_with(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        assert result["team_traded_with"][0] == "DAL"

    def test_picks_received(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        assert result["picks_received"][0] == "10"

    def test_picks_gave(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        assert result["picks_gave"][0] == "12,84"

    def test_jj_value(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        # PHI received pick 10 (1300.0), gave picks 12 (1200.0) + 84 (170.0) = 1370.0
        assert result["jj_value"][0] == pytest.approx(-70.0)

    def test_jj_picks_uses_abs_net_value(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        # find_pick_combination(abs(-70.0)=70.0, "jimmy_johnson") should return picks
        assert result["jj_picks"][0] != ""

    def test_all_14_columns_present(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        for col in _EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_excludes_player_from_wrong_draft_year(self, mocker):
        mocker.patch("nflreadpy.load_trades", return_value=_make_ec_trades())
        bad_players = pl.DataFrame({
            "pfr_id": ["SmitDe07", "ParsMi00", "GolsCh00"],
            "draft_year": [2019, 2021, 2021],  # DeVonta Smith drafted 2019
        })
        mocker.patch("nflreadpy.load_players", return_value=bad_players)
        assert len(analyze_draft_trades("PHI", 2021)) == 0

    def test_null_pfr_id_rows_are_valid(self, mocker):
        pure_pick_trade = pl.DataFrame({
            "trade_id": [9999.0, 9999.0],
            "season": [2021, 2021],
            "trade_date": [date(2021, 4, 29), date(2021, 4, 29)],
            "gave": ["PHI", "DAL"],
            "received": ["DAL", "PHI"],
            "pick_season": [2021.0, 2021.0],
            "pick_round": [2.0, 1.0],
            "pick_number": [40.0, 15.0],
            "conditional": [0.0, 0.0],
            "pfr_id": pl.Series([None, None], dtype=pl.String),
            "pfr_name": pl.Series([None, None], dtype=pl.String),
        })
        mocker.patch("nflreadpy.load_trades", return_value=pure_pick_trade)
        mocker.patch("nflreadpy.load_players", return_value=pl.DataFrame(
            {"pfr_id": pl.Series([], dtype=pl.String), "draft_year": pl.Series([], dtype=pl.Int64)}
        ))
        assert len(analyze_draft_trades("PHI", 2021)) == 1

    def test_excludes_trade_with_no_picks(self, mocker):
        no_pick_trade = pl.DataFrame({
            "trade_id": [8888.0],
            "season": [2021],
            "trade_date": [date(2021, 4, 29)],
            "gave": ["PHI"],
            "received": ["DAL"],
            "pick_season": pl.Series([None], dtype=pl.Float64),
            "pick_round": pl.Series([None], dtype=pl.Float64),
            "pick_number": pl.Series([None], dtype=pl.Float64),
            "conditional": pl.Series([None], dtype=pl.Float64),
            "pfr_id": ["FlacJo00"],
            "pfr_name": ["Joe Flacco"],
        })
        mocker.patch("nflreadpy.load_trades", return_value=no_pick_trade)
        mocker.patch("nflreadpy.load_players", return_value=pl.DataFrame(
            {"pfr_id": ["FlacJo00"], "draft_year": [2021]}
        ))
        assert len(analyze_draft_trades("PHI", 2021)) == 0

    def test_round_only_pick_uses_mid_round_estimate(self, mocker):
        # PHI gives a round-2 pick (no pick_number), receives pick 10
        round_only_trade = pl.DataFrame({
            "trade_id": [7777.0, 7777.0],
            "season": [2021, 2021],
            "trade_date": [date(2021, 4, 29), date(2021, 4, 29)],
            "gave": ["PHI", "DAL"],
            "received": ["DAL", "PHI"],
            "pick_season": [2021.0, 2021.0],
            "pick_round": [2.0, 1.0],
            "pick_number": pl.Series([None, 10.0], dtype=pl.Float64),
            "conditional": [0.0, 0.0],
            "pfr_id": pl.Series([None, None], dtype=pl.String),
            "pfr_name": pl.Series([None, None], dtype=pl.String),
        })
        mocker.patch("nflreadpy.load_trades", return_value=round_only_trade)
        mocker.patch("nflreadpy.load_players", return_value=pl.DataFrame(
            {"pfr_id": pl.Series([], dtype=pl.String), "draft_year": pl.Series([], dtype=pl.Int64)}
        ))
        result = analyze_draft_trades("PHI", 2021)
        # Round 2 estimate = (2 - 1) * 32 + 16 = 48
        assert result["picks_gave"][0] == "48"

    def test_empty_result_schema(self, mocker):
        # Trades from a different year — filter returns no rows
        other_year_trades = pl.DataFrame({
            "trade_id": [1558.0, 1558.0, 1558.0],
            "season": [2020, 2020, 2020],
            "trade_date": [date(2020, 4, 23), date(2020, 4, 23), date(2020, 4, 23)],
            "gave": ["DAL", "PHI", "PHI"],
            "received": ["PHI", "DAL", "DAL"],
            "pick_season": [2020.0, 2020.0, 2020.0],
            "pick_round": [1.0, 1.0, 3.0],
            "pick_number": [10.0, 12.0, 84.0],
            "conditional": [0.0, 0.0, 0.0],
            "pfr_id": ["SmitDe07", "ParsMi00", "GolsCh00"],
            "pfr_name": ["DeVonta Smith", "Micah Parsons", "Chauncey Golston"],
        })
        mocker.patch("nflreadpy.load_trades", return_value=other_year_trades)
        mocker.patch("nflreadpy.load_players", return_value=_make_players_all_2021())
        result = analyze_draft_trades("PHI", 2021)
        assert len(result) == 0
        for col in _EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing column in empty result: {col}"
