"""Tests for src/pfr_downloader.py and src/scraper_utils.py (shared utilities)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pfr_downloader import (
    build_pfr_url,
    expand_iterations,
    is_pfr_blocked,
    load_pfr_config,
    make_pfr_output_path,
    parse_pfr_table,
    unwrap_pfr_comments,
)
from src.scraper_utils import load_progress, save_progress

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pfr"


# =============================================================================
# URL building
# =============================================================================


class TestBuildPfrUrl:
    def test_team_substitution(self):
        url = build_pfr_url(
            "https://www.pro-football-reference.com/teams/{team}/executives.htm",
            {"team": "det"},
        )
        assert url == "https://www.pro-football-reference.com/teams/det/executives.htm"

    def test_year_substitution(self):
        url = build_pfr_url(
            "https://www.pro-football-reference.com/years/{year}/",
            {"year": 2025},
        )
        assert url == "https://www.pro-football-reference.com/years/2025/"

    def test_multiple_variables(self):
        url = build_pfr_url(
            "https://www.pro-football-reference.com/teams/{team}/{year}.htm",
            {"team": "det", "year": 2025},
        )
        assert url == "https://www.pro-football-reference.com/teams/det/2025.htm"

    def test_missing_variable_raises_key_error(self):
        with pytest.raises(KeyError):
            build_pfr_url(
                "https://www.pro-football-reference.com/teams/{team}/",
                {},
            )

    def test_extra_variable_ignored(self):
        url = build_pfr_url(
            "https://www.pro-football-reference.com/teams/{team}/",
            {"team": "det", "year": 2025},
        )
        assert url == "https://www.pro-football-reference.com/teams/det/"


# =============================================================================
# Iteration expansion
# =============================================================================


class TestExpandIterations:
    def test_explicit_team_list(self):
        result = expand_iterations({"team": ["det", "chi", "gb"]})
        assert result == [{"team": "det"}, {"team": "chi"}, {"team": "gb"}]

    def test_year_range(self):
        result = expand_iterations({"year": {"start": 2020, "end": 2022}})
        assert result == [{"year": 2020}, {"year": 2021}, {"year": 2022}]

    def test_year_range_inclusive_both_ends(self):
        result = expand_iterations({"year": {"start": 2024, "end": 2024}})
        assert result == [{"year": 2024}]

    def test_empty_explicit_list(self):
        result = expand_iterations({"team": []})
        assert result == []

    def test_list_order_preserved(self):
        teams = ["det", "chi", "gb", "min"]
        result = expand_iterations({"team": teams})
        assert [r["team"] for r in result] == teams

    def test_year_range_ascending(self):
        result = expand_iterations({"year": {"start": 2021, "end": 2023}})
        years = [r["year"] for r in result]
        assert years == sorted(years)

    def test_invalid_spec_raises_value_error(self):
        with pytest.raises((ValueError, AssertionError)):
            expand_iterations({"year": "bad_value"})


# =============================================================================
# Comment unwrapping
# =============================================================================


class TestUnwrapPfrComments:
    def test_unwraps_single_commented_table(self):
        html = "<div><!-- <table id='x'><tr><td>data</td></tr></table> --></div>"
        result = unwrap_pfr_comments(html)
        assert "<table" in result
        assert "<!--" not in result

    def test_normal_table_is_unchanged(self):
        html = "<div><table id='x'><tr><td>data</td></tr></table></div>"
        result = unwrap_pfr_comments(html)
        assert result == html

    def test_multiple_commented_tables(self):
        html = (
            "<div>"
            "<!-- <table id='a'><tr><td>1</td></tr></table> -->"
            "<!-- <table id='b'><tr><td>2</td></tr></table> -->"
            "</div>"
        )
        result = unwrap_pfr_comments(html)
        assert result.count("<table") == 2

    def test_unwrapped_table_is_parseable(self):
        html = "<div><!-- <table id='t'><thead><tr><th>Col</th></tr></thead><tbody><tr><td>val</td></tr></tbody></table> --></div>"
        result = unwrap_pfr_comments(html)
        df = parse_pfr_table(result, table_id="t")
        assert df is not None
        assert not df.empty

    def test_non_table_comments_are_preserved(self):
        html = "<!-- a comment --> <div><table><tr><td>x</td></tr></table></div>"
        result = unwrap_pfr_comments(html)
        # regular (non-table) comment should remain or become harmless — no crash
        assert "<table" in result


# =============================================================================
# Table selection / parsing
# =============================================================================


class TestParsePfrTable:
    @staticmethod
    def _execs_html() -> str:
        return (FIXTURE_DIR / "executives_det.html").read_text()

    @staticmethod
    def _standings_html() -> str:
        return (FIXTURE_DIR / "standings_2025.html").read_text()

    def test_select_visible_table_by_id(self):
        df = parse_pfr_table(self._execs_html(), table_id="executives")
        assert df is not None
        assert not df.empty

    def test_select_by_index_zero(self):
        df = parse_pfr_table(self._execs_html(), table_index=0)
        assert df is not None
        assert not df.empty

    def test_missing_id_returns_none(self):
        df = parse_pfr_table(self._execs_html(), table_id="nonexistent_table")
        assert df is None

    def test_comment_wrapped_afc_table(self):
        df = parse_pfr_table(self._standings_html(), table_id="AFC")
        assert df is not None
        assert not df.empty

    def test_comment_wrapped_nfc_table(self):
        df = parse_pfr_table(self._standings_html(), table_id="NFC")
        assert df is not None
        assert not df.empty

    def test_no_selector_returns_first_table(self):
        html = "<html><body><table id='a'><tr><th>X</th></tr><tr><td>1</td></tr></table><table id='b'><tr><th>Y</th></tr><tr><td>2</td></tr></table></body></html>"
        df = parse_pfr_table(html)
        assert df is not None

    def test_empty_html_returns_none(self):
        assert parse_pfr_table("<html><body></body></html>") is None

    def test_returns_dataframe(self):
        df = parse_pfr_table(self._execs_html(), table_id="executives")
        assert isinstance(df, pd.DataFrame)

    def test_all_columns_are_strings(self):
        df = parse_pfr_table(self._execs_html(), table_id="executives")
        for col in df.columns:
            assert df[col].dtype == object, f"Column {col!r} is not string/object"


# =============================================================================
# Access / rate-limit wall detection
# =============================================================================


class TestIsPfrBlocked:
    def test_clean_page_not_blocked(self):
        html = "<html><body><table><tr><th>Tm</th></tr><tr><td>Detroit Lions</td></tr></table></body></html>"
        assert not is_pfr_blocked(html)

    def test_rate_limit_text_detected(self):
        html = "<html><body><p>You are being rate limited. Please try again later.</p></body></html>"
        assert is_pfr_blocked(html)

    def test_too_many_requests_detected(self):
        html = "<html><body>Too many requests. Please slow down.</body></html>"
        assert is_pfr_blocked(html)

    def test_access_denied_detected(self):
        html = "<html><body><h1>Access Denied</h1><p>You do not have permission.</p></body></html>"
        assert is_pfr_blocked(html)


# =============================================================================
# Output path naming
# =============================================================================


class TestMakePfrOutputPath:
    def test_team_with_suffix_csv(self, tmp_path):
        p = make_pfr_output_path(tmp_path, {"team": "det"}, "executives", use_csv=True)
        assert p == tmp_path / "det_executives.csv"

    def test_year_with_suffix_csv(self, tmp_path):
        p = make_pfr_output_path(tmp_path, {"year": 2025}, "afc", use_csv=True)
        assert p == tmp_path / "2025_afc.csv"

    def test_no_suffix(self, tmp_path):
        p = make_pfr_output_path(tmp_path, {"team": "det"}, table_suffix=None, use_csv=True)
        assert p == tmp_path / "det.csv"

    def test_parquet_extension(self, tmp_path):
        p = make_pfr_output_path(tmp_path, {"team": "det"}, "executives", use_csv=False)
        assert p.suffix == ".parquet"

    def test_multi_table_yield_distinct_paths(self, tmp_path):
        p_afc = make_pfr_output_path(tmp_path, {"year": 2025}, "afc", use_csv=True)
        p_nfc = make_pfr_output_path(tmp_path, {"year": 2025}, "nfc", use_csv=True)
        assert p_afc != p_nfc
        assert p_afc == tmp_path / "2025_afc.csv"
        assert p_nfc == tmp_path / "2025_nfc.csv"


# =============================================================================
# Progress tracking (scraper_utils)
# =============================================================================


class TestProgressTracking:
    def test_load_progress_missing_file_returns_empty_set(self, tmp_path):
        result = load_progress(tmp_path)
        assert result == set()

    def test_save_then_reload_round_trip(self, tmp_path):
        keys = {"det_executives", "chi_executives"}
        save_progress(tmp_path, keys)
        assert load_progress(tmp_path) == keys

    def test_completed_key_is_preserved_across_saves(self, tmp_path):
        save_progress(tmp_path, {"key1"})
        loaded = load_progress(tmp_path)
        assert "key1" in loaded

    def test_overwrite_replaces_previous_state(self, tmp_path):
        save_progress(tmp_path, {"old_key"})
        save_progress(tmp_path, {"new_key"})
        loaded = load_progress(tmp_path)
        assert loaded == {"new_key"}


# =============================================================================
# Config loading
# =============================================================================


class TestLoadPfrConfig:
    def _write_cfg(self, tmp_path, cfg: dict) -> str:
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg))
        return str(p)

    def test_valid_config_loads(self, tmp_path):
        cfg = {
            "source_name": "pfr_test",
            "url_template": "https://www.pro-football-reference.com/teams/{team}/executives.htm",
            "iterate": {"team": ["det", "chi"]},
            "tables": [{"id": "executives", "output_suffix": "executives"}],
            "output_dir": "data/raw/pfr/test",
            "sleep_between_requests": 4.0,
            "max_retries": 3,
            "retry_backoff": 10.0,
        }
        loaded = load_pfr_config(self._write_cfg(tmp_path, cfg))
        assert loaded["source_name"] == "pfr_test"
        assert loaded["url_template"] == "https://www.pro-football-reference.com/teams/{team}/executives.htm"

    def test_iterations_expanded_in_config(self, tmp_path):
        cfg = {
            "source_name": "pfr_test",
            "url_template": "https://pfr.com/{team}/",
            "iterate": {"team": ["det", "chi"]},
            "tables": [],
            "output_dir": "data/raw/pfr/test",
            "sleep_between_requests": 4.0,
            "max_retries": 3,
            "retry_backoff": 10.0,
        }
        loaded = load_pfr_config(self._write_cfg(tmp_path, cfg))
        assert loaded["_iterations"] == [{"team": "det"}, {"team": "chi"}]

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pfr_config(str(tmp_path / "nonexistent.json"))


# =============================================================================
# Content extraction — executives reference fixture (det/executives.htm)
# =============================================================================


class TestExecutivesContent:
    @pytest.fixture(scope="class")
    def df(self):
        html = (FIXTURE_DIR / "executives_det.html").read_text()
        return parse_pfr_table(html, table_id="executives")

    def test_table_is_not_none(self, df):
        assert df is not None

    def test_table_has_rows(self, df):
        assert len(df) > 0

    def test_year_column_present(self, df):
        assert any("year" in c.lower() for c in df.columns)

    def test_pos_column_present(self, df):
        assert any("pos" in c.lower() for c in df.columns)

    def test_person_column_present(self, df):
        assert any(c.lower() in ("person", "player") for c in df.columns)

    def test_head_coach_row_exists(self, df):
        pos_col = next(c for c in df.columns if "pos" in c.lower())
        assert df[pos_col].str.contains("HC").any()

    def test_general_manager_row_exists(self, df):
        pos_col = next(c for c in df.columns if "pos" in c.lower())
        assert df[pos_col].str.contains("GM").any()

    def test_dan_campbell_present(self, df):
        person_col = next(c for c in df.columns if c.lower() in ("person", "player"))
        assert df[person_col].str.contains("Dan Campbell").any()

    def test_brad_holmes_present(self, df):
        person_col = next(c for c in df.columns if c.lower() in ("person", "player"))
        assert df[person_col].str.contains("Brad Holmes").any()

    def test_year_2024_present(self, df):
        year_col = next(c for c in df.columns if "year" in c.lower())
        assert "2024" in df[year_col].values


# =============================================================================
# Content extraction — standings reference fixture (years/2025/)
# =============================================================================


class TestStandingsContent:
    @pytest.fixture(scope="class")
    def afc(self):
        html = (FIXTURE_DIR / "standings_2025.html").read_text()
        return parse_pfr_table(html, table_id="AFC")

    @pytest.fixture(scope="class")
    def nfc(self):
        html = (FIXTURE_DIR / "standings_2025.html").read_text()
        return parse_pfr_table(html, table_id="NFC")

    def test_afc_is_not_none(self, afc):
        assert afc is not None

    def test_nfc_is_not_none(self, nfc):
        assert nfc is not None

    def test_afc_has_rows(self, afc):
        assert len(afc) > 0

    def test_nfc_has_rows(self, nfc):
        assert len(nfc) > 0

    def test_afc_has_wins_column(self, afc):
        assert "W" in afc.columns

    def test_afc_has_losses_column(self, afc):
        assert "L" in afc.columns

    def test_nfc_has_wins_column(self, nfc):
        assert "W" in nfc.columns

    def test_team_column_present_afc(self, afc):
        assert any("Tm" in c or "Team" in c for c in afc.columns)

    def test_buffalo_bills_in_afc(self, afc):
        team_col = next(c for c in afc.columns if "Tm" in c)
        assert afc[team_col].str.contains("Buffalo").any()

    def test_detroit_lions_in_nfc(self, nfc):
        team_col = next(c for c in nfc.columns if "Tm" in c)
        assert nfc[team_col].str.contains("Detroit").any()

    def test_afc_and_nfc_are_distinct(self, afc, nfc):
        assert not afc.equals(nfc)

    def test_no_team_appears_in_both_conferences(self, afc, nfc):
        afc_team_col = next(c for c in afc.columns if "Tm" in c)
        nfc_team_col = next(c for c in nfc.columns if "Tm" in c)
        afc_teams = set(afc[afc_team_col].tolist())
        nfc_teams = set(nfc[nfc_team_col].tolist())
        assert afc_teams.isdisjoint(nfc_teams)
