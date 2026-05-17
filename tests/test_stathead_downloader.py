"""Tests for src/stathead_downloader.py — covers pure functions only (no HTTP)."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.stathead_downloader import (
    build_url,
    combo_key,
    detect_total_results,
    is_login_wall,
    load_config,
    make_output_path,
    parse_table,
)
from src.scraper_utils import load_progress, save_progress


# =============================================================================
# load_config
# =============================================================================


class TestLoadConfig:
    def _write_cfg(self, tmp_path, data: dict) -> str:
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_draft_year_range_expanded_from_start_end(self, tmp_path):
        cfg_data = {
            "draft_year_start": 2020,
            "draft_year_end": 2022,
            "fixed_params": {},
            "page_size": 200,
            "sleep_between_requests": 3.0,
            "max_retries": 3,
            "retry_backoff": 10.0,
        }
        cfg = load_config(self._write_cfg(tmp_path, cfg_data))
        assert cfg["draft_year_ranges"] == [(2020, 2020), (2021, 2021), (2022, 2022)]

    def test_explicit_draft_year_ranges_preserved(self, tmp_path):
        cfg_data = {
            "draft_year_ranges": [[2018, 2020], [2021, 2023]],
            "fixed_params": {},
            "page_size": 200,
            "sleep_between_requests": 3.0,
            "max_retries": 3,
            "retry_backoff": 10.0,
        }
        cfg = load_config(self._write_cfg(tmp_path, cfg_data))
        assert cfg["draft_year_ranges"] == [(2018, 2020), (2021, 2023)]

    def test_single_year_range_when_start_equals_end(self, tmp_path):
        cfg_data = {
            "draft_year_start": 2024,
            "draft_year_end": 2024,
            "fixed_params": {},
            "page_size": 200,
            "sleep_between_requests": 3.0,
            "max_retries": 3,
            "retry_backoff": 10.0,
        }
        cfg = load_config(self._write_cfg(tmp_path, cfg_data))
        assert cfg["draft_year_ranges"] == [(2024, 2024)]

    def test_missing_draft_ranges_defaults_to_empty(self, tmp_path):
        cfg_data = {
            "fixed_params": {},
            "page_size": 200,
            "sleep_between_requests": 3.0,
            "max_retries": 3,
            "retry_backoff": 10.0,
        }
        cfg = load_config(self._write_cfg(tmp_path, cfg_data))
        assert cfg["draft_year_ranges"] == []


# =============================================================================
# build_url
# =============================================================================


class TestBuildUrl:
    _BASE_CFG = {
        "fixed_params": {
            "request": "1",
            "order_by": "av",
        }
    }

    def test_url_contains_base_url(self):
        url = build_url(self._BASE_CFG, 2020, 2020, 2021, 2021, 0)
        assert "stathead" in url or "sports-reference" in url

    def test_draft_year_params_in_url(self):
        url = build_url(self._BASE_CFG, 2020, 2020, 2021, 2021, 0)
        assert "draft_year_min=2020" in url
        assert "draft_year_max=2020" in url

    def test_season_year_params_in_url(self):
        url = build_url(self._BASE_CFG, 2020, 2020, 2021, 2021, 0)
        assert "year_min=2021" in url
        assert "year_max=2021" in url

    def test_offset_zero_in_url(self):
        url = build_url(self._BASE_CFG, 2020, 2020, 2021, 2021, 0)
        assert "offset=0" in url

    def test_nonzero_offset_in_url(self):
        url = build_url(self._BASE_CFG, 2020, 2020, 2021, 2021, 200)
        assert "offset=200" in url

    def test_fixed_params_in_url(self):
        url = build_url(self._BASE_CFG, 2020, 2020, 2021, 2021, 0)
        assert "request=1" in url
        assert "order_by=av" in url


# =============================================================================
# parse_table
# =============================================================================


class TestParseTable:
    _SIMPLE_TABLE = """
    <html><body>
    <table id="stats">
      <thead><tr><th>Rk</th><th>Player</th><th>AV.1</th></tr></thead>
      <tbody>
        <tr><td>1</td><td>Alice</td><td>12</td></tr>
        <tr><td>2</td><td>Bob</td><td>8</td></tr>
        <tr><td>Rk</td><td>Player</td><td>AV.1</td></tr>
        <tr><td>3</td><td>Carol</td><td>5</td></tr>
      </tbody>
    </table>
    </body></html>
    """

    def test_returns_dataframe_for_valid_html(self):
        df = parse_table(self._SIMPLE_TABLE)
        assert df is not None

    def test_header_repeat_rows_removed(self):
        df = parse_table(self._SIMPLE_TABLE)
        assert "Rk" not in df["Rk"].values

    def test_empty_rk_rows_removed(self):
        html = """
        <html><body>
        <table id="stats">
          <thead><tr><th>Rk</th><th>Player</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>Alice</td></tr>
            <tr><td></td><td></td></tr>
          </tbody>
        </table>
        </body></html>
        """
        df = parse_table(html)
        assert not df["Rk"].str.strip().eq("").any()

    def test_returns_none_for_no_table(self):
        result = parse_table("<html><body><p>Nothing here</p></body></html>")
        assert result is None

    def test_all_columns_are_strings(self):
        df = parse_table(self._SIMPLE_TABLE)
        for col in df.columns:
            assert pd.api.types.is_string_dtype(df[col]), f"Column {col!r} is not a string dtype"

    def test_correct_row_count(self):
        df = parse_table(self._SIMPLE_TABLE)
        assert len(df) == 3


# =============================================================================
# detect_total_results
# =============================================================================


class TestDetectTotalResults:
    def test_pattern_of_results(self):
        html = "<p>Showing 1 to 200 of 1,234 results</p>"
        assert detect_total_results(html) == 1234

    def test_pattern_results_found(self):
        html = "<p>456 results found</p>"
        assert detect_total_results(html) == 456

    def test_pattern_of_N_results(self):
        html = "<span>of 789 results</span>"
        assert detect_total_results(html) == 789

    def test_no_match_returns_none(self):
        assert detect_total_results("<p>No data here</p>") is None

    def test_comma_separated_number_parsed_correctly(self):
        html = "<p>of 12,345 results</p>"
        assert detect_total_results(html) == 12345


# =============================================================================
# is_login_wall
# =============================================================================


class TestIsLoginWall:
    def test_clean_page_not_flagged(self):
        html = "<html><body><table><tr><th>Rk</th></tr><tr><td>1</td></tr></table></body></html>"
        assert not is_login_wall(html)

    def test_sign_in_text_flagged(self):
        html = "<html><body><p>Please sign in to view this content.</p></body></html>"
        assert is_login_wall(html)

    def test_subscribe_text_flagged(self):
        html = "<html><body><p>Subscribe to access Stathead membership content.</p></body></html>"
        assert is_login_wall(html)

    def test_log_in_text_flagged(self):
        html = "<html><body><p>You must log in to continue.</p></body></html>"
        assert is_login_wall(html)

    def test_page_with_rk_column_not_flagged(self):
        # Even if login text is present, as long as "rk" is in the page it's data
        html = "<html><body><p>sign in</p><table><tr><th>Rk</th></tr></table></body></html>"
        assert not is_login_wall(html)


# =============================================================================
# make_output_path
# =============================================================================


class TestMakeOutputPath:
    def test_same_draft_year_format(self, tmp_path):
        p = make_output_path(tmp_path, 2020, 2020, 2021, 2021, use_csv=True)
        assert p.name == "draft2020_season2021.csv"

    def test_different_draft_years_format(self, tmp_path):
        p = make_output_path(tmp_path, 2018, 2020, 2021, 2021, use_csv=True)
        assert p.name == "draft2018-2020_season2021.csv"

    def test_parquet_extension(self, tmp_path):
        p = make_output_path(tmp_path, 2020, 2020, 2021, 2021, use_csv=False)
        assert p.suffix == ".parquet"

    def test_csv_extension(self, tmp_path):
        p = make_output_path(tmp_path, 2020, 2020, 2021, 2021, use_csv=True)
        assert p.suffix == ".csv"

    def test_path_inside_output_dir(self, tmp_path):
        p = make_output_path(tmp_path, 2020, 2020, 2021, 2021)
        assert p.parent == tmp_path


# =============================================================================
# combo_key
# =============================================================================


class TestComboKey:
    def test_key_contains_draft_and_season(self):
        key = combo_key(2020, 2020, 2021, 2021)
        assert "draft" in key
        assert "season" in key
        assert "2020" in key
        assert "2021" in key

    def test_different_inputs_produce_different_keys(self):
        k1 = combo_key(2020, 2020, 2021, 2021)
        k2 = combo_key(2021, 2021, 2022, 2022)
        assert k1 != k2

    def test_same_inputs_produce_same_key(self):
        assert combo_key(2020, 2020, 2021, 2021) == combo_key(2020, 2020, 2021, 2021)


# =============================================================================
# Progress tracking (shared scraper_utils — tested here in stathead context)
# =============================================================================


class TestStatheadProgressTracking:
    def test_no_file_returns_empty_set(self, tmp_path):
        assert load_progress(tmp_path) == set()

    def test_save_and_load_round_trip(self, tmp_path):
        keys = {"draft2020-2020_season2021-2021", "draft2021-2021_season2022-2022"}
        save_progress(tmp_path, keys)
        assert load_progress(tmp_path) == keys

    def test_multiple_saves_accumulate(self, tmp_path):
        save_progress(tmp_path, {"key1"})
        loaded = load_progress(tmp_path)
        loaded.add("key2")
        save_progress(tmp_path, loaded)
        assert load_progress(tmp_path) == {"key1", "key2"}
