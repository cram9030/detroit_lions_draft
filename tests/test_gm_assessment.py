"""Tests for src/gm_assessment.py and related aggregation functions.

Uses real data files where available (TDD with no mocked data per project convention).
Synthetic fixtures cover edge-case logic (e.g. two GMs in same year).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import pytest

from src.gm_assessment import (
    fetch_gm,
    filter_gm_records,
    find_overlapping_gms,
    get_gm_year_range,
    get_surplus_value,
    get_trade_value,
    load_executives,
)
from src.surplus_av import aggregate_drafts
from src.trade_value import aggregate_trade_value

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXECUTIVES_DIR = _PROJECT_ROOT / "data" / "raw" / "pfr" / "executives"
_MODELS_DIR = _PROJECT_ROOT / "models"
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "stathead" / "annual_av"
_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executives_parquet(tmp_path: Path, pfr_code: str, rows: list[dict]) -> Path:
    """Write a minimal executives parquet to tmp_path for testing edge cases."""
    df = pl.DataFrame(
        {
            "Person": [r["Person"] for r in rows],
            "Teams": [r.get("Teams", "Lions") for r in rows],
            "From": [str(r["From"]) for r in rows],
            "To": [str(r["To"]) for r in rows],
            "Titles": [r["Titles"] for r in rows],
            "Notes": [r.get("Notes") for r in rows],
        }
    )
    out = tmp_path / f"{pfr_code}_executives.parquet"
    df.write_parquet(out)
    return out


# ---------------------------------------------------------------------------
# TestLoadExecutives
# ---------------------------------------------------------------------------


class TestLoadExecutives:
    def test_columns_present(self):
        df = load_executives("det", _EXECUTIVES_DIR)
        assert set(df.columns) >= {"Person", "Teams", "From", "To", "Titles", "Notes"}

    def test_det_has_rows(self):
        df = load_executives("det", _EXECUTIVES_DIR)
        assert len(df) > 0

    def test_brad_holmes_in_det(self):
        df = load_executives("det", _EXECUTIVES_DIR)
        assert "Brad Holmes" in df["Person"].to_list()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_executives("zzz", tmp_path)


# ---------------------------------------------------------------------------
# TestFetchGM
# ---------------------------------------------------------------------------


class TestFetchGM:
    def test_det_2024_is_brad_holmes(self):
        result = fetch_gm("det", 2024, _EXECUTIVES_DIR)
        assert result is not None
        assert result["person"] == "Brad Holmes"

    def test_det_2021_is_brad_holmes(self):
        result = fetch_gm("det", 2021, _EXECUTIVES_DIR)
        assert result is not None
        assert result["person"] == "Brad Holmes"

    def test_det_2015_excludes_interim(self):
        """Sheldon White (Interim GM) must be excluded; Martin Mayhew returned."""
        result = fetch_gm("det", 2015, _EXECUTIVES_DIR)
        assert result is not None
        assert "Interim" not in result["person"]
        assert result["person"] == "Martin Mayhew"

    def test_no_gm_for_ancient_year_returns_none(self):
        result = fetch_gm("det", 1900, _EXECUTIVES_DIR)
        assert result is None

    def test_result_has_expected_keys(self):
        result = fetch_gm("det", 2024, _EXECUTIVES_DIR)
        assert result is not None
        assert {"person", "from_year", "to_year"} <= set(result.keys())

    def test_year_range_populated(self):
        result = fetch_gm("det", 2024, _EXECUTIVES_DIR)
        assert result is not None
        assert result["from_year"] <= 2024 <= result["to_year"]

    def test_april_criteria_fired_before_april(self, tmp_path):
        """GM fired in February → not April GM; second GM returned."""
        _make_executives_parquet(
            tmp_path,
            "tst",
            [
                {
                    "Person": "Early Fired GM",
                    "From": 2020,
                    "To": 2022,
                    "Titles": "General Manager",
                    "Notes": "Fired on 2/15/22",
                },
                {
                    "Person": "Hired March GM",
                    "From": 2022,
                    "To": 2025,
                    "Titles": "General Manager",
                    "Notes": "Named on 3/1/22",
                },
            ],
        )
        result = fetch_gm("tst", 2022, tmp_path)
        assert result is not None
        assert result["person"] == "Hired March GM"

    def test_april_criteria_fired_after_april(self, tmp_path):
        """GM fired in October → was April GM; they should be returned."""
        _make_executives_parquet(
            tmp_path,
            "tst",
            [
                {
                    "Person": "October Fired GM",
                    "From": 2020,
                    "To": 2022,
                    "Titles": "General Manager",
                    "Notes": "Fired on 10/5/22",
                },
                {
                    "Person": "Late Hired GM",
                    "From": 2022,
                    "To": 2025,
                    "Titles": "General Manager",
                    "Notes": "Named on 11/1/22",
                },
            ],
        )
        result = fetch_gm("tst", 2022, tmp_path)
        assert result is not None
        assert result["person"] == "October Fired GM"

    def test_interim_gm_excluded(self, tmp_path):
        """Interim GM title must never be returned."""
        _make_executives_parquet(
            tmp_path,
            "tst",
            [
                {
                    "Person": "Interim Guy",
                    "From": 2022,
                    "To": 2022,
                    "Titles": "Interim General Manager",
                    "Notes": None,
                },
            ],
        )
        result = fetch_gm("tst", 2022, tmp_path)
        assert result is None

    def test_assistant_gm_excluded(self, tmp_path):
        _make_executives_parquet(
            tmp_path,
            "tst",
            [
                {
                    "Person": "Asst Guy",
                    "From": 2022,
                    "To": 2022,
                    "Titles": "Assistant General Manager",
                    "Notes": None,
                },
            ],
        )
        result = fetch_gm("tst", 2022, tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# TestGetGMYearRange
# ---------------------------------------------------------------------------


class TestGetGMYearRange:
    def test_det_2024_brad_holmes(self):
        result = get_gm_year_range("det", 2024, _EXECUTIVES_DIR)
        assert result is not None
        gm_name, from_yr, to_yr = result
        assert gm_name == "Brad Holmes"
        assert from_yr == 2021
        assert to_yr == 2026

    def test_returns_tuple_of_three(self):
        result = get_gm_year_range("det", 2024, _EXECUTIVES_DIR)
        assert result is not None
        assert len(result) == 3

    def test_no_gm_returns_none(self):
        result = get_gm_year_range("det", 1900, _EXECUTIVES_DIR)
        assert result is None


# ---------------------------------------------------------------------------
# TestFindOverlappingGMs
# ---------------------------------------------------------------------------


class TestFindOverlappingGMs:
    def test_returns_dataframe(self):
        df = find_overlapping_gms(2021, 2024, _EXECUTIVES_DIR)
        assert isinstance(df, pl.DataFrame)

    def test_required_columns_present(self):
        df = find_overlapping_gms(2021, 2024, _EXECUTIVES_DIR)
        required = {"pfr_code", "stathead_code", "gm_name", "gm_from", "gm_to", "overlap_from", "overlap_to"}
        assert required <= set(df.columns)

    def test_covers_all_teams(self):
        """At least 31 unique franchises — some teams (e.g. CIN) use a non-GM title
        in PFR data (Duke Tobin is 'Director of Player Personnel'), so ≥31 is correct."""
        df = find_overlapping_gms(2021, 2024, _EXECUTIVES_DIR)
        assert df["pfr_code"].n_unique() >= 31

    def test_overlap_from_lte_overlap_to(self):
        df = find_overlapping_gms(2021, 2024, _EXECUTIVES_DIR)
        assert (df["overlap_from"] <= df["overlap_to"]).all()

    def test_overlap_within_requested_range(self):
        """overlap_from >= data_start_year (2010 default); overlap_to <= years_cap."""
        df = find_overlapping_gms(2021, 2026, _EXECUTIVES_DIR, years_cap=2024)
        assert (df["overlap_from"] >= 2010).all()
        assert (df["overlap_to"] <= 2024).all()

    def test_full_career_window(self):
        """Kevin Colbert (PIT 2000-2021) overlapping 2021-2026 should be assessed 2010-2021."""
        df = find_overlapping_gms(2021, 2026, _EXECUTIVES_DIR, years_cap=2024)
        colbert = df.filter(
            (pl.col("pfr_code") == "pit") & (pl.col("gm_name") == "Kevin Colbert")
        )
        assert len(colbert) == 1
        assert colbert["overlap_from"][0] == 2010  # capped to data_start_year, not 2021
        assert colbert["overlap_to"][0] == 2021    # capped to gm_to

    def test_no_interim_gms(self):
        df = find_overlapping_gms(2021, 2024, _EXECUTIVES_DIR)
        for name in df["gm_name"].to_list():
            assert "Interim" not in name
            assert "Assistant" not in name

    def test_det_brad_holmes_present(self):
        df = find_overlapping_gms(2021, 2024, _EXECUTIVES_DIR)
        det_rows = df.filter(pl.col("pfr_code") == "det")
        assert "Brad Holmes" in det_rows["gm_name"].to_list()


# ---------------------------------------------------------------------------
# TestAggregateDrafts  (tests the new aggregate_drafts in src/surplus_av.py)
# ---------------------------------------------------------------------------


class TestAggregateDrafts:
    def test_returns_dict_with_expected_keys(self):
        result = aggregate_drafts("DET", [2021], models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert isinstance(result, dict)
        for key in ("total_surplus_av", "per_year", "n_years", "avg_surplus_per_year"):
            assert key in result, f"Missing key: {key}"

    def test_per_year_is_dict(self):
        result = aggregate_drafts("DET", [2021], models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert isinstance(result["per_year"], dict)

    def test_n_years_for_valid_year(self):
        result = aggregate_drafts("DET", [2021], models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert result["n_years"] == 1

    def test_avg_surplus_is_float(self):
        result = aggregate_drafts("DET", [2021], models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert isinstance(result["avg_surplus_per_year"], float)

    def test_invalid_year_skipped(self):
        """Years with no data don't crash — they're skipped."""
        result = aggregate_drafts("DET", [2021, 1960], models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert result["n_years"] >= 1  # only 2021 counts

    def test_empty_years_returns_zero_totals(self):
        result = aggregate_drafts("DET", [], models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert result["n_years"] == 0
        assert result["total_surplus_av"] == 0.0


# ---------------------------------------------------------------------------
# TestAggregateTradeValue  (tests the new aggregate_trade_value in src/trade_value.py)
# ---------------------------------------------------------------------------


class TestAggregateTradeValue:
    def test_returns_dict_with_expected_keys(self):
        result = aggregate_trade_value("DET", [2021, 2022, 2023], data_dir=_DATA_DIR)
        assert isinstance(result, dict)
        for key in ("total_trade_value", "per_year", "n_years", "avg_trade_per_year"):
            assert key in result, f"Missing key: {key}"

    def test_per_year_is_dict(self):
        result = aggregate_trade_value("DET", [2021, 2022, 2023], data_dir=_DATA_DIR)
        assert isinstance(result["per_year"], dict)

    def test_n_years_equals_requested(self):
        result = aggregate_trade_value("DET", [2021, 2022, 2023], data_dir=_DATA_DIR)
        assert result["n_years"] == 3

    def test_avg_trade_is_float(self):
        result = aggregate_trade_value("DET", [2021, 2022], data_dir=_DATA_DIR)
        assert isinstance(result["avg_trade_per_year"], float)

    def test_empty_years_returns_zero_totals(self):
        result = aggregate_trade_value("DET", [], data_dir=_DATA_DIR)
        assert result["n_years"] == 0
        assert result["total_trade_value"] == 0.0


# ---------------------------------------------------------------------------
# TestGetSurplusValue
# ---------------------------------------------------------------------------


class TestGetSurplusValue:
    def _make_gm_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "pfr_code": ["det"],
                "stathead_code": ["DET"],
                "gm_name": ["Brad Holmes"],
                "gm_from": [2021],
                "gm_to": [2026],
                "overlap_from": [2021],
                "overlap_to": [2022],
            }
        )

    def test_adds_surplus_columns(self):
        gm_df = self._make_gm_df()
        result = get_surplus_value(gm_df, models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        for col in ("total_surplus_av", "n_draft_years", "avg_surplus_per_year"):
            assert col in result.columns, f"Missing column: {col}"

    def test_returns_dataframe(self):
        gm_df = self._make_gm_df()
        result = get_surplus_value(gm_df, models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert isinstance(result, pl.DataFrame)

    def test_same_row_count(self):
        gm_df = self._make_gm_df()
        result = get_surplus_value(gm_df, models_dir=_MODELS_DIR, raw_dir=_RAW_DIR)
        assert len(result) == len(gm_df)


# ---------------------------------------------------------------------------
# TestGetTradeValue
# ---------------------------------------------------------------------------


class TestGetTradeValue:
    def _make_gm_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "pfr_code": ["det"],
                "stathead_code": ["DET"],
                "gm_name": ["Brad Holmes"],
                "gm_from": [2021],
                "gm_to": [2026],
                "overlap_from": [2021],
                "overlap_to": [2023],
            }
        )

    def test_adds_trade_columns(self):
        gm_df = self._make_gm_df()
        result = get_trade_value(gm_df, data_dir=_DATA_DIR)
        for col in ("total_trade_value", "n_trade_years", "avg_trade_per_year"):
            assert col in result.columns, f"Missing column: {col}"

    def test_returns_dataframe(self):
        gm_df = self._make_gm_df()
        result = get_trade_value(gm_df, data_dir=_DATA_DIR)
        assert isinstance(result, pl.DataFrame)

    def test_same_row_count(self):
        gm_df = self._make_gm_df()
        result = get_trade_value(gm_df, data_dir=_DATA_DIR)
        assert len(result) == len(gm_df)


# ---------------------------------------------------------------------------
# TestFilterGMRecords
# ---------------------------------------------------------------------------


class TestFilterGMRecords:
    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "gm_name": ["Brad Holmes", "New Hire GM"],
                "pfr_code": ["det", "oti"],
                "stathead_code": ["DET", "TEN"],
                "gm_from": [2021, 2025],
                "gm_to": [2026, 2030],
                "overlap_from": [2021, 2025],
                "overlap_to": [2024, 2026],
                "n_draft_years": [4, 0],
                "avg_surplus_per_year": [5.0, 0.0],
            }
        )

    def test_removes_zero_draft_year_rows(self):
        result = filter_gm_records(self._make_df())
        assert len(result) == 1
        assert result["gm_name"][0] == "Brad Holmes"

    def test_returns_dataframe(self):
        result = filter_gm_records(self._make_df())
        assert isinstance(result, pl.DataFrame)

    def test_all_have_n_draft_years_gt_zero(self):
        result = filter_gm_records(self._make_df())
        assert (result["n_draft_years"] > 0).all()


# ---------------------------------------------------------------------------
# TestPlotGMQuadrant
# ---------------------------------------------------------------------------


class TestPlotGMQuadrant:
    def _make_gm_plot_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "gm_name": ["Brad Holmes", "Howie Roseman"],
                "pfr_code": ["det", "phi"],
                "stathead_code": ["DET", "PHI"],
                "avg_surplus_per_year": [5.2, -1.3],
                "avg_trade_per_year": [120.0, -45.0],
                "total_surplus_av": [10.4, -2.6],
                "total_trade_value": [240.0, -90.0],
                "n_draft_years": [2, 2],
                "n_trade_years": [2, 2],
                "gm_from": [2021, 2016],
                "gm_to": [2026, 2030],
                "overlap_from": [2021, 2021],
                "overlap_to": [2022, 2022],
            }
        )

    def test_returns_figure(self):
        from src.plot_av import plot_gm_quadrant

        fig = plot_gm_quadrant(self._make_gm_plot_df())
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self):
        from src.plot_av import plot_gm_quadrant

        fig = plot_gm_quadrant(self._make_gm_plot_df())
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1

    def test_uses_plotly_white_template(self):
        from src.plot_av import plot_gm_quadrant

        fig = plot_gm_quadrant(self._make_gm_plot_df())
        assert fig.layout.template is not None

    def test_layout_images_added(self):
        from src.plot_av import plot_gm_quadrant

        fig = plot_gm_quadrant(self._make_gm_plot_df())
        assert len(fig.layout.images) >= 1

    def test_quadrant_lines_present(self):
        from src.plot_av import plot_gm_quadrant

        fig = plot_gm_quadrant(self._make_gm_plot_df())
        shapes = fig.layout.shapes
        assert len(shapes) >= 2  # one vline + one hline

    def test_custom_title(self):
        from src.plot_av import plot_gm_quadrant

        fig = plot_gm_quadrant(self._make_gm_plot_df(), title="My Test Title")
        assert fig.layout.title.text == "My Test Title"
