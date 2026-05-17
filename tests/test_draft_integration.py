"""Tests for src/draft_integration.py."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.draft_integration import (
    add_new_trades,
    find_incomplete_trades,
    populate_pick_numbers,
    apply_trade_patch,
    generate_trade_patch,
)

_DRAFT_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "nfl_draft_2026.json"
_PATCH_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "trade_patch_2026.json"


def _load_2026_picks() -> list[dict]:
    with open(_DRAFT_JSON_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Trade 1929 fixture (DET ↔ JAX, 2025 draft, includes 2026 future picks)
# ---------------------------------------------------------------------------

def _make_trade_1929() -> pl.DataFrame:
    """Six rows from trade 1929: three 2025 picks with numbers, three 2026 picks null."""
    return pl.DataFrame({
        "trade_id": [1929.0] * 6,
        "season": [2025] * 6,
        "trade_date": [date(2025, 4, 25)] * 6,
        "gave": ["JAX", "JAX", "JAX", "DET", "DET", "DET"],
        "received": ["DET", "DET", "DET", "JAX", "JAX", "JAX"],
        "pick_season": [2025.0, 2025.0, 2026.0, 2025.0, 2026.0, 2026.0],
        "pick_round": [3.0, 6.0, 6.0, 3.0, 3.0, 3.0],
        "pick_number": pl.Series([70.0, 182.0, None, 102.0, None, None], dtype=pl.Float64),
        "conditional": [0.0] * 6,
        "pfr_id": pl.Series(["TeSlIs00", "BorrAn00", None, "FeltTa00", None, None], dtype=pl.String),
        "pfr_name": pl.Series(["Isaac TeSlaa", "Andres Borregales", None, "Tai Felton", None, None], dtype=pl.String),
    })


# ---------------------------------------------------------------------------
# Base trades fixture for new-trade tests (max trade_id = 1977)
# ---------------------------------------------------------------------------

def _make_base_trades() -> pl.DataFrame:
    """Minimal trades DataFrame with max trade_id=1977 and no DET-BUF 2026 rows."""
    return pl.DataFrame({
        "trade_id": [1977.0],
        "season": [2025],
        "trade_date": [date(2025, 1, 1)],
        "gave": ["KC"],
        "received": ["NE"],
        "pick_season": [2025.0],
        "pick_round": [1.0],
        "pick_number": [1.0],
        "conditional": [0.0],
        "pfr_id": pl.Series([None], dtype=pl.String),
        "pfr_name": pl.Series([None], dtype=pl.String),
    })


# ---------------------------------------------------------------------------
# TestFindIncompleteTrades
# ---------------------------------------------------------------------------

class TestFindIncompleteTrades:
    def test_returns_dataframe(self):
        result = find_incomplete_trades(_make_trade_1929(), 2026)
        assert isinstance(result, pl.DataFrame)

    def test_finds_three_null_rows_for_2026(self):
        result = find_incomplete_trades(_make_trade_1929(), 2026)
        assert len(result) == 3

    def test_only_returns_year_matches(self):
        result = find_incomplete_trades(_make_trade_1929(), 2026)
        assert result["pick_season"].unique().to_list() == [2026.0]

    def test_no_rows_when_year_has_no_nulls(self):
        result = find_incomplete_trades(_make_trade_1929(), 2025)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestIncompleteTrades2026_Trade1929
# ---------------------------------------------------------------------------

class TestIncompleteTrades2026_Trade1929:
    """Verify populate_pick_numbers fills in trade 1929's 2026 null picks."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _make_trade_1929()
        picks = _load_2026_picks()
        return populate_pick_numbers(df, picks, 2026)

    def test_jax_to_det_sixth_round_is_213(self, result):
        row = result.filter(
            (pl.col("gave") == "JAX") &
            (pl.col("received") == "DET") &
            (pl.col("pick_season") == 2026.0) &
            (pl.col("pick_round") == 6.0)
        )
        assert len(row) == 1
        assert row["pick_number"][0] == 213.0

    def test_det_to_jax_third_round_pick_numbers(self, result):
        rows = result.filter(
            (pl.col("gave") == "DET") &
            (pl.col("received") == "JAX") &
            (pl.col("pick_season") == 2026.0) &
            (pl.col("pick_round") == 3.0)
        )
        assert len(rows) == 2
        assert set(rows["pick_number"].to_list()) == {81.0, 100.0}

    def test_pfr_name_for_pick_213_is_jordan_van_den_berg(self, result):
        row = result.filter(pl.col("pick_number") == 213.0)
        assert row["pfr_name"][0] == "Jordan van den Berg"

    def test_pfr_name_for_pick_81_is_albert_regis(self, result):
        row = result.filter(pl.col("pick_number") == 81.0)
        assert row["pfr_name"][0] == "Albert Regis"

    def test_pfr_name_for_pick_100_is_jalen_huskey(self, result):
        row = result.filter(pl.col("pick_number") == 100.0)
        assert row["pfr_name"][0] == "Jalen Huskey"

    def test_no_null_pick_numbers_remain_for_2026(self, result):
        nulls = result.filter(
            (pl.col("pick_season") == 2026.0) & pl.col("pick_number").is_null()
        )
        assert len(nulls) == 0

    def test_existing_2025_picks_unchanged(self, result):
        row_70 = result.filter(pl.col("pick_number") == 70.0)
        assert len(row_70) == 1
        assert row_70["pfr_name"][0] == "Isaac TeSlaa"


# ---------------------------------------------------------------------------
# TestNewTrades2026_DETvsBUF
# ---------------------------------------------------------------------------

class TestNewTrades2026_DETvsBUF:
    """Verify add_new_trades creates the 2026 DET-BUF draft-day trade correctly."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _make_base_trades()
        picks = _load_2026_picks()
        return add_new_trades(df, picks, 2026)

    def test_lions_received_pick_168_kendrick_law(self, result):
        row = result.filter(
            (pl.col("gave") == "BUF") &
            (pl.col("received") == "DET") &
            (pl.col("pick_number") == 168.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Kendrick Law"

    def test_bills_received_pick_181_zane_durant(self, result):
        row = result.filter(
            (pl.col("gave") == "DET") &
            (pl.col("received") == "BUF") &
            (pl.col("pick_number") == 181.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Zane Durant"

    def test_bills_received_pick_213_jordan_van_den_berg(self, result):
        row = result.filter(
            (pl.col("gave") == "DET") &
            (pl.col("received") == "BUF") &
            (pl.col("pick_number") == 213.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Jordan van den Berg"

    def test_all_three_picks_share_same_trade_id(self, result):
        det_buf_rows = result.filter(
            ((pl.col("gave") == "BUF") & (pl.col("received") == "DET") & (pl.col("pick_number") == 168.0)) |
            ((pl.col("gave") == "DET") & (pl.col("received") == "BUF") & (pl.col("pick_number") == 181.0)) |
            ((pl.col("gave") == "DET") & (pl.col("received") == "BUF") & (pl.col("pick_number") == 213.0))
        )
        assert len(det_buf_rows) == 3
        assert det_buf_rows["trade_id"].n_unique() == 1

    def test_new_trade_id_greater_than_1977(self, result):
        det_buf_row = result.filter(
            (pl.col("gave") == "BUF") &
            (pl.col("received") == "DET") &
            (pl.col("pick_number") == 168.0)
        )
        assert det_buf_row["trade_id"][0] > 1977.0

    def test_original_row_unchanged(self, result):
        original = result.filter(pl.col("trade_id") == 1977.0)
        assert len(original) == 1
        assert original["gave"][0] == "KC"


# ---------------------------------------------------------------------------
# TestApplyTradePatch_Trade1929
# ---------------------------------------------------------------------------

class TestApplyTradePatch_Trade1929:
    """Verify apply_trade_patch handles multiple patches in the same round correctly.

    Trade 1929 (DET ↔ JAX) has two round-3 patches for DET→JAX (picks #81 and #100).
    The bug caused both rows to receive pick_number=100.0 (last patch applied twice).
    """

    @pytest.fixture(scope="class")
    def result(self):
        df = _make_trade_1929()
        with open(_PATCH_JSON_PATH) as f:
            patch = json.load(f)
        return apply_trade_patch(df, patch)

    def test_det_to_jax_third_round_has_two_distinct_picks(self, result):
        rows = result.filter(
            (pl.col("trade_id") == 1929.0) &
            (pl.col("gave") == "DET") &
            (pl.col("received") == "JAX") &
            (pl.col("pick_season") == 2026.0) &
            (pl.col("pick_round") == 3.0)
        )
        assert len(rows) == 2
        assert set(rows["pick_number"].to_list()) == {81.0, 100.0}

    def test_pick_81_is_albert_regis(self, result):
        row = result.filter(
            (pl.col("trade_id") == 1929.0) &
            (pl.col("pick_number") == 81.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Albert Regis"

    def test_pick_100_is_jalen_huskey(self, result):
        row = result.filter(
            (pl.col("trade_id") == 1929.0) &
            (pl.col("pick_number") == 100.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Jalen Huskey"


# ---------------------------------------------------------------------------
# TestGenerateTradePatch_KCvsLA
# ---------------------------------------------------------------------------

class TestGenerateTradePatch_KCvsLA:
    """Verify generate_trade_patch includes veteran players and future picks.

    Trade: LA gave picks #29, #169, #210 (2026) + 2027 third round pick to KC.
           KC gave Trent McDuffie to LA. Dated 2026-03-11.
    """

    @pytest.fixture(scope="class")
    def patched_df(self):
        base = _make_base_trades()
        patch = generate_trade_patch(_DRAFT_JSON_PATH, 2026, base)
        return apply_trade_patch(base, patch)

    def test_kc_gave_trent_mcduffie_to_la(self, patched_df):
        row = patched_df.filter(
            (pl.col("gave") == "KC") &
            (pl.col("received") == "LA") &
            (pl.col("pfr_name") == "Trent McDuffie")
        )
        assert len(row) == 1

    def test_la_gave_2027_third_round_to_kc(self, patched_df):
        row = patched_df.filter(
            (pl.col("gave") == "LA") &
            (pl.col("received") == "KC") &
            (pl.col("pick_season") == 2027.0) &
            (pl.col("pick_round") == 3.0)
        )
        assert len(row) == 1

    def test_la_gave_pick_29_peter_woods_to_kc(self, patched_df):
        row = patched_df.filter(
            (pl.col("gave") == "LA") &
            (pl.col("received") == "KC") &
            (pl.col("pick_number") == 29.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Peter Woods"

    def test_la_gave_pick_169_riley_nowakowski_to_kc(self, patched_df):
        row = patched_df.filter(
            (pl.col("gave") == "LA") &
            (pl.col("received") == "KC") &
            (pl.col("pick_number") == 169.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Riley Nowakowski"

    def test_la_gave_pick_210_gabriel_rubio_to_kc(self, patched_df):
        row = patched_df.filter(
            (pl.col("gave") == "LA") &
            (pl.col("received") == "KC") &
            (pl.col("pick_number") == 210.0)
        )
        assert len(row) == 1
        assert row["pfr_name"][0] == "Gabriel Rubio"

    def test_all_five_rows_share_trade_id(self, patched_df):
        kc_la_rows = patched_df.filter(
            (
                (pl.col("gave") == "KC") & (pl.col("received") == "LA") &
                (pl.col("pfr_name") == "Trent McDuffie")
            ) | (
                (pl.col("gave") == "LA") & (pl.col("received") == "KC") &
                (pl.col("pick_season") == 2027.0) & (pl.col("pick_round") == 3.0)
            ) | (
                (pl.col("gave") == "LA") & (pl.col("received") == "KC") &
                (pl.col("pick_number") == 29.0)
            ) | (
                (pl.col("gave") == "LA") & (pl.col("received") == "KC") &
                (pl.col("pick_number") == 169.0)
            ) | (
                (pl.col("gave") == "LA") & (pl.col("received") == "KC") &
                (pl.col("pick_number") == 210.0)
            )
        )
        assert len(kc_la_rows) == 5
        assert kc_la_rows["trade_id"].n_unique() == 1
