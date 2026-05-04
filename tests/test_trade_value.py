"""Tests for src/trade_value.py."""

import pytest
import polars as pl

from src.trade_value import (
    PickCombinationResult,
    _extended_two_pointer,
    find_pick_combination,
    load_trade_chart,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic pick lists
# ---------------------------------------------------------------------------


def _picks(values: list[float]) -> list[tuple[int, float]]:
    """Build (pick_number, value) pairs from a list of values, sorted ascending."""
    return sorted(enumerate(values, start=1), key=lambda x: x[1])


# ---------------------------------------------------------------------------
# load_trade_chart
# ---------------------------------------------------------------------------


class TestLoadTradeChart:
    def test_returns_pick_and_value_columns(self):
        df = load_trade_chart("jimmy_johnson")
        assert df.columns == ["Pick", "Value"]

    def test_pick_is_int64(self):
        df = load_trade_chart("jimmy_johnson")
        assert df["Pick"].dtype == pl.Int64

    def test_value_is_float64(self):
        df = load_trade_chart("jimmy_johnson")
        assert df["Value"].dtype == pl.Float64

    def test_sorted_ascending_by_pick(self):
        df = load_trade_chart("jimmy_johnson")
        picks = df["Pick"].to_list()
        assert picks == sorted(picks)

    @pytest.mark.parametrize(
        "chart_name",
        [
            "jimmy_johnson",
            "fitzgerald_spielberger",
            "pff_war",
            "rich_hill",
            "eavar",
            "5_year_av",
        ],
    )
    def test_all_charts_load_successfully(self, chart_name):
        df = load_trade_chart(chart_name)
        assert df.columns == ["Pick", "Value"]
        assert len(df) > 0

    def test_unknown_chart_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown chart"):
            load_trade_chart("nonexistent_chart")

    def test_jimmy_johnson_no_duplicate_picks(self):
        df = load_trade_chart("jimmy_johnson")
        assert df["Pick"].n_unique() == len(df)

    def test_pick_1_has_highest_value(self):
        for chart_name in ["jimmy_johnson", "fitzgerald_spielberger", "rich_hill"]:
            df = load_trade_chart(chart_name)
            max_val = df["Value"].max()
            pick1_val = df.filter(pl.col("Pick") == 1)["Value"][0]
            assert pick1_val == max_val, f"{chart_name}: pick 1 should be max value"


# ---------------------------------------------------------------------------
# _extended_two_pointer (unit tests on synthetic data)
# ---------------------------------------------------------------------------


class TestExtendedTwoPointer:
    def test_exact_single_pick_match(self):
        pv = _picks([10.0, 20.0, 30.0, 40.0])
        picks, values, total = _extended_two_pointer(pv, 30.0, 5, 0.0)
        assert len(picks) == 1
        assert total == pytest.approx(30.0)

    def test_exact_two_pick_match(self):
        pv = _picks([10.0, 20.0, 30.0, 40.0])
        picks, values, total = _extended_two_pointer(pv, 50.0, 5, 0.0)
        assert total == pytest.approx(50.0)  # 10+40 or 20+30
        assert len(picks) == 2

    def test_two_pick_beats_one_pick(self):
        # target=55, single picks: 10,20,30,40 -> best single is 40 (error=15)
        # two picks: 20+40=60 (error=5) or 30+40=70 (error=15) — best pair 20+30=50 err 5
        pv = _picks([10.0, 20.0, 30.0, 40.0])
        _, _, total = _extended_two_pointer(pv, 55.0, 5, 0.0)
        assert abs(total - 55.0) <= 5.0

    def test_three_pick_beats_two_pick(self):
        # picks: 1, 2, 3, 10, 11 (ascending)
        # target=6, best single=3 (err=3), best pair=1+3=4 (err=2) or 2+3=5 (err=1)
        # best triple=1+2+3=6 (err=0) — exact!
        pv = _picks([1.0, 2.0, 3.0, 10.0, 11.0])
        picks, values, total = _extended_two_pointer(pv, 6.0, 5, 0.0)
        assert total == pytest.approx(6.0)
        assert len(picks) == 3

    def test_max_picks_limits_combination_size(self):
        # With max_picks=1 we can only return a single pick
        pv = _picks([1.0, 2.0, 3.0, 10.0, 11.0])
        picks, values, total = _extended_two_pointer(pv, 6.0, 1, 0.0)
        assert len(picks) == 1

    def test_tolerance_stops_early(self):
        # target=55, tolerance=20 — best single (40, err=15) already within tolerance
        pv = _picks([10.0, 20.0, 30.0, 40.0])
        picks, values, total = _extended_two_pointer(pv, 55.0, 5, 20.0)
        assert abs(total - 55.0) <= 20.0

    def test_total_equals_sum_of_values(self):
        pv = _picks([5.0, 15.0, 25.0, 35.0, 45.0])
        picks, values, total = _extended_two_pointer(pv, 40.0, 5, 0.0)
        assert total == pytest.approx(sum(values))

    def test_empty_pick_list(self):
        picks, values, total = _extended_two_pointer([], 100.0, 5, 0.0)
        assert picks == []
        assert values == []
        assert total == 0.0

    def test_single_pick_list(self):
        pv = [(1, 100.0)]
        picks, values, total = _extended_two_pointer(pv, 999.0, 5, 0.0)
        assert picks == [1]
        assert total == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# find_pick_combination (integration tests against real CSVs)
# ---------------------------------------------------------------------------


class TestFindPickCombination:
    def test_returns_pick_combination_result_type(self):
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert isinstance(result, dict)
        for key in ("chart_name", "target_value", "picks", "values", "total_value", "error", "n_picks"):
            assert key in result

    def test_chart_name_preserved_in_result(self):
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert result["chart_name"] == "jimmy_johnson"

    def test_target_value_preserved_in_result(self):
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert result["target_value"] == pytest.approx(2600.0)

    def test_total_value_equals_sum_of_values(self):
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert result["total_value"] == pytest.approx(sum(result["values"]))

    def test_n_picks_matches_picks_length(self):
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert result["n_picks"] == len(result["picks"])

    def test_error_is_signed_difference(self):
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert result["error"] == pytest.approx(result["total_value"] - result["target_value"])

    def test_exact_match_pick_2_jimmy_johnson(self):
        # Pick 2 on JJ chart = 2600
        result = find_pick_combination(2600.0, "jimmy_johnson")
        assert result["error"] == pytest.approx(0.0, abs=1e-6)
        assert result["n_picks"] == 1
        assert result["picks"] == [2]

    def test_max_picks_one_returns_single_pick(self):
        result = find_pick_combination(2800.0, "jimmy_johnson", max_picks=1)
        assert result["n_picks"] == 1

    def test_no_duplicate_picks_in_result(self):
        result = find_pick_combination(2800.0, "jimmy_johnson")
        assert len(result["picks"]) == len(set(result["picks"]))

    def test_unknown_chart_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown chart"):
            find_pick_combination(1000.0, "bad_chart")

    @pytest.mark.parametrize(
        "chart_name",
        [
            "jimmy_johnson",
            "fitzgerald_spielberger",
            "pff_war",
            "rich_hill",
            "eavar",
            "5_year_av",
        ],
    )
    def test_all_charts_return_valid_result(self, chart_name):
        df = load_trade_chart(chart_name)
        # Use the midpoint of the chart's value range as target
        min_val = float(df["Value"].min())
        max_val = float(df["Value"].max())
        target = (min_val + max_val) / 2.0
        result = find_pick_combination(target, chart_name)
        assert result["n_picks"] >= 1
        assert len(result["picks"]) == len(result["values"])

    def test_tolerance_stops_search(self):
        # With large tolerance, a single pick should suffice
        result = find_pick_combination(2800.0, "jimmy_johnson", tolerance=500.0)
        assert abs(result["error"]) <= 500.0
