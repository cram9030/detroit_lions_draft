"""Tests for scripts/baked_draft_data.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "baked_draft_data.py"
_BAKED_DIR = _REPO_ROOT / "data" / "processed" / "baked"

sys.path.insert(0, str(_REPO_ROOT))

from scripts.baked_draft_data import (
    build_class_summary,
    get_gm_for_team_year,
    pfr_to_stathead,
    player_row_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec_parquet(path: Path, rows: list[dict]) -> None:
    schema = {
        "Person": pl.String,
        "Teams": pl.String,
        "From": pl.String,
        "To": pl.String,
        "Titles": pl.String,
        "Notes": pl.String,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(path)


def _exec_row(person: str, from_yr: str, to_yr: str, title: str) -> dict:
    return {
        "Person": person,
        "Teams": "Test",
        "From": from_yr,
        "To": to_yr,
        "Titles": title,
        "Notes": None,
    }


# ---------------------------------------------------------------------------
# TestPfrToStathead
# ---------------------------------------------------------------------------


class TestPfrToStathead:
    def test_static_det(self):
        assert pfr_to_stathead("det", 2020) == "DET"

    def test_static_gnb(self):
        assert pfr_to_stathead("gnb", 2020) == "GNB"

    def test_static_rav(self):
        assert pfr_to_stathead("rav", 2020) == "BAL"

    def test_static_clt(self):
        assert pfr_to_stathead("clt", 2020) == "IND"

    def test_static_crd(self):
        assert pfr_to_stathead("crd", 2020) == "ARI"

    def test_static_htx(self):
        assert pfr_to_stathead("htx", 2020) == "HOU"

    def test_static_oti(self):
        assert pfr_to_stathead("oti", 2020) == "TEN"

    def test_static_sfo(self):
        assert pfr_to_stathead("sfo", 2020) == "SFO"

    def test_static_sea(self):
        assert pfr_to_stathead("sea", 2020) == "SEA"

    def test_raiders_pre_relocation(self):
        assert pfr_to_stathead("rai", 2019) == "OAK"

    def test_raiders_post_relocation(self):
        assert pfr_to_stathead("rai", 2020) == "LVR"

    def test_rams_pre_relocation(self):
        assert pfr_to_stathead("ram", 2015) == "STL"

    def test_rams_post_relocation(self):
        assert pfr_to_stathead("ram", 2016) == "LAR"

    def test_chargers_pre_relocation(self):
        assert pfr_to_stathead("sdg", 2016) == "SDG"

    def test_chargers_post_relocation(self):
        assert pfr_to_stathead("sdg", 2017) == "LAC"

    def test_unknown_code_raises(self):
        with pytest.raises(KeyError):
            pfr_to_stathead("xyz", 2020)


# ---------------------------------------------------------------------------
# TestGetGmForTeamYear
# ---------------------------------------------------------------------------


class TestGetGmForTeamYear:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = get_gm_for_team_year("det", 2020, tmp_path)
        assert result is None

    def test_single_match(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [_exec_row("Brad Holmes", "2021", "2026", "General Manager")],
        )
        assert get_gm_for_team_year("det", 2023, tmp_path) == "Brad Holmes"

    def test_no_year_overlap_before(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [_exec_row("Brad Holmes", "2021", "2026", "General Manager")],
        )
        assert get_gm_for_team_year("det", 2019, tmp_path) is None

    def test_no_year_overlap_after(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [_exec_row("Brad Holmes", "2021", "2024", "General Manager")],
        )
        assert get_gm_for_team_year("det", 2025, tmp_path) is None

    def test_compound_title_matches(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [_exec_row("Martin Mayhew", "2008", "2015", "Executive VP of Football Operations/General Manager")],
        )
        assert get_gm_for_team_year("det", 2010, tmp_path) == "Martin Mayhew"

    def test_head_coach_gm_compound_matches(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [_exec_row("Old Timer", "1950", "1960", "Head Coach/General Manager")],
        )
        assert get_gm_for_team_year("det", 1955, tmp_path) == "Old Timer"

    def test_non_gm_title_excluded(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [_exec_row("Scouting Guy", "2015", "2020", "Director of Player Personnel")],
        )
        assert get_gm_for_team_year("det", 2018, tmp_path) is None

    def test_multi_row_overlap_returns_last(self, tmp_path):
        _make_exec_parquet(
            tmp_path / "det_executives.parquet",
            [
                _exec_row("Outgoing GM", "2018", "2020", "General Manager"),
                _exec_row("Interim GM", "2020", "2021", "Interim General Manager"),
            ],
        )
        assert get_gm_for_team_year("det", 2020, tmp_path) == "Interim GM"


# ---------------------------------------------------------------------------
# TestPlayerRowToDict
# ---------------------------------------------------------------------------


class TestPlayerRowToDict:
    def _fully_observed_row(self) -> dict:
        return {
            "Player": "Test Player",
            "Pos": "QB",
            "Pick": 1,
            "Draft Year": 2010,
            "obs_yr0": 5.0,
            "obs_yr1": 6.0,
            "obs_yr2": 7.0,
            "obs_yr3": 4.0,
            "total_4yr_av": 22.0,
            "total_4yr_av_above_replacement": 17.4,
            "eavar": 15.0,
            "eavar_upper": 18.0,
            "eavar_lower": 12.0,
            "replacement_level": 4.586,
            "surplus_av": 2.4,
        }

    def _projected_row(self) -> dict:
        return {
            "Player": "Test Player",
            "Pos": "CB",
            "Pick": 24,
            "Draft Year": 2024,
            "obs_yr0": 5.0,
            "obs_yr1": 6.0,
            "obs_yr2": None,
            "obs_yr3": None,
            "proj_yr2": 5.5,
            "proj_yr3": 3.8,
            "is_projected": True,
            "total_4yr_av": 20.3,
            "total_4yr_av_above_replacement": 15.7,
            "eavar": 10.2,
            "eavar_upper": 12.5,
            "eavar_lower": 8.0,
            "replacement_level": 4.586,
            "surplus_av": 5.5,
        }

    def test_fully_observed_keys(self):
        row = player_row_to_dict(self._fully_observed_row(), is_projected_class=False)
        assert "obs_yr2" in row
        assert "obs_yr3" in row
        assert "proj_yr2" not in row
        assert "proj_yr3" not in row
        assert "is_projected" not in row

    def test_projected_keys_present(self):
        row = player_row_to_dict(self._projected_row(), is_projected_class=True)
        assert "proj_yr2" in row
        assert "proj_yr3" in row
        assert "is_projected" in row
        assert "obs_yr2" in row
        assert "obs_yr3" in row

    def test_projected_obs_yr2_null(self):
        row = player_row_to_dict(self._projected_row(), is_projected_class=True)
        assert row["obs_yr2"] is None
        assert row["obs_yr3"] is None

    def test_float_values_are_python_float(self):
        row = player_row_to_dict(self._fully_observed_row(), is_projected_class=False)
        for key in ("obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3", "total_4yr_av", "surplus_av", "eavar"):
            assert isinstance(row[key], float), f"{key} should be float"

    def test_pick_is_int(self):
        row = player_row_to_dict(self._fully_observed_row(), is_projected_class=False)
        assert isinstance(row["pick"], int)

    def test_player_name_key(self):
        row = player_row_to_dict(self._fully_observed_row(), is_projected_class=False)
        assert row["player"] == "Test Player"
        assert "Player" not in row


# ---------------------------------------------------------------------------
# TestBuildClassSummary
# ---------------------------------------------------------------------------


class TestBuildClassSummary:
    def _make_results(self, rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows)

    def test_empty_valid_rows_returns_none(self):
        df = self._make_results([
            {"surplus_av": None, "total_4yr_av": 10.0, "total_4yr_av_above_replacement": 5.0, "eavar": 8.0, "is_projected": True},
        ])
        assert build_class_summary(df, is_projected_class=True) is None

    def test_sums_valid_rows(self):
        df = self._make_results([
            {"surplus_av": 3.0, "total_4yr_av": 20.0, "total_4yr_av_above_replacement": 15.0, "eavar": 12.0, "is_projected": True},
            {"surplus_av": -1.0, "total_4yr_av": 10.0, "total_4yr_av_above_replacement": 5.0, "eavar": 6.0, "is_projected": False},
        ])
        summary = build_class_summary(df, is_projected_class=True)
        assert summary is not None
        assert summary["total_4yr_av"] == pytest.approx(30.0, abs=0.01)
        assert summary["total_above_replacement"] == pytest.approx(20.0, abs=0.01)
        assert summary["total_eavar"] == pytest.approx(18.0, abs=0.01)
        assert summary["class_surplus"] == pytest.approx(2.0, abs=0.01)
        assert summary["n_players"] == 2

    def test_n_projected_present_when_projected(self):
        df = self._make_results([
            {"surplus_av": 2.0, "total_4yr_av": 10.0, "total_4yr_av_above_replacement": 5.0, "eavar": 3.0, "is_projected": True},
            {"surplus_av": 1.0, "total_4yr_av": 8.0, "total_4yr_av_above_replacement": 3.0, "eavar": 2.0, "is_projected": False},
        ])
        summary = build_class_summary(df, is_projected_class=True)
        assert "n_projected" in summary
        assert summary["n_projected"] == 1

    def test_n_projected_absent_when_fully_observed(self):
        df = self._make_results([
            {"surplus_av": 2.0, "total_4yr_av": 10.0, "total_4yr_av_above_replacement": 5.0, "eavar": 3.0},
        ])
        summary = build_class_summary(df, is_projected_class=False)
        assert "n_projected" not in summary

    def test_mixed_null_only_sums_valid(self):
        df = self._make_results([
            {"surplus_av": 5.0, "total_4yr_av": 20.0, "total_4yr_av_above_replacement": 15.0, "eavar": 10.0},
            {"surplus_av": None, "total_4yr_av": 99.0, "total_4yr_av_above_replacement": 99.0, "eavar": 99.0},
        ])
        summary = build_class_summary(df, is_projected_class=False)
        assert summary["n_players"] == 1
        assert summary["class_surplus"] == pytest.approx(5.0, abs=0.01)

    def test_class_surplus_equals_sum_of_player_surplus(self):
        surpluses = [3.5, -1.2, 7.8, 0.0]
        df = self._make_results([
            {"surplus_av": s, "total_4yr_av": 10.0, "total_4yr_av_above_replacement": 5.0, "eavar": 5.0 - s}
            for s in surpluses
        ])
        summary = build_class_summary(df, is_projected_class=False)
        assert summary["class_surplus"] == pytest.approx(sum(surpluses), abs=0.01)


# ---------------------------------------------------------------------------
# Script smoke test (always runs — no data required)
# ---------------------------------------------------------------------------


class TestScriptInvocable:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--year" in result.stdout
        assert "--output-dir" in result.stdout


# ---------------------------------------------------------------------------
# Integration tests — require generated baked data
# ---------------------------------------------------------------------------

_BAKED_AVAILABLE = _BAKED_DIR.exists() and any(_BAKED_DIR.glob("draft_*.json"))
_SKIP_REASON = f"Baked data not present at {_BAKED_DIR}; run scripts/baked_draft_data.py first"


def _load_year(year: int) -> dict:
    path = _BAKED_DIR / f"draft_{year}.json"
    with open(path) as f:
        return json.load(f)


def _expected_stathead_codes(year: int) -> set[str]:
    """Return the 32 expected stathead codes for a given year, accounting for relocations."""
    codes = {
        "ATL", "BUF", "CAR", "CHI", "CIN", "CLE", "IND", "ARI",
        "DAL", "DEN", "DET", "GNB", "HOU", "JAX", "KAN", "MIA",
        "MIN", "NOR", "NWE", "NYG", "NYJ", "TEN", "PHI", "PIT",
        "BAL", "SEA", "SFO", "TAM", "WAS",
    }
    # Raiders
    codes.add("OAK" if year <= 2019 else "LVR")
    # Rams
    codes.add("STL" if year <= 2015 else "LAR")
    # Chargers
    codes.add("SDG" if year <= 2016 else "LAC")
    return codes


@pytest.mark.skipif(not _BAKED_AVAILABLE, reason=_SKIP_REASON)
class TestBakedDraftDataIntegration:
    def test_all_years_present(self):
        for year in range(2010, 2025):
            assert (_BAKED_DIR / f"draft_{year}.json").exists(), f"Missing draft_{year}.json"

    def test_metadata_keys(self):
        data = _load_year(2020)
        for key in ("generated_at", "year", "models"):
            assert key in data["metadata"], f"metadata missing key: {key}"

    def test_metadata_models_list(self):
        data = _load_year(2020)
        assert set(data["metadata"]["models"]) == {"parametric", "knn", "linear"}

    def test_all_32_teams_per_year(self):
        for year in range(2010, 2025):
            data = _load_year(year)
            expected = _expected_stathead_codes(year)
            present = set(data["teams"].keys())
            missing = expected - present
            assert not missing, f"{year}: missing teams {missing}"

    @pytest.mark.parametrize("year", range(2010, 2023))
    def test_fully_observed_for_pre_2023(self, year):
        data = _load_year(year)
        for team, entry in data["teams"].items():
            if entry.get("fully_observed") is None:
                continue  # no-picks sentinel — skip
            assert entry["fully_observed"] is True, f"{year} {team} should be fully observed"

    @pytest.mark.parametrize("year", [2023, 2024])
    def test_projected_for_2023_2024(self, year):
        data = _load_year(year)
        for team, entry in data["teams"].items():
            if entry.get("fully_observed") is None:
                continue
            assert entry["fully_observed"] is False, f"{year} {team} should not be fully observed"

    def test_fully_observed_entry_shape(self):
        data = _load_year(2015)
        entry = data["teams"]["DET"]
        assert "players" in entry
        assert "class_summary" in entry
        assert "models" not in entry

    def test_projected_entry_shape(self):
        data = _load_year(2024)
        entry = data["teams"]["DET"]
        assert "models" in entry
        assert set(entry["models"].keys()) == {"parametric", "knn", "linear"}
        assert "players" not in entry

    def test_each_model_has_players_and_summary(self):
        data = _load_year(2024)
        models = data["teams"]["DET"]["models"]
        # Non-parametric models are flat: {players, class_summary}
        for model_name in ("knn", "linear"):
            assert "players" in models[model_name], f"{model_name} missing players"
            assert "class_summary" in models[model_name], f"{model_name} missing class_summary"
        # Parametric is nested by curve: {curve_name: {players, class_summary}}
        for curve_name, curve_block in models["parametric"].items():
            assert "players" in curve_block, f"parametric/{curve_name} missing players"
            assert "class_summary" in curve_block, f"parametric/{curve_name} missing class_summary"

    def test_gm_spot_check_det_2010(self):
        data = _load_year(2010)
        assert data["teams"]["DET"]["gm"] == "Martin Mayhew"

    def test_gm_spot_check_det_2024(self):
        data = _load_year(2024)
        assert data["teams"]["DET"]["gm"] == "Brad Holmes"

    def test_players_list_nonempty(self):
        data = _load_year(2010)
        assert len(data["teams"]["DET"]["players"]) > 0

    def test_class_surplus_equals_sum_of_player_surpluses_fully_observed(self):
        data = _load_year(2010)
        entry = data["teams"]["DET"]
        player_sum = sum(
            p["surplus_av"] for p in entry["players"] if p["surplus_av"] is not None
        )
        assert entry["class_summary"]["class_surplus"] == pytest.approx(player_sum, abs=0.5)

    def test_class_surplus_equals_sum_of_player_surpluses_projected(self):
        data = _load_year(2024)
        models = data["teams"]["DET"]["models"]
        # Flatten parametric curves alongside non-parametric models
        blocks: list[tuple[str, dict]] = []
        for name, block in models.items():
            if name == "parametric":
                for curve, cb in block.items():
                    blocks.append((f"parametric/{curve}", cb))
            else:
                blocks.append((name, block))
        for label, model_data in blocks:
            player_sum = sum(
                p["surplus_av"] for p in model_data["players"] if p["surplus_av"] is not None
            )
            summary_surplus = model_data["class_summary"]["class_surplus"]
            assert summary_surplus == pytest.approx(player_sum, abs=0.5), (
                f"{label} class_surplus {summary_surplus} != player sum {player_sum}"
            )

    def test_player_keys_fully_observed(self):
        data = _load_year(2010)
        player = data["teams"]["DET"]["players"][0]
        for key in ("player", "pos", "pick", "obs_yr0", "obs_yr1", "obs_yr2", "obs_yr3",
                    "total_4yr_av", "eavar", "surplus_av"):
            assert key in player, f"player missing key: {key}"
        assert "proj_yr2" not in player

    def test_player_keys_projected(self):
        data = _load_year(2024)
        # knn is a flat block; use it to check projected player keys
        player = data["teams"]["DET"]["models"]["knn"]["players"][0]
        for key in ("player", "pos", "pick", "obs_yr0", "obs_yr1",
                    "total_4yr_av", "eavar", "surplus_av", "proj_yr2", "proj_yr3", "is_projected"):
            assert key in player, f"projected player missing key: {key}"

    # Relocated team spot checks
    def test_raiders_2019_uses_oak(self):
        data = _load_year(2019)
        assert "OAK" in data["teams"]
        assert "LVR" not in data["teams"]

    def test_raiders_2020_uses_lvr(self):
        data = _load_year(2020)
        assert "LVR" in data["teams"]
        assert "OAK" not in data["teams"]

    def test_rams_2015_uses_stl(self):
        data = _load_year(2015)
        assert "STL" in data["teams"]
        assert "LAR" not in data["teams"]

    def test_rams_2016_uses_lar(self):
        data = _load_year(2016)
        assert "LAR" in data["teams"]
        assert "STL" not in data["teams"]

    def test_chargers_2016_uses_sdg(self):
        data = _load_year(2016)
        assert "SDG" in data["teams"]
        assert "LAC" not in data["teams"]

    def test_chargers_2017_uses_lac(self):
        data = _load_year(2017)
        assert "LAC" in data["teams"]
        assert "SDG" not in data["teams"]

    def test_2023_has_n_projected_in_summary(self):
        data = _load_year(2023)
        models = data["teams"]["DET"]["models"]
        for model_name in ("knn", "linear"):
            summary = models[model_name]["class_summary"]
            assert "n_projected" in summary, f"{model_name} summary missing n_projected"
        for curve_name, curve_block in models["parametric"].items():
            summary = curve_block["class_summary"]
            assert "n_projected" in summary, f"parametric/{curve_name} summary missing n_projected"

    def test_2010_no_n_projected_in_summary(self):
        data = _load_year(2010)
        summary = data["teams"]["DET"]["class_summary"]
        assert "n_projected" not in summary

    def test_giovanni_manu_present_in_det_2024(self):
        """Regression: Giovanni Manu (pick 126) earned 0 AV in yr0 and was absent from the
        yr0 Stathead parquet.  canonicalize_positions previously dropped him via an inner join
        on yr0 rows only; _POSITION_OVERRIDES was not forwarded to load_team_draft_class so his
        generalist OL code was never resolved to OT before canonicalization.  Both bugs together
        caused him to be silently omitted from all model outputs for DET 2024."""
        data = _load_year(2024)
        det = data["teams"]["DET"]
        assert not det["fully_observed"]
        # Flatten parametric curves alongside non-parametric models for uniform checking
        blocks: list[tuple[str, dict]] = []
        for model_name, model_block in det["models"].items():
            if model_name == "parametric":
                for curve, cb in model_block.items():
                    blocks.append((f"parametric/{curve}", cb))
            else:
                blocks.append((model_name, model_block))
        for label, block in blocks:
            players = block["players"]
            names = [p["player"] for p in players]
            assert "Giovanni Manu" in names, (
                f"Giovanni Manu missing from DET 2024 {label} players"
            )
            manu = next(p for p in players if p["player"] == "Giovanni Manu")
            assert manu["pick"] == 126
            assert manu["pos"] == "OT", (
                f"Expected OT (from _POSITION_OVERRIDES) but got {manu['pos']} "
                f"in {label} — position override not applied at load time"
            )
