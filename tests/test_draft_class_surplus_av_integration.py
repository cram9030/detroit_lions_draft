"""Integration tests for scripts/draft_class_surplus_av.py.

These tests exercise the full script end-to-end via subprocess.  They require:
  - data/raw/stathead/annual_av/draft2024_season2024.parquet
  - data/raw/stathead/annual_av/draft2024_season2025.parquet
  - models/parametric/params.json
  - data/processed/expected_av_above_replacement.csv

All tests are skipped when any of these artefacts are absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "draft_class_surplus_av.py"

_REQUIRED_FILES = [
    _REPO_ROOT / "data/raw/stathead/annual_av/draft2024_season2024.parquet",
    _REPO_ROOT / "data/raw/stathead/annual_av/draft2024_season2025.parquet",
    _REPO_ROOT / "models/parametric/params.json",
    _REPO_ROOT / "data/processed/expected_av_above_replacement.csv",
]

_data_available = all(p.exists() for p in _REQUIRED_FILES)
_skip_reason = (
    "Integration data/models not present: "
    + ", ".join(str(p) for p in _REQUIRED_FILES if not p.exists())
)


@pytest.mark.skipif(not _data_available, reason=_skip_reason)
class TestDraftClassSurplusAvScript:
    def test_script_exits_zero(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--team", "DET",
                "--year", "2024",
                "--model", "parametric",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Script exited non-zero.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_player_chart_created(self, tmp_path):
        subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--team", "DET",
                "--year", "2024",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=True,
        )
        assert (tmp_path / "DET_2024_player_surplus_av.html").exists()

    def test_class_chart_created(self, tmp_path):
        subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--team", "DET",
                "--year", "2024",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=True,
        )
        assert (tmp_path / "DET_2024_class_surplus_av.html").exists()

    def test_stdout_contains_class_surplus(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--team", "DET",
                "--year", "2024",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=True,
        )
        assert "Class surplus" in result.stdout

    def test_invalid_year_raises(self, tmp_path):
        """Draft class with fewer than 2 completed seasons must exit non-zero."""
        result = subprocess.run(
            [
                sys.executable, str(_SCRIPT),
                "--team", "DET",
                "--year", "2099",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0


class TestScriptInvocable:
    """Smoke test: --help works without any data present."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--team" in result.stdout
        assert "--year" in result.stdout
