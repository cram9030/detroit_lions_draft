"""Unit tests for _validate_model in scripts/train_models.py."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_models import _validate_model

_MAX_YEARS = 10
_N_OBS = 3  # hardcoded in _validate_model


class _ConstantModel:
    """Predicts a fixed constant for all future years."""

    def predict(self, pos, observed_av):
        n_obs = len(observed_av)
        future = list(range(n_obs, _MAX_YEARS))
        return {
            "position": pos,
            "observed_years": list(range(n_obs)),
            "observed_av": observed_av,
            "predicted_years": future,
            "y_pred": [5.0] * len(future),
            "y_upper": [6.0] * len(future),
            "y_lower": [4.0] * len(future),
        }


def _make_rows(player, pos, n_years, av=10.0):
    return [
        {"Player": player, "Pos": pos, "Draft Year": 2011,
         "years_from_draft": yr, "AV.1": av}
        for yr in range(n_years)
    ]


def test_validate_includes_short_career_players():
    """Players with fewer than max_years seasons must be included in validation."""
    # Only one player, 4 seasons (< max_years=10).
    # Old code filtered to n_years >= max_years, so this player would be excluded → mae = {}.
    rows = _make_rows("ShortCareer", "QB", n_years=4, av=10.0)
    val_df = pl.DataFrame(rows)

    mae = _validate_model(_ConstantModel(), val_df, max_years=_MAX_YEARS)

    assert "QB" in mae  # old code: {} because no player has 10 seasons


def test_validate_zeros_out_missing_years():
    """Late years a short-career player didn't play must count as actual=0."""
    # Player played exactly n_obs=3 seasons with AV=0; years 3-9 are absent in data.
    # After zero-padding: actual[3..9] = 0.0, model predicts 5.0 → MAE = 5.0.
    rows = _make_rows("Bust", "RB", n_years=_N_OBS, av=0.0)
    val_df = pl.DataFrame(rows)

    mae = _validate_model(_ConstantModel(), val_df, max_years=_MAX_YEARS)

    assert "RB" in mae
    assert abs(mae["RB"] - 5.0) < 1e-6
