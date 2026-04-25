"""Train CareerAV trajectory models on historical draft data.

Usage
-----
Train parametric model (default)::

    python scripts/train_models.py

Train a specific model::

    python scripts/train_models.py --model parametric
    python scripts/train_models.py --model knn
    python scripts/train_models.py --model ridge
    python scripts/train_models.py --model all

Control training window and round filter::

    python scripts/train_models.py --model parametric --train-years 1970 2010
    python scripts/train_models.py --model parametric --rounds 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.annual_av_analysis import _aggregate_career_av_by_position, _prepare_av_data
from src.data_ingest import load_parquets_from_dir
from src.models.factory import make_career_av_model

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
MODELS_DIR = PROJECT_ROOT / "models"

_SUPPORTED = ["parametric", "knn", "ridge"]
_VAL_YEARS = (2011, 2015)  # holdout validation window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CareerAV trajectory models.")
    parser.add_argument(
        "--model",
        choices=_SUPPORTED + ["all"],
        default="parametric",
        help="Model to train (default: parametric).",
    )
    parser.add_argument(
        "--train-years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=[1970, 2010],
        help="Inclusive draft-year range for training (default: 1970 2010).",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        metavar="ROUND",
        default=None,
        help="Draft rounds to include (default: all rounds).",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=10,
        metavar="N",
        help="Number of career years to model (default: 10).",
    )
    return parser.parse_args()


def _build_trajectory_df(
    train_years: tuple[int, int],
    rounds: list[int] | None,
) -> pl.DataFrame:
    raw_lf = load_parquets_from_dir(RAW_DIR)
    prepared_lf = _prepare_av_data(raw_lf)
    career_lf = _aggregate_career_av_by_position(prepared_lf, normalize=True, rounds=rounds)

    start, end = train_years
    val_start, val_end = _VAL_YEARS
    return (
        career_lf
        .filter(
            (pl.col("Draft Year") >= start) & (pl.col("Draft Year") <= end) & ~((pl.col("Draft Year") >= val_start) & (pl.col("Draft Year") <= val_end))
        )
        .collect()
    )


def _build_val_df(rounds: list[int] | None) -> pl.DataFrame:
    raw_lf = load_parquets_from_dir(RAW_DIR)
    prepared_lf = _prepare_av_data(raw_lf)
    career_lf = _aggregate_career_av_by_position(prepared_lf, normalize=True, rounds=rounds)

    val_start, val_end = _VAL_YEARS
    return (
        career_lf
        .filter(
            (pl.col("Draft Year") >= val_start) & (pl.col("Draft Year") <= val_end)
        )
        .collect()
    )


def _validate_model(model, val_df: pl.DataFrame, max_years: int) -> dict[str, float]:
    """Predict years 3–(max_years-1) given years 0–2; return MAE per position."""
    n_obs = 3
    mae_by_pos: dict[str, list[float]] = {}

    players = (
        val_df
        .group_by(["Player", "Pos"])
        .agg(pl.col("years_from_draft").n_unique().alias("n_years"))
        .filter(pl.col("n_years") >= max_years)
    )

    for row in players.iter_rows(named=True):
        player, pos = row["Player"], row["Pos"]
        player_df = (
            val_df
            .filter((pl.col("Player") == player) & (pl.col("Pos") == pos))
            .sort("years_from_draft")
        )
        if len(player_df) < max_years:
            continue

        actual = player_df["AV.1"].to_list()[:max_years]
        try:
            result = model.predict(pos, actual[:n_obs])
        except ValueError:
            continue

        # Compare predicted years n_obs..9 against actual
        for pred_yr, pred_val in zip(result["predicted_years"], result["y_pred"]):
            if pred_yr < len(actual):
                err = abs(actual[pred_yr] - pred_val)
                mae_by_pos.setdefault(pos, []).append(err)

    return {pos: sum(errs) / len(errs) for pos, errs in mae_by_pos.items()}


def train_one(
    name: str,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    train_years: tuple[int, int],
    rounds: list[int] | None,
    max_years: int,
) -> None:
    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] Training on {len(train_df)} player-seasons "
          f"({train_years[0]}–{train_years[1]})...")
    model = make_career_av_model(name, max_years=max_years)
    model.fit(train_df)
    print(f"[{name}] Fit complete.")

    print(f"[{name}] Validating on {_VAL_YEARS[0]}–{_VAL_YEARS[1]} draft class...")
    val_mae = _validate_model(model, val_df, max_years)

    # Print MAE table
    print(f"\n{'Position':<12} {'Val MAE':>8}")
    print("-" * 22)
    for pos in sorted(val_mae):
        print(f"{pos:<12} {val_mae[pos]:>8.3f}")
    if val_mae:
        overall = sum(val_mae.values()) / len(val_mae)
        print(f"{'OVERALL':<12} {overall:>8.3f}")

    model.save(model_dir)
    print(f"\n[{name}] Saved to {model_dir}")

    # Update metadata.json
    metadata = {
        "model": name,
        "trained_on": date.today().isoformat(),
        "train_years": list(train_years),
        "val_years": list(_VAL_YEARS),
        "val_mae_by_position": {pos: round(v, 4) for pos, v in val_mae.items()},
        "positions": sorted(val_mae.keys()),
        "normalize": True,
        "rounds": rounds,
        "max_years": max_years,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[{name}] Metadata written.")


def main() -> None:
    args = parse_args()
    to_train = _SUPPORTED if args.model == "all" else [args.model]
    train_years = tuple(args.train_years)
    rounds = args.rounds
    max_years = args.max_years

    print("Building training dataset...")
    train_df = _build_trajectory_df(train_years, rounds)
    print(f"  {len(train_df)} rows, {train_df['Pos'].n_unique()} positions")

    print("Building validation dataset...")
    val_df = _build_val_df(rounds)
    print(f"  {len(val_df)} rows")

    for name in to_train:
        train_one(name, train_df, val_df, train_years, rounds, max_years)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
