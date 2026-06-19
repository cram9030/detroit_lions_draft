"""Generate one JSON file per draft year covering all 32 NFL franchises.

Each file contains per-player and class-level surplus AV data for that year's
draft class, plus the GM for each franchise.  Fully-observed classes (≥4
completed seasons) use raw AV data only.  Partially-observed classes include
projections from all available career AV models.

Output files
------------
``data/baked/draft_{YEAR}.json`` for each year in [2010, --year].

JSON shape — fully observed::

    {
      "metadata": {"generated_at": "...", "year": 2024, "models": {...}},
      "teams": {
        "DET": {
          "gm": "Brad Holmes",
          "fully_observed": true,
          "players": [...],
          "class_summary": {...}
        }
      }
    }

JSON shape — partially observed (2023/2024)::

    "DET": {
      "gm": "Brad Holmes",
      "fully_observed": false,
      "models": {
        "parametric": {
          "gamma":    {"players": [...], "class_summary": {...}},
          "exp_decay": {"players": [...], "class_summary": {...}}
        },
        "knn":   {"players": [...], "class_summary": {...}},
        "linear": {"players": [...], "class_summary": {...}}
      }
    }

All trained parametric curve variants in ``models/parametric/`` are discovered
automatically — train a new variant with ``train_models.py --curve <name>`` and
it will appear in the next bake without any code changes.

Prerequisites
-------------
1. Stathead AV data (2+ seasons per class): ``data/raw/stathead/annual_av/``
2. PFR executives data: ``data/raw/pfr/executives/``
3. EAVAR table: ``data/processed/expected_av_above_replacement.csv``
4. Trained models (for 2023/2024 projections)::

       python scripts/train_models.py --model parametric --curve gamma
       python scripts/train_models.py --model knn
       python scripts/train_models.py --model linear

Usage
-----
    python scripts/baked_draft_data.py
    python scripts/baked_draft_data.py --year 2023
    python scripts/baked_draft_data.py --output-dir outputs/baked
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.models.factory import make_career_av_model
from src.models.utils import check_model_exists, discover_parametric_curves
from src.surplus_av import (
    aggregate_model_av,
    aggregate_observed_av,
    compute_surplus_av,
    load_team_draft_class,
)
from src.teams import ALL_PFR_CODES, pfr_to_stathead

MODELS_DIR = PROJECT_ROOT / "models"
BAKED_DIR = PROJECT_ROOT / "data" / "processed" / "baked"
EXECUTIVES_DIR = PROJECT_ROOT / "data" / "raw" / "pfr" / "executives"
STATHEAD_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stathead" / "annual_av"

START_YEAR = 2000
_NON_PARAMETRIC_MODELS = ["knn", "linear"]

# Per-player position overrides for players Stathead records with a generalist
# catch-all code ("OL", "DL") that the career-AV models do not recognise.
# Keys are exact player names; values are the normalised position group to use.
_POSITION_OVERRIDES: dict[str, str] = {
    # 2024 draft
    "Christian Mahogany": "OG",
    "Mekhi Wingo": "DT",
    "Giovanni Manu": "OT",
    # 2023 draft
    "Juice Scruggs": "OC",
    "Zach Harrison": "DE",
    "Nick Saldiveri": "OT",
    "Colby Wooden": "DE",
    "Colby Sorsdal": "OT",
    "Jordan McFadden": "OG",
    "Asim Richards": "OT",
    "Robert Beal": "DE",
    "Karl Brooks": "DT",
    "Jovaughn Gwyn": "OG",
    "Jordon Riley": "DT",
    "Spencer Anderson": "OG",
}


def get_gm_for_team_year(
    pfr_code: str,
    year: int,
    executives_dir: Path,
) -> str | None:
    """Return the GM name for a franchise/year, or None if not found."""
    path = executives_dir / f"{pfr_code}_executives.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    matches = df.filter(
        pl.col("Titles").str.contains("General Manager")
        & (pl.col("From").cast(pl.Int32) <= year)
        & (pl.col("To").cast(pl.Int32) >= year)
    )
    if matches.is_empty():
        return None
    return matches.tail(1)["Person"][0]


def player_row_to_dict(row: dict, is_projected_class: bool) -> dict:
    """Serialize a compute_surplus_av result row to a JSON-ready dict."""

    def _f(v) -> float | None:
        return None if v is None else round(float(v), 3)

    out: dict = {
        "player": str(row["Player"]),
        "pos": str(row["Pos"]),
        "pick": int(row["Pick"]),
        "obs_yr0": _f(row["obs_yr0"]),
        "obs_yr1": _f(row["obs_yr1"]),
    }

    if is_projected_class:
        out["obs_yr2"] = _f(row.get("obs_yr2"))
        out["obs_yr3"] = _f(row.get("obs_yr3"))
        out["proj_yr2"] = _f(row.get("proj_yr2"))
        out["proj_yr3"] = _f(row.get("proj_yr3"))
        out["is_projected"] = bool(row.get("is_projected", False))
    else:
        out["obs_yr2"] = _f(row.get("obs_yr2"))
        out["obs_yr3"] = _f(row.get("obs_yr3"))

    out["total_4yr_av"] = _f(row["total_4yr_av"])
    out["total_4yr_av_above_replacement"] = _f(row["total_4yr_av_above_replacement"])
    out["eavar"] = _f(row["eavar"])
    out["eavar_upper"] = _f(row["eavar_upper"])
    out["eavar_lower"] = _f(row["eavar_lower"])
    out["replacement_level"] = _f(row["replacement_level"])
    out["surplus_av"] = _f(row["surplus_av"])
    return out


def build_class_summary(
    results_df: pl.DataFrame,
    is_projected_class: bool,
) -> dict | None:
    """Build a class-level summary dict from compute_surplus_av output."""
    valid = results_df.filter(pl.col("surplus_av").is_not_null())
    if valid.is_empty():
        return None

    summary: dict = {
        "total_4yr_av": round(float(valid["total_4yr_av"].sum()), 3),
        "total_above_replacement": round(float(valid["total_4yr_av_above_replacement"].sum()), 3),
        "total_eavar": round(float(valid["eavar"].sum()), 3),
        "class_surplus": round(float(valid["surplus_av"].sum()), 3),
        "n_players": len(valid),
    }
    if is_projected_class:
        summary["n_projected"] = int(valid["is_projected"].sum())
    return summary


def _build_model_block(
    draft_df: pl.DataFrame,
    model_name: str,
    models_dir: Path,
    position_overrides: dict[str, str] | None = None,
    curve_name: str | None = None,
) -> dict:
    """Run one model projection and return its players + class_summary."""
    model = make_career_av_model(model_name)
    if model_name == "parametric" and curve_name:
        load_path = models_dir / "parametric" / curve_name
    else:
        load_path = models_dir / model_name
    model.load(load_path)
    players_df = aggregate_model_av(draft_df, model, position_overrides)
    results = compute_surplus_av(players_df)
    players = [player_row_to_dict(r, is_projected_class=True) for r in results.iter_rows(named=True)]
    return {
        "players": players,
        "class_summary": build_class_summary(results, is_projected_class=True),
    }


def build_team_entry(
    pfr_code: str,
    stathead_code: str,
    year: int,
    models_dir: Path,
    executives_dir: Path,
    parametric_curves: list[str],
) -> dict:
    """Build the full JSON entry for one team/year."""
    gm = get_gm_for_team_year(pfr_code, year, executives_dir)

    try:
        draft_df = load_team_draft_class(stathead_code, year, position_overrides=_POSITION_OVERRIDES)
    except ValueError as exc:
        warnings.warn(f"No valid draft data for {stathead_code} {year}: {exc}")
        return {"gm": gm, "fully_observed": None, "players": [], "class_summary": None}

    # A class is fully observed when the yr3 season file exists on disk — even if
    # this team's players all had 0 AV that year (and thus have no rows in it).
    fully_observed = (STATHEAD_RAW_DIR / f"draft{year}_season{year + 3}.parquet").exists()

    if fully_observed:
        players_df = aggregate_observed_av(draft_df)
        results = compute_surplus_av(players_df)
        players = [player_row_to_dict(r, is_projected_class=False) for r in results.iter_rows(named=True)]
        return {
            "gm": gm,
            "fully_observed": True,
            "players": players,
            "class_summary": build_class_summary(results, is_projected_class=False),
        }
    else:
        model_results: dict = {}

        # Each discovered parametric curve variant gets its own nested block.
        if parametric_curves:
            model_results["parametric"] = {
                curve: _build_model_block(
                    draft_df, "parametric", models_dir, _POSITION_OVERRIDES, curve_name=curve
                )
                for curve in parametric_curves
            }

        for model_name in _NON_PARAMETRIC_MODELS:
            model_results[model_name] = _build_model_block(
                draft_df, model_name, models_dir, _POSITION_OVERRIDES
            )

        return {
            "gm": gm,
            "fully_observed": False,
            "models": model_results,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-year baked draft data JSON for static site")
    p.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Last draft year to include (inclusive, default: 2024)",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing trained model artifacts (default: models/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write draft_{YEAR}.json files (default: data/processed/baked/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end_year = args.year
    models_dir = args.models_dir or MODELS_DIR
    output_dir = args.output_dir or BAKED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    parametric_curves = discover_parametric_curves(models_dir)
    if parametric_curves:
        print(f"Discovered parametric variants: {parametric_curves}")
    else:
        warnings.warn(
            "No trained parametric variants found in models/parametric/ — "
            "parametric projections will be omitted from partially-observed classes. "
            "Train one with: python scripts/train_models.py --model parametric --curve gamma"
        )

    for model_name in _NON_PARAMETRIC_MODELS:
        check_model_exists(model_name, models_dir)

    models_metadata: dict = {
        "parametric": parametric_curves,
        "knn": True,
        "linear": True,
    }

    generated_at = datetime.now(timezone.utc).isoformat()

    for year in range(START_YEAR, end_year + 1):
        print(f"\n--- {year} ---")
        teams: dict = {}
        for pfr_code in ALL_PFR_CODES:
            stathead_code = pfr_to_stathead(pfr_code, year)
            print(f"  {stathead_code}", end=" ", flush=True)
            try:
                entry = build_team_entry(
                    pfr_code, stathead_code, year, models_dir, EXECUTIVES_DIR, parametric_curves
                )
            except Exception as exc:
                warnings.warn(f"Unexpected error for {stathead_code} {year}: {exc}")
                continue
            teams[stathead_code] = entry

        payload = {
            "metadata": {
                "generated_at": generated_at,
                "year": year,
                "models": models_metadata,
            },
            "teams": teams,
        }
        out_path = output_dir / f"draft_{year}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  → {out_path.name}")

    print(f"\nDone. {end_year - START_YEAR + 1} files written to {output_dir}/")


if __name__ == "__main__":
    main()
