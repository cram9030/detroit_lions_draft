"""Detroit Lions 2024 draft class — projected 4-year AV vs pick expectation.

This script loads observed AV for the Lions' 2024 draft picks (years 0 and 1),
projects years 2–3 using the trained ParametricCurveModel, and compares the
projected 4-year total to the historical expectation derived from pick position.

Prerequisites
-------------
1. **2024 draft data** — the raw store must contain:
       data/raw/stathead/annual_av/draft2024_season2024.parquet
       data/raw/stathead/annual_av/draft2024_season2025.parquet

   To download, update ``config/stathead_annual_av.json``::

       "draft_year_start": 2024,
       "draft_year_end":   2024

   then run::

       python -m src.stathead_downloader --config config/stathead_annual_av.json

2. **Trained parametric model** — ``models/parametric/params.json`` must exist::

       python scripts/train_models.py --model parametric

Outputs
-------
- Console: per-player comparison table + class-level summary
- ``outputs/figures/lions_2024_player_comparison.html``
- ``outputs/figures/lions_2024_class_comparison.html``
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.annual_av_analysis import exponential_av_fit_means
from src.models.parametric import ParametricCurveModel

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

_DRAFT_YEAR = 2024
_OBS_SEASONS = [2024, 2025]  # years_from_draft 0 and 1

# Normalized position mapping matching _POSITION_GROUPS in annual_av_analysis
_POSITION_GROUPS: dict[str, str] = {
    "FL": "WR", "SE": "WR",
    "HB": "RB", "FB": "RB", "RH": "RB", "LH": "RB",
    "LDE": "DE", "RDE": "DE",
    "NT": "DT", "LDT": "DT", "RDT": "DT",
    "LG": "OG", "RG": "OG", "G": "OG",
    "LT": "OT", "RT": "OT", "T": "OT",
    "C": "OC",
    "LCB": "CB", "RCB": "CB",
    "LILB": "LB", "RILB": "LB", "LOLB": "LB", "ROLB": "LB",
    "LLB": "LB", "ILB": "LB", "OLB": "LB", "RLB": "LB", "MLB": "LB",
    "FS": "S", "SS": "S", "DB": "S",
}

# Generalist codes (OL, DL) aren't resolvable without knowing the specific position.
# Override per-player where the actual position is known.
_PLAYER_POSITION_OVERRIDES: dict[str, str] = {
    "Giovanni Manu": "OT",
    "Mekhi Wingo": "DT",
    "Christian Mahogany": "OG",
}


def _normalize_pos(player: str, pos: str) -> str:
    """Return the normalized position, applying per-player overrides first."""
    if player in _PLAYER_POSITION_OVERRIDES:
        return _PLAYER_POSITION_OVERRIDES[player]
    first = pos.replace("-", "/").split("/")[0].strip()
    return _POSITION_GROUPS.get(first, first)


def _check_prerequisites() -> None:
    missing = []
    for season in _OBS_SEASONS:
        path = RAW_DIR / f"draft{_DRAFT_YEAR}_season{season}.parquet"
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Missing 2024 draft data files:\n  " + "\n  ".join(missing) + "\n\n"
            "To download, update config/stathead_annual_av.json:\n"
            '  "draft_year_start": 2024, "draft_year_end": 2024\n'
            "then run:\n"
            "  python -m src.stathead_downloader --config config/stathead_annual_av.json"
        )

    params_path = MODELS_DIR / "parametric" / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Parametric model not found at {params_path}\n"
            "Train it first:\n"
            "  python scripts/train_models.py --model parametric"
        )


def _load_lions_observed() -> pl.DataFrame:
    """Load DET 2024 picks with their observed AV for years 0 and 1."""
    frames = []
    for season in _OBS_SEASONS:
        path = RAW_DIR / f"draft{_DRAFT_YEAR}_season{season}.parquet"
        df = pl.read_parquet(path)
        frames.append(df.filter(pl.col("Draft Team") == "DET"))

    raw = pl.concat(frames)

    prepared = (
        raw
        .with_columns([
            pl.col("Pick").cast(pl.Int64, strict=False),
            pl.col("Season").cast(pl.Int64, strict=False),
            pl.col("Draft Year").cast(pl.Int64, strict=False),
            pl.col("AV.1").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("Pos").str.strip_chars(),
        ])
        .with_columns(
            (pl.col("Season") - pl.col("Draft Year")).alias("years_from_draft")
        )
        .filter(pl.col("years_from_draft").is_in([0, 1]))
    )

    # Normalize positions so compound/variant codes resolve consistently across seasons
    # (e.g. "RCB" and "CB" both become "CB" for the same player)
    prepared = prepared.with_columns(
        pl.struct(["Player", "Pos"]).map_elements(
            lambda s: _normalize_pos(s["Player"], s["Pos"]),
            return_dtype=pl.String,
        ).alias("Pos")
    )

    return prepared.select(["Player", "Pos", "Pick", "Draft Year", "years_from_draft", "AV.1"])


def _build_pick_expectation(max_pick: int = 260) -> dict[int, float]:
    """Return expected 4-year cumulative AV per pick number from the exp-fit model."""
    pick_stats_path = PROCESSED_DIR / "pick_stats.csv"
    if not pick_stats_path.exists():
        raise FileNotFoundError(
            f"pick_stats.csv not found at {pick_stats_path}\n"
            "Run the analysis pipeline first:\n"
            "  python scripts/run_analysis.py"
        )

    stats_df = pl.read_csv(pick_stats_path)
    fit = exponential_av_fit_means(stats_df, max_pick=max_pick)
    import numpy as np
    a, b, c = fit["popt"]
    picks = stats_df.filter(pl.col("Pick") <= max_pick)["Pick"].to_list()
    return {int(p): float(max(0.0, a * np.exp(-b * p) + c)) for p in picks}


def _project_player(
    model: ParametricCurveModel,
    player: str,
    pos: str,
    obs_av: list[float],
) -> tuple[float, float] | None:
    """Return (proj_yr2, proj_yr3) or None if position unknown to model."""
    norm_pos = _normalize_pos(player, pos)
    try:
        result = model.predict(norm_pos, obs_av)
    except ValueError:
        return None
    preds = dict(zip(result["predicted_years"], result["y_pred"]))
    return preds.get(2, 0.0), preds.get(3, 0.0)


def main() -> None:
    _check_prerequisites()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Lions 2024 observed AV...")
    obs_df = _load_lions_observed()

    # Pivot to one row per player with yr0 and yr1 AV
    wide = (
        obs_df
        .pivot(index=["Player", "Pos", "Pick"], on="years_from_draft", values="AV.1")
        .rename({"0": "obs_yr0", "1": "obs_yr1"})
        .with_columns([
            pl.col("obs_yr0").fill_null(0.0),
            pl.col("obs_yr1").fill_null(0.0),
        ])
        .sort("Pick")
    )

    print(f"  {len(wide)} Lions 2024 picks with observed data")

    print("Loading parametric model...")
    model = ParametricCurveModel()
    model.load(MODELS_DIR / "parametric")

    print("Building pick expectation baseline...")
    pick_exp = _build_pick_expectation()

    # Build comparison table
    rows = []
    for row in wide.iter_rows(named=True):
        player = row["Player"]
        pos = row["Pos"]
        pick = row["Pick"]
        obs0 = row["obs_yr0"]
        obs1 = row["obs_yr1"]

        proj = _project_player(model, player, pos, [obs0, obs1])
        if proj is None:
            proj_yr2, proj_yr3 = 0.0, 0.0
            note = f"(pos '{pos}' not in model)"
        else:
            proj_yr2, proj_yr3 = proj
            note = ""

        model_4yr = obs0 + obs1 + proj_yr2 + proj_yr3
        exp_4yr = pick_exp.get(int(pick), 0.0) if pick is not None else 0.0
        delta = model_4yr - exp_4yr

        rows.append({
            "Player": player,
            "Pos": pos,
            "Pick": pick,
            "Obs yr0": round(obs0, 1),
            "Obs yr1": round(obs1, 1),
            "Proj yr2": round(proj_yr2, 1),
            "Proj yr3": round(proj_yr3, 1),
            "Model 4yr": round(model_4yr, 1),
            "Exp 4yr": round(exp_4yr, 1),
            "Delta": round(delta, 1),
            "_note": note,
        })

    results_df = pl.DataFrame(rows)

    print("\n" + "=" * 90)
    print("Detroit Lions 2024 Draft Class — Projected vs Expected 4-Year AV")
    print("=" * 90)
    display_cols = ["Player", "Pos", "Pick", "Obs yr0", "Obs yr1",
                    "Proj yr2", "Proj yr3", "Model 4yr", "Exp 4yr", "Delta"]
    with pl.Config(tbl_width_chars=200, tbl_cols=-1):
        print(results_df.select(display_cols))

    for row in results_df.iter_rows(named=True):
        if row["_note"]:
            print(f"  Note — {row['Player']}: {row['_note']}")

    # Class summary
    class_model = results_df["Model 4yr"].sum()
    class_exp = results_df["Exp 4yr"].sum()
    class_delta = class_model - class_exp
    pct = 100 * class_delta / class_exp if class_exp > 0 else float("nan")
    print(f"\nClass totals:")
    print(f"  Model 4yr AV : {class_model:.1f}")
    print(f"  Exp by pick  : {class_exp:.1f}")
    print(f"  Delta        : {class_delta:+.1f}  ({pct:+.1f}%)")

    # ------------------------------------------------------------------
    # Player comparison chart
    # ------------------------------------------------------------------
    players = results_df["Player"].to_list()
    model_vals = results_df["Model 4yr"].to_list()
    exp_vals = results_df["Exp 4yr"].to_list()
    deltas = results_df["Delta"].to_list()
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    fig_players = go.Figure()
    fig_players.add_bar(name="Model 4yr AV", x=players, y=model_vals, marker_color="#1f77b4")
    fig_players.add_bar(name="Expected by Pick", x=players, y=exp_vals, marker_color="#ff7f0e")
    fig_players.update_layout(
        title="Lions 2024 Draft — Model 4yr AV vs Pick Expectation (per player)",
        xaxis_title="Player",
        yaxis_title="4-Year AV",
        barmode="group",
        xaxis_tickangle=-35,
        height=500,
    )
    fig_players.write_html(str(FIGURES_DIR / "lions_2024_player_comparison.html"))
    print("\n  Saved lions_2024_player_comparison.html")

    # ------------------------------------------------------------------
    # Class-level bar chart
    # ------------------------------------------------------------------
    fig_class = go.Figure()
    fig_class.add_bar(
        x=["Model Projection", "Expected by Pick"],
        y=[class_model, class_exp],
        marker_color=["#1f77b4", "#ff7f0e"],
        text=[f"{class_model:.1f}", f"{class_exp:.1f}"],
        textposition="outside",
    )
    fig_class.update_layout(
        title=f"Lions 2024 Draft Class — Total 4yr AV  (Δ = {class_delta:+.1f}, {pct:+.1f}%)",
        yaxis_title="Total 4-Year AV",
        height=400,
    )
    fig_class.write_html(str(FIGURES_DIR / "lions_2024_class_comparison.html"))
    print("  Saved lions_2024_class_comparison.html")


if __name__ == "__main__":
    main()
