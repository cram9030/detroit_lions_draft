"""Required rookie-contract pace vs. closest historical career comps.

For a team/draft-year whose rookie contracts have **not** all completed
(fewer than 4 observed seasons), computes the annual AV each player must
still produce to close the gap to their pick's Expected AV Above Replacement
(EAVAR) — i.e. to reach ``surplus_av == 0`` — using the position-based
career-year AV shape (``pos_stats_norm``) to weight how that gap is spread
across the remaining years. It then finds the 3 historical players at the
same position whose actual, fully-realized rookie-contract trajectory most
closely matches the generated (observed + required) profile, and plots each
player against their comps.

Prerequisites
-------------
1. **Draft data** — the raw store must contain at least 2 completed seasons
   (and fewer than 4, or there's nothing to project)::

       data/raw/stathead/annual_av/draft{YEAR}_season{YEAR}.parquet
       data/raw/stathead/annual_av/draft{YEAR}_season{YEAR+1}.parquet

2. **Processed EAVAR** — ``data/processed/expected_av_above_replacement.csv``::

       python scripts/run_analysis.py

Usage
-----
    python scripts/rookie_contract_pace.py --team DET --year 2023
    python scripts/rookie_contract_pace.py --team DET --year 2023 --n-neighbors 3

Outputs
-------
- Console: per-player required-pace table + closest comps
- ``outputs/figures/{TEAM}_{YEAR}_{player}_pace_comparison.html`` (one per player)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.career_av import position_career_stats
from src.rookie_contract_pace import (
    ROOKIE_CONTRACT_YEARS,
    available_rookie_years,
    build_position_reference_trajectories,
    compute_pace_requirements,
    find_closest_career_comps,
)

RAW_DIR = PROJECT_ROOT / "data/raw/stathead/annual_av"
EAVAR_PATH = PROJECT_ROOT / "data/processed/expected_av_above_replacement.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"

_TARGET_COLOR = "#440154"  # Viridis dark purple — matches src/plot_av.py convention
_COMP_DASH = "dot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Required rookie-contract AV pace vs. EAVAR, with closest career comps."
    )
    p.add_argument("--team", required=True, help="Three-letter team code, e.g. DET")
    p.add_argument("--year", required=True, type=int, help="Draft year, e.g. 2023")
    p.add_argument("--raw-dir", type=Path, default=None, help="Directory of annual AV parquets")
    p.add_argument("--eavar-path", type=Path, default=None, help="Path to expected_av_above_replacement.csv")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory to write HTML charts")
    p.add_argument("--n-neighbors", type=int, default=3, help="Number of closest career comps per player")
    p.add_argument(
        "--close-tolerance",
        type=float,
        default=1.0,
        help="AV distance within which comps are treated as tied for recency/team tie-breaking",
    )
    return p.parse_args()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_pace_comparison(
    player: str,
    position: str,
    team: str,
    draft_year: int,
    year_types: list[str],
    target_profile: list[float],
    comps_df: pl.DataFrame,
    export_path: Path,
) -> go.Figure:
    """Plot a player's required pace profile against their closest career comps."""
    years = list(range(len(target_profile)))
    obs_years = [y for y, t in zip(years, year_types) if t == "observed"]
    req_years = [y for y, t in zip(years, year_types) if t == "required"]
    obs_vals = [target_profile[y] for y in obs_years]
    req_vals = [target_profile[y] for y in req_years]

    # Bridge the dashed "required" segment to the last observed point so the line is continuous.
    if obs_years and req_years:
        req_years = [obs_years[-1]] + req_years
        req_vals = [target_profile[obs_years[-1]]] + req_vals

    fig = go.Figure()

    if obs_years:
        fig.add_trace(
            go.Scatter(
                x=obs_years,
                y=obs_vals,
                mode="lines+markers",
                line=dict(color=_TARGET_COLOR, width=3),
                marker=dict(size=8),
                name=f"{player} — observed",
            )
        )
    if req_years:
        fig.add_trace(
            go.Scatter(
                x=req_years,
                y=req_vals,
                mode="lines+markers",
                line=dict(color=_TARGET_COLOR, width=3, dash="dash"),
                marker=dict(size=9, symbol="diamond"),
                name=f"{player} — required pace",
            )
        )

    n_comps = len(comps_df)
    comp_colors = (
        px.colors.sample_colorscale("Viridis", n_comps) if n_comps > 1 else ["#21908c"]
    )
    for color, row in zip(comp_colors, comps_df.iter_rows(named=True)):
        y_vals = [row[f"yr{y}"] for y in years]
        dist = row["distance"]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=color, width=2, dash=_COMP_DASH),
                marker=dict(size=6),
                name=(
                    f"{row['Player']} ({row['Draft Team']} {row['Draft Year']}) "
                    f"— dist {dist:.1f}"
                ),
            )
        )

    fig.update_layout(
        title=f"{player} ({position}, {team} {draft_year}) — Required Pace vs. Closest Career Comps",
        xaxis=dict(title="Years from Draft", dtick=1),
        yaxis_title="Annual AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    export_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(export_path))
    return fig


def main() -> None:
    args = parse_args()
    team = args.team.upper()
    year = args.year
    raw_dir = args.raw_dir or RAW_DIR
    eavar_path = args.eavar_path or EAVAR_PATH
    output_dir = args.output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking rookie-contract status for {team} {year}...")
    available_years = available_rookie_years(year, raw_dir)
    if len(available_years) >= len(ROOKIE_CONTRACT_YEARS):
        print(f"{team} {year} has completed all 4 rookie-contract seasons — no pace projection needed.")
        return
    remaining_years = [y for y in ROOKIE_CONTRACT_YEARS if y not in available_years]
    print(f"  Observed years:  {available_years}")
    print(f"  Remaining years: {remaining_years}")

    print("Computing position-based career AV curves (pos_stats_norm)...")
    pos_stats_norm = position_career_stats(raw_dir, normalize=True)

    print(f"Computing required pace for {team} {year}...")
    pace_df = compute_pace_requirements(
        team, year, raw_dir=raw_dir, eavar_path=eavar_path, pos_stats=pos_stats_norm
    )

    display_cols = [
        "Player", "Pos", "Pick",
        "total_observed_av", "target_total_av", "required_av_remaining",
    ] + [f"yr{y}_av" for y in ROOKIE_CONTRACT_YEARS]
    print()
    print("=" * 110)
    print(f"{team} {year} — Required Rookie-Contract Pace vs. EAVAR")
    print("=" * 110)
    with pl.Config(tbl_width_chars=220, tbl_cols=-1):
        print(pace_df.select(display_cols))

    print("\nBuilding historical position reference trajectories (fully-realized rookie contracts)...")
    reference_df = build_position_reference_trajectories(raw_dir=raw_dir)
    class_players = set(pace_df["Player"].to_list())

    print(f"\nFinding closest career comps (top {args.n_neighbors}) and generating plots...\n")
    for row in pace_df.iter_rows(named=True):
        player = row["Player"]
        position = row["Pos"]
        target_profile = [row[f"yr{y}_av"] for y in ROOKIE_CONTRACT_YEARS]
        year_types = [row[f"yr{y}_type"] for y in ROOKIE_CONTRACT_YEARS]

        comps_df = find_closest_career_comps(
            target_profile,
            position,
            team,
            reference_df,
            n_neighbors=args.n_neighbors,
            close_tolerance=args.close_tolerance,
            exclude_players=class_players,
        )

        if comps_df.is_empty():
            print(f"  {player} ({position}): no historical comps found at this position — skipping plot.")
            continue

        comp_names = ", ".join(comps_df["Player"].to_list())
        print(
            f"  {player} ({position}, pick {row['Pick']}): needs {row['required_av_remaining']:+.1f} AV "
            f"over years {remaining_years} -> comps: {comp_names}"
        )

        export_path = output_dir / f"{team}_{year}_{_slugify(player)}_pace_comparison.html"
        plot_pace_comparison(player, position, team, year, year_types, target_profile, comps_df, export_path)
        print(f"    Saved {export_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
