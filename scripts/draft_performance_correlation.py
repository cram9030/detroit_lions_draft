"""Correlate draft class metrics with season performance.

Approach 3 (Lagged): for each lag k=0..3, how does a draft class from year Y
correlate with season Y+k win %, point differential, and SRS?

Approach 2 (Year-Specific Composition): for each season Y, the composite draft
contribution is obs_yr0(class Y) + obs_yr1(class Y-1) + obs_yr2(class Y-2) +
obs_yr3(class Y-3). How does this correlate with season performance?

Usage:
    python scripts/draft_performance_correlation.py
    python scripts/draft_performance_correlation.py --min-year 2010
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gm_assessment import STATHEAD_TO_NFLREADPY
from src.trade_value import load_trade_chart

BAKED_DIR = PROJECT_ROOT / "data" / "processed" / "baked"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

PERF_METRICS = ["win_pct", "point_diff", "srs"]
PERF_LABELS = {"win_pct": "Win %", "point_diff": "Point Differential", "srs": "SRS"}
DRAFT_METRICS = ["class_surplus", "trade_av", "net_av"]
DRAFT_LABELS = {"class_surplus": "Surplus AV", "trade_av": "Trade AV", "net_av": "Net AV"}


# ---------------------------------------------------------------------------
# Season performance (win%, point_diff, SRS) from nflreadpy schedules
# ---------------------------------------------------------------------------

def _srs_for_season(games: pl.DataFrame) -> dict[str, float]:
    """Solve the SRS linear system for one season using a normalised schedule matrix."""
    teams = sorted(games["team"].unique().to_list())
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    margin_sum = np.zeros(n)
    A = np.zeros((n, n))  # A[i,j] = games team i played vs team j

    for row in games.iter_rows(named=True):
        i = idx[row["team"]]
        j = idx.get(row["opponent"])
        if j is not None:
            A[i, j] += 1
            margin_sum[i] += row["pf"] - row["pa"]

    games_played = A.sum(axis=1)
    safe = np.where(games_played == 0, 1.0, games_played)
    mov = margin_sum / safe
    A_norm = A / safe[:, np.newaxis]

    # Solve (I - A_norm) @ SRS = MOV subject to sum(SRS) = 0
    lhs = np.eye(n) - A_norm
    rhs = mov.copy()
    lhs[-1, :] = 1.0
    rhs[-1] = 0.0

    try:
        srs_vals = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        srs_vals = mov  # fallback: uncorrected MOV

    return {t: round(float(s), 3) for t, s in zip(teams, srs_vals)}


def compute_season_performance() -> pl.DataFrame:
    """Return (team, season, win_pct, point_diff, srs) using nflreadpy schedules.

    Team codes are nflreadpy abbreviations (GB, KC, etc.).
    """
    import nflreadpy

    sched = nflreadpy.load_schedules()
    reg = (
        sched
        .filter(pl.col("game_type") == "REG")
        .filter(pl.col("home_score").is_not_null())
        .select(["season", "home_team", "home_score", "away_team", "away_score"])
    )

    home = reg.select(
        pl.col("season"),
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("pf"),
        pl.col("away_score").alias("pa"),
        pl.col("away_team").alias("opponent"),
    )
    away = reg.select(
        pl.col("season"),
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("pf"),
        pl.col("home_score").alias("pa"),
        pl.col("home_team").alias("opponent"),
    )
    all_games = pl.concat([home, away])

    agg = (
        all_games
        .with_columns(
            (pl.col("pf") > pl.col("pa")).cast(pl.Float64).alias("win"),
            (pl.col("pf") == pl.col("pa")).cast(pl.Float64).alias("tie"),
        )
        .group_by(["season", "team"])
        .agg(
            pl.sum("win").alias("wins"),
            pl.sum("tie").alias("ties"),
            pl.len().alias("games"),
            pl.sum("pf").alias("total_pf"),
            pl.sum("pa").alias("total_pa"),
        )
        .with_columns(
            ((pl.col("wins") + 0.5 * pl.col("ties")) / pl.col("games")).alias("win_pct"),
            (pl.col("total_pf") - pl.col("total_pa")).alias("point_diff"),
        )
    )

    srs_rows: list[dict] = []
    for season in sorted(all_games["season"].unique().to_list()):
        sg = all_games.filter(pl.col("season") == season)
        for team, srs in _srs_for_season(sg).items():
            srs_rows.append({"season": season, "team": team, "srs": srs})

    srs_df = pl.DataFrame(srs_rows)
    return (
        agg
        .join(srs_df, on=["season", "team"], how="left")
        .select(["season", "team", "win_pct", "point_diff", "srs"])
        .sort(["season", "team"])
    )


# ---------------------------------------------------------------------------
# Baked draft data loading
# ---------------------------------------------------------------------------

def _obs_players(team_data: dict) -> list[dict]:
    """Return the player list with obs_yr* values (model-independent)."""
    if team_data.get("fully_observed"):
        return team_data.get("players", [])
    models = team_data.get("models", {})
    if not models:
        return []
    # obs values are identical across models; use whichever is present
    return next(iter(models.values())).get("players", [])


def load_baked_data(years: list[int]) -> pl.DataFrame:
    """Load per-(team, draft_year) class surplus and obs_yr* totals from baked JSONs.

    class_surplus is None for partially-observed years (models are not committed
    for this analysis since surplus depends on the projection chosen).
    """
    rows = []
    for year in years:
        path = BAKED_DIR / f"draft_{year}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for team, tdata in data["teams"].items():
            if tdata.get("fully_observed"):
                surplus = tdata.get("class_summary", {}).get("class_surplus")
            else:
                surplus = None

            players = _obs_players(tdata)
            rows.append({
                "stathead_team": team,
                "draft_year": year,
                "class_surplus": surplus,
                "obs_yr0": sum(p.get("obs_yr0") or 0.0 for p in players),
                "obs_yr1": sum(p.get("obs_yr1") or 0.0 for p in players),
                "obs_yr2": sum(p.get("obs_yr2") or 0.0 for p in players),
                "obs_yr3": sum(p.get("obs_yr3") or 0.0 for p in players),
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade value computation
# ---------------------------------------------------------------------------

def compute_trade_values(stathead_teams: list[str], draft_years: list[int]) -> pl.DataFrame:
    """Return net EAVAR trade value per (stathead_team, draft_year).

    Filters the nflreadpy trades table by pick_season == draft_year so that
    picks associated with a given draft class are correctly attributed regardless
    of when the trade occurred.
    """
    import nflreadpy

    trades = nflreadpy.load_trades().filter(pl.col("pick_number").is_not_null() | pl.col("pick_round").is_not_null())

    eavar = load_trade_chart("eavar")
    max_pick = int(eavar["Pick"].max())
    eavar_dict: dict[int, float] = dict(zip(eavar["Pick"].to_list(), eavar["Value"].to_list()))

    def _val(pick_num, pick_rnd) -> float:
        if pick_num is not None:
            p = min(int(pick_num), max_pick)
        elif pick_rnd is not None:
            p = min(int((int(pick_rnd) - 1) * 32 + 16), max_pick)
        else:
            return 0.0
        return eavar_dict.get(p, 0.0)

    rows = []
    for sh_team in stathead_teams:
        nfl_team = STATHEAD_TO_NFLREADPY.get(sh_team, sh_team)
        team_trades = trades.filter(
            (pl.col("gave") == nfl_team) | (pl.col("received") == nfl_team)
        )
        for year in draft_years:
            year_trades = team_trades.filter(pl.col("pick_season") == year)
            rcv = sum(
                _val(r["pick_number"], r["pick_round"])
                for r in year_trades.filter(pl.col("received") == nfl_team).iter_rows(named=True)
            )
            gave = sum(
                _val(r["pick_number"], r["pick_round"])
                for r in year_trades.filter(pl.col("gave") == nfl_team).iter_rows(named=True)
            )
            rows.append({
                "stathead_team": sh_team,
                "draft_year": year,
                "trade_av": round(rcv - gave, 3),
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach 3: Lagged correlation
# ---------------------------------------------------------------------------

def approach3_lagged_correlation(
    draft_df: pl.DataFrame,
    season_perf: pl.DataFrame,
) -> dict:
    """Compute Pearson r and Spearman ρ for each (draft_metric, lag, perf_metric).

    Returns a nested dict: results[dm][lag][pm] = (pearson_r, spearman_r, n).
    """
    nfl_map = STATHEAD_TO_NFLREADPY
    df = draft_df.with_columns(
        pl.col("stathead_team")
        .map_elements(lambda t: nfl_map.get(t, t), return_dtype=pl.String)
        .alias("team")
    )

    results: dict[str, dict[int, dict[str, tuple]]] = {
        dm: {lag: {} for lag in range(4)} for dm in DRAFT_METRICS
    }

    for lag in range(4):
        lag_df = (
            df.with_columns((pl.col("draft_year") + lag).alias("season"))
            .join(season_perf.select(["season", "team"] + PERF_METRICS), on=["season", "team"], how="inner")
        )
        for dm in DRAFT_METRICS:
            for pm in PERF_METRICS:
                pair = lag_df.select([dm, pm]).drop_nulls()
                if len(pair) < 10:
                    continue
                x = pair[dm].to_numpy()
                y = pair[pm].to_numpy()
                pr, _ = stats.pearsonr(x, y)
                sr, _ = stats.spearmanr(x, y)
                results[dm][lag][pm] = (round(pr, 3), round(sr, 3), len(pair))

    return results


def print_lag_correlations(results: dict) -> None:
    col_w = 14
    header = f"{'Metric':<16} {'Lag':<5}" + "".join(
        f"  {PERF_LABELS[pm]:<{col_w}}" for pm in PERF_METRICS
    )

    for stat_name, stat_idx in [("Pearson r", 0), ("Spearman ρ", 1)]:
        print(f"\n=== Approach 3 — {stat_name} ===\n")
        print(header)
        print("-" * len(header))
        for dm in DRAFT_METRICS:
            for lag in range(4):
                vals = []
                n = 0
                for pm in PERF_METRICS:
                    entry = results[dm][lag].get(pm)
                    if entry:
                        vals.append(f"{entry[stat_idx]:>{col_w}.3f}")
                        n = entry[2]
                    else:
                        vals.append(f"{'N/A':>{col_w}}")
                row = f"{DRAFT_LABELS[dm]:<16} {lag:<5}" + "  ".join(vals) + f"  n={n}"
                print(row)
            print()


def plot_lag_heatmap(results: dict) -> None:
    """Save a Pearson r heatmap across all draft metrics, lags, and perf metrics."""
    y_labels = [f"{DRAFT_LABELS[dm]} | lag {lag}" for dm in DRAFT_METRICS for lag in range(4)]
    z_matrix: list[list[float | None]] = []
    text_matrix: list[list[str]] = []

    for dm in DRAFT_METRICS:
        for lag in range(4):
            row_z = []
            row_t = []
            for pm in PERF_METRICS:
                entry = results[dm][lag].get(pm)
                if entry:
                    row_z.append(entry[0])
                    row_t.append(f"{entry[0]:.3f}<br>n={entry[2]}")
                else:
                    row_z.append(None)
                    row_t.append("")
            z_matrix.append(row_z)
            text_matrix.append(row_t)

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=[PERF_LABELS[pm] for pm in PERF_METRICS],
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-0.5,
            zmax=0.5,
            text=text_matrix,
            texttemplate="%{text}",
            hovertemplate="x: %{x}<br>y: %{y}<br>r: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Pearson r"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Approach 3: Lagged Correlation — Pearson r (Draft Metrics vs Season Performance)",
        xaxis_title="Season Performance Metric",
        yaxis_title="Draft Metric | Lag (years)",
        yaxis=dict(autorange="reversed"),
    )
    out = FIGURES_DIR / "draft_lag_correlation_heatmap.html"
    fig.write_html(str(out))
    print(f"\nSaved: {out}")


def plot_lag_scatter(draft_df: pl.DataFrame, season_perf: pl.DataFrame, results: dict) -> None:
    """Save one scatter per (draft_metric, perf_metric) showing all 4 lags as subplots."""
    nfl_map = STATHEAD_TO_NFLREADPY
    df = draft_df.with_columns(
        pl.col("stathead_team")
        .map_elements(lambda t: nfl_map.get(t, t), return_dtype=pl.String)
        .alias("team")
    )

    for dm in DRAFT_METRICS:
        for pm in PERF_METRICS:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[f"Lag {lag}" for lag in range(4)],
                shared_xaxes=False,
            )
            for lag in range(4):
                row, col = divmod(lag, 2)
                lag_df = (
                    df.with_columns((pl.col("draft_year") + lag).alias("season"))
                    .join(season_perf.select(["season", "team", pm]), on=["season", "team"], how="inner")
                    .select([dm, pm, "team", "draft_year"])
                    .drop_nulls()
                )
                if len(lag_df) < 5:
                    continue

                x = lag_df[dm].to_numpy()
                y = lag_df[pm].to_numpy()
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)

                entry = results[dm][lag].get(pm)
                r_label = f"r={entry[0]:.3f}" if entry else ""

                fig.add_trace(
                    go.Scatter(
                        x=x.tolist(),
                        y=y.tolist(),
                        mode="markers",
                        text=[f"{r['team']} {r['draft_year']}" for r in lag_df.iter_rows(named=True)],
                        hovertemplate="%{text}<br>" + f"{dm}=%{{x:.1f}}<br>{pm}=%{{y:.3f}}<extra></extra>",
                        marker=dict(size=4, opacity=0.55, color="#1f77b4"),
                        showlegend=False,
                    ),
                    row=row + 1,
                    col=col + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_line.tolist(),
                        y=(m * x_line + b).tolist(),
                        mode="lines",
                        line=dict(color="red", dash="dash", width=1.5),
                        name=r_label,
                        showlegend=True,
                    ),
                    row=row + 1,
                    col=col + 1,
                )

            dm_slug = dm.replace("_", "-")
            pm_slug = pm.replace("_", "-")
            fig.update_layout(
                template="plotly_white",
                title=f"Approach 3: {DRAFT_LABELS[dm]} vs {PERF_LABELS[pm]} (by lag)",
            )
            fig.update_xaxes(title_text=DRAFT_LABELS[dm])
            fig.update_yaxes(title_text=PERF_LABELS[pm])
            out = FIGURES_DIR / f"draft_lag_{dm_slug}_vs_{pm_slug}.html"
            fig.write_html(str(out))
            print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Approach 2: Year-specific AV composition
# ---------------------------------------------------------------------------

def build_approach2_composite(baked_df: pl.DataFrame, season_perf: pl.DataFrame) -> pl.DataFrame:
    """For each (team, season), compute composite = yr0(Y) + yr1(Y-1) + yr2(Y-2) + yr3(Y-3).

    Requires that obs data for all 4 draft classes is present in baked_df.
    """
    nfl_map = STATHEAD_TO_NFLREADPY
    df = baked_df.with_columns(
        pl.col("stathead_team")
        .map_elements(lambda t: nfl_map.get(t, t), return_dtype=pl.String)
        .alias("team")
    )

    yr0 = df.select(["team", "draft_year", "obs_yr0"]).rename({"draft_year": "season", "obs_yr0": "c_yr0"})
    yr1 = df.select(["team", "draft_year", "obs_yr1"]).with_columns((pl.col("draft_year") + 1).alias("season")).drop("draft_year").rename({"obs_yr1": "c_yr1"})
    yr2 = df.select(["team", "draft_year", "obs_yr2"]).with_columns((pl.col("draft_year") + 2).alias("season")).drop("draft_year").rename({"obs_yr2": "c_yr2"})
    yr3 = df.select(["team", "draft_year", "obs_yr3"]).with_columns((pl.col("draft_year") + 3).alias("season")).drop("draft_year").rename({"obs_yr3": "c_yr3"})

    composite = (
        yr0
        .join(yr1, on=["team", "season"], how="inner")
        .join(yr2, on=["team", "season"], how="inner")
        .join(yr3, on=["team", "season"], how="inner")
        .with_columns(
            (pl.col("c_yr0") + pl.col("c_yr1") + pl.col("c_yr2") + pl.col("c_yr3")).alias("composite_av")
        )
        .join(season_perf.select(["season", "team"] + PERF_METRICS), on=["season", "team"], how="inner")
    )
    return composite


def approach2_correlation(composite: pl.DataFrame) -> dict:
    """Return Pearson r and Spearman ρ for composite_av vs each perf metric."""
    results = {}
    for pm in PERF_METRICS:
        pair = composite.select(["composite_av", pm]).drop_nulls()
        if len(pair) < 10:
            continue
        x = pair["composite_av"].to_numpy()
        y = pair[pm].to_numpy()
        pr, _ = stats.pearsonr(x, y)
        sr, _ = stats.spearmanr(x, y)
        results[pm] = (round(pr, 3), round(sr, 3), len(pair))
    return results


def print_approach2_correlations(results: dict) -> None:
    print("\n=== Approach 2 — Year-Specific AV Composition ===\n")
    print(f"{'Metric':<20} {'Pearson r':>12} {'Spearman ρ':>12} {'n':>6}")
    print("-" * 54)
    for pm in PERF_METRICS:
        entry = results.get(pm)
        if entry:
            print(f"{PERF_LABELS[pm]:<20} {entry[0]:>12.3f} {entry[1]:>12.3f} {entry[2]:>6}")


def plot_approach2_scatter(composite: pl.DataFrame, results: dict) -> None:
    """Save a 1×3 scatter figure for Approach 2 composite vs each performance metric."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[PERF_LABELS[pm] for pm in PERF_METRICS],
    )
    for col_idx, pm in enumerate(PERF_METRICS, start=1):
        pair = composite.select(["composite_av", pm, "team", "season"]).drop_nulls()
        if len(pair) < 5:
            continue
        x = pair["composite_av"].to_numpy()
        y = pair[pm].to_numpy()
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)

        entry = results.get(pm)
        r_label = f"r={entry[0]:.3f}, ρ={entry[1]:.3f}" if entry else ""

        fig.add_trace(
            go.Scatter(
                x=x.tolist(),
                y=y.tolist(),
                mode="markers",
                text=[f"{r['team']} {r['season']}" for r in pair.iter_rows(named=True)],
                hovertemplate="%{text}<br>composite_av=%{x:.1f}<br>" + f"{pm}=%{{y:.3f}}<extra></extra>",
                marker=dict(size=4, opacity=0.55, color="#1f77b4"),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=x_line.tolist(),
                y=(m * x_line + b).tolist(),
                mode="lines",
                line=dict(color="red", dash="dash", width=1.5),
                name=r_label,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        template="plotly_white",
        title="Approach 2: Year-Specific Draft AV Composite vs Season Performance",
    )
    fig.update_xaxes(title_text="Composite Draft AV")
    out = FIGURES_DIR / "draft_approach2_composite_scatter.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Earliest draft class year to include (default: 2000)",
    )
    args = parser.parse_args()

    observed_years = list(range(args.min_year, 2023))  # fully-observed classes
    all_baked_years = list(range(args.min_year, 2025))  # includes 2023, 2024 for obs_yr*

    print("Loading season performance from nflreadpy schedules...")
    season_perf = compute_season_performance()

    print("Loading baked draft data...")
    baked_df = load_baked_data(all_baked_years)

    stathead_teams = sorted(baked_df["stathead_team"].unique().to_list())

    print(f"Computing trade values for {len(stathead_teams)} teams × {len(observed_years)} years...")
    trade_df = compute_trade_values(stathead_teams, observed_years)

    # Build Approach 3 dataset (fully observed only, with trade AV)
    observed_df = (
        baked_df
        .filter(pl.col("class_surplus").is_not_null())
        .join(trade_df, on=["stathead_team", "draft_year"], how="left")
        .with_columns(
            (pl.col("class_surplus") + pl.col("trade_av").fill_null(0.0)).alias("net_av")
        )
    )

    print(f"\nApproach 3 data: {len(observed_df)} team-draft-class rows")

    # --- Approach 3 ---
    lag_results = approach3_lagged_correlation(observed_df, season_perf)
    print_lag_correlations(lag_results)
    plot_lag_heatmap(lag_results)
    plot_lag_scatter(observed_df, season_perf, lag_results)

    # --- Approach 2 ---
    print("\nBuilding year-specific AV composite...")
    composite = build_approach2_composite(baked_df, season_perf)
    print(f"Approach 2 data: {len(composite)} team-season rows (seasons {composite['season'].min()}–{composite['season'].max()})")

    a2_results = approach2_correlation(composite)
    print_approach2_correlations(a2_results)
    plot_approach2_scatter(composite, a2_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
