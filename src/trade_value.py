"""Trade chart loading and pick combination finder for draft trade evaluation."""

from pathlib import Path
from typing import TypedDict

import nflreadpy
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED_DATA_DIR = _REPO_ROOT / "data" / "processed"

# Registry: chart_name -> (filename, pick_col, value_col, deduplicate_on_pick)
_CHART_REGISTRY: dict[str, tuple[str, str, str, bool]] = {
    "jimmy_johnson": (
        "jimmy_johnson_trade_chart.csv",
        "Pick",
        "Value",
        True,
    ),
    "fitzgerald_spielberger": (
        "fitzgerald_spielberger_trade_chart.csv",
        "Pick",
        "Value",
        False,
    ),
    "pff_war": (
        "pff_war_draft_chart.csv",
        "Pick",
        "PFF_WAR_Normalized",
        False,
    ),
    "rich_hill": (
        "Rich-Hill.csv",
        "pick",
        "value",
        False,
    ),
    "eavar": (
        "expected_av_above_replacement.csv",
        "pick",
        "eavar",
        False,
    ),
    "5_year_av": (
        "5_year_av_chart.csv",
        "Pk",
        "FP Val",
        False,
    ),
}


class PickCombinationResult(TypedDict):
    chart_name: str
    target_value: float
    picks: list[int]
    values: list[float]
    total_value: float
    error: float  # total_value - target_value (signed)
    n_picks: int


def load_trade_chart(
    chart_name: str,
    data_dir: Path | str = _PROCESSED_DATA_DIR,
) -> pl.DataFrame:
    """Load a trade chart CSV and return a normalised ``[Pick, Value]`` DataFrame."""
    if chart_name not in _CHART_REGISTRY:
        valid = ", ".join(sorted(_CHART_REGISTRY))
        raise ValueError(f"Unknown chart '{chart_name}'. Valid names: {valid}")

    filename, pick_col, value_col, dedup = _CHART_REGISTRY[chart_name]
    path = Path(data_dir) / filename

    df = pl.read_csv(path, infer_schema_length=10000)

    if dedup:
        df = df.unique(subset=[pick_col], keep="first")

    df = (
        df.select([pick_col, value_col])
        .rename({pick_col: "Pick", value_col: "Value"})
        .with_columns([
            pl.col("Pick").cast(pl.Int64),
            pl.col("Value").cast(pl.Float64),
        ])
        .sort("Pick")
    )

    return df


def _extended_two_pointer(
    pick_values: list[tuple[int, float]],
    target: float,
    max_picks: int,
    tolerance: float,
) -> tuple[list[int], list[float], float]:
    """Find the combination of picks whose sum is closest to ``target``.

    Uses an extended two-pointer strategy: for each depth k=1..max_picks,
    fixes k-2 outer elements and runs a two-pointer sweep over the remaining
    tail. Stops early when criteria are met (see inline comments).

    ``pick_values`` must be sorted by value ascending.
    """
    n = len(pick_values)
    if n == 0:
        return [], [], 0.0

    min_val = pick_values[0][1]

    best_picks: list[int] = []
    best_values: list[float] = []
    best_total: float = 0.0
    best_error: float = float("inf")

    def _update_best(candidate_indices: list[int]) -> None:
        nonlocal best_picks, best_values, best_total, best_error
        total = sum(pick_values[i][1] for i in candidate_indices)
        err = abs(total - target)
        if err < best_error:
            best_error = err
            best_total = total
            best_picks = [pick_values[i][0] for i in candidate_indices]
            best_values = [pick_values[i][1] for i in candidate_indices]

    def _two_pointer_sweep(fixed: list[int], lo: int, hi: int) -> None:
        fixed_sum = sum(pick_values[i][1] for i in fixed)
        remaining = target - fixed_sum
        left, right = lo, hi
        while left < right:
            s = pick_values[left][1] + pick_values[right][1]
            _update_best(fixed + [left, right])
            if abs(s - remaining) < 1e-12:
                return
            if s < remaining:
                left += 1
            else:
                right -= 1

    def _recurse(fixed: list[int], start: int, depth_remaining: int) -> None:
        if depth_remaining == 2:
            _two_pointer_sweep(fixed, start, n - 1)
            return
        for i in range(start, n - depth_remaining + 1):
            _update_best(fixed + [i])
            if depth_remaining > 1:
                _recurse(fixed + [i], i + 1, depth_remaining - 1)
            fixed_so_far = sum(pick_values[j][1] for j in fixed) + pick_values[i][1]
            if fixed_so_far > target + best_error:
                break

    for i in range(n):
        _update_best([i])
        if pick_values[i][1] > target + best_error:
            break

    prev_error = best_error

    for k in range(2, max_picks + 1):
        if best_error <= tolerance:
            break
        if best_error < min_val:
            break
        if k > 2 and best_error >= prev_error:
            break

        prev_error = best_error

        if k == 2:
            _two_pointer_sweep([], 0, n - 1)
        else:
            _recurse([], 0, k)

    return best_picks, best_values, best_total


def find_pick_combination(
    target_value: float,
    chart_name: str,
    max_picks: int = 5,
    tolerance: float = 0.0,
    data_dir: Path | str = _PROCESSED_DATA_DIR,
) -> PickCombinationResult:
    """Find the combination of picks whose chart values sum closest to ``target_value``."""
    df = load_trade_chart(chart_name, data_dir=data_dir)

    pick_values: list[tuple[int, float]] = sorted(
        [(row["Pick"], row["Value"]) for row in df.iter_rows(named=True)],
        key=lambda x: x[1],
    )

    picks, values, total = _extended_two_pointer(
        pick_values, target_value, max_picks, tolerance
    )

    return PickCombinationResult(
        chart_name=chart_name,
        target_value=target_value,
        picks=picks,
        values=values,
        total_value=total,
        error=total - target_value,
        n_picks=len(picks),
    )


# ---------------------------------------------------------------------------
# Draft trade analysis
# ---------------------------------------------------------------------------

_TRADE_CHARTS: list[tuple[str, str]] = [
    ("fitz_spiel", "fitzgerald_spielberger"),
    ("jj", "jimmy_johnson"),
    ("pff", "pff_war"),
    ("rich_hill", "rich_hill"),
    ("eaar", "eavar"),
]


def _empty_trade_df() -> pl.DataFrame:
    schema: dict[str, pl.PolarsDataType] = {
        "trade_id": pl.Int64,
        "team_traded_with": pl.String,
        "picks_received": pl.String,
        "picks_gave": pl.String,
        "picks_received_seasons": pl.String,
        "picks_gave_seasons": pl.String,
    }
    for prefix, _ in _TRADE_CHARTS:
        schema[f"{prefix}_value"] = pl.Float64
        schema[f"{prefix}_picks"] = pl.String
    return pl.DataFrame(schema=schema)


def _resolve_pick_number(
    pick_number: float | None,
    pick_round: float | None,
) -> int | None:
    """Return a concrete pick number, estimating mid-round when only round is known."""
    if pick_number is not None:
        return int(pick_number)
    if pick_round is not None:
        return int((pick_round - 1) * 32 + 16)
    return None


def _compute_chart_value(
    picks_received: list[int],
    picks_gave: list[int],
    chart_name: str,
    data_dir: Path | str = _PROCESSED_DATA_DIR,
) -> tuple[float, str]:
    chart = load_trade_chart(chart_name, data_dir=data_dir)

    def _val(p: int) -> float:
        row = chart.filter(pl.col("Pick") == p)
        return float(row["Value"][0]) if len(row) > 0 else 0.0

    net = sum(_val(p) for p in picks_received) - sum(_val(p) for p in picks_gave)

    if net != 0:
        result = find_pick_combination(abs(net), chart_name, data_dir=data_dir)
        equiv = ",".join(str(p) for p in sorted(result["picks"]))
    else:
        equiv = ""

    return net, equiv


def analyze_draft_trades(
    team: str,
    year: int,
    data_dir: Path | str = _PROCESSED_DATA_DIR,
    trades_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return a DataFrame of draft trades involving team in the given year."""
    all_trades = trades_df if trades_df is not None else nflreadpy.load_trades()
    team_trades = all_trades.filter(
        (pl.col("season") == year)
        & ((pl.col("gave") == team) | (pl.col("received") == team))
    )

    if len(team_trades) == 0:
        return _empty_trade_df()

    output_rows: list[dict] = []

    for tid in team_trades["trade_id"].unique().to_list():
        trade_rows = team_trades.filter(pl.col("trade_id") == tid)

        if trade_rows.filter(
            pl.col("pick_season").is_null()
            & pl.col("pick_round").is_null()
            & pl.col("pick_number").is_null()
        ).height > 0:
            continue

        rcv_pick_data: list[tuple[int, int]] = []  # (overall, season)
        gave_pick_data: list[tuple[int, int]] = []  # (overall, season)
        for row in trade_rows.iter_rows(named=True):
            resolved = _resolve_pick_number(row["pick_number"], row["pick_round"])
            if resolved is None:
                continue
            season = int(row["pick_season"]) if row["pick_season"] is not None else year
            if row["received"] == team:
                rcv_pick_data.append((resolved, season))
            if row["gave"] == team:
                gave_pick_data.append((resolved, season))

        if not rcv_pick_data and not gave_pick_data:
            continue

        rcv_pick_data.sort()
        gave_pick_data.sort()
        rcv_picks = [p for p, _ in rcv_pick_data]
        gave_picks = [p for p, _ in gave_pick_data]

        other_teams: set[str] = set()
        for gave_col, rcv_col in zip(
            trade_rows["gave"].to_list(), trade_rows["received"].to_list()
        ):
            if gave_col != team:
                other_teams.add(gave_col)
            if rcv_col != team:
                other_teams.add(rcv_col)

        row_dict: dict = {
            "trade_id": int(tid),
            "team_traded_with": ",".join(sorted(other_teams)),
            "picks_received": ",".join(str(p) for p, _ in rcv_pick_data),
            "picks_gave": ",".join(str(p) for p, _ in gave_pick_data),
            "picks_received_seasons": ",".join(str(s) for _, s in rcv_pick_data),
            "picks_gave_seasons": ",".join(str(s) for _, s in gave_pick_data),
        }

        for prefix, chart_name in _TRADE_CHARTS:
            val, equiv = _compute_chart_value(rcv_picks, gave_picks, chart_name, data_dir)
            row_dict[f"{prefix}_value"] = val
            row_dict[f"{prefix}_picks"] = equiv

        output_rows.append(row_dict)

    if not output_rows:
        return _empty_trade_df()

    return pl.DataFrame(output_rows)


def aggregate_trade_value(
    team: str,
    years: list[int],
    data_dir: Path | str = _PROCESSED_DATA_DIR,
    chart_name: str = "eaar",
) -> dict:
    """Aggregate net trade value across multiple draft years for a team.

    Calls :func:`analyze_draft_trades` for each year, extracts the column
    ``{chart_name}_value``, and sums across all trades in that year.  Years
    with no draft trades count as 0.0 net value and are included in ``n_years``.

    Args:
        team: Stathead team code (e.g. ``"DET"``).
        years: List of draft years to aggregate.
        data_dir: Directory containing trade chart CSVs.
        chart_name: Column prefix in :func:`analyze_draft_trades` output
            to sum (default: ``"eaar"`` — EAVAR units, same scale as surplus AV).

    Returns:
        Dict with keys:
        - ``total_trade_value``: sum of yearly net values.
        - ``per_year``: ``{year: net_value}`` for every requested year.
        - ``n_years``: number of requested years (including 0-trade years).
        - ``avg_trade_per_year``: ``total / n_years`` (0.0 when n_years==0).
    """
    per_year: dict[int, float] = {}
    col = f"{chart_name}_value"

    for year in years:
        try:
            trades_df = analyze_draft_trades(team, year, data_dir=data_dir)
        except Exception:
            per_year[year] = 0.0
            continue

        if col in trades_df.columns and len(trades_df) > 0:
            per_year[year] = round(float(trades_df[col].drop_nulls().sum()), 3)
        else:
            per_year[year] = 0.0

    total = sum(per_year.values())
    n = len(per_year)
    return {
        "total_trade_value": round(total, 3),
        "per_year": per_year,
        "n_years": n,
        "avg_trade_per_year": round(total / n, 3) if n > 0 else 0.0,
    }
