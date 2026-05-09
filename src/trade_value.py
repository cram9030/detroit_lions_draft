"""Trade chart loading and pick combination finder for draft trade evaluation."""

from pathlib import Path
from typing import TypedDict

import nflreadpy
import polars as pl

from src.data_ingest import load_csv

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
    """Load a trade chart CSV and return a normalised ``[Pick, Value]`` DataFrame.

    All six supported charts are coerced to ``Pick`` (Int64) and ``Value``
    (Float64), sorted ascending by ``Pick``.

    Args:
        chart_name: One of ``"jimmy_johnson"``, ``"fitzgerald_spielberger"``,
            ``"pff_war"``, ``"rich_hill"``, ``"eavar"``, ``"5_year_av"``.
        data_dir: Directory containing the processed CSV files.

    Returns:
        DataFrame with columns ``["Pick", "Value"]``, sorted by ``Pick``.

    Raises:
        ValueError: If ``chart_name`` is not in the registry.
        FileNotFoundError: If the CSV file does not exist.
    """
    if chart_name not in _CHART_REGISTRY:
        valid = ", ".join(sorted(_CHART_REGISTRY))
        raise ValueError(
            f"Unknown chart '{chart_name}'. Valid names: {valid}"
        )

    filename, pick_col, value_col, dedup = _CHART_REGISTRY[chart_name]
    path = Path(data_dir) / filename

    df = load_csv(path)

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
    tail. Stops early when four criteria are met (see inline comments).

    ``pick_values`` must be sorted by value ascending.

    Returns:
        (best_picks, best_values, best_total) — the combination whose sum
        minimises ``|sum - target|``.
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
        """Run two-pointer over pick_values[lo..hi] with fixed outer indices."""
        fixed_sum = sum(pick_values[i][1] for i in fixed)
        remaining = target - fixed_sum
        left, right = lo, hi
        while left < right:
            s = pick_values[left][1] + pick_values[right][1]
            _update_best(fixed + [left, right])
            if abs(s - remaining) < 1e-12:  # exact pair match
                return
            if s < remaining:
                left += 1
            else:
                right -= 1

    def _recurse(fixed: list[int], start: int, depth_remaining: int) -> None:
        """Fix one more element then either two-pointer or recurse deeper."""
        if depth_remaining == 2:
            _two_pointer_sweep(fixed, start, n - 1)
            return
        for i in range(start, n - depth_remaining + 1):
            _update_best(fixed + [i])
            if depth_remaining > 1:
                _recurse(fixed + [i], i + 1, depth_remaining - 1)
            # Pruning: if fixing this element alone already exceeds the target
            # by more than the current best error, no smaller-index element
            # paired with later (larger-value) picks can improve — but since
            # we iterate ascending and the remaining picks are >= current, we
            # can stop early if the fixed sum already overshoots.
            fixed_so_far = sum(pick_values[j][1] for j in fixed) + pick_values[i][1]
            if fixed_so_far > target + best_error:
                break

    # k=1: linear scan
    for i in range(n):
        _update_best([i])
        # Once the single-pick value exceeds target + current best, stop
        if pick_values[i][1] > target + best_error:
            break

    prev_error = best_error

    for k in range(2, max_picks + 1):
        # Stopping criterion 1: exact or within tolerance
        if best_error <= tolerance:
            break
        # Stopping criterion 2: gap smaller than the smallest available pick
        if best_error < min_val:
            break
        # Stopping criterion 3: no improvement from adding another pick
        if k > 2 and best_error >= prev_error:
            break

        prev_error = best_error

        if k == 2:
            _two_pointer_sweep([], 0, n - 1)
        else:
            # Fix k-2 outer elements, then two-pointer
            _recurse([], 0, k)

    return best_picks, best_values, best_total


def find_pick_combination(
    target_value: float,
    chart_name: str,
    max_picks: int = 5,
    tolerance: float = 0.0,
    data_dir: Path | str = _PROCESSED_DATA_DIR,
) -> PickCombinationResult:
    """Find the combination of picks whose chart values sum closest to ``target_value``.

    Uses an extended two-pointer algorithm that searches combinations of
    k=1..max_picks picks, stopping early when the approximation error falls
    within ``tolerance`` or cannot be improved further.

    Picks are treated as distinct — no pick number appears more than once in a
    result.

    Args:
        target_value: The value to approximate (must be within the chart's
            range, i.e. between the value of pick 1 and the last pick).
        chart_name: Trade chart to use (see ``load_trade_chart``).
        max_picks: Maximum number of picks allowed in the returned combination.
            Default 5.
        tolerance: Stop searching once ``|total - target_value| <= tolerance``.
            Default 0.0 (find the best possible match).
        data_dir: Directory containing the processed CSV files.

    Returns:
        ``PickCombinationResult`` with the best combination found.

    Raises:
        ValueError: If ``chart_name`` is unknown.
        FileNotFoundError: If the chart CSV does not exist.
    """
    df = load_trade_chart(chart_name, data_dir=data_dir)

    # Sort ascending by value so two-pointer invariants hold
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
    """Return a zero-row DataFrame with the analyze_draft_trades schema."""
    schema: dict[str, pl.PolarsDataType] = {
        "trade_id": pl.Int64,
        "team_traded_with": pl.String,
        "picks_received": pl.String,
        "picks_gave": pl.String,
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
    """Return (net_value, equivalent_picks_str) for one trade chart.

    net_value = sum of received pick values - sum of gave pick values.
    equivalent_picks_str = find_pick_combination(abs(net_value)) when net != 0,
    empty string when net == 0.
    """
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
) -> pl.DataFrame:
    """Return a DataFrame of draft trades involving team in the given year.

    Each row represents one trade. Trades are excluded when:
    - Any player asset (non-null pfr_id) has a draft_year ≠ year.
    - After pick number resolution, no picks were exchanged on either side.

    Rows with null pfr_id are pure pick rows and never trigger exclusion.
    When only pick_round is known (pick_number is null), pick number is
    estimated as (round - 1) * 32 + 16.

    Args:
        team: 3-letter NFL team abbreviation (e.g. "PHI", "DAL").
        year: Draft year to filter by.
        data_dir: Directory containing trade chart CSVs.

    Returns:
        DataFrame with columns: trade_id, team_traded_with, picks_received,
        picks_gave, and for each of 5 trade charts a {prefix}_value (Float64)
        and {prefix}_picks (String) column. picks_* columns are comma-separated
        pick numbers sorted ascending. {prefix}_picks is the combination from
        find_pick_combination(abs(net_value)), or "" when net_value == 0.
    """
    all_trades = nflreadpy.load_trades()
    team_trades = all_trades.filter(
        (pl.col("season") == year)
        & ((pl.col("gave") == team) | (pl.col("received") == team))
    )

    if len(team_trades) == 0:
        return _empty_trade_df()

    all_players = nflreadpy.load_players()
    pfr_to_draft_year: dict[str, int | None] = {
        pfr_id: dy
        for pfr_id, dy in zip(
            all_players["pfr_id"].to_list(),
            all_players["draft_year"].to_list(),
        )
        if pfr_id is not None
    }

    output_rows: list[dict] = []

    for tid in team_trades["trade_id"].unique().to_list():
        trade_rows = team_trades.filter(pl.col("trade_id") == tid)

        # Exclusion rule 1: player drafted in a different year
        player_rows = trade_rows.filter(pl.col("pfr_id").is_not_null())
        if any(
            pfr_to_draft_year.get(pfr_id) != year
            for pfr_id in player_rows["pfr_id"].to_list()
        ):
            continue

        # Resolve pick numbers for each row
        rcv_picks: list[int] = []
        gave_picks: list[int] = []
        for row in trade_rows.iter_rows(named=True):
            resolved = _resolve_pick_number(row["pick_number"], row["pick_round"])
            if resolved is None:
                continue
            if row["received"] == team:
                rcv_picks.append(resolved)
            if row["gave"] == team:
                gave_picks.append(resolved)

        # Exclusion rule 2: no picks exchanged
        if not rcv_picks and not gave_picks:
            continue

        rcv_picks.sort()
        gave_picks.sort()

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
            "picks_received": ",".join(str(p) for p in rcv_picks),
            "picks_gave": ",".join(str(p) for p in gave_picks),
        }

        for prefix, chart_name in _TRADE_CHARTS:
            val, equiv = _compute_chart_value(rcv_picks, gave_picks, chart_name, data_dir)
            row_dict[f"{prefix}_value"] = val
            row_dict[f"{prefix}_picks"] = equiv

        output_rows.append(row_dict)

    if not output_rows:
        return _empty_trade_df()

    return pl.DataFrame(output_rows)
