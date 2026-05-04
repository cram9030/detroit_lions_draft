"""Trade chart loading and pick combination finder for draft trade evaluation."""

from pathlib import Path
from typing import TypedDict

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
