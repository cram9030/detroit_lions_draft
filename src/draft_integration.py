"""Integration of season draft JSON data into the nflreadpy load_trades() format."""

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import nflreadpy
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]

_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "49ers": "SF", "Bears": "CHI", "Bengals": "CIN", "Bills": "BUF",
    "Broncos": "DEN", "Browns": "CLE", "Buccaneers": "TB", "Cardinals": "ARI",
    "Chargers": "LAC", "Chiefs": "KC", "Colts": "IND", "Cowboys": "DAL",
    "Dolphins": "MIA", "Eagles": "PHI", "Falcons": "ATL", "Giants": "NYG",
    "Jaguars": "JAX", "Jets": "NYJ", "Lions": "DET", "Packers": "GB",
    "Panthers": "CAR", "Patriots": "NE", "Raiders": "LV", "Rams": "LA",
    "Ravens": "BAL", "Saints": "NO", "Seahawks": "SEA", "Steelers": "PIT",
    "Texans": "HOU", "Titans": "TEN", "Vikings": "MIN", "Commanders": "WAS",
}
_TEAM_NAME_TO_ABBR_LOWER: dict[str, str] = {
    k.lower(): v for k, v in _TEAM_NAME_TO_ABBR.items()
}

_ORDINAL_TO_NUM: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7,
}

_PICK_REF_RE = re.compile(r"(\d{4})\s+(\w+)\s+round\s+pick\s+\(#(\d+)-(.+)\)")


def _team_to_abbr(name: str) -> str | None:
    if not name:
        return None
    return _TEAM_NAME_TO_ABBR_LOWER.get(name.lower().strip())


def _parse_pick_ref(item: str, year: int) -> tuple[int, int, str] | None:
    """Parse '2026 sixth round pick (#213-Jordan van den Berg)' -> (round, pick_num, player).

    Returns None for player-name items, picks from other years, or missing pick numbers.
    """
    cleaned = item.strip().replace("conditional ", "")
    m = _PICK_REF_RE.match(cleaned)
    if not m:
        return None
    if int(m.group(1)) != year:
        return None
    round_num = _ORDINAL_TO_NUM.get(m.group(2).lower())
    if round_num is None:
        return None
    return (round_num, int(m.group(3)), m.group(4).strip())


def _build_sent_lookup(
    draft_picks: list[dict],
    year: int,
) -> dict[tuple[str, str, int, str], list[tuple[int, str]]]:
    """Build lookup: (gave_abbr, received_abbr, round, trade_date_str) -> sorted [(pick_number, pfr_name)].

    Including the trade date disambiguates cases where the same team pair exchanged
    picks in the same round across multiple trades (e.g., JAX→DET in Apr and Aug).
    """
    raw: dict[tuple[str, str, int, str], set[tuple[int, str]]] = defaultdict(set)

    for pick in draft_picks:
        for trade in pick.get("trades", []):
            date_str = trade.get("date", "")
            if not date_str or date_str == "Various":
                continue
            from_abbr = _team_to_abbr(trade.get("from", ""))
            to_abbr = _team_to_abbr(trade.get("to", ""))
            if not from_abbr or not to_abbr:
                continue
            for key, items in trade.items():
                if not key.endswith("_sent"):
                    continue
                sent_by = _team_to_abbr(key[:-5])
                if not sent_by:
                    continue
                other = to_abbr if sent_by == from_abbr else from_abbr
                for item in items:
                    parsed = _parse_pick_ref(item, year)
                    if parsed:
                        rnd, pnum, player = parsed
                        raw[(sent_by, other, rnd, date_str)].add((pnum, player))

    return {k: sorted(v, key=lambda x: x[0]) for k, v in raw.items()}


def find_incomplete_trades(trades_df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Return rows where pick_season == year and pick_number is null."""
    return trades_df.filter(
        (pl.col("pick_season") == float(year)) & pl.col("pick_number").is_null()
    )


def populate_pick_numbers(
    trades_df: pl.DataFrame,
    draft_picks: list[dict],
    year: int,
) -> pl.DataFrame:
    """Fill null pick_number and pfr_name for year-picks using draft JSON data.

    Rows within the same (trade_id, gave, received, pick_round) group that are
    still null are assigned pick numbers from the lookup in ascending order.
    """
    lookup = _build_sent_lookup(draft_picks, year)
    rows = trades_df.to_dicts()

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("pick_season") == float(year) and row.get("pick_number") is None:
            key = (
                row["trade_id"],
                row["gave"],
                row["received"],
                int(row["pick_round"]),
                str(row["trade_date"]),
            )
            groups[key].append(i)

    updates: dict[int, tuple[float, str]] = {}
    for (_, gave, received, pick_round, trade_date_str), indices in groups.items():
        candidates = lookup.get((gave, received, pick_round, trade_date_str), [])
        for pos, idx in enumerate(sorted(indices)):
            if pos < len(candidates):
                pnum, player = candidates[pos]
                updates[idx] = (float(pnum), player)

    new_pick_numbers = [
        updates[i][0] if i in updates else row.get("pick_number")
        for i, row in enumerate(rows)
    ]
    new_pfr_names = [
        updates[i][1] if i in updates else row.get("pfr_name")
        for i, row in enumerate(rows)
    ]

    return trades_df.with_columns([
        pl.Series("pick_number", new_pick_numbers, dtype=pl.Float64),
        pl.Series("pfr_name", new_pfr_names, dtype=pl.String),
    ])


# ---------------------------------------------------------------------------
# New-trade helpers
# ---------------------------------------------------------------------------

def _build_pick_selector_lookup(draft_picks: list[dict]) -> dict[int, str | None]:
    return {
        p["pick"]: _team_to_abbr(p.get("selecting_team", ""))
        for p in draft_picks
        if "pick" in p
    }


def _collect_year_new_trades(
    draft_picks: list[dict],
    year: int,
) -> dict[tuple, dict[str, set[tuple[int, int, str]]]]:
    """Return unique year-dated trades not already representing prior-year entries.

    Uses the primary-pick convention check to correct inverted {team}_sent labels:
    if the primary pick appears in {team}_sent and that team IS the selecting_team,
    the entire trade's sent labels are inverted.

    Returns {(frozenset_teams, date_str): {gave_abbr: set((round, pick_num, player))}}
    """
    selector = _build_pick_selector_lookup(draft_picks)
    trade_map: dict = defaultdict(lambda: defaultdict(set))

    for pick_entry in draft_picks:
        primary_num = pick_entry.get("pick")
        primary_sel = _team_to_abbr(pick_entry.get("selecting_team", ""))

        for trade in pick_entry.get("trades", []):
            date_str = trade.get("date", "")
            if not date_str.startswith(str(year)) or date_str == "Various":
                continue
            from_abbr = _team_to_abbr(trade.get("from", ""))
            to_abbr = _team_to_abbr(trade.get("to", ""))
            if not from_abbr or not to_abbr:
                continue
            trade_key = (frozenset({from_abbr, to_abbr}), date_str)

            # Determine if convention is inverted by checking which {team}_sent
            # contains the primary pick and whether that team is the selecting_team.
            inverted = False
            if primary_num is not None and primary_sel is not None:
                for key, items in trade.items():
                    if not key.endswith("_sent"):
                        continue
                    sent_by = _team_to_abbr(key[:-5])
                    if sent_by not in {from_abbr, to_abbr}:
                        continue
                    for item in items:
                        p = _parse_pick_ref(item, year)
                        if p and p[1] == primary_num and sent_by == primary_sel:
                            inverted = True
                            break
                    if inverted:
                        break

            for key, items in trade.items():
                if not key.endswith("_sent"):
                    continue
                sent_by = _team_to_abbr(key[:-5])
                if not sent_by or sent_by not in {from_abbr, to_abbr}:
                    continue
                other = to_abbr if sent_by == from_abbr else from_abbr

                for item in items:
                    parsed = _parse_pick_ref(item, year)
                    if not parsed:
                        continue
                    rnd, pnum, player = parsed

                    if inverted:
                        actual_gave, actual_received = other, sent_by
                    else:
                        actual_gave, actual_received = sent_by, other

                    trade_map[trade_key][actual_gave].add((rnd, pnum, player))

    return trade_map


def add_new_trades(
    trades_df: pl.DataFrame,
    draft_picks: list[dict],
    year: int,
) -> pl.DataFrame:
    """Append new year-dated draft trade rows to trades_df.

    Assigns sequential trade_ids starting from max(trade_id) + 1.
    For picks already resolved (same year), pick_number and pfr_name are populated.
    """
    trade_map = _collect_year_new_trades(draft_picks, year)
    if not trade_map:
        return trades_df

    existing_year = trades_df.filter(pl.col("season") == year)
    new_rows: list[dict] = []
    next_id = int(trades_df["trade_id"].max()) + 1

    for (teams_frozen, date_str), sent_by_dict in sorted(
        trade_map.items(), key=lambda x: x[0][1]
    ):
        trade_date = date.fromisoformat(date_str)
        teams_list = list(teams_frozen)

        if len(existing_year) > 0:
            match = existing_year.filter(
                (pl.col("trade_date") == trade_date) &
                (pl.col("gave").is_in(teams_list) | pl.col("received").is_in(teams_list))
            )
            if len(match) > 0:
                continue

        rows_for_trade: list[dict] = []
        for gave_abbr, picks_given in sent_by_dict.items():
            rcv = next(iter(teams_frozen - {gave_abbr}))
            for rnd, pnum, player in sorted(picks_given, key=lambda x: x[1]):
                rows_for_trade.append({
                    "trade_id": float(next_id),
                    "season": year,
                    "trade_date": trade_date,
                    "gave": gave_abbr,
                    "received": rcv,
                    "pick_season": float(year),
                    "pick_round": float(rnd),
                    "pick_number": float(pnum),
                    "conditional": 0.0,
                    "pfr_name": player,
                })

        if rows_for_trade:
            new_rows.extend(rows_for_trade)
            next_id += 1

    if not new_rows:
        return trades_df

    new_df = pl.DataFrame({
        "trade_id":   pl.Series([r["trade_id"]   for r in new_rows], dtype=pl.Float64),
        "season":     pl.Series([r["season"]      for r in new_rows], dtype=pl.Int32),
        "trade_date": pl.Series([r["trade_date"]  for r in new_rows], dtype=pl.Date),
        "gave":       pl.Series([r["gave"]         for r in new_rows], dtype=pl.String),
        "received":   pl.Series([r["received"]     for r in new_rows], dtype=pl.String),
        "pick_season":pl.Series([r["pick_season"]  for r in new_rows], dtype=pl.Float64),
        "pick_round": pl.Series([r["pick_round"]   for r in new_rows], dtype=pl.Float64),
        "pick_number":pl.Series([r["pick_number"]  for r in new_rows], dtype=pl.Float64),
        "conditional":pl.Series([r["conditional"]  for r in new_rows], dtype=pl.Float64),
        "pfr_id":     pl.Series([None] * len(new_rows),                dtype=pl.String),
        "pfr_name":   pl.Series([r["pfr_name"]     for r in new_rows], dtype=pl.String),
    })
    return pl.concat([trades_df, new_df], how="vertical_relaxed")


# ---------------------------------------------------------------------------
# Patch apply / generate
# ---------------------------------------------------------------------------

def apply_trade_patch(trades_df: pl.DataFrame, patch: dict) -> pl.DataFrame:
    """Apply an extended trade patch dict to trades_df.

    Handles two sections:
    - patches: update existing rows (pick_number, optionally pfr_name) matched by
      (trade_id, gave, received, pick_season, pick_round).
    - new_trades: append new trade rows, assigning sequential IDs from max+1.
    """
    rows = trades_df.to_dicts()

    # Build index for fast lookup
    row_index: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("trade_id") is not None:
            key = (
                row["trade_id"],
                row["gave"],
                row["received"],
                row.get("pick_season"),
                row.get("pick_round"),
            )
            row_index[key].append(i)

    for p in patch.get("patches", []):
        key = (p["trade_id"], p["gave"], p["received"], p["pick_season"], p["pick_round"])
        for idx in row_index.get(key, []):
            rows[idx]["pick_number"] = float(p["pick_number"])
            if "pfr_name" in p and p["pfr_name"] is not None:
                rows[idx]["pfr_name"] = p["pfr_name"]

    next_id = int(trades_df["trade_id"].max()) + 1
    new_rows: list[dict] = []
    for trade in patch.get("new_trades", []):
        trade_date = date.fromisoformat(trade["trade_date"])
        season = int(trade["season"])
        for r in trade["rows"]:
            new_rows.append({
                "trade_id":    float(next_id),
                "season":      season,
                "trade_date":  trade_date,
                "gave":        r["gave"],
                "received":    r["received"],
                "pick_season": float(r["pick_season"]),
                "pick_round":  float(r["pick_round"]),
                "pick_number": float(r["pick_number"]) if r.get("pick_number") is not None else None,
                "conditional": 0.0,
                "pfr_id":      None,
                "pfr_name":    r.get("pfr_name"),
            })
        next_id += 1

    updated = pl.DataFrame(rows, schema=trades_df.schema)

    if not new_rows:
        return updated

    new_df = pl.DataFrame({
        "trade_id":    pl.Series([r["trade_id"]    for r in new_rows], dtype=pl.Float64),
        "season":      pl.Series([r["season"]       for r in new_rows], dtype=pl.Int32),
        "trade_date":  pl.Series([r["trade_date"]   for r in new_rows], dtype=pl.Date),
        "gave":        pl.Series([r["gave"]          for r in new_rows], dtype=pl.String),
        "received":    pl.Series([r["received"]      for r in new_rows], dtype=pl.String),
        "pick_season": pl.Series([r["pick_season"]   for r in new_rows], dtype=pl.Float64),
        "pick_round":  pl.Series([r["pick_round"]    for r in new_rows], dtype=pl.Float64),
        "pick_number": pl.Series([r["pick_number"]   for r in new_rows], dtype=pl.Float64),
        "conditional": pl.Series([r["conditional"]   for r in new_rows], dtype=pl.Float64),
        "pfr_id":      pl.Series([None] * len(new_rows),                 dtype=pl.String),
        "pfr_name":    pl.Series([r["pfr_name"]      for r in new_rows], dtype=pl.String),
    })
    return pl.concat([updated, new_df], how="vertical_relaxed")


def generate_trade_patch(
    draft_json_path: str | Path,
    year: int,
    trades_df: pl.DataFrame | None = None,
) -> dict:
    """Generate an extended trade patch dict from a draft JSON file.

    The returned dict has keys: metadata, patches, new_trades, warnings.
    - patches: unambiguous pick_number updates for pre-existing nflreadpy rows.
    - new_trades: draft-day trades not present in nflreadpy.
    - warnings: cases skipped due to ambiguous team/round count mismatches.
    """
    import datetime

    with open(draft_json_path) as f:
        draft_picks: list[dict] = json.load(f)

    if trades_df is None:
        trades_df = nflreadpy.load_trades()

    lookup = _build_sent_lookup(draft_picks, year)
    incomplete = find_incomplete_trades(trades_df, year)

    patches: list[dict] = []
    warnings: list[dict] = []

    # Group incomplete rows by (trade_id, gave, received, pick_round)
    groups: dict[tuple, list] = defaultdict(list)
    for row in incomplete.iter_rows(named=True):
        key = (row["trade_id"], row["gave"], row["received"], int(row["pick_round"]), str(row["trade_date"]))
        groups[key].append(row)

    for (trade_id, gave, received, pick_round, trade_date_str), group_rows in groups.items():
        candidates = lookup.get((gave, received, pick_round, trade_date_str), [])
        if len(candidates) == 0:
            warnings.append({
                "trade_id": trade_id, "gave": gave, "received": received,
                "pick_season": float(year), "pick_round": float(pick_round),
                "message": "no matching pick found in draft JSON",
            })
            continue
        if len(candidates) != len(group_rows):
            warnings.append({
                "trade_id": trade_id, "gave": gave, "received": received,
                "pick_season": float(year), "pick_round": float(pick_round),
                "nflreadr_count": len(group_rows),
                "draft_order_count": len(candidates),
                "message": "ambiguous match: counts differ, skipped",
            })
            continue
        for row, (pnum, player) in zip(group_rows, candidates):
            patches.append({
                "trade_id":   row["trade_id"],
                "gave":       gave,
                "received":   received,
                "pick_season": float(year),
                "pick_round":  float(pick_round),
                "pick_number": float(pnum),
                "pfr_name":    player,
            })

    # New trades
    trade_map = _collect_year_new_trades(draft_picks, year)
    existing_year = trades_df.filter(pl.col("season") == year)
    new_trades: list[dict] = []

    for (teams_frozen, date_str), sent_by_dict in sorted(
        trade_map.items(), key=lambda x: x[0][1]
    ):
        trade_date_obj = date.fromisoformat(date_str)
        teams_list = list(teams_frozen)

        if len(existing_year) > 0:
            match = existing_year.filter(
                (pl.col("trade_date") == trade_date_obj) &
                (pl.col("gave").is_in(teams_list) | pl.col("received").is_in(teams_list))
            )
            if len(match) > 0:
                continue

        trade_rows: list[dict] = []
        for gave_abbr, picks_given in sent_by_dict.items():
            rcv = next(iter(teams_frozen - {gave_abbr}))
            for rnd, pnum, player in sorted(picks_given, key=lambda x: x[1]):
                trade_rows.append({
                    "gave": gave_abbr, "received": rcv,
                    "pick_season": float(year), "pick_round": float(rnd),
                    "pick_number": float(pnum), "pfr_name": player,
                })

        if trade_rows:
            new_trades.append({
                "trade_date": date_str,
                "season": year,
                "rows": trade_rows,
            })

    return {
        "metadata": {"year": year, "created": str(datetime.date.today())},
        "patches": patches,
        "new_trades": new_trades,
        "warnings": warnings,
    }
