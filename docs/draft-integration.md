# Draft Integration

`src/draft_integration.py` bridges the gap between a season-specific draft JSON (e.g. `data/processed/nfl_draft_2026.json`) and the nflreadpy `load_trades()` Polars DataFrame. It solves two problems:

1. **Incomplete prior-year trades** — nflreadpy records trades made before the draft with `pick_number = null` for future-year picks. This module fills in those numbers and the player ultimately drafted with each pick.

2. **Missing draft-day trades** — trades made on draft day itself often don't appear in nflreadpy at all. This module constructs those rows from the draft JSON.

---

## Data source

The draft JSON is an array of pick objects. Each pick includes `pick`, `round`, `original_team`, `selecting_team`, `player`, and a `trades` list. Each trade entry has `date`, `from`, `to`, and `{team_name}_sent` lists describing what each side gave up.

Example pick entry (abbreviated):

```json
{
  "pick": 81,
  "round": 3,
  "original_team": "Lions",
  "selecting_team": "Jaguars",
  "player": "Albert Regis",
  "trades": [
    {
      "date": "2025-04-25",
      "from": "Lions",
      "to": "Jaguars",
      "jaguars_sent": ["2025 third round pick (#70-Isaac TeSlaa)", "2026 sixth round pick (#213-Jordan van den Berg)"],
      "lions_sent": ["2025 third round pick (#102-Tai Felton)", "2026 third round pick (#81-Albert Regis)", "2026 third round pick (#100-Jalen Huskey)"]
    }
  ]
}
```

---

## Functions

### `find_incomplete_trades(trades_df, year)`

Returns rows from `trades_df` where `pick_season == year` and `pick_number` is null — i.e. future picks that haven't been resolved yet.

```python
import nflreadpy
from src.draft_integration import find_incomplete_trades

trades = nflreadpy.load_trades()
incomplete = find_incomplete_trades(trades, 2026)
print(f"{len(incomplete)} unresolved 2026 pick rows")
```

---

### `populate_pick_numbers(trades_df, draft_picks, year)`

Fills in `pick_number` and `pfr_name` for null rows matched against the draft JSON. Rows are matched on `(gave, received, pick_round, trade_date)`. When multiple null rows exist for the same group (e.g. two 3rd-round picks DET→JAX), they are assigned pick numbers in ascending order.

`pfr_id` is left null since PFR IDs for newly-drafted players are not available in the JSON.

```python
import json
import nflreadpy
from src.draft_integration import populate_pick_numbers

with open("data/processed/nfl_draft_2026.json") as f:
    draft_picks = json.load(f)

trades = nflreadpy.load_trades()
trades_filled = populate_pick_numbers(trades, draft_picks, 2026)
```

---

### `add_new_trades(trades_df, draft_picks, year)`

Appends draft-day trades from the JSON that are not already present in `trades_df`. Assigns new `trade_id` values sequentially from `max(trade_id) + 1`.

A trade is skipped if `trades_df` already contains any row with the same `(trade_date, gave/received teams)` for the given year. Only trades dated within `year` are considered (pre-draft trades with future picks are handled by `populate_pick_numbers` instead).

`pfr_id` is null for all new rows. `pick_number` and `pfr_name` are populated for picks already resolved at draft time.

```python
from src.draft_integration import add_new_trades

trades_extended = add_new_trades(trades, draft_picks, 2026)
print(f"Now {len(trades_extended)} total trade rows (was {len(trades)})")
```

---

### `apply_trade_patch(trades_df, patch)`

Applies a pre-generated patch dict to `trades_df`. The patch format has three sections:

| Key | Effect |
|---|---|
| `patches` | Update `pick_number` (and optionally `pfr_name`) on existing rows, matched by `(trade_id, gave, received, pick_season, pick_round)` |
| `new_trades` | Append entirely new trade rows; sequential `trade_id` values are assigned from `max + 1` |
| `warnings` | Informational only — logged but not applied |

```python
import json
import nflreadpy
from src.draft_integration import apply_trade_patch

with open("data/processed/trade_patch_2026.json") as f:
    patch = json.load(f)

trades = nflreadpy.load_trades()
trades_patched = apply_trade_patch(trades, patch)
```

#### Patch JSON format

```json
{
  "metadata": {"year": 2026, "created": "2026-05-17"},
  "patches": [
    {
      "trade_id": 1929.0,
      "gave": "JAX",
      "received": "DET",
      "pick_season": 2026.0,
      "pick_round": 6.0,
      "pick_number": 213.0,
      "pfr_name": "Jordan van den Berg"
    }
  ],
  "new_trades": [
    {
      "trade_date": "2026-04-25",
      "season": 2026,
      "rows": [
        {"gave": "BUF", "received": "DET", "pick_season": 2026.0, "pick_round": 5.0, "pick_number": 168.0, "pfr_name": "Kendrick Law"},
        {"gave": "DET", "received": "BUF", "pick_season": 2026.0, "pick_round": 5.0, "pick_number": 181.0, "pfr_name": "Zane Durant"},
        {"gave": "DET", "received": "BUF", "pick_season": 2026.0, "pick_round": 6.0, "pick_number": 213.0, "pfr_name": "Jordan van den Berg"}
      ]
    }
  ],
  "warnings": []
}
```

---

### `generate_trade_patch(draft_json_path, year, trades_df=None)`

Generates the full patch dict automatically by comparing the draft JSON against nflreadpy (or a supplied `trades_df`). Returns a dict with the same structure as the patch JSON above.

- **patches** — unambiguous matches only (count of null rows equals count of candidates from JSON)
- **new_trades** — draft-day trades not present in nflreadpy
- **warnings** — skipped entries where the JSON and nflreadpy row counts differ

```python
from src.draft_integration import generate_trade_patch

patch = generate_trade_patch("data/processed/nfl_draft_2026.json", 2026)
print(f"{len(patch['patches'])} patches, {len(patch['new_trades'])} new trades, {len(patch['warnings'])} warnings")
```

---

## CLI: `scripts/generate_trade_patch.py`

Generates a patch JSON file from a draft order JSON and writes it to disk.

```bash
# Default output: data/processed/trade_patch_2026.json
python scripts/generate_trade_patch.py data/processed/nfl_draft_2026.json 2026

# Custom output path
python scripts/generate_trade_patch.py data/processed/nfl_draft_2026.json 2026 \
    --output data/processed/trade_patch_2026_v2.json
```

The script prints a summary on completion:

```
Wrote 19 patches, 54 new trades, 50 warnings → data/processed/trade_patch_2026.json
```

---

## Typical workflow

```python
import json
import nflreadpy
from src.draft_integration import (
    find_incomplete_trades,
    populate_pick_numbers,
    add_new_trades,
)

with open("data/processed/nfl_draft_2026.json") as f:
    draft_picks = json.load(f)

trades = nflreadpy.load_trades()

# 1. Fill in null pick numbers from prior-year trades
trades = populate_pick_numbers(trades, draft_picks, 2026)

# 2. Append draft-day trades not in nflreadpy
trades = add_new_trades(trades, draft_picks, 2026)

# Verify no 2026 nulls remain in the resolved set
still_null = find_incomplete_trades(trades, 2026)
print(f"{len(still_null)} rows still unresolved (expected: warnings count)")
```

---

## Convention notes

- **Team names** — the draft JSON uses full names ("Lions", "Jaguars"); all functions convert these to nflreadpy 3-letter abbreviations ("DET", "JAX") via an internal mapping.
- **pfr_name convention** — `pfr_name` is the player ultimately drafted with the pick, regardless of which team holds the row. For example, the DET→BUF row for pick 213 has `pfr_name = "Jordan van den Berg"` even though the Bears (via a subsequent trade) made the actual selection.
- **Inverted `{team}_sent` labels** — some JSON entries have inconsistent sent labels (where the selecting team appears to have sent its own pick). The module detects and corrects these inversions automatically using the primary-pick convention check.
- **Trade date disambiguation** — the same team pair can exchange picks in the same round across multiple trades. The lookup is keyed on `(gave, received, round, trade_date)` to prevent cross-assignment.
