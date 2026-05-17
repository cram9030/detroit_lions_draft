# Draft Trade Analysis

`analyze_draft_trades(team, year)` in `src/trade_value.py` fetches all NFL trades involving a team in a given draft year, filters to draft-day pick trades, and evaluates each trade across five trade charts.

---

## Usage

```python
from src.trade_value import analyze_draft_trades

# Evaluate all 2021 draft trades involving the Philadelphia Eagles
df = analyze_draft_trades("PHI", 2021)
print(df)
```

The function uses `nflreadpy.load_trades()` and `nflreadpy.load_players()` at call time — no local data download is required.

---

## Output schema

Each row is one trade. All `*_picks` columns are comma-separated pick numbers sorted ascending.

| Column | Type | Description |
|---|---|---|
| `trade_id` | Int64 | Trade identifier from nflreadpy |
| `team_traded_with` | String | Other team(s) in the trade, comma-separated |
| `picks_received` | String | Overall pick numbers the team received |
| `picks_gave` | String | Overall pick numbers the team gave away |
| `fitz_spiel_value` | Float64 | Net value per Fitzgerald-Spielberger chart |
| `fitz_spiel_picks` | String | Picks equivalent to `abs(net_value)` |
| `jj_value` | Float64 | Net value per Jimmy Johnson chart |
| `jj_picks` | String | Equivalent picks |
| `pff_value` | Float64 | Net value per PFF WAR chart |
| `pff_picks` | String | Equivalent picks |
| `rich_hill_value` | Float64 | Net value per Rich Hill chart |
| `rich_hill_picks` | String | Equivalent picks |
| `eaar_value` | Float64 | Net value per Expected AV Above Replacement chart |
| `eaar_picks` | String | Equivalent picks |

`*_value` is `received_total − gave_total` from the team's perspective. A positive value means the team received more chart value than it gave; negative means the opposite.

`*_picks` is the output of `find_pick_combination(abs(net_value), chart_name)` — the combination of picks whose total chart value best approximates the absolute net. This is always computed when `net_value != 0`; it is an empty string when `net_value == 0`.

---

## Supported trade charts

| Column prefix | Chart | Description |
|---|---|---|
| `fitz_spiel` | Fitzgerald-Spielberger | Modern empirical chart based on actual trade data |
| `jj` | Jimmy Johnson | Classic chart; highest absolute values |
| `pff` | PFF WAR | Normalized 0–1 scale based on PFF WAR |
| `rich_hill` | Rich Hill | Alternative empirical chart |
| `eaar` | Expected AV Above Replacement | Project-internal chart from `expected_av_above_replacement.csv` |

All charts are loaded via `load_trade_chart(chart_name)` from `src/trade_value.py`.

---

## Filtering rules

A trade is **excluded** from the output in exactly two cases:

1. **Wrong draft year player** — a non-null `pfr_id` in the trade data belongs to a player whose `draft_year` (from `nflreadpy.load_players()`) does not equal `year`. Rows with null `pfr_id` (pure pick rows) are always valid and never trigger this exclusion.

2. **No picks exchanged** — after pick number resolution, neither side of the trade has any pick numbers.

---

## Pick number estimation

The nflreadpy trades data sometimes records only a round number without a specific pick number (e.g., a conditional pick). When `pick_number` is null but `pick_round` is not, the pick number is estimated as the mid-point of that round:

```
estimated_pick = (pick_round - 1) * 32 + 16
```

| Round | Estimated pick |
|---|---|
| 1 | 16 |
| 2 | 48 |
| 3 | 80 |
| 4 | 112 |
| 5 | 144 |
| 6 | 176 |
| 7 | 208 |

---

## Example: 2021 Eagles-Cowboys

On April 29, 2021 (draft day), the Eagles (PHI) traded picks 12 and 84 to the Cowboys (DAL) in exchange for pick 10.

```python
df = analyze_draft_trades("PHI", 2021)
# trade_id            1558
# team_traded_with    DAL
# picks_received      10
# picks_gave          12,84
# jj_value            -70.0   (1300 received − 1370 gave)
# jj_picks            <pick equivalent to 70 JJ points>
```

The negative `jj_value` indicates PHI gave more Jimmy Johnson value than it received. The Cowboys gained +70 JJ points on paper, acquiring Micah Parsons (pick 12) and Chauncey Golston (pick 84) while trading DeVonta Smith's slot (pick 10).

---

## Using with current-year data

nflreadpy's `load_trades()` has two gaps for the current draft year:

1. Prior-year trades with future picks have `pick_number = null`.
2. Draft-day trades from the current year are absent entirely.

Use `src/draft_integration.py` to fill both before calling `analyze_draft_trades`:

```python
import json
import nflreadpy
from src.draft_integration import populate_pick_numbers, add_new_trades
from src.trade_value import analyze_draft_trades

with open("data/processed/nfl_draft_2026.json") as f:
    draft_picks = json.load(f)

trades = nflreadpy.load_trades()
trades = populate_pick_numbers(trades, draft_picks, 2026)
trades = add_new_trades(trades, draft_picks, 2026)

# analyze_draft_trades calls nflreadpy internally, so pass the enriched df directly
# by patching or using the lower-level functions — see docs/draft-integration.md
```

See [docs/draft-integration.md](docs/draft-integration.md) for the full workflow.

---

## Related functions

- `load_trade_chart(chart_name)` — load any of the six charts as a normalized `[Pick, Value]` DataFrame
- `find_pick_combination(target, chart_name)` — find the set of picks whose chart values sum closest to a target
- `src.draft_integration.populate_pick_numbers` — fill null pick_numbers from a draft JSON
- `src.draft_integration.add_new_trades` — append draft-day trades not in nflreadpy
- `src.draft_integration.apply_trade_patch` — apply a pre-generated patch file
