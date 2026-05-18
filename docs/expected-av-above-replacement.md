# Expected AV Above Replacement (EAVAR)

## What is EAVAR?

**Expected AV Above Replacement (EAVAR)** measures how much more value a drafted player is expected to produce over their rookie contract than a freely available replacement-level player. It answers the question: *how valuable is this draft pick relative to the baseline of what any team can acquire for near-nothing?*

A pick with EAVAR = 20 means the player selected there is expected to contribute 20 more Approximate Value units over their rookie contract window than a replacement-level player would.

## Formula

```
EAVAR(pick) = f(pick) − replacement_level

where:
  f(pick) = a · exp(−b · pick) + c      (exponential decay fit)
  replacement_level = mean_season_AV(replacement players) × 4
```

The replacement level is scaled by 4 to match the 4-year rookie contract window used in the exponential fit.

## Data sources

| Data | Source | Variable |
|---|---|---|
| Rookie contract AV | Stathead season AV (`AV.1`), summed over draft years 0–3 | `rookie_contract_av` |
| Replacement-level contracts | Over-the-Cap 1-year contracts | `AV.1` for matched player-seasons |

## Replacement level methodology

A **replacement-level player** is defined as one who signed a 1-year contract valued at or below **120% of the NFL minimum salary** for their experience tier (the default; configurable via `--replacement-percent`).

The replacement level baseline is:
1. Identify all such contracts (using `src/contracts.py`)
2. Join to Stathead season AV data to get the `AV.1` for each matched player-season
3. Take the **mean** `AV.1` across all matched player-seasons
4. Multiply by **4** to convert to the 4-year rookie contract scale

## How to run

```bash
# Full run (computes CSV + both HTML figures)
python scripts/expected_av_above_replacement.py

# Use a stricter replacement-level threshold (110% of minimum)
python scripts/expected_av_above_replacement.py --replacement-percent 1.10

# Skip figure generation (CSV only)
python scripts/expected_av_above_replacement.py --skip-plots
```

## Outputs

| File | Description |
|---|---|
| `data/processed/expected_av_above_replacement.csv` | Per-pick EAVAR with 1σ confidence bounds and the replacement level scalar used |
| `outputs/figures/expected_av_above_replacement.html` | EAVAR curve + confidence band + per-player scatter + replacement level baseline |
| `outputs/figures/eavar_vs_trade_charts.html` | EAVAR normalized to pick 1 = 1.0, overlaid against Jimmy Johnson, Fitzgerald-Spielberger, PFF WAR, 5-Year AV, and Rich Hill trade charts |

### CSV columns

| Column | Type | Description |
|---|---|---|
| `pick` | Int64 | Overall draft pick number (1–250) |
| `eavar` | Float64 | Expected AV above replacement at this pick |
| `eavar_upper` | Float64 | Upper 1σ confidence bound |
| `eavar_lower` | Float64 | Lower 1σ confidence bound |
| `replacement_level` | Float64 | The replacement level scalar subtracted (same for all rows; 4-year total) |

Draft picks beyond pick 250 are not in the table.  `compute_surplus_av` in
`src/surplus_av.py` handles this by capping such picks at 250 before joining,
so late-round selections (e.g. pick 257) receive the pick-250 EAVAR as a
conservative floor rather than a null value.

## Comparison interpretation

The `eavar_vs_trade_charts.html` figure normalizes all series to pick 1 = 1.0. This lets you compare the *shape* of the EAVAR curve — how pick value drops off — against established trade charts. Key observations to look for:

- **Steeper curves** (Jimmy Johnson, Rich Hill) imply the top picks are worth disproportionately more than later picks.
- **Flatter curves** (EAVAR, Fitzgerald-Spielberger) suggest later picks retain more relative value after accounting for replacement-level availability.
- **EAVAR's curve** may be flatter than raw AV curves because subtracting a fixed constant compresses the ratio between early and late picks.
