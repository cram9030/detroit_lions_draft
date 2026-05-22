# Baked Draft Data

This document describes the approach behind `scripts/baked_draft_data.py` and
how to run it.

---

## What it does

`baked_draft_data.py` generates one JSON file per draft year (2010 through
`--year`, default 2024) covering all 32 NFL franchises.  Each file contains:

- Per-player AV data for every pick in that year's class
- Class-level summary statistics (total AV, total surplus, etc.)
- The GM for each franchise at the time of the draft

The script distinguishes between two kinds of draft classes:

**Fully observed** — four complete seasons of AV data exist on disk (i.e. the
`draft{YEAR}_season{YEAR+3}.parquet` file is present).  For these classes, raw
observed AV is used as-is; no model is involved.

**Partially observed** — fewer than four seasons have been played.  For these
classes, missing seasons are projected using all three career trajectory models
(parametric, knn, ridge), and the output contains a separate block per model so
consumers can choose or blend them.

As of 2025, draft classes through 2021 are fully observed.  The 2022–2024
classes are partially observed.

---

## Output format

Files are written to `data/processed/baked/draft_{YEAR}.json` (or
`--output-dir`).

### Fully-observed class

```json
{
  "metadata": {
    "generated_at": "2025-05-22T00:00:00+00:00",
    "year": 2021,
    "models": ["parametric", "knn", "ridge"]
  },
  "teams": {
    "DET": {
      "gm": "Brad Holmes",
      "fully_observed": true,
      "players": [
        {
          "player": "Penei Sewell",
          "pos": "OT",
          "pick": 7,
          "obs_yr0": 8.0,
          "obs_yr1": 9.0,
          "obs_yr2": 11.0,
          "obs_yr3": 12.0,
          "total_4yr_av": 40.0,
          "total_4yr_av_above_replacement": 35.4,
          "eavar": 21.3,
          "eavar_upper": 27.1,
          "eavar_lower": 15.5,
          "replacement_level": 4.6,
          "surplus_av": 14.1
        }
      ],
      "class_summary": {
        "total_4yr_av": 95.0,
        "total_above_replacement": 72.4,
        "total_eavar": 60.2,
        "class_surplus": 12.2,
        "n_players": 7
      }
    }
  }
}
```

### Partially-observed class

```json
{
  "metadata": { "..." : "..." },
  "teams": {
    "DET": {
      "gm": "Brad Holmes",
      "fully_observed": false,
      "models": {
        "parametric": {
          "players": [
            {
              "player": "Terrion Arnold",
              "pos": "CB",
              "pick": 24,
              "obs_yr0": 10.0,
              "obs_yr1": 12.0,
              "obs_yr2": null,
              "obs_yr3": null,
              "proj_yr2": 8.2,
              "proj_yr3": 7.5,
              "is_projected": true,
              "total_4yr_av": 37.7,
              "total_4yr_av_above_replacement": 33.1,
              "eavar": 15.1,
              "eavar_upper": 19.4,
              "eavar_lower": 10.8,
              "replacement_level": 4.6,
              "surplus_av": 18.0
            }
          ],
          "class_summary": {
            "total_4yr_av": 184.3,
            "total_above_replacement": 138.3,
            "total_eavar": 148.6,
            "class_surplus": -10.3,
            "n_players": 10,
            "n_projected": 10
          }
        },
        "knn":   { "players": ["..."], "class_summary": { "..." : "..." } },
        "ridge": { "players": ["..."], "class_summary": { "..." : "..." } }
      }
    }
  }
}
```

### Per-player fields

| Field | Type | Present when |
|---|---|---|
| `player` | string | always |
| `pos` | string | always |
| `pick` | int | always |
| `obs_yr0` | float\|null | always |
| `obs_yr1` | float\|null | always |
| `obs_yr2` | float\|null | always |
| `obs_yr3` | float\|null | always |
| `proj_yr2` | float\|null | partially observed only |
| `proj_yr3` | float\|null | partially observed only |
| `is_projected` | bool | partially observed only |
| `total_4yr_av` | float\|null | always |
| `total_4yr_av_above_replacement` | float\|null | always |
| `eavar` | float\|null | always |
| `eavar_upper` | float\|null | always |
| `eavar_lower` | float\|null | always |
| `replacement_level` | float\|null | always |
| `surplus_av` | float\|null | always |

`is_projected` is `true` when at least one of the player's seasons was filled
by model projection rather than observed data.  A `null` numeric value means
the player had no data for that season (e.g. they were cut before the season).

---

## Prerequisites

1. **Stathead AV data** — at least two season files per draft year:

   ```
   data/raw/stathead/annual_av/draft{YEAR}_season{YEAR}.parquet
   data/raw/stathead/annual_av/draft{YEAR}_season{YEAR+1}.parquet
   ```

   Download via:
   ```bash
   python src/stathead_downloader.py --config config/stathead_annual_av.json
   ```

2. **PFR executives data** — one parquet per franchise:

   ```
   data/raw/pfr/executives/{team}_executives.parquet
   ```

   Download via:
   ```bash
   python src/pfr_downloader.py --config config/pfr_executives.json
   ```

3. **EAVAR table** — generated by the analysis pipeline:

   ```bash
   python scripts/run_analysis.py
   ```

4. **Trained models** — required for partially-observed classes (2022 and
   later):

   ```bash
   python scripts/train_models.py --model all
   ```

---

## Usage

```bash
# Bake all years 2010–2024 (default)
python scripts/baked_draft_data.py

# Stop at a different end year (still starts from 2010)
python scripts/baked_draft_data.py --year 2023

# Write to a custom output directory
python scripts/baked_draft_data.py --output-dir outputs/baked

# Use a custom models directory
python scripts/baked_draft_data.py --models-dir /path/to/models
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--year` | no | `2024` | Last draft year to include (inclusive) |
| `--models-dir` | no | `models/` | Path to trained model artifacts |
| `--output-dir` | no | `data/processed/baked/` | Directory to write JSON files |

---

## How observability is detected

A class is treated as fully observed when the year-3 season parquet file exists
on disk:

```
data/raw/stathead/annual_av/draft{YEAR}_season{YEAR+3}.parquet
```

The file's existence is the signal — not whether any given team's players
appear in it (a team with an unusually low-AV class can produce zero rows in
that file and still be correctly classified as fully observed).  This means
re-running the script after downloading new season data will automatically
upgrade classes from partially to fully observed.

---

## Position overrides

Stathead occasionally records players with generalist position codes (`"OL"`,
`"DL"`) that the career trajectory models do not recognise.  `baked_draft_data.py`
maintains a hardcoded `_POSITION_OVERRIDES` dict mapping player names to
specific position groups (e.g. `"Christian Mahogany" → "OG"`).

When a new draft class is added and any partially-observed players are missing
projections, check whether their position is one the models know.  If not, add
an entry to `_POSITION_OVERRIDES` near the top of the script.

---

## Team code mapping

PFR and Stathead use different franchise codes, and some franchises relocated
during the covered range.  `pfr_to_stathead()` handles the translation:

| Franchise | PFR code | Stathead code | Note |
|---|---|---|---|
| Raiders | `rai` | `OAK` / `LVR` | Split at 2020 |
| Rams | `ram` | `STL` / `LAR` | Split at 2016 |
| Chargers | `sdg` | `SDG` / `LAC` | Split at 2017 |
| All others | — | static mapping | See `_PFR_TO_STATHEAD_STATIC` |

---

## Relationship to `draft_class_surplus_av.py`

`draft_class_surplus_av.py` is designed for ad-hoc, single-team exploration
and produces interactive HTML charts.  `baked_draft_data.py` is the batch
counterpart — it loops over every team and year, skips charting, and writes
structured JSON intended for downstream consumption (e.g. a static website).

Both scripts share the same underlying surplus AV logic from `src/surplus_av.py`.

See [docs/surplus-av.md](surplus-av.md) for metric details and interpretation.
