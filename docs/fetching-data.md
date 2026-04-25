# Fetching Data

## Prerequisites

- Python 3.10 or later
- A valid Stathead subscription (logged in via your browser)
- The **Cookie-Editor** browser extension
  ([Chrome](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm) |
  [Firefox](https://addons.mozilla.org/en-US/firefox/addon/cookie-editor/))

---

## Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Export your browser session cookies

The script authenticates as you by replaying your browser's login cookies.
You need to export them once and re-export if they expire.

1. Go to **https://www.sports-reference.com/stathead/** and confirm you are logged in.
2. Click the **Cookie-Editor** extension icon.
3. Click **Export → Export All** (copies to clipboard or writes a file directly).
4. Paste the contents into **`secrets/cookies.json`** (create the file if needed).

The file should be a JSON array: `[{"name": "...", "value": "..."}, ...]`

> **Cookie lifetime:** Stathead session cookies typically last a few days.
> If the script logs `LOGIN WALL DETECTED`, your cookies have expired — repeat this step.

> **Security:** `secrets/` is gitignored and will never be committed. Do not move
> `cookies.json` outside that folder.

---

## Step 3 — Configure the query

Edit **`config/stathead_annual_av.json`** (or copy it to create a new config for a
different query type). The fields you are most likely to change:

| Field | What it controls | Default |
|---|---|---|
| `output_dir` | Where Parquet files are saved | `data/raw/stathead/annual_av` |
| `draft_year_ranges` | List of `[min, max]` draft year pairs | `[[2021, 2021]]` |
| `season_years` | Season years to query | `[2021, 2022, 2023, 2024, 2025]` |
| `sleep_between_requests` | Seconds to wait between requests | `3.0` |

### Typical configuration examples

**Query each draft class against multiple seasons:**
```json
"draft_year_ranges": [[2018, 2018], [2019, 2019], [2020, 2020], [2021, 2021], [2022, 2022]],
"season_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024]
```
This produces 5 draft ranges × 7 season years = **35 combinations**.

**Query a single wide draft window:**
```json
"draft_year_ranges": [[2018, 2022]],
"season_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024]
```
This produces 1 range × 7 seasons = **7 combinations**.

The `fixed_params` block mirrors Stathead URL parameters that never change
(`order_by=av`, `comp_type=reg`, etc.). Edit only for a fundamentally different query type.

---

## Step 4 — Run the script

```bash
python src/stathead_downloader.py
```

Override defaults with flags:

```bash
python src/stathead_downloader.py \
  --config config/stathead_annual_av.json \
  --cookies secrets/cookies.json
```

The script will:

1. Load cookies and the query config.
2. Iterate over every draft-range × season-year combination.
3. For each combination, fetch all paginated pages (200 rows each).
4. Parse the HTML table and write a single **Parquet** file per combination.
5. Log progress to the terminal and to `stathead_downloader.log`.
6. Record completed combinations in `.progress.json` so interrupted runs
   can be safely restarted without re-downloading anything.

---

## Output structure

```
data/raw/stathead/annual_av/
├── draft2021_season2021.parquet
├── draft2021_season2022.parquet
├── draft2021_season2023.parquet
└── .progress.json              ← tracks completed combinations for resumability
```

Each Parquet file contains all pages for that draft × season combination, with
repeating Stathead header rows stripped out.

---

## Loading the data

```python
import polars as pl

# Load a single file
df = pl.read_parquet("data/raw/stathead/annual_av/draft2021_season2021.parquet")

# Load all files at once
df = pl.scan_parquet("data/raw/stathead/annual_av/*.parquet").collect()
```

---

## Resuming an interrupted run

Simply re-run the script. Completed combinations are recorded in `.progress.json`
and skipped automatically — only missing combinations are fetched.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `LOGIN WALL DETECTED` | Cookies expired | Re-export cookies (Step 2) |
| `Cookie file not found` | Wrong path | Confirm `secrets/cookies.json` exists |
| `No data for this combination` | Draft/season combo has 0 results | Normal — script skips and continues |
| `HTTP 429` | Too many requests | Increase `sleep_between_requests` to `5.0` or higher |
| Wrong table parsed | Stathead updated their HTML | Inspect the table `id` in DevTools and update `parse_table()` |
| Script is slow | Intentional rate limiting | Do not reduce `sleep_between_requests` below `2.0` |

---

## Rate limiting and terms of service

The default 3-second delay between requests is intentional. Stathead is a
paid service and aggressive scraping can get your account flagged. Do not
reduce the delay below **2 seconds**. This tool is intended for personal
automation of data you are entitled to access as a subscriber.
