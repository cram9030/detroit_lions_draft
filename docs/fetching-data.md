# Fetching Data

This project has two downloaders that write to separate folders under `data/raw/`:

| Downloader | Source | Auth | Output |
|---|---|---|---|
| `stathead_downloader.py` | Stathead (subscription) | Browser cookies | `data/raw/stathead/` |
| `pfr_downloader.py` | Pro-Football-Reference (public) | None | `data/raw/pfr/` |

Both support resumable runs via `.progress.json` and write Parquet by default (pass `--csv` for CSV).

---

# Stathead

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

---

# Pro-Football-Reference (PFR)

PFR is a publicly accessible site — no subscription or cookies are required.

## Prerequisites

- Python 3.10 or later
- Dependencies installed (`pip install -r requirements.txt`)

---

## Step 1 — Choose or create a config

Two configs are included out of the box:

| Config | URL pattern | Iterates over | Output |
|---|---|---|---|
| `config/pfr_executives.json` | `/teams/{team}/executives.htm` | All 32 NFL team codes | `data/raw/pfr/executives/{team}_executives.parquet` |
| `config/pfr_standings.json` | `/years/{year}/` | Years 2000–2025 | `data/raw/pfr/standings/{year}_{afc\|nfc}.parquet` |

To fetch a **different PFR page**, see [Adding a new PFR data source](#adding-a-new-pfr-data-source) below.

---

## Step 2 — Run the downloader

```bash
# Team executives / front-office history for all 32 teams
python src/pfr_downloader.py --config config/pfr_executives.json

# AFC + NFC standings for years 2000–2025
python src/pfr_downloader.py --config config/pfr_standings.json

# Save as CSV instead of Parquet (useful for quick inspection)
python src/pfr_downloader.py --config config/pfr_standings.json --csv
```

The script will:

1. Load the config and expand the iteration list.
2. For each iteration value, build the URL and fetch the page.
3. Extract each configured table (unwrapping PFR comment-hidden tables automatically).
4. Write one file per `(iteration_value, table)` pair.
5. Log progress to the terminal and to `pfr_downloader.log`.
6. Record completed keys in `.progress.json` so interrupted runs resume without re-downloading.

---

## Output structure

```
data/raw/pfr/
├── executives/
│   ├── det_executives.parquet   ← one file per team
│   ├── chi_executives.parquet
│   └── .progress.json
└── standings/
    ├── 2025_afc.parquet         ← one file per (year, conference)
    ├── 2025_nfc.parquet
    ├── 2024_afc.parquet
    └── .progress.json
```

---

## Loading the data

```python
import pandas as pd

# Load a single file
df = pd.read_parquet("data/raw/pfr/executives/det_executives.parquet")

# Load all executives files at once
import glob
dfs = [pd.read_parquet(f) for f in glob.glob("data/raw/pfr/executives/*.parquet")]
executives = pd.concat(dfs, ignore_index=True)

# Or with polars
import polars as pl
standings = pl.scan_parquet("data/raw/pfr/standings/*.parquet").collect()
```

> **Note:** All columns are stored as strings, consistent with the Stathead raw data schema.
> Apply your own type casting before numeric analysis.

---

## Adding a new PFR data source

**No code changes are required.** Create a JSON config in `config/` following this template:

```json
{
  "source_name": "pfr_my_table",
  "url_template": "https://www.pro-football-reference.com/teams/{team}/some_page.htm",

  "iterate": {
    "team": ["det", "chi", "gb", "min"]
  },

  "tables": [
    { "id": "html_table_id", "output_suffix": "my_table" }
  ],

  "output_dir": "data/raw/pfr/my_table",

  "sleep_between_requests": 4.0,
  "max_retries": 3,
  "retry_backoff": 10.0
}
```

Then run:

```bash
python src/pfr_downloader.py --config config/pfr_my_table.json
```

### Config reference

| Field | Required | Description |
|---|---|---|
| `source_name` | Yes | Human-readable label used in logs |
| `url_template` | Yes | URL with `{variable}` placeholders |
| `iterate` | Yes | Iteration dimension — see formats below |
| `tables` | Yes | List of tables to extract per page |
| `output_dir` | Yes | Path relative to project root |
| `sleep_between_requests` | Yes | Seconds between fetches; keep ≥ 4.0 for PFR |
| `max_retries` | Yes | Retry count on non-200 responses |
| `retry_backoff` | Yes | Initial backoff seconds (doubles on each retry) |

### `iterate` formats

**Explicit list** — one request per value:
```json
"iterate": { "team": ["det", "chi", "gb", "min"] }
```

**Integer range** — one request per year, inclusive on both ends:
```json
"iterate": { "year": { "start": 2000, "end": 2025 } }
```

Any single key name works (`team`, `year`, `season`, `week`, etc.) as long as it matches
the placeholder in `url_template`.

### `tables` entry fields

| Field | Description |
|---|---|
| `id` | The HTML `id=` attribute of the target table (preferred) |
| `index` | 0-based position of the table in the page (fallback if no id) |
| `output_suffix` | Appended to the iteration value in the output filename |

PFR hides certain tables inside HTML comments (`<!-- <table ...> -->`). The downloader
unwraps these automatically — no special config is needed.

### Output file naming

`{output_dir}/{iteration_value}_{output_suffix}.parquet`

Examples:
- `{"team": "det"}` + suffix `"executives"` → `det_executives.parquet`
- `{"year": 2025}` + suffix `"afc"` → `2025_afc.parquet`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `BLOCKED RESPONSE DETECTED` | Rate-limited by PFR | Wait a few minutes; increase `sleep_between_requests` to `6.0` or higher |
| Table not found or empty | Wrong table `id`, or table is seasonal and absent for that year | Inspect the page in DevTools to find the correct `id` |
| All rows are strings | Expected — raw schema stores everything as `str` | Cast columns after loading |
| Script is slow | Intentional rate limiting | Keep `sleep_between_requests` ≥ 4.0; PFR can throttle aggressive bots |

---

## Rate limiting and terms of service

The default 4-second delay is intentional. PFR is a free public resource that
relies on advertising revenue. Aggressive scraping hurts the site and risks an
IP ban. Do not reduce the delay below **3 seconds**. This tool is intended for
personal, non-commercial research use.
