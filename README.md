# Detroit Lions Draft Analysis

Analysis of Detroit Lions NFL draft history and data. The project fetches historical Approximate Value (AV) data from Stathead, runs a suite of pick-value analyses, and provides three position-aware career trajectory models (Parametric, KNN, Ridge) for projecting future player value.

## Project Structure

```
detroit_lions_draft/
├── config/               # Query configurations (version-controlled)
│   └── stathead_annual_av.json
├── data/
│   ├── raw/              # Raw source data (not tracked by git)
│   └── processed/        # Cleaned and transformed data
├── docs/                 # Extended documentation
│   ├── draft-integration.md
│   ├── fetching-data.md
│   ├── modeling.md
│   ├── running-analysis.md
│   └── trade-analysis.md
├── models/               # Trained model artifacts
│   ├── knn/
│   ├── parametric/
│   └── ridge/
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/
│   ├── figures/          # Generated interactive HTML plots
│   └── reports/
├── scripts/              # Runnable entry points
│   ├── example_lions_2024.py
│   ├── generate_trade_patch.py
│   ├── run_analysis.py
│   └── train_models.py
├── secrets/              # Local credentials — gitignored, never committed
│   └── cookies.json      # (you create this — see Fetching Data below)
├── src/                  # Python source modules
│   ├── models/           # CareerAV model implementations
│   ├── draft_integration.py
│   ├── stathead_downloader.py
│   └── trade_value.py
├── tests/                # Unit tests
│   ├── models/
│   ├── test_draft_integration.py
│   └── test_draft_trade_analysis.py
├── requirements.txt
└── README.md
```

## Requirements

- Python **3.10+** (required by `nflreadpy`)
- Recommended: use the Dev Container (see below) to avoid managing Python versions locally

---

# Setup

## Option 1: Dev Container (recommended)

The easiest way to get a fully isolated, correctly versioned environment.

**Prerequisites:** [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

1. Open the repo in VS Code
2. When prompted, click **Reopen in Container** (or run `Dev Containers: Reopen in Container` from the command palette)
3. The container builds with Python 3.11 and installs all dependencies automatically

Jupyter runs on port **8888** and will open in your browser automatically.

---

## Option 2: Local virtual environment

Requires Python 3.10+ installed locally. Check your version with `python3 --version`.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the kernel for Jupyter

```bash
python -m ipykernel install --user --name detroit-lions-draft --display-name "Detroit Lions Draft"
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Select the **Detroit Lions Draft** kernel when creating or opening a notebook.

### Deactivating the virtual environment

```bash
deactivate
```

---

# Quick Start

1. **Fetch data** — export Stathead cookies and run the downloader. See [docs/fetching-data.md](docs/fetching-data.md).

2. **Run analysis**:
   ```bash
   python scripts/run_analysis.py
   ```
   See [docs/running-analysis.md](docs/running-analysis.md) for pipeline flags and output plot descriptions.

3. **Train models**:
   ```bash
   python scripts/train_models.py --model all
   ```
   See [docs/modeling.md](docs/modeling.md) for model details, training options, and how to add a new model.

---

# Documentation

| Topic | File |
|---|---|
| Fetching data from Stathead | [docs/fetching-data.md](docs/fetching-data.md) |
| Running the analysis pipeline and output plots | [docs/running-analysis.md](docs/running-analysis.md) |
| Career trajectory models (Parametric, KNN, Ridge) | [docs/modeling.md](docs/modeling.md) |
| Draft trade analysis across 5 trade charts | [docs/trade-analysis.md](docs/trade-analysis.md) |
| Integrating draft JSON into nflreadpy trades | [docs/draft-integration.md](docs/draft-integration.md) |

---

# Trade Analysis

`analyze_draft_trades(team, year)` in `src/trade_value.py` evaluates all draft-day trades involving a team in a given year across five trade charts.

```python
from src.trade_value import analyze_draft_trades

df = analyze_draft_trades("PHI", 2021)
print(df)
```

Each row in the output represents one trade. Output columns:

| Column | Type | Description |
|---|---|---|
| `trade_id` | Int64 | Identifier from nflreadpy |
| `team_traded_with` | String | Other team(s), comma-separated |
| `picks_received` | String | Pick numbers received, sorted ascending, comma-separated |
| `picks_gave` | String | Pick numbers given, sorted ascending, comma-separated |
| `fitz_spiel_value` | Float64 | Net value (received − gave) per Fitzgerald-Spielberger chart |
| `fitz_spiel_picks` | String | Picks equivalent to `abs(net_value)` on that chart |
| `jj_value` | Float64 | Net value per Jimmy Johnson chart |
| `jj_picks` | String | Equivalent picks |
| `pff_value` | Float64 | Net value per PFF WAR chart |
| `pff_picks` | String | Equivalent picks |
| `rich_hill_value` | Float64 | Net value per Rich Hill chart |
| `rich_hill_picks` | String | Equivalent picks |
| `eaar_value` | Float64 | Net value per Expected AV Above Replacement chart |
| `eaar_picks` | String | Equivalent picks |

A trade is excluded when any player asset was not drafted in `year`, or when no picks were exchanged. See [docs/trade-analysis.md](docs/trade-analysis.md) for full details.

---

# Draft Integration

`src/draft_integration.py` fills in missing data that nflreadpy doesn't have: prior-year trades with null pick numbers, and draft-day trades that don't exist in nflreadpy at all. It reads from a season draft JSON (e.g. `data/processed/nfl_draft_2026.json`).

**Generate a patch file from the command line:**

```bash
python scripts/generate_trade_patch.py data/processed/nfl_draft_2026.json 2026
# Writes data/processed/trade_patch_2026.json
```

**Apply in Python:**

```python
import json
import nflreadpy
from src.draft_integration import populate_pick_numbers, add_new_trades

with open("data/processed/nfl_draft_2026.json") as f:
    draft_picks = json.load(f)

trades = nflreadpy.load_trades()
trades = populate_pick_numbers(trades, draft_picks, 2026)  # fill null pick_numbers
trades = add_new_trades(trades, draft_picks, 2026)         # add draft-day trades

# Then pass the enriched trades DataFrame to analyze_draft_trades or your own analysis
```

See [docs/draft-integration.md](docs/draft-integration.md) for the full API, patch JSON format, and workflow details.
