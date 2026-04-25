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
│   ├── fetching-data.md
│   ├── modeling.md
│   └── running-analysis.md
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
│   ├── run_analysis.py
│   └── train_models.py
├── secrets/              # Local credentials — gitignored, never committed
│   └── cookies.json      # (you create this — see Fetching Data below)
├── src/                  # Python source modules
│   ├── models/           # CareerAV model implementations
│   └── stathead_downloader.py
├── tests/                # Unit tests
│   └── models/
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
