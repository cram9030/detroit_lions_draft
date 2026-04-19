# Detroit Lions Draft Analysis

Analysis of Detroit Lions NFL draft history and data.

## Project Structure

```
detroit_lions_draft/
├── data/
│   ├── raw/          # Raw source data (not tracked by git)
│   └── processed/    # Cleaned and transformed data
├── notebooks/        # Jupyter notebooks for prototyping and exploration
├── src/              # Python source modules
├── outputs/
│   ├── figures/      # Generated plots and charts
│   └── reports/      # Generated reports
├── requirements.txt
└── README.md
```

## Requirements

- Python **3.10+** (required by `nflreadpy`)
- Recommended: use the Dev Container (see below) to avoid managing Python versions locally

---

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
