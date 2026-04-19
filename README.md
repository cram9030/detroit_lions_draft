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

## Setup

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

### 4. Register the kernel for Jupyter (optional, for notebook use)

```bash
python -m ipykernel install --user --name detroit-lions-draft --display-name "Detroit Lions Draft"
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Select the **Detroit Lions Draft** kernel when creating or opening a notebook.

## Deactivating the virtual environment

```bash
deactivate
```
