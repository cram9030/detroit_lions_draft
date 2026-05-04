# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Tests
python -m pytest tests/                          # full suite
python -m pytest tests/test_curve_fitting.py -v  # single file
python -m pytest tests/models/test_parametric.py::test_fit_populates_params  # single test

# Analysis pipeline
python scripts/run_analysis.py                   # full pipeline
python scripts/run_analysis.py --skip-skew --skip-rolling  # skip expensive steps

# Model training
python scripts/train_models.py --model parametric
python scripts/train_models.py --model all --train-years 1970 2010 --rounds 1 2

# Data fetching (requires secrets/cookies.json with Stathead session cookies)
python src/stathead_downloader.py --config config/stathead_annual_av.json

# Example inference
python scripts/example_lions_2024.py
```

## Architecture

### Data flow

```
Stathead (browser cookies) → stathead_downloader.py → data/raw/stathead/annual_av/*.parquet
                                                              ↓
                                              annual_av_analysis.py  ←  curve_fitting.py
                                                    ↓             ↓
                                            data/processed/    outputs/figures/
                                                    ↓
                                          train_models.py → models/{parametric,knn,ridge}/
                                                    ↓
                                        example_lions_2024.py (inference)
```

### Raw data schema

Each Parquet file covers one `(draft_year, season_year)` pair. **All columns are stored as strings** — `prepare_av_data()` in `annual_av_analysis.py` must be applied before any numeric work. Key column distinction:

- `AV` — career approximate value (cumulative, from Stathead header)
- `AV.1` — season-level AV for the specific season row ← this is what the analysis uses

### `src/` module roles

| Module | Role |
|---|---|
| `data_ingest.py` | `load_parquets_from_dir()`, `load_nflreadr_draft_picks()` — all I/O |
| `annual_av_analysis.py` | Data prep, aggregation, per-pick stats, skew/exp/log fits, position career stats |
| `curve_fitting.py` | Generic fit engine; `ExpDecayModel` / `LogDecayModel` descriptors used by `annual_av_analysis.py` |
| `plot_av.py` | All Plotly figure generation; receives fit result dicts directly |
| `data_output.py` | `save_data()` — writes CSV or Parquet, auto-creates parent dirs |
| `stathead_downloader.py` | Paginated Stathead scraper; resumes via `.progress.json` |
| `models/` | `CareerAVModel` protocol + Parametric / KNN / Ridge implementations |
| `trade_value.py` | `load_trade_chart(chart_name)` — normalises any of the 6 trade chart CSVs to `[Pick, Value]`; `find_pick_combination(target, chart_name)` — extended two-pointer search returning the set of picks summing closest to `target` |

### Key types in `curve_fitting.py`

Two TypedDicts replace what used to be four separate return types:
- `IndividualFitResult` — from `fit_individuals()`; has `av_values` key (one row per player)
- `StatsFitResult` — from `fit_stats()`; has `stat_values`, `iqr_picks`, `q25`, `q75` keys

`annual_av_analysis.py` functions (`exponential_av_fit`, `logarithmic_av_fit`, etc.) are thin wrappers that pass `ExpDecayModel` or `LogDecayModel` to the generic engine. To add a new model shape, define `(model_fn, jacobian_fn, p0_fn)` in `curve_fitting.py`.

### Career trajectory models

All three models implement `CareerAVModel` (protocol in `src/models/protocol.py`). They are instantiated via `make_career_av_model(name)` from `src/models/factory.py`.

`fit()` expects a `trajectory_df` — the output of `aggregate_career_av_by_position()` with columns `[Player, Pos, Draft Year, years_from_draft, AV.1]`. Positions are normalized through `_POSITION_GROUPS` before training.

`predict(position, observed_av)` returns a `PredictionResult` TypedDict with `predicted_years`, `y_pred`, `y_upper`, `y_lower`.

Parametric model artifacts are committed to git as human-readable JSON in `models/parametric/`. KNN and Ridge artifacts are `.joblib` binaries in `models/knn/` and `models/ridge/` and are **not** committed.

### Position normalization

Raw `Pos` values like `"LDE"`, `"LOLB"`, `"RG"` are mapped to 12 standard groups via `_POSITION_GROUPS` in `annual_av_analysis.py`. Compound positions (`"LDE/LOLB"`) are split and exploded. `_SPECALIST` (K, P, KR, etc.) and `_GENERALIST` (DL, OL) positions are excluded from normalized analyses because they don't appear in year-0 data.

### nflreadr vs Stathead AV

`load_nflreadr_draft_picks()` provides `dr_av` — the AV a player produced specifically for the team that drafted them. This differs from Stathead's `AV.1` (total season AV, including traded seasons). Both are used in `run_analysis.py` for comparison plots.
