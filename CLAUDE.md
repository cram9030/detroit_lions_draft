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

# PFR data fetching (public — no cookies required)
python src/pfr_downloader.py --config config/pfr_executives.json   # team executives/staff history
python src/pfr_downloader.py --config config/pfr_standings.json    # AFC + NFC standings by year
python src/pfr_downloader.py --config config/pfr_executives.json --csv  # save as CSV instead of Parquet

# Draft class surplus AV analysis (any team/year with 2+ completed seasons)
python scripts/draft_class_surplus_av.py --team DET --year 2024
python scripts/draft_class_surplus_av.py --team DET --year 2024 --model knn

# Example inference (Lions 2024 — 3-model comparison vs pick expectation)
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

Pro-Football-Reference (public) → pfr_downloader.py → data/raw/pfr/{source}/*.parquet
    config/pfr_executives.json  →  data/raw/pfr/executives/{team}_executives.parquet
    config/pfr_standings.json   →  data/raw/pfr/standings/{year}_{afc|nfc}.parquet
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
| `scraper_utils.py` | Shared HTTP utilities: `build_session()`, `fetch_page()` (retry/backoff), `load_progress()` / `save_progress()` — imported by both downloaders |
| `stathead_downloader.py` | Paginated Stathead scraper; resumes via `.progress.json` |
| `pfr_downloader.py` | Config-driven PFR scraper; URL template + {variable} substitution, team-list or year-range iteration, PFR comment-unwrapping, multi-table per page |
| `models/` | `CareerAVModel` protocol + Parametric / KNN / Ridge implementations |
| `surplus_av.py` | `load_team_draft_class(team, year)` — generalized draft class loader (requires 2+ completed seasons); `project_player_seasons(model, player, pos, obs_av)` — projects years 2/3 via any CareerAVModel; `aggregate_observed_av(df)` — pure-Polars 4yr totals when all 4 seasons are present (no model needed); `aggregate_model_av(df, model)` — combines observed + projected into per-player 4yr totals when < 4 seasons available; `compute_surplus_av(df)` — joins against `expected_av_above_replacement.csv` (caps picks > 250 at the last pick), adds `total_4yr_av_above_replacement = total_4yr_av − replacement_level` and `surplus_av = total_4yr_av_above_replacement − eavar` |
| `trade_value.py` | `load_trade_chart(chart_name)` — normalises any of the 6 trade chart CSVs to `[Pick, Value]`; `find_pick_combination(target, chart_name)` — extended two-pointer search returning the set of picks summing closest to `target`; `analyze_draft_trades(team, year)` — fetches trades via nflreadpy, filters to same-year draft picks, returns per-trade DataFrame with net value and equivalent picks (via `abs(net_value)`) across 5 trade charts |

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

### Adding a new PFR data source

No code changes are needed. Create a JSON config in `config/` with these fields:

```json
{
  "source_name": "pfr_my_table",
  "url_template": "https://www.pro-football-reference.com/teams/{team}/some_page.htm",
  "iterate": {
    "team": ["det", "chi", "gb"]
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

**`iterate` forms:**
- Explicit list: `{"team": ["det", "chi"]}` → one URL per team
- Year range: `{"year": {"start": 2000, "end": 2025}}` → one URL per year (inclusive)
- Any single key with either form works (`{"season": [...]}`, `{"week": {...}}`, etc.)

**`tables` selection:** each entry can specify `"id"` (HTML `id=` attribute) or `"index"` (0-based position in page). PFR comment-wrapped tables are handled automatically. Omit `tables` entirely to skip saving (useful for debugging).

**Output naming:** `{output_dir}/{iter_value}_{output_suffix}.parquet`  
e.g. `data/raw/pfr/my_table/det_my_table.parquet`

**Re-runs are idempotent** — completed keys are stored in `.progress.json` inside the output directory and skipped on subsequent runs.

### nflreadr vs Stathead AV

`load_nflreadr_draft_picks()` provides `dr_av` — the AV a player produced specifically for the team that drafted them. This differs from Stathead's `AV.1` (total season AV, including traded seasons). Both are used in `run_analysis.py` for comparison plots.
