# Modeling

The `src/models/` package provides position-aware trajectory models that project a player's future AV given their position and first *X* observed seasons.

---

## The Parametric Model

`ParametricCurveModel` fits a Gamma-shaped curve to the population mean AV trajectory for each normalized position group:

```
f(t) = a · t^α · exp(−b·t) + c
```

Unlike a pure exponential, this shape rises to a peak (typically years 3–5) then decays, matching the observed career arc. Parameters `a`, `α`, `b`, `c` are fitted per position using `scipy.optimize.curve_fit`.

**At inference**, the curve shape is held fixed and a single scale factor `s` is computed from the player's observed seasons to personalise the projection:

```
s = mean(observed_av / f(t_observed))
projected_av[t] = s · f(t)
```

The uncertainty band is derived from the fit's covariance matrix via Jacobian propagation.

Model artifacts (human-readable JSON, committed to git) are stored in `models/parametric/`:
- `params.json` — fitted `popt` and `pcov` per position
- `metadata.json` — training date, year range, validation MAE by position

---

## The KNN Model

`KNNTrajectoryModel` takes a non-parametric approach: it stores the full career trajectories of all players in the training set and, at inference time, finds the K most similar historical players based on the observed seasons only.

**At training**, for each position the model builds a reference matrix of shape `(n_complete_players, max_years)` — one row per player who has a full `max_years` career recorded.

**At inference**, similarity is measured as Euclidean distance on the *observed* dimensions only, so the model works regardless of how many seasons have been seen:

```
dists[i] = ‖ ref_matrix[i, :n_obs] − observed_av ‖₂
```

The K nearest neighbors are selected and their future seasons are averaged with inverse-distance weights:

```
weights[i] = 1 / (dists[i] + ε)
projected_av[t] = Σ weights[i] · ref_matrix[i, t]  for t ≥ n_obs
```

The uncertainty band is ±1 std dev across the K neighbors' future AV values.

**Key parameter**: `n_neighbors` (default 10) — fewer neighbors produces projections that more closely mirror a specific player comp; more neighbors gives a smoother, population-level estimate.

Model artifacts are stored in `models/knn/_config.joblib` (binary, not committed to git).

---

## The Ridge Model

`RidgeRegressionModel` takes a linear approach: it trains one multi-output `RidgeCV` model per position that maps a small number of observed early-career seasons directly to all remaining future seasons.

**At training**, players with a complete `max_years` career window are pivoted into a matrix of shape `(n_players, max_years)`. The first `n_input` columns become features `X`; the remaining columns become targets `Y`:

```
X = AV[0 : n_input]          # observed early seasons
Y = AV[n_input : max_years]  # all future seasons to predict
```

`RidgeCV` selects the best regularisation strength from `[0.1, 1.0, 10.0, 100.0]` via cross-validation. Training residuals `Y − Ŷ` are stored per future year so the model can report per-season uncertainty at inference time.

**At inference**, only the first `n_input` observed seasons are used as features — future seasons are never seen by the model, so there is no leakage when the remaining feature slots are zero-padded:

```
x = [observed_av[0], ..., observed_av[n_input - 1]]
y_pred[t] = RidgeCV.predict(x)   for t in n_input .. max_years - 1
```

The uncertainty band is ±1 std dev of training residuals for each future year.

**Key parameters**:
- `n_input` (default 2) — number of early-career seasons used as features. Increasing this makes the model more informative but limits how early a projection can be made.
- `max_years` (default 10) — total career window modelled; predictions cover years `n_input` through `max_years − 1`.

Model artifacts are stored in `models/ridge/_config.joblib` (binary, not committed to git).

---

## Training a Model

```
python scripts/train_models.py [--model parametric|knn|ridge|all]
                               [--train-years START END]
                               [--rounds ROUND ...]
                               [--max-years N]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `parametric` | Which model(s) to train |
| `--train-years` | `1970 2010` | Inclusive draft-year training window |
| `--rounds` | all | Draft rounds to include |
| `--max-years` | `10` | Number of career years to model |

The script trains on `START`–`END` draft classes, validates on 2011–2015 picks (predicting years 3–(N-1) given years 0–2), prints a per-position MAE table, and writes trained artifacts to `models/<name>/`.

Example:

```bash
python scripts/train_models.py --model parametric
```

Output:

```
Position      Val MAE
----------------------
CB              5.201
DE              4.653
...
OVERALL         4.412
```

---

## Example Script — Lions 2024 Draft Class

`scripts/example_lions_2024.py` runs all three models (Parametric, KNN, Ridge) on the Lions 2024 draft picks, using years 0 and 1 as observed input and projecting years 2–3. Each model's 4-year cumulative AV is compared against the historical expectation derived from pick position.

**Prerequisites:**

1. **2024 draft data** — update `config/stathead_annual_av.json` with `"draft_year_start": 2024, "draft_year_end": 2024`, then run:
   ```bash
   python src/stathead_downloader.py --config config/stathead_annual_av.json
   ```

2. **Trained models** — run:
   ```bash
   python scripts/train_models.py --model parametric
   python scripts/train_models.py --model knn
   python scripts/train_models.py --model ridge
   ```

**Run:**

```bash
python scripts/example_lions_2024.py
```

The script prints a per-player table with observed AV, each model's year-2 and year-3 projections, cumulative 4-year totals, and deltas vs pick expectation, followed by a class-level summary across all three models. It then saves:
- `outputs/figures/lions_2024_player_comparison.html` — grouped bar chart, all three models vs expectation per player
- `outputs/figures/lions_2024_class_comparison.html` — class total AV bar chart with all three models

---

## Adding a New Model to the Factory

1. Create `src/models/<name>.py` implementing the `CareerAVModel` Protocol:
   ```python
   class MyModel:
       def fit(self, trajectory_df: pl.DataFrame) -> None: ...
       def predict(self, position: str, observed_av: list[float]) -> PredictionResult: ...
       def save(self, model_dir: str | Path) -> None: ...
       def load(self, model_dir: str | Path) -> None: ...
   ```

2. Register it in `src/models/factory.py`:
   ```python
   _REGISTRY = {
       ...,
       "<name>": MyModel,
   }
   ```

3. Add a placeholder `models/<name>/metadata.json`.

4. Add unit tests in `tests/models/test_<name>.py` following the existing pattern (see `test_parametric.py`).
