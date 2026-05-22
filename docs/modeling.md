# Modeling

The `src/models/` package provides position-aware trajectory models that project a player's future AV given their position and first *X* observed seasons.

---

## The Parametric Model

`ParametricCurveModel` fits a population-mean trajectory curve per position using `scipy.optimize.curve_fit`, then scales the fitted shape to individual players at inference time.

**Pluggable curve shape** — the curve function is selected at construction time via `curve_name` (default `"gamma"`). All curve descriptors live in `src/curve_fitting.py`.

| `curve_name` | Formula | Notes |
|---|---|---|
| `gamma` *(default)* | `a · t^α · exp(−b·t) + c` | Rises to a peak then decays; best match for career AV arcs |
| `exp_decay` | `a · exp(−b·t) + c` | Monotone decay from year 0 |
| `log_decay` | `a · ln(t) + b` | Slow log growth / decay |
| `quadratic` | `a·t² + b·t + c` | Unconstrained polynomial |
| `cubic` | `a·t³ + b·t² + c·t + d` | Unconstrained polynomial |
| `quartic` | `a·t⁴ + b·t³ + c·t² + d·t + e` | Unconstrained polynomial |

**At inference**, the fitted curve is held fixed and a single scale factor `s` personalises the projection:

```
s = mean(observed_av / f(t_observed))
projected_av[t] = s · f(t)
```

The uncertainty band is derived from the fit's covariance matrix via ±1σ parameter perturbation.

**Model artifacts** (human-readable JSON, committed to git) are stored in `models/parametric/<curve_name>/` — one sub-directory per trained curve variant, so multiple variants coexist without overwriting each other:

```
models/parametric/
  gamma/
    params.json      # fitted popt and pcov per position, plus curve_name
    metadata.json    # training date, year range, validation MAE, curve_name
  exp_decay/
    params.json
    metadata.json
```

**Using a custom curve at construction time:**

```python
from src.curve_fitting import GammaCurveModel, ExpDecayModel
from src.models.parametric import ParametricCurveModel

# Named lookup (recommended — curve_name is serialised with the model)
model = ParametricCurveModel(curve_name="exp_decay")

# Custom descriptor (use curve_name to label it for save/load)
model = ParametricCurveModel(
    curve=GammaCurveModel,
    curve_name="gamma",
    bounds=([0, 0.1, 0.01, -1], [50, 5, 5, 10]),
)
```

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
                               [--curve gamma|exp_decay|log_decay|quadratic|cubic|quartic]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `parametric` | Which model(s) to train |
| `--train-years` | `1970 2010` | Inclusive draft-year training window |
| `--rounds` | all | Draft rounds to include |
| `--max-years` | `10` | Number of career years to model |
| `--curve` | `gamma` | Curve shape for the parametric model (ignored for knn/ridge) |

The script trains on `START`–`END` draft classes, validates on 2011–2015 picks (predicting years 3–(N-1) given years 0–2), prints a per-position MAE table, and writes trained artifacts to:
- `models/parametric/<curve>/` for the parametric model
- `models/<name>/` for knn and ridge

Each parametric curve variant gets its own sub-directory, so multiple variants can be trained and compared without overwriting each other.

Example:

```bash
# Train gamma (default) — saves to models/parametric/gamma/
python scripts/train_models.py --model parametric

# Train exp_decay alongside it — saves to models/parametric/exp_decay/
python scripts/train_models.py --model parametric --curve exp_decay
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
   python scripts/train_models.py --model parametric --curve gamma
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

## Projection Comparison Script

`scripts/model_projection_comparison.py` generates per-player projection plots for any team/year, showing 1/2/3-year input windows for each model.

```bash
python scripts/model_projection_comparison.py --year 2022 [--team DET] [--model {parametric,knn,ridge,all}]
                                               [--parametric-curve CURVE]
```

| Option | Default | Description |
|---|---|---|
| `--year` | required | Draft year |
| `--team` | `DET` | Three-letter team code |
| `--model` | `all` | Which model(s) to plot |
| `--parametric-curve` | `gamma` | Curve variant to load from `models/parametric/<curve>/` |

To compare two parametric curve variants side-by-side, run the script twice with different `--parametric-curve` values:

```bash
python scripts/model_projection_comparison.py --year 2022 --model parametric --parametric-curve gamma
python scripts/model_projection_comparison.py --year 2022 --model parametric --parametric-curve exp_decay
```

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
