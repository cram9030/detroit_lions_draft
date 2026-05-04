# Running the Analysis

Once you have data in `data/raw/stathead/annual_av/`, run the full pipeline with:

```bash
python scripts/run_analysis.py
```

This runs every analysis step in sequence and writes processed data to `data/processed/` and figures to `outputs/figures/`.

## Pipeline flags

Use `--skip-*` flags to bypass expensive steps during iteration:

| Flag | What it skips |
|---|---|
| `--skip-skew` | Full-dataset and rolling-window skew-normal distribution fits |
| `--skip-rolling` | All rolling-window analyses (stats, skew fit, and animated plot) |
| `--skip-exp-fit` | Exponential decay curve fits and their plots |
| `--skip-plots` | All Plotly figure generation (analysis steps still run) |

Flags can be combined:

```bash
python scripts/run_analysis.py --skip-skew --skip-rolling
```

---

## Output plots

All figures are saved as interactive HTML files in `outputs/figures/`. Open them in any browser — hover over data points for exact values, click legend items to show/hide traces, and use the toolbar to zoom or pan.

---

### `pick_av_static.html` — Mean AV by Draft Pick

Shows the full historical dataset (1970–2022) summarized at the pick level.

- **Dark blue line** — mean rookie contract AV at each pick number (career AV accumulated through the end of the rookie contract, typically four years).
- **Teal shaded band** — 25th–75th percentile range at each pick. Narrow bands indicate consistent outcomes; wide bands reflect high variance (e.g. early first-round picks where busts and stars diverge sharply).

The steep drop-off in the first 30–40 picks and the flattening toward the later rounds reflects the diminishing expected value as picks get later.

---

### `pick_av_animated.html` — Rolling-Window Exponential Fit

An animated version of the exponential fit (see below) computed over an 11-year rolling window centered on each draft year. Use the **Play** button or the year slider to step through time.

Each frame shows three layers for that window's data:

- **Teal shaded band** — 25th–75th percentile IQR of AV at each pick position.
- **Dark blue curve** — fitted exponential decay: `f(pick) = a·exp(−b·pick) + c`. The legend shows the fitted parameters for that window.
- **Teal scatter points** — per-pick mean AV (one dot per pick number).

Watch for shifts in the curve parameters over time to identify eras where early picks were more or less predictably valuable (e.g., changes in scouting, the rookie wage scale, or rule changes that amplified certain positions).

---

### `pick_av_exp_fit.html` — Exponential Fit (Individual Players)

Fits `f(pick) = a·exp(−b·pick) + c` against every individual player observation rather than per-pick means, so each player contributes one data point.

- **Teal shaded band** — ±1σ confidence band derived from the fit's covariance matrix via error propagation. Reflects uncertainty in the fitted curve, not spread in player outcomes.
- **Dark blue curve** — fitted exponential.
- **Semi-transparent scatter** — individual player AV values. The dense cloud at low AV values throughout the pick range illustrates how most picks produce little to no contribution.

Use this plot to see the raw distribution of outcomes before aggregation. The wide vertical spread at any given pick number shows that draft position is a noisy signal — the curve captures expected value, not certainty.

---

### `pick_av_exp_fit_means.html` — Exponential Fit on Per-Pick Means

Identical model to the individual-player fit above, but fitted against per-pick mean AV (one data point per pick position, each pick weighted equally regardless of how many players were drafted there).

- **Teal shaded band** — 25th–75th percentile IQR of player AV at each pick, showing the actual spread of outcomes around the mean.
- **Dark blue curve** — fitted exponential decay with parameters shown in the legend.
- **Teal scatter points** — per-pick mean AV (the points the curve is fitted to).

This plot is better for assessing the smoothness of the value curve and for reading off expected AV at a specific pick. The IQR band gives a practical sense of the range of outcomes a team should plan for — a pick where the band sits well above zero still carries meaningful bust risk.

---

### `pick_av_all_fits_comparison.html` — All-Model Comparison

Overlays all five decay models on one figure against the individual player scatter, allowing direct visual comparison of how each functional form tracks the data.

Models shown (each as a solid line across picks 1–250):

| Model | Form |
|---|---|
| Exponential | `a · exp(−b · pick) + c` |
| Logarithmic | `a · ln(pick) + b` |
| Quadratic | `a · pick² + b · pick + c` |
| Cubic | `a · pick³ + b · pick² + c · pick + d` |
| Quartic | `a · pick⁴ + b · pick³ + c · pick² + d · pick + e` |

All five models fit the data comparably well (R² ≈ 0.28–0.29 on individual player AV, which is inherently noisy). The quartic edges out exponential by a small margin. Use this plot to assess where models diverge — particularly in the late rounds — and whether any functional form produces implausible values outside the fitted range.

---

## Processed data outputs

In addition to the figures, `run_analysis.py` writes per-pick fit curves to `data/processed/`:

| File | Contents |
|---|---|
| `exp_fit_rookie_contract_av.csv` | Exponential fit evaluated at picks 1–250 |
| `log_fit_rookie_contract_av.csv` | Logarithmic fit evaluated at picks 1–250 |
| `poly_fit_quadratic_rookie_contract_av.csv` | Quadratic fit evaluated at picks 1–250 |
| `poly_fit_cubic_rookie_contract_av.csv` | Cubic fit evaluated at picks 1–250 |
| `poly_fit_quartic_rookie_contract_av.csv` | Quartic fit evaluated at picks 1–250 |
| `fit_metrics.csv` | R² and RMSE for all five models |

All fit CSVs share the same schema: `pick` (integer), `y_fit`, `y_upper`, `y_lower` (Float64, 1-sigma confidence band).
