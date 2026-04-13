# F1 Tyre Degradation — Gradient Boosting Model

**Model Type:** Gradient Boosting Regressor  
**Inputs (Features):** `TyreLife`, `Initial_Life`, `Stint_Usage`, `Expected_Max_Life`, `LapNumber`, `Compound_Enc`, `Track_Enc`, `Year`  
**Target Variable:** `degradation_stint_%` (0.0 to 100.0)

This model predicts tyre degradation as a percentage of the expected maximum life of a tyre compound for a given track and year. It captures the non-linear wear patterns across different compounds and circuits, allowing the simulation engine to determine optimal pit windows.

---

## 🔬 Model Architecture

The model is built using a **Gradient Boosting Regressor**, which was selected over Ridge and Random Forest due to its superior performance in capturing the complex, circuit-specific degradation slopes.

### Feature Engineering
- **`Initial_Life`**: The age of the tyre at the start of the current stint (important for used tyres).
- **`Stint_Usage`**: Number of laps completed on the current set of tyres.
- **`Expected_Max_Life`**: A benchmark calculated as the 75th percentile of tyre life seen for a specific track/compound combination.
- **`degradation_stint_%`**: Scaled wear relative to the benchmark, clipped between 0 and 100.

---

## 📊 Performance Metrics

The model was evaluated using a year-based holdout strategy (2025 as the test set). Performance is exceptionally high due to the high correlation between lap count and physical wear.

| Metric | Training Set | Test Set (2025) |
| :--- | :--- | :--- |
| **MAE** | 1.45% | 1.91% |
| **RMSE** | 2.88% | 4.16% |
| **R²** | 0.991 | 0.981 |

### Breakdown by Track (MAE)
- **Hungary**: 1.16%
- **Saudi Arabia**: 1.68%
- **Italy (Monza)**: 2.18%
- **Australia**: 22.74% (Note: outlier due to extreme track-specific conditions)

### Breakdown by Compound (MAE)
- **HARD**: 1.33%
- **MEDIUM**: 2.39%
- **SOFT**: 6.93%

---

## 🖼️ Visualizations

### Tyre Degradation Curves
The model identifies distinct slopes for different compounds. Harder compounds exhibit flatter curves (lower degradation per lap), while soft compounds show accelerated wear.

![Tyre Degradation Curves](file:///c:/ROSHITH2/NMIMS/Sem%202/Mini%20Project/tyre_degradation_curves.png)

### Actual vs Predicted
The high fidelity in the 0-80% range ensures the simulation engine makes accurate pit stop decisions before the "cliff" is reached.

---

## 🛠️ Usage in Simulator

This model is serialized as `tyre_deg_model.pkl`. In the simulation engine, it is called every lap to check if the current `degradation_stint_%` has exceeded the **`DEG_THRESHOLD`** (configured at 95%).

```python
# Integration Example
deg = td_model.predict(current_lap_features)
if deg >= 95.0:
    trigger_pit_stop()
```
