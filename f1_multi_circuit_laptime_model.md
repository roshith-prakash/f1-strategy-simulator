# F1 Lap Time Prediction — Multi-Circuit XGBoost Model

**Model Type:** XGBoost Regressor  
**Inputs (Features):** `Driver_encoded`, `Track_encoded`, `LapNumber`, `TyreLife`, `Year`, `Compound_HARD`, `Compound_MEDIUM`, `Compound_SOFT`  
**Target Variable:** `LapTime` (seconds)

This model predicts the lap time for a given driver at a specific track, accounting for tyre compound, tyre age (life), and the progression of the race. It serves as the primary performance estimator in the strategy simulation engine.

---

## 🔬 Model Architecture

The model utilizes **XGBoost (Extreme Gradient Boosting)**, optimized for high-dimensional tabular data. It was trained on over 10,000 clean racing laps from 2022-2025 across 12 different circuits.

### Key Characteristics
- **Multi-Circuit Coverage**: Encodes 12 global tracks, allowing for circuit-specific pace variations.
- **Driver Profiling**: Includes 5 top-tier drivers (ALB, LEC, NOR, RUS, VER) who maintained team consistency across the dataset period.
- **One-Hot Compounds**: Directly models the performance delta between HARD, MEDIUM, and SOFT tyres.

---

## 📊 Performance Metrics

The model demonstrates exceptional accuracy, with an overall R² of 0.9966, meaning it explains over 99.6% of the variance in lap times.

| Metric | Value |
| :--- | :--- |
| **RMSE** | 0.6604 s |
| **MAE** | 0.4223 s |
| **R²** | 0.9966 |

### Per-Driver Accuracy (MAE)
- **VER**: 0.3972 s
- **RUS**: 0.3963 s
- **LEC**: 0.4049 s
- **ALB**: 0.4300 s
- **NOR**: 0.4810 s

### Per-Track Accuracy (MAE)
- **Austria**: 0.2902 s (Highest accuracy)
- **Italy**: 0.3139 s
- **Hungary**: 0.5255 s
- **Australia**: 0.4926 s

---

## 🖼️ Visualizations

### Actual vs Predicted Lap Times
The scatter plots across different tracks show a near-perfect linear alignment along the identity line ($y=x$), indicating the model correctly handles the different time scales of various circuits (e.g., 75s at Brazil vs 115s at Belgium).

![Actual vs Predicted Lap Time per Track & Year](file:///c:/ROSHITH2/NMIMS/Sem%202/Mini%20Project/actual_vs_predicted_laps.png)

### Prediction Error by Year
Boxplots show that prediction errors are tightly centered around zero across all years (2022-2025), confirming the model's robustness to yearly regulation changes and car development.

---

## 🛠️ Usage in Simulator

This model is serialized as `laptime_model.pkl`. It is invoked by the `StrategySimulator` to calculate the "cost" of every lap in a simulated race strategy.

```python
# Integration Example
predicted_seconds = lt_model.predict(lap_features)
total_race_time += predicted_seconds
```
