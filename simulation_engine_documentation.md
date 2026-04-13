# F1 Strategy Simulation Engine (Implementation)

**Module:** `simulation_engine.py`  
**Core Logic:** Monte Carlo-style exhaustive search over valid compound sequences.  
**ML Integration:** Laptime Regression (Gradient Boosting) & Tyre Degradation (Gradient Boosting).

This engine serves as the deployment environment for the trained F1 models. It simulates full race distances by predicting lap-by-lap performance and wear, then optimizes for the minimum total race duration.

## 🏎️ Engine Architecture

The simulator is built around the `StrategySimulator` class, which encapsulates the model loading, strategy generation, and per-lap simulation loop.

### 1. Circuit & Environmental Constants
The engine uses hardcoded circuit data to define the boundaries of the simulation.

| Circuit | Laps | Pit Loss (s) |
| :--- | :--- | :--- |
| Italy (Monza) | 53 | 21.0 |
| Australia | 58 | 22.0 |
| Hungary | 70 | 22.0 |
| Saudi Arabia | 50 | 19.0 |

**Degradation Threshold:** Set to `95.0%`. Crossing this value triggers a "Critical Wear" state, forcing a pit stop if another compound is available in the sequence.

### 2. Strategy Generation Logic
The engine generates all permutations of 2-stint and 3-stint strategies that satisfy FIA regulation (≥ 2 different compounds) and physical constraints (no consecutive identical compounds).

```python
def generate_strategies(self):
    # 2-stint (1 pit stop)
    for a, b in itertools.permutations(COMPOUNDS, 2):
        strategies.append([a, b])

    # 3-stint (2 pit stops)
    for combo in itertools.product(COMPOUNDS, repeat=3):
        a, b, c = combo
        if a == b or b == c: continue
        if len(set(combo)) < 2: continue
        strategies.append(list(combo))
```

---

## 🔬 Simulation Methodology

### Prediction Features
For every lap $L$, the engine constructs a feature vector for the ML models:

- **Laptime Model ($M_{LT}$):** `[Driver, Track, LapNumber, TyreLife, Year, Compound_OneHot]`
- **Degradation Model ($M_{TD}$):** `[TyreLife, Initial_Life, Stint_Usage, LapNumber, Compound_Enc, Track_Enc, Year]`

### The Simulation Loop
The simulation proceeds linearly. If the predicted degradation % meets the threshold, the `stint_index` is incremented, and the `PITSTOP_TIME` overhead is added to the running total.

```python
# Core Loop Snippet
while lap_number <= total_laps:
    lap_time = self._predict_laptime(lap_number, tyre_life, current_compound)
    total_time += lap_time
    
    deg = self._predict_tyre_deg(lap_number, tyre_life, initial_life, current_compound)

    if deg >= DEG_THRESHOLD and lap_number < total_laps and not on_last_compound:
        # Pit Stop Logic
        stint_index += 1
        current_compound = strategy[stint_index]
        total_time += PITSTOP_TIME[self.race]
        n_pitstops += 1
```

---

## 📊 Output & Results

The engine produces a ranked leaderboard of strategies. A typical output for a race like **Italy (53 Laps)** might look like this:

```text
==============================================================
         F1 RACE STRATEGY OPTIMISATION RESULTS
==============================================================
  Driver : VER    Race : Italy    Year : 2025
==============================================================
  ★ BEST  MEDIUM → HARD               73m 28.75s  (4408.75 s)  |  Pit Stops: 1
    #2   MEDIUM → SOFT               73m 31.68s  (4411.68 s)  |  Pit Stops: 1
    #3   SOFT → HARD                 73m 35.02s  (4415.02 s)  |  Pit Stops: 1
```

### Best Strategy Detail
The engine re-simulates the winning strategy with `record_laps=True` to generate a high-fidelity CSV/DataFrame containing the telemetry for every single lap, including the exact point of compound crossover.

---

## 🛠️ Usage Instructions

1.  **Dependencies**: `numpy`, `pandas`, `joblib`, `scikit-learn`.
2.  **Model Placement**: Ensure `.pkl` and `metadata.pkl` files for both models are located in the `models/` subdirectory.
3.  **Run**: Update the `DRIVER`, `RACE`, and `YEAR` variables in the `__main__` block and execute:
    ```bash
    python simulation_engine.py
    ```

## 📂 File Structure
```text
.
├── simulation_engine.py
├── simulation_engine_documentation.md
└── models/
    ├── laptime_model.pkl
    ├── laptime_metadata.pkl
    ├── tyre_deg_model.pkl
    └── tyre_deg_metadata.pkl
```
