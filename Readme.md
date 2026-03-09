# 🏎️ F1 Strategy Simulator

**Formula 1 Strategy Recommendation System using Machine Learning**

---

## 👥 Contributors

### Core Contributors
- [**Roshith Prakash**](https://github.com/roshith-prakash)
- [**Rushil Patel**](https://github.com/rushilpatel11)
- [**Achal Chhag**](https://github.com/achalchhag)

### Secondary Contributors
- [**Soumyadeep Das**](https://github.com/s-h-u-v)
- [**Shlok Sukhija**](https://github.com/shlokk25)
- [**Tamanna Jain**](https://github.com/Tamanna950)

---

## 📌 Core Concept

This project aims to build a **machine-learning-powered F1 race strategy simulator** that can predict lap times under different conditions and, ultimately, recommend optimal pit-stop strategies (tyre compound, stint length, number of stops) for a given Grand Prix.

The system works by:

1. **Collecting historical lap data** from recent F1 seasons (2022–2025) via the [FastF1](https://github.com/theOehrly/Fast-F1) Python library.
2. **Cleaning and preprocessing** the data — removing pit-in/pit-out laps, first laps, laps under non-green-flag conditions, and handling `NaT`/`NaN` values.
3. **Training regression models** (starting with XGBoost) to predict lap times based on key features like driver, tyre compound, tyre life, lap number, and season year.
4. **Analysing year-over-year trends** — computing per-driver improvement rates to support extrapolation of predictions to future seasons.

---

## 💡 Key Ideas

| Idea | Details |
|------|---------|
| **Lap time prediction** | Predict lap times in seconds using features like `Driver`, `LapNumber`, `Compound`, `TyreLife`, and `Year`. |
| **Driver-consistent modelling** | Only include drivers who remained with the same team across all seasons (ALB / Williams, LEC / Ferrari, NOR / McLaren, RUS / Mercedes, VER / Red Bull) to control for car-performance confounders. |
| **Multi-model exploration** | Iteratively developed four modelling approaches — a basic model, a multi-year model, a per-driver model, and an XGBoost regressor — to find the best fit. |
| **Year-over-year trend analysis** | Fit a linear regression on each driver's average lap time per year to quantify yearly improvement and enable future-year extrapolation. |
| **Full-calendar data pipeline** | Automated download of race lap data for **25 Grand Prix circuits** across four seasons, with caching and skip logic for efficiency. |
| **Tyre degradation curves** | Capture tyre wear effects through `TyreLife` and compound type, enabling strategy simulations around optimal stint lengths. |

---

## 📂 Project Structure

```
Mini Project/
├── Readme.md                                              # This file
├── .gitignore
│
├── datasets/                                              # Race lap data (CSV per year per circuit)
│   ├── download_race_data.py                              # Download & save single-circuit data
│   ├── download_all_races.py                              # Batch download all 25 circuits
│   ├── Bahrain/
│   │   ├── 2022_Bahrain_Laps.csv
│   │   ├── 2023_Bahrain_Laps.csv
│   │   ├── 2024_Bahrain_Laps.csv
│   │   └── 2025_Bahrain_Laps.csv
│   ├── Abu_Dhabi/ ...
│   ├── Australia/ ...
│   └── ... (25 circuits total)
│
├── f1_laptime_model.ipynb                                 # Initial lap time model
├── f1_laptime_multiyear_model.ipynb                       # Multi-year lap time model
├── f1_laptime_per_driver_model.ipynb                      # Per-driver lap time model
├── f1_laptime_xgboost_model.ipynb                         # XGBoost-based lap time model (current best)
│
└── Formula 1 Strategy Recommendation System ... .pdf/.pptx  # Project presentation
```

---

## ✅ Current Progress

- [x] **Data acquisition pipeline** — Automated scripts to download race lap data for all 25 Grand Prix circuits (2022–2025) using FastF1, with SSL workarounds and caching.
- [x] **Data cleaning & preprocessing** — Handles `NaT`→`NaN` conversion, removes pit-in/out laps, first laps, non-green-flag laps, and filters for accurate laps only.
- [x] **Exploratory modelling** — Four modelling notebooks developed:
  - Basic lap time model
  - Multi-year model
  - Per-driver model
  - **XGBoost regressor** (current primary model)
- [x] **Feature engineering** — Label-encoding for drivers, one-hot encoding for tyre compound, year-over-year trend analysis via linear regression.
- [x] **XGBoost model trained & evaluated** — Trained on Bahrain GP data with 80/20 train-test split, stratified by driver. Evaluation metrics include RMSE, MAE, and R² (overall and per-driver).
- [x] **Year-over-year trend analysis** — Per-driver improvement rates computed (e.g., NOR ≈ −1.16 s/year, VER ≈ −0.11 s/year).

---

## 🔮 Next Steps

- [ ] **Multi-track models** — Extend the lap time prediction model to all 25 downloaded Grand Prix circuits (currently trained on Bahrain only).
- [ ] **Tyre degradation model** — Build a dedicated model that captures how lap times evolve with tyre life for each compound, enabling accurate stint-length estimation.
- [ ] **Race simulation & strategy prediction** — Develop a simulation engine that uses the lap time and tyre degradation models to recommend optimal pit-stop strategies (number of stops, compound choices, stint lengths) for a given circuit and driver.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Data source | [FastF1](https://github.com/theOehrly/Fast-F1) (Python library for F1 telemetry data) |
| Data handling | Pandas, NumPy |
| Machine learning | XGBoost, scikit-learn, SciPy |
| Visualisation | Matplotlib, Seaborn |
| Notebooks | Jupyter |

---

> **Note:** This is an ongoing academic mini-project. Contributions, suggestions, and feedback are welcome!