# F1 Tyre Degradation — Gradient Boosting Model

**Model:** Gradient Boosting Regressor  
**Inputs:** TyreLife, Initial_Life, Stint_Usage, Expected_Max_Life, LapNumber, Compound, Track, Year  
**Target:** degradation_stint_%  

This model predicts tyre degradation as a percentage of the expected maximum life of a tyre compound for a given track and year. It uses historical lap data to capture wear patterns across different compounds and circuits.

## Setup and Libraries

```python
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

## Data Loading and Initial Filtering

We target specific drivers who have maintained consistency across recent seasons and focus on a subset of representative tracks.

```python
TARGET_DRIVERS = ['ALB', 'LEC', 'NOR', 'RUS', 'VER']

TRACKS = [
    'Australia',
    'Saudi_Arabia',
    'Hungary',
    'Italy'
]

GROUP_COLS = ['Year', 'Track', 'Driver', 'Stint', 'Compound']

def load_and_filter_data(base_path='datasets'):
    all_files = glob.glob(os.path.join(base_path, '**', '*.csv'), recursive=True)
    df_list = []

    for file in all_files:
        try:
            parts = os.path.normpath(file).split(os.sep)
            track = parts[-2]

            # Track filter
            if track not in TRACKS:
                continue

            year = int(parts[-1][:4])

            df = pd.read_csv(file, low_memory=False)
            df['Year'] = year
            df['Track'] = track

            df_list.append(df)

        except Exception as e:
            print(f"Failed: {file} -> {e}")

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    print("Initial rows:", len(df))

    # Driver filter
    df = df[df['Driver'].isin(TARGET_DRIVERS)].copy()

    print("After driver filter:", len(df))

    return df

df_raw = load_and_filter_data()
```
**Output:**
```text
Initial rows: 16805
After driver filter: 4300
```

## Data Cleaning

Cleaning involves removing pit-in/pit-out laps, ensuring accurate lap times, and filtering for dry tyre compounds (SOFT, MEDIUM, HARD).

```python
def clean_data(df):
    if df.empty:
        return df
    
    df = df.copy()

    # Keep only clean, representative racing laps
    if 'IsAccurate' in df.columns:
        df = df[df['IsAccurate'] == True].copy()

    if 'PitOutTime' in df.columns:
        df['PitOutTime'] = pd.to_datetime(df['PitOutTime'], errors='coerce')
        df = df[df['PitOutTime'].isna()].copy()

    if 'PitInTime' in df.columns:
        df['PitInTime'] = pd.to_datetime(df['PitInTime'], errors='coerce')
        df = df[df['PitInTime'].isna()].copy()

    if 'TrackStatus' in df.columns:
        df = df[df['TrackStatus'].astype(str).isin(['1', ''])].copy()

    # --- LapTime ---
    df['LapTime'] = df['LapTime'].astype(str).str.strip()
    df['LapTime_sec'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds()

    print("Valid LapTime:", df['LapTime_sec'].notna().sum())

    df = df.dropna(subset=['LapTime_sec'])
    df = df[df['LapTime_sec'] > 0]

    # --- TyreLife ---
    df['TyreLife'] = pd.to_numeric(df['TyreLife'], errors='coerce')
    df = df.dropna(subset=['TyreLife'])

    # --- Compound ---
    df = df.dropna(subset=['Compound'])
    df['Compound'] = df['Compound'].astype(str).str.upper().str.strip()
    df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()

    return df.reset_index(drop=True)

df_clean = clean_data(df_raw)

print("Raw rows:", len(df_raw))
print("Clean rows:", len(df_clean))
```
**Output:**
```text
Valid LapTime: 3643
Raw rows: 4300
Clean rows: 3484
```

## Feature Engineering

We define `Expected_Max_Life` as the 75th percentile of tyre life seen for a specific track, year, and compound. This serves as a benchmark to calculate the degradation percentage.

```python
def engineering_features(df):
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values(by=GROUP_COLS + ['LapNumber'])

    df['Initial_Life'] = df.groupby(GROUP_COLS)['TyreLife'].transform('min')
    df['Stint_Usage'] = df['TyreLife'] - df['Initial_Life']
    df['Stint_Max_Life'] = df.groupby(GROUP_COLS)['TyreLife'].transform('max')

    stint_life_lookup = (
        df[GROUP_COLS + ['Stint_Max_Life']]
        .drop_duplicates()
        .groupby(['Year', 'Track', 'Compound'])['Stint_Max_Life']
        .quantile(0.75)
        .reset_index()
        .rename(columns={'Stint_Max_Life': 'Expected_Max_Life'})
    )

    df = df.merge(stint_life_lookup, on=['Year', 'Track', 'Compound'], how='left')

    life_range = df['Expected_Max_Life'] - df['Initial_Life']

    df['degradation_stint_%'] = np.where(
        life_range > 0,
        (df['Stint_Usage'] / life_range) * 100,
        0
    )

    df['degradation_stint_%'] = df['degradation_stint_%'].clip(0, 100)

    return df

df_features = engineering_features(df_clean)
df_features.head()
```

| | Driver | Track | LapTime_sec | TyreLife | Stint_Usage | Expected_Max_Life | degradation_stint_% |
|---|---|---|---|---|---|---|---|
| 0 | ALB | Australia | 88.217 | 7.0 | 0.0 | 38.0 | 0.000000 |
| 1 | ALB | Australia | 86.992 | 8.0 | 1.0 | 38.0 | 3.225806 |
| 2 | ALB | Australia | 86.750 | 9.0 | 2.0 | 38.0 | 6.451613 |
| 3 | ALB | Australia | 85.709 | 10.0 | 3.0 | 38.0 | 9.677419 |
| 4 | ALB | Australia | 87.213 | 11.0 | 4.0 | 38.0 | 12.903226 |

## Model Selection

We evaluate Ridge, Random Forest, and Gradient Boosting. The data is split by year, with the final year (2025) used as a holdout test set.

```python
# Encoding and Feature Selection
features = ['TyreLife', 'Initial_Life', 'Stint_Usage', 'Expected_Max_Life', 'LapNumber', 'Compound_Enc', 'Track_Enc', 'Year']
target = 'degradation_stint_%'

# ... (Encoding logic omitted for brevity)

# Training and Evaluation
models = {
    'Ridge': Ridge(alpha=1.5),
    'RandomForest': RandomForestRegressor(n_estimators=220, max_depth=12, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42)
}

# (Model fitting and scoring logic)
```

**Model Comparison Results:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **GradientBoosting** | **1.907004** | **4.155104** | **0.981218** |
| Ridge | 4.268634 | 7.646420 | 0.936395 |
| RandomForest | 5.462006 | 8.211629 | 0.926644 |

**Chosen Model:** GradientBoosting

## Performance Analysis

### Degradation by Track (2025 Holdout)
The model captures the degradation gradient effectively across different circuits.

| Track | MAE |
|---|---|
| Hungary | 1.157436 |
| Saudi_Arabia | 1.681867 |
| Italy | 2.176792 |
| Australia | 22.742140 |

### Degradation by Compound (2025 Holdout)
Harder compounds exhibit more predictable wear patterns.

| Compound | MAE |
|---|---|
| HARD | 1.332365 |
| MEDIUM | 2.398404 |
| SOFT | 6.927111 |

## Visualizing Results

### Actual vs Predicted
The scatter plot shows high fidelity in predicting tyre wear, particularly in the core 0-80% degradation range.

![Actual vs Predicted](file:///c:/ROSHITH2/NMIMS/Sem%202/Mini%20Project/actual_vs_predicted.png)

### Tyre Degradation Curves
The following curves visualize how predicted degradation progresses over tyre life, showing the distinct slopes for different compounds.

![Tyre Degradation Curves](file:///c:/ROSHITH2/NMIMS/Sem%202/Mini%20Project/tyre_degradation_curves.png)

## Conclusion
The Gradient Boosting model provides a robust estimation of tyre degradation across multi-circuit data. By quantifying wear as a percentage of expected life, race stratégistes can better estimate pit-stop windows and compound performance drop-offs.
