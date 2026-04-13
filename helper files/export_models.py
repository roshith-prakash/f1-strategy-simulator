import os
import glob
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Common Config ---
TARGET_DRIVERS = ['ALB', 'LEC', 'NOR', 'RUS', 'VER']
TRACKS = ['Australia', 'Saudi_Arabia', 'Hungary', 'Italy']
DATASETS_PATH = 'datasets'
MODELS_DIR = 'models'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_all_data():
    df_list = []
    for track in TRACKS:
        files = glob.glob(os.path.join(DATASETS_PATH, track, '*_Laps.csv'))
        for f in files:
            temp_df = pd.read_csv(f, low_memory=False)
            year = int(os.path.basename(f).split('_')[0])
            temp_df['Year'] = year
            temp_df['Track'] = track
            df_list.append(temp_df)
    return pd.concat(df_list, ignore_index=True)

# ==========================================
# 1. Laptime Model Export
# ==========================================
def export_laptime_model(df_raw):
    print("Training Laptime Model...")
    df = df_raw.copy()
    df = df.replace("NaT", np.nan)
    
    # Filtering
    df = df[df['Driver'].isin(TARGET_DRIVERS)].copy()
    if df['LapTime'].dtype == 'object':
        df['LapTime'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    df = df.dropna(subset=['LapTime'])
    df = df[df['IsAccurate'] == True]
    df = df[df['PitOutTime'].isna()]
    df = df[df['PitInTime'].isna()]
    df = df[df['LapNumber'] > 1]
    df['TrackStatus'] = df['TrackStatus'].astype(str).str.replace('.0', '', regex=False)
    df = df[df['TrackStatus'] == '1']
    df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    df = df.dropna(subset=['Compound'])

    # Encoding
    le_driver = LabelEncoder()
    df['Driver_encoded'] = le_driver.fit_transform(df['Driver'])
    le_track = LabelEncoder()
    df['Track_encoded'] = le_track.fit_transform(df['Track'])
    
    compound_dummies = pd.get_dummies(df['Compound'], prefix='Compound')
    df = pd.concat([df, compound_dummies], axis=1)
    
    feature_cols = ['Driver_encoded', 'Track_encoded', 'LapNumber', 'TyreLife', 'Year'] + \
                   [c for c in df.columns if c.startswith('Compound_')]
    target_col = 'LapTime'
    
    X = df[feature_cols]
    y = df[target_col]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X, y)
    
    # Export
    joblib.dump(model, os.path.join(MODELS_DIR, 'laptime_model.pkl'))
    metadata = {
        'le_driver': le_driver,
        'le_track': le_track,
        'feature_cols': feature_cols
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'laptime_metadata.pkl'))
    print("Done: Laptime model exported.")

# ==========================================
# 2. Tyre Degradation Model Export
# ==========================================
def export_tyre_deg_model(df_raw):
    print("Training Tyre Deg Model...")
    df = df_raw.copy()
    
    # Cleaning
    df = df[df['Driver'].isin(TARGET_DRIVERS)].copy()
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

    df['LapTime_sec'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds()
    df = df.dropna(subset=['LapTime_sec', 'TyreLife', 'Compound'])
    df['Compound'] = df['Compound'].astype(str).str.upper().str.strip()
    df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()

    # Feature Engineering
    group_cols = ['Year', 'Track', 'Driver', 'Stint', 'Compound']
    df = df.sort_values(by=group_cols + ['LapNumber'])
    df['Initial_Life'] = df.groupby(group_cols)['TyreLife'].transform('min')
    df['Stint_Usage'] = df['TyreLife'] - df['Initial_Life']
    
    stint_life_lookup = (
        df[group_cols + ['TyreLife']]
        .groupby(['Year', 'Track', 'Compound'])['TyreLife']
        .quantile(0.75)
        .reset_index()
        .rename(columns={'TyreLife': 'Expected_Max_Life'})
    )
    df = df.merge(stint_life_lookup, on=['Year', 'Track', 'Compound'], how='left')
    life_range = df['Expected_Max_Life'] - df['Initial_Life']
    df['degradation_stint_%'] = np.where(life_range > 0, (df['Stint_Usage'] / life_range) * 100, 0)
    df['degradation_stint_%'] = df['degradation_stint_%'].clip(0, 100)

    # Encoding
    le_comp = LabelEncoder()
    df['Compound_Enc'] = le_comp.fit_transform(df['Compound'])
    le_track = LabelEncoder()
    df['Track_Enc'] = le_track.fit_transform(df['Track'])

    feature_cols = ['TyreLife', 'Initial_Life', 'Stint_Usage', 'LapNumber', 'Compound_Enc', 'Track_Enc', 'Year']
    target = 'degradation_stint_%'
    
    df_model = df.dropna(subset=feature_cols + [target])
    X = df_model[feature_cols]
    y = df_model[target]

    model = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=4,
        subsample=0.85,
        random_state=42
    )
    model.fit(X, y)
    
    # Export
    joblib.dump(model, os.path.join(MODELS_DIR, 'tyre_deg_model.pkl'))
    metadata = {
        'le_comp': le_comp,
        'le_track': le_track,
        'feature_cols': feature_cols
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'tyre_deg_metadata.pkl'))
    print("Done: Tyre degradation model exported.")

if __name__ == "__main__":
    raw_data = load_all_data()
    export_laptime_model(raw_data)
    export_tyre_deg_model(raw_data)
