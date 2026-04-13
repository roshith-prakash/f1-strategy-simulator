import joblib
import pandas as pd
import numpy as np
import os

MODELS_DIR = 'models'

def run_combined_example():
    print("--- F1 Model Prediction Example (Internal Tyre Deg Logic) ---")
    
    # Common Input Data
    driver = "VER"
    track = "Italy"
    lap_number = 25
    tyre_life = 10
    year = 2025
    compound = "MEDIUM" # Try changing this to SOFT or MEDIUM
    
    initial_life = 0
    stint_usage = tyre_life - initial_life
    
    print(f"Scenario: {driver} at {track}, Year {year}")
    print(f"Details: Lap {lap_number}, {compound} Tyres (Age: {tyre_life} laps)\n")

    # 1. Load Models and Metadata
    laptime_model = joblib.load(os.path.join(MODELS_DIR, 'laptime_model.pkl'))
    laptime_meta = joblib.load(os.path.join(MODELS_DIR, 'laptime_metadata.pkl'))
    
    tyre_deg_model = joblib.load(os.path.join(MODELS_DIR, 'tyre_deg_model.pkl'))
    tyre_deg_meta = joblib.load(os.path.join(MODELS_DIR, 'tyre_deg_metadata.pkl'))
    
    # 2. Laptime Prediction
    le_driver = laptime_meta['le_driver']
    le_track_lap = laptime_meta['le_track']
    lap_feature_cols = laptime_meta['feature_cols']
    
    driver_encoded = le_driver.transform([driver])[0]
    track_encoded_lap = le_track_lap.transform([track])[0]
    
    lap_input = {
        'Driver_encoded': driver_encoded,
        'Track_encoded': track_encoded_lap,
        'LapNumber': lap_number,
        'TyreLife': tyre_life,
        'Year': year,
        'Compound_HARD': 1 if compound == "HARD" else 0,
        'Compound_MEDIUM': 1 if compound == "MEDIUM" else 0,
        'Compound_SOFT': 1 if compound == "SOFT" else 0
    }
    X_lap = pd.DataFrame([lap_input])[lap_feature_cols]
    lap_pred = laptime_model.predict(X_lap)[0]

    # 3. Tyre Degradation Prediction
    le_comp = tyre_deg_meta['le_comp']
    le_track_deg = tyre_deg_meta['le_track']
    deg_feature_cols = tyre_deg_meta['feature_cols']
    
    track_encoded_deg = le_track_deg.transform([track])[0]
    compound_encoded = le_comp.transform([compound])[0]
    
    deg_input = {
        'TyreLife': tyre_life,
        'Initial_Life': initial_life,
        'Stint_Usage': stint_usage,
        'LapNumber': lap_number,
        'Compound_Enc': compound_encoded,
        'Track_Enc': track_encoded_deg,
        'Year': year
    }
    X_deg = pd.DataFrame([deg_input])[deg_feature_cols]
    deg_pred = tyre_deg_model.predict(X_deg)[0]

    # 4. Results
    print(f"Results for this stint:")
    print(f">> Predicted Lap Time    : {lap_pred:.3f} s")
    print(f">> Predicted Tyre Deg (%) : {deg_pred:.2f}%")
    print("-" * 50)
    print("Note: The model now determines degradation internally based on")
    print("the track and compound, without requiring external baselines.")


if __name__ == "__main__":
    try:
        run_combined_example()
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure you have run 'python export_models.py' first.")
