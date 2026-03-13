import fastf1
import os

fastf1.Cache.enable_cache('cache')

years = [2024, 2025]
track = 'Silverstone'
track_dir = 'Great_Britain'

os.makedirs(f'datasets/{track_dir}', exist_ok=True)

for year in years:
    try:
        print(f"Downloading {year} {track}...")
        session = fastf1.get_session(year, track, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        
        file_path = f'datasets/{track_dir}/{year}_{track_dir}_Laps.csv'
        laps.to_csv(file_path, index=False)
        print(f"Successfully saved {file_path}")
    except Exception as e:
        print(f"Failed to download {year} {track}: {e}")
