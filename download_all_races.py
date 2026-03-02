"""
download_all_races.py
---------------------
Batch-downloads lap data for ALL F1 races (2022-2025) by reusing
the download logic from download_race_data.py.

Skips any country whose folder already exists under datasets/.
"""

import os
import fastf1

# Reuse the download functions from the existing script
from download_race_data import download_all_years, download_single_year

# All Grand Prix event names that FastF1 can fuzzy-match
# Covers races from the 2022-2025 calendars
ALL_RACES = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Emilia Romagna",   # Imola
    "Miami",
    "Spain",
    "Monaco",
    "Azerbaijan",
    "Canada",
    "Great Britain",
    "Austria",
    "France",           # 2022 only
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",            # Monza
    "Singapore",
    "Japan",
    "United States",    # COTA
    "Mexico",
    "Brazil",           # São Paulo
    "Abu Dhabi",
    "Qatar",
    "Las Vegas",        # 2023 onwards
    "China",            # 2024 onwards
]

YEARS = [2022, 2023, 2024, 2025]

def main():
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")

    # Enable cache once
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    for country in ALL_RACES:
        safe_name = country.strip().replace(" ", "_")
        country_dir = os.path.join(datasets_dir, safe_name)
        os.makedirs(country_dir, exist_ok=True)

        # Figure out which years are missing
        missing_years = []
        for year in YEARS:
            csv_file = os.path.join(country_dir, f"{year}_{safe_name}_Laps.csv")
            if not os.path.isfile(csv_file):
                missing_years.append(year)

        if not missing_years:
            print(f"\n{'='*55}")
            print(f"  ⏭️  Skipping {country} — all years already downloaded")
            print(f"{'='*55}")
            continue

        print(f"\n{'='*55}")
        print(f"  Downloading {country} GP — missing years: {missing_years}")
        print(f"{'='*55}")

        for year in missing_years:
            download_single_year(year, country, country_dir)


if __name__ == "__main__":
    main()
