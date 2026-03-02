"""
download_all_races.py
---------------------
Batch-downloads lap data for ALL F1 races (2022-2025) by reusing
the download logic from download_race_data.py.

Skips any country whose folder already exists under datasets/.
"""

import os

# Reuse the download function from the existing script
from download_race_data import download_all_years

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

    for country in ALL_RACES:
        safe_name = country.strip().replace(" ", "_")
        country_dir = os.path.join(datasets_dir, safe_name)

        # Skip if already downloaded
        if os.path.isdir(country_dir) and len(os.listdir(country_dir)) > 0:
            print(f"\n{'='*55}")
            print(f"  ⏭️  Skipping {country} — already downloaded")
            print(f"{'='*55}")
            continue

        download_all_years(country, YEARS)


if __name__ == "__main__":
    main()
