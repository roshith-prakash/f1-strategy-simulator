"""
download_race_data.py
---------------------
Downloads lap data for a specific F1 race across all years (2022–2025)
using the FastF1 library and saves each year as a CSV inside a dedicated
subfolder under `datasets/`.

Folder structure created:
    datasets/
    └── Bahrain/
        ├── 2022_Bahrain_Laps.csv
        ├── 2023_Bahrain_Laps.csv
        ├── 2024_Bahrain_Laps.csv
        └── 2025_Bahrain_Laps.csv

The race is identified by country/event name (e.g. "Spain", "Bahrain").
FastF1 will fuzzy-match the name against the official event schedule.

Usage:
    python download_race_data.py --country "Bahrain"
    python download_race_data.py --country "Spain" --years 2022 2023 2024 2025

    OR edit the CONFIG section below and run:
    python download_race_data.py
"""

import os
import ssl
import warnings
import argparse

# ── Suppress SSL warnings before anything else ─────────────────────────────
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Disable SSL verification globally for requests ─────────────────────────
# Needed because the F1 live-timing API SSL certificate cannot be verified
# on some corporate / university networks.
import requests
from requests import Session as _ReqSession
_orig_request = _ReqSession.request
def _no_verify_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)
_ReqSession.request = _no_verify_request

# ── Now import FastF1 (it uses requests internally) ────────────────────────
import fastf1
import pandas as pd


# ===========================================================
# ⚙️  CONFIG — edit these if you don't want to use CLI args
# ===========================================================
DEFAULT_COUNTRY = "Bahrain"                 # Country / event name (FastF1 fuzzy-matches this)
DEFAULT_YEARS   = [2022, 2023, 2024, 2025]  # Years to download
# ===========================================================


def download_single_year(year: int, country: str, output_dir: str) -> bool:
    """
    Download lap data for one year of a race and save it to output_dir.

    Returns True on success, False if the race could not be loaded.

    Parameters
    ----------
    year       : F1 season year (e.g. 2023)
    country    : Country or event name (e.g. "Bahrain")
    output_dir : Directory to save the CSV into
    """
    print(f"\n  [fastf1] Loading {year} {country} Grand Prix — Race session ...")
    try:
        session = fastf1.get_session(year, country, "R")
        # Skip telemetry/weather to speed things up - we only need laps
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  ⚠️  Could not load {year} {country}: {e}")
        return False

    event_name = session.event["EventName"]
    print(f"  [fastf1] Session loaded: {event_name} {year}")

    # Extract lap data — FastF1 v3.x can raise DataNotLoadedError when some
    # ancillary API endpoints are unavailable; fall back to _laps if needed.
    try:
        laps = session.laps.copy()
    except fastf1.exceptions.DataNotLoadedError:
        if hasattr(session, "_laps") and session._laps is not None:
            warnings.warn(
                f"  {year}: session.laps raised DataNotLoadedError; "
                "using session._laps fallback.",
                stacklevel=2,
            )
            laps = pd.DataFrame(session._laps).copy()
        else:
            print(f"  ⚠️  Laps data unavailable for {year} {country} (DataNotLoadedError).")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not retrieve laps for {year} {country}: {e}")
        return False

    print(f"  [data]   Total laps fetched: {len(laps)}")

    # Convert timedelta / datetime columns to strings so they round-trip
    # cleanly through CSV (same format that the existing datasets use).
    for col in laps.select_dtypes(include=["timedelta64[ns]"]).columns:
        laps[col] = laps[col].astype(str)
    for col in laps.select_dtypes(include=["datetime64[ns]"]).columns:
        laps[col] = laps[col].astype(str)

    # Save CSV
    safe_name = country.strip().replace(" ", "_")
    filename  = f"{year}_{safe_name}_Laps.csv"
    filepath  = os.path.join(output_dir, filename)
    laps.to_csv(filepath, index=False)
    print(f"  ✅  Saved {len(laps)} laps → {filepath}")
    return True


def download_all_years(country: str, years: list) -> None:
    """
    Download race lap data for all specified years and save each to:
        datasets/<Country>/<year>_<Country>_Laps.csv

    Parameters
    ----------
    country : Country or event name (e.g. "Bahrain", "Great Britain")
    years   : List of season years to download
    """

    # ----------------------------------------------------------
    # 1. Enable cache so repeated runs don't re-download data
    # ----------------------------------------------------------
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"[cache]  Using cache directory: {cache_dir}")

    # ----------------------------------------------------------
    # 2. Create subfolder: datasets/<Country>/
    # ----------------------------------------------------------
    safe_name  = country.strip().replace(" ", "_")
    output_dir = os.path.join(os.path.dirname(__file__), "datasets", safe_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[output] Saving to: {output_dir}")

    # ----------------------------------------------------------
    # 3. Loop over all years
    # ----------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Downloading {country} GP — years: {years}")
    print(f"{'='*55}")

    results = {}
    for year in years:
        ok = download_single_year(year, country, output_dir)
        results[year] = "✅ OK" if ok else "⚠️  Skipped"

    # ----------------------------------------------------------
    # 4. Summary
    # ----------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Download Summary — {country} GP")
    print(f"{'='*55}")
    for year, status in results.items():
        print(f"  {year}: {status}")
    print(f"\n  Files saved in: {output_dir}")


# ===========================================================
# CLI entry point
# ===========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download F1 race lap data (all years) and save to datasets/<Country>/"
    )
    parser.add_argument(
        "--country", type=str, default=DEFAULT_COUNTRY,
        help=f'Country / event name — FastF1 fuzzy-matches this (default: "{DEFAULT_COUNTRY}")'
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=DEFAULT_YEARS,
        help=f"Years to download, space-separated (default: {DEFAULT_YEARS})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_all_years(
        country = args.country,
        years   = args.years,
    )
