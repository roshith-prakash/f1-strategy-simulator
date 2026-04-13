import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

RACE_LAPS: dict[str, int] = {
    "Australia":    58,
    "Italy":        53,
    "Hungary":      70,
    "Saudi_Arabia": 50,
}

ALLOWED_RACES   = list(RACE_LAPS.keys())
ALLOWED_DRIVERS = ["ALB", "LEC", "NOR", "RUS", "VER"]
COMPOUNDS       = ["HARD", "MEDIUM", "SOFT"]

PITSTOP_TIME: dict[str, float] = {  # pit loss time (seconds) per circuit
    "Australia":    22.0,
    "Italy":        21.0,
    "Hungary":      22.0,
    "Saudi_Arabia": 19.0,
}
DEG_THRESHOLD:    float = 95.0   # % degradation that triggers a pit stop
TOP_N_STRATEGIES: int   = 3      # number of top strategies printed in summary

# Paths to model artefacts — place the 4 .pkl files alongside this script,
# or update these paths to wherever they are stored.
MODEL_DIR          = Path(__file__).parent
LAPTIME_MODEL_PATH = MODEL_DIR / "models/laptime_model.pkl"
LAPTIME_META_PATH  = MODEL_DIR / "models/laptime_metadata.pkl"
TYREDEG_MODEL_PATH = MODEL_DIR / "models/tyre_deg_model.pkl"
TYREDEG_META_PATH  = MODEL_DIR / "models/tyre_deg_metadata.pkl"


# ---------------------------------------------------------------------------
# StrategySimulator
# ---------------------------------------------------------------------------

class StrategySimulator:
    """
    Simulates F1 race strategies for one driver at one track.

    Parameters
    ----------
    driver : str
        Three-letter driver code, e.g. "VER".
    race : str
        Race name, one of: Australia, Italy, Hungary, Saudi_Arabia.
    year : int
        Season year used as a model feature (e.g. 2023, 2024, 2025).
    """

    def __init__(self, driver: str, race: str, year: int) -> None:
        self._validate_inputs(driver, race, year)

        self.driver = driver
        self.race   = race
        self.year   = year

        # Populated after run_race()
        self._top_strategies: list[dict]    = []
        self.best_strategy_laps: pd.DataFrame = pd.DataFrame()

        self._load_models()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, driver: str, race: str, year: int) -> None:
        if driver not in ALLOWED_DRIVERS:
            raise ValueError(
                f"Driver '{driver}' not recognised. Allowed: {ALLOWED_DRIVERS}"
            )
        if race not in ALLOWED_RACES:
            raise ValueError(
                f"Race '{race}' not recognised. Allowed: {ALLOWED_RACES}"
            )
        if not isinstance(year, int) or year < 2000:
            raise ValueError("Year must be an integer >= 2000.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load both models and their metadata from disk."""
        print("Loading models…")
        self.lt_model = joblib.load(LAPTIME_MODEL_PATH)
        self.lt_meta  = joblib.load(LAPTIME_META_PATH)
        self.td_model = joblib.load(TYREDEG_MODEL_PATH)
        self.td_meta  = joblib.load(TYREDEG_META_PATH)

        # Cache feature column order from metadata
        self.lt_feature_cols = self.lt_meta["feature_cols"]
        self.td_feature_cols = self.td_meta["feature_cols"]

        # Pre-encode driver and track — constant for the whole simulation run
        self._driver_enc    = int(self.lt_meta["le_driver"].transform([self.driver])[0])
        self._lt_track_enc  = int(self.lt_meta["le_track"].transform([self.race])[0])
        self._td_track_enc  = int(self.td_meta["le_track"].transform([self.race])[0])

        print("Models loaded successfully.\n")

    # ------------------------------------------------------------------
    # Strategy generation
    # ------------------------------------------------------------------

    def generate_strategies(self) -> list[list[str]]:
        """
        Return all valid 2-stint and 3-stint compound sequences.

        Rules
        -----
        - At least 2 different compounds used across the race.
        - Maximum 3 stints (≤ 2 pit stops).
        - No two consecutive identical compounds.
        """
        strategies: list[list[str]] = []

        # 2-stint (1 pit stop) — all ordered pairs of distinct compounds
        for a, b in itertools.permutations(COMPOUNDS, 2):
            strategies.append([a, b])

        # 3-stint (2 pit stops) — repeated compounds allowed (e.g. SOFT→HARD→SOFT)
        # but never two identical compounds back-to-back
        for combo in itertools.product(COMPOUNDS, repeat=3):
            a, b, c = combo
            if a == b or b == c:
                continue
            if len(set(combo)) < 2:   # must use ≥ 2 different types overall
                continue
            strategies.append(list(combo))

        return strategies

    # ------------------------------------------------------------------
    # Per-lap ML predictions
    # ------------------------------------------------------------------

    def _predict_laptime(
        self,
        lap_number: int,
        tyre_life: int,
        compound: str,
    ) -> float:
        """Return predicted lap time (seconds) for a single lap."""
        row = {
            "Driver_encoded":  self._driver_enc,
            "Track_encoded":   self._lt_track_enc,
            "LapNumber":       lap_number,
            "TyreLife":        tyre_life,
            "Year":            self.year,
            "Compound_HARD":   int(compound == "HARD"),
            "Compound_MEDIUM": int(compound == "MEDIUM"),
            "Compound_SOFT":   int(compound == "SOFT"),
        }
        X = pd.DataFrame([row])[self.lt_feature_cols]
        return float(self.lt_model.predict(X)[0])

    def _predict_tyre_deg(
        self,
        lap_number: int,
        tyre_life: int,
        initial_life: int,
        compound: str,
    ) -> float:
        """Return predicted tyre degradation (%) for the current lap."""
        comp_enc    = int(self.td_meta["le_comp"].transform([compound])[0])
        stint_usage = tyre_life - initial_life

        row = {
            "TyreLife":     tyre_life,
            "Initial_Life": initial_life,
            "Stint_Usage":  stint_usage,
            "LapNumber":    lap_number,
            "Compound_Enc": comp_enc,
            "Track_Enc":    self._td_track_enc,
            "Year":         self.year,
        }
        X = pd.DataFrame([row])[self.td_feature_cols]
        return float(self.td_model.predict(X)[0])

    # ------------------------------------------------------------------
    # Single strategy simulation
    # ------------------------------------------------------------------

    def simulate_strategy(
        self,
        strategy: list[str],
        record_laps: bool = False,
    ) -> dict | None:
        """
        Simulate one full race distance with the given compound sequence.

        Parameters
        ----------
        strategy : list[str]
            Ordered list of compounds, e.g. ["SOFT", "HARD"].
        record_laps : bool
            When True, collect a per-lap data record and include it in the
            returned dict as 'laps_df'. Used for the best-strategy table.

        Returns
        -------
        dict
            Keys: strategy, total_time, n_pitstops, [laps_df if record_laps]
        None
            Returned when compounds run out before the race ends (invalid).
        """
        total_laps = RACE_LAPS[self.race]
        total_time = 0.0
        n_pitstops = 0

        lap_number   = 1
        tyre_life    = 1   # 1-indexed: fresh tyres start at life = 1
        initial_life = 0   # tyre_life value at the start of the current stint
        stint_index  = 0   # which entry in the strategy list we are running
        stint_number = 1   # 1-based stint counter (increments on each pit stop)

        current_compound = strategy[stint_index]

        lap_records: list[dict] = []   # only populated when record_laps=True

        while lap_number <= total_laps:
            # ── 1. Predict lap time ───────────────────────────────────
            lap_time    = self._predict_laptime(lap_number, tyre_life, current_compound)
            total_time += lap_time

            # ── 2. Predict tyre degradation ───────────────────────────
            deg = self._predict_tyre_deg(
                lap_number, tyre_life, initial_life, current_compound
            )

            # ── 3. Record this lap ────────────────────────────────────
            if record_laps:
                lap_records.append(
                    {
                        "Lap":      lap_number,
                        "LapTime":  round(lap_time, 3),
                        "Compound": current_compound,
                        "TyreDeg":  round(deg, 2),
                        "Stint":    stint_number,
                    }
                )

            # ── 4. Pit stop check ─────────────────────────────────────
            # Pit only when: degradation threshold is crossed, laps remain,
            # AND there is a next compound to switch to.
            # If already on the last compound and deg crosses the threshold,
            # push on to the flag — do NOT discard the strategy.
            on_last_compound = (stint_index + 1 >= len(strategy))

            if deg >= DEG_THRESHOLD and lap_number < total_laps and not on_last_compound:
                stint_index      += 1
                stint_number     += 1
                current_compound  = strategy[stint_index]
                tyre_life         = 0   # will become 1 after increment below
                initial_life      = 0
                total_time       += PITSTOP_TIME[self.race]
                n_pitstops       += 1

            lap_number += 1
            tyre_life  += 1

        result: dict = {
            "strategy":   strategy,
            "total_time": total_time,
            "n_pitstops": n_pitstops,
        }
        if record_laps:
            result["laps_df"] = pd.DataFrame(lap_records)

        return result

    # ------------------------------------------------------------------
    # Race optimisation
    # ------------------------------------------------------------------

    def run_race(self) -> None:
        """
        Evaluate all valid strategies for the selected race.

        After this call:
          - self._top_strategies    : ranked list of top-N strategy summary dicts
          - self.best_strategy_laps : DataFrame with per-lap detail of the winner
        """
        strategies  = self.generate_strategies()
        all_results: list[dict] = []

        print(
            f"Simulating {len(strategies)} strategies for "
            f"{self.race} ({self.year}) — driver: {self.driver}…"
        )

        for strategy in strategies:
            outcome = self.simulate_strategy(strategy, record_laps=False)
            if outcome is not None:
                all_results.append(outcome)

        if not all_results:
            print("No valid strategies found for this race.")
            return

        # Sort ascending by total race time
        all_results.sort(key=lambda r: r["total_time"])
        self._top_strategies = all_results[:TOP_N_STRATEGIES]

        # Re-simulate the winner with full lap-by-lap recording
        best_detailed = self.simulate_strategy(
            self._top_strategies[0]["strategy"], record_laps=True
        )
        self.best_strategy_laps = best_detailed["laps_df"]

        print(f"Done — {len(all_results)} valid strategies evaluated.\n")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_results(self) -> None:
        """Print the strategy leaderboard and the best-strategy lap table."""
        if not self._top_strategies:
            print("No results yet. Call run_race() first.")
            return

        best = self._top_strategies[0]

        # ── Strategy leaderboard ─────────────────────────────────────
        print("=" * 62)
        print("         F1 RACE STRATEGY OPTIMISATION RESULTS")
        print("=" * 62)
        print(
            f"  Driver : {self.driver}    "
            f"Race : {self.race}    "
            f"Year : {self.year}"
        )
        print("=" * 62)

        for rank, result in enumerate(self._top_strategies, start=1):
            label        = "★ BEST" if rank == 1 else f"  #{rank} "
            compound_str = " → ".join(result["strategy"])
            total_time   = result["total_time"]
            mins, secs   = divmod(total_time, 60)
            print(
                f"  {label}  {compound_str:<26}"
                f"  {int(mins)}m {secs:05.2f}s  ({total_time:.2f} s)"
                f"  |  Pit Stops: {result['n_pitstops']}"
            )

        # ── Per-lap detail for the best strategy ─────────────────────
        print(f"\n{'─' * 62}")
        print(
            f"  Best Strategy Lap Table : "
            f"{' → '.join(best['strategy'])}"
            f"  ({best['n_pitstops']} pit stop(s))"
        )
        print(f"{'─' * 62}")

        # Format numeric columns for display
        df_display = self.best_strategy_laps.copy()
        df_display["LapTime"] = df_display["LapTime"].map(lambda t: f"{t:.3f}s")
        df_display["TyreDeg"] = df_display["TyreDeg"].map(lambda d: f"{d:.2f}%")
        print(df_display.to_string(index=False))
        print(f"{'=' * 62}\n")

    def save_lap_table(
        self, filepath: str | Path = "best_strategy_laps.csv"
    ) -> None:
        """
        Save the best-strategy lap-by-lap table to a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Destination path (default: 'best_strategy_laps.csv').
        """
        if self.best_strategy_laps.empty:
            print("No lap data available. Call run_race() first.")
            return

        out = Path(filepath)
        self.best_strategy_laps.to_csv(out, index=False)
        print(f"Lap table saved to: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── CONFIGURE YOUR RUN HERE ──────────────────────────────────────────
    DRIVER = "VER"
    RACE   = "Saudi_Arabia"          # Australia | Italy | Hungary | Saudi_Arabia
    YEAR   = 2025
    # ─────────────────────────────────────────────────────────────────────

    simulator = StrategySimulator(driver=DRIVER, race=RACE, year=YEAR)
    simulator.run_race()
    simulator.get_results()

    # Optional: save the best-strategy lap table to CSV
    # simulator.save_lap_table(f"{RACE}_{YEAR}_{DRIVER}_best_strategy.csv")