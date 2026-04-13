import pandas as pd
from simulation_engine import StrategySimulator, ALLOWED_RACES, ALLOWED_DRIVERS
import warnings

# Suppress pandas performance warnings if any
warnings.filterwarnings("ignore")

def run_bulk_validation(year=2024):
    results = []
    
    total_runs = len(ALLOWED_RACES) * len(ALLOWED_DRIVERS)
    current_run = 0
    
    print("=" * 75)
    print(f"         F1 STRATEGY SIMULATOR: PROJECT-WIDE VALIDATION ({year})")
    print("=" * 75)
    print(f" Total Scenarios: {total_runs}")
    print("-" * 75)
    print(f"{'Race':<18} | {'Driver':<6} | {'Pred (s)':<10} | {'Act (s)':<10} | {'Error %':<8}")
    print("-" * 75)
    
    for race in ALLOWED_RACES:
        for driver in ALLOWED_DRIVERS:
            current_run += 1
            try:
                # Suppress simulator prints for cleaner bulk view
                import sys, os
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                try:
                    simulator = StrategySimulator(driver=driver, race=race, year=year)
                    simulator.run_race()
                    
                    # Restore stdout
                    sys.stdout.close()
                    sys.stdout = old_stdout
                    
                    # Get best predicted strategy
                    if not simulator._top_strategies:
                        print(f"{race:<18} | {driver:<6} | {'NO STRAT':<10} | {'-':<10} | {'-':<8}")
                        continue
                        
                    best_pred = simulator._top_strategies[0]["total_time"]
                    
                    # Get actual historical strategy
                    actual_res = simulator.get_actual_strategy()
                    if not actual_res:
                        print(f"{race:<18} | {driver:<6} | {best_pred:<10.2f} | {'NO DATA':<10} | {'-':<8}")
                        continue
                        
                    actual_time = actual_res["total_time"]
                    
                    # Calculate metrics
                    diff = abs(best_pred - actual_time)
                    error_pct = (diff / actual_time) * 100
                    
                    results.append({
                        "Race": race,
                        "Driver": driver,
                        "Predicted_Time": round(best_pred, 2),
                        "Actual_Time": round(actual_time, 2),
                        "Difference": round(diff, 2),
                        "Error_Pct": round(error_pct, 4)
                    })
                    
                    print(f"{race:<18} | {driver:<6} | {best_pred:<10.2f} | {actual_time:<10.2f} | {error_pct:>7.2f}%")
                
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"{race:<18} | {driver:<6} | {'ERR_RUN':<10} | {'-':<10} | {'-':<8} ({str(e)[:20]})")
                
            except Exception as e:
                print(f"{race:<18} | {driver:<6} | {'ERR_INIT':<10} | {'-':<10} | {'-':<8}")
                
    if not results:
        print("\nNo valid validation data points found. Check your datasets directory.")
        return

    # Grand Average Calculation
    df = pd.DataFrame(results)
    full_avg = df["Error_Pct"].mean()
    
    # Cleaned Mean (Exclude DNFs where error > 50%)
    cleaned_df = df[df["Error_Pct"] <= 50.0]
    cleaned_avg = cleaned_df["Error_Pct"].mean()
    mae_seconds = (cleaned_df["Pred_s"] - cleaned_df["Act_s"]).abs().mean()
    
    print("-" * 75)
    print(f" FULL GRAND AVERAGE ERROR  : {full_avg:.4f}%")
    print(f" CLEANED MEAN ERROR (FINISHERS ONLY): {cleaned_avg:.4f}% (~{mae_seconds:.2f} s absolute)")
    print("=" * 75)
    
    # Save Results
    df["Status"] = df["Error_Pct"].apply(lambda x: "PASS" if x <= 50.0 else "OUTLIER/DNF")
    df.to_csv("validation_results.csv", index=False)
    print(f"\n[INFO] Detailed scenario report saved to: validation_results.csv\n")

if __name__ == "__main__":
    run_bulk_validation()
