from flask import Flask, render_template, request, jsonify
from simulation_engine import StrategySimulator, ALLOWED_RACES, ALLOWED_DRIVERS
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", races=ALLOWED_RACES, drivers=ALLOWED_DRIVERS)

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.json
    driver = data.get("driver")
    race = data.get("race")
    year = int(data.get("year", 2025))

    try:
        simulator = StrategySimulator(driver=driver, race=race, year=year)
        simulator.run_race()
        
        # Prepare leaderboard
        leaderboard = []
        for res in simulator._top_strategies:
            leaderboard.append({
                "strategy": " → ".join(res["strategy"]),
                "total_time": res["total_time"],
                "total_time_str": f"{int(res['total_time'] // 60)}m {res['total_time'] % 60:05.2f}s",
                "n_pitstops": res["n_pitstops"]
            })
            
        # Prepare best strategy details
        laps_df = simulator.best_strategy_laps.copy()
        
        # Convert to list of dicts for JSON
        laps_data = laps_df.to_dict(orient="records")
        
        return jsonify({
            "success": True,
            "leaderboard": leaderboard,
            "best_laps": laps_data,
            "config": {
                "driver": driver,
                "race": race,
                "year": year
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
