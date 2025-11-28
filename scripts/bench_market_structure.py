"""
A simple script to benchmark the performance of the Market Structure engine.
"""
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.market_structure_features import add_market_structure_features
from finantradealgo.market_structure.config import MarketStructureConfig

# --- Configuration ---
# Using a larger data file for a more meaningful benchmark
DATA_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m.csv"


def run_benchmark():
    """
    Loads a large OHLCV dataset and measures the execution time of the
    Market Structure feature engine.
    """
    print("--- Market Structure Engine Benchmark ---")

    try:
        print(f"Loading data from: {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        num_bars = len(df)
        if num_bars == 0:
            print("Error: Data file is empty.")
            return
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure you have the necessary data.")
        return

    print(f"Loaded {num_bars} bars.")
    print("Running benchmark...")

    # --- Run the benchmark ---
    cfg = MarketStructureConfig()  # Use default config
    start_time = time.perf_counter()

    # We only want to time the feature calculation
    _, _ = add_market_structure_features(df, cfg)

    end_time = time.perf_counter()
    # --- End of benchmark ---

    total_time_ms = (end_time - start_time) * 1000
    ms_per_bar = total_time_ms / num_bars

    print("\n--- Benchmark Results ---")
    print(f"Total bars processed: {num_bars}")
    print(f"Total execution time: {total_time_ms:.2f} ms")
    print(f"Performance:          {ms_per_bar:.4f} ms per bar")
    print("-------------------------")


if __name__ == "__main__":
    run_benchmark()
