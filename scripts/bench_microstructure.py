import sys
import time
from pathlib import Path
from typing import Optional

# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.microstructure.engine import compute_microstructure_df


def run_benchmark(file_path: Path):
    """
    Runs a performance benchmark on the microstructure feature computation.
    """
    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return

    print(f"Loading data from {file_path}...")
    df = load_ohlcv_csv(str(file_path))
    if df is None or df.empty:
        print("Failed to load data or data is empty.")
        return

    num_bars = len(df)
    print(f"Loaded {num_bars} bars.")
    print("Starting microstructure benchmark...")

    # --- Time the computation ---
    start_time = time.time()
    ms_df = compute_microstructure_df(df)
    end_time = time.time()
    # --------------------------

    total_time_s = end_time - start_time
    ms_per_bar = (total_time_s * 1000) / num_bars

    print("\n--- Benchmark Results ---")
    print(f"Total bars processed: {num_bars}")
    print(f"Total computation time: {total_time_s:.4f} seconds")
    print(f"Time per 1k bars: {ms_per_bar * 1000:.4f} ms")
    print("-------------------------\n")

    print("Resulting features shape:", ms_df.shape)
    print("Sample of calculated features:")
    print(ms_df.dropna().tail())


def main(symbol: Optional[str] = None, timeframe: Optional[str] = None):
    """
    Main function to select the dataset and run the benchmark.
    """
    resolved_symbol = symbol or "BTCUSDT"
    resolved_timeframe = timeframe or "15m"
    
    file_path = ROOT / "data" / "ohlcv" / f"{resolved_symbol}_{resolved_timeframe}.csv"
    run_benchmark(file_path)


if __name__ == "__main__":
    # You can change the symbol and timeframe here for the benchmark
    # e.g., main(symbol="AIAUSDT", timeframe="15m")
    main()
