import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

# Assuming EventBarConfig and build_event_bars are correctly imported
# from the finantradealgo package structure.
# Adjust imports if the structure is different in the actual project.
from finantradealgo.system.config_loader import EventBarConfig
from finantradealgo.data_engine.event_bars import build_event_bars

def load_data(file_path: str) -> pd.DataFrame:
    """Loads OHLCV data from a CSV file."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    return df

def benchmark_bar_mode(df: pd.DataFrame, mode: Literal["time", "volume"], target_value: Optional[float] = None):
    """
    Benchmarks the event bar construction for a given mode.
    """
    print(f"Benchmarking {mode.capitalize()} Bars...")
    
    cfg = EventBarConfig(mode=mode)
    if mode == "volume":
        cfg.target_volume = target_value
    
    start_time = time.perf_counter()
    event_df = build_event_bars(df.copy(), cfg) # Pass a copy to avoid modifying original df
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    bar_count = len(event_df)
    
    # Calculate total duration of the original data for average bar duration
    if not df.empty and not event_df.empty:
        total_data_duration = (df.index.max() - df.index.min()).total_seconds()
        if bar_count > 0:
            avg_bar_duration = total_data_duration / bar_count
        else:
            avg_bar_duration = 0
    else:
        total_data_duration = 0
        avg_bar_duration = 0

    print(f"  Mode: {mode.capitalize()}")
    if target_value:
        print(f"  Target: {target_value}")
    print(f"  Bar Count: {bar_count}")
    print(f"  Runtime: {runtime:.4f} seconds")
    print(f"  Average Bar Duration: {avg_bar_duration:.2f} seconds\n")

    return {
        "mode": mode,
        "target": target_value,
        "bar_count": bar_count,
        "runtime": runtime,
        "avg_bar_duration": avg_bar_duration
    }

if __name__ == "__main__":
    data_file_path = 'data/ohlcv/BTCUSDT_15m.csv' # Assuming this path is correct relative to project root
    
    print(f"Loading data from {data_file_path}...")
    try:
        raw_df = load_data(data_file_path)
        print(f"Raw data loaded: {len(raw_df)} rows from {raw_df.index.min()} to {raw_df.index.max()}\n")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
        exit(1)

    target_volume = 5000 # Example target volume, adjust as needed

    time_results = benchmark_bar_mode(raw_df, "time")
    volume_results = benchmark_bar_mode(raw_df, "volume", target_volume)

    print("-" * 30)
    print("Benchmark Summary:")
    print("-" * 30)
    print(f"Time Bars:")
    print(f"  Bar Count: {time_results['bar_count']}")
    print(f"  Runtime: {time_results['runtime']:.4f} seconds")
    print(f"  Average Bar Duration: {time_results['avg_bar_duration']:.2f} seconds")
    print("\n")
    print(f"Volume Bars (Target: {target_volume}):")
    print(f"  Bar Count: {volume_results['bar_count']}")
    print(f"  Runtime: {volume_results['runtime']:.4f} seconds")
    print(f"  Average Bar Duration: {volume_results['avg_bar_duration']:.2f} seconds")
    print("-" * 30)
