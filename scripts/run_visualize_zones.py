"""
A script to generate a visual chart of the market structure zones for sanity checking.
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finantradealgo.features.market_structure_features import (
    add_market_structure_features,
)
from finantradealgo.market_structure.config import MarketStructureConfig


def visualize_zones(
    data_path: str = "data/ohlcv/BTCUSDT_15m.csv",
    output_path: str = "outputs/zones_visualization.png",
    num_bars: int = 1000,
):
    """
    Loads OHLCV data, computes market structure zones, and saves a visualization.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please run `python fetch_binance_data.py` to download sample data.")
        return

    df = df.tail(num_bars).reset_index(drop=True)
    df = df.set_index("timestamp")

    print("Computing market structure features and zones...")
    cfg = MarketStructureConfig()
    cfg.swing.lookback = 10  # A reasonable lookback for visualization
    cfg.zone.min_touches = 2
    cfg.zone.window_bars = 500
    cfg.zone.price_proximity_pct = 0.005 # 0.5%

    df_features, zones = add_market_structure_features(df, cfg)
    
    print(f"Found {len(zones)} zones. Generating plot...")

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 9))

    # Plot price
    ax.plot(df_features.index, df_features["close"], color="black", linewidth=0.8, label="Close Price")
    ax.fill_between(df_features.index, df_features["high"], df_features["low"], color="gray", alpha=0.2, label="High-Low Range")

    # Plot zones
    for zone in zones:
        color = "green" if zone.type == "demand" else "red"
        
        # Get the start and end datetime from the DataFrame's index
        # We need to handle potential out-of-bounds index
        try:
            start_time = df_features.index[zone.first_ts]
            end_time = df_features.index[zone.last_ts]
        except IndexError:
            continue # Skip zones whose timestamps are out of the df slice

        # The width is the duration in days for matplotlib's date formatter
        width = mdates.date2num(end_time) - mdates.date2num(start_time)

        rect = patches.Rectangle(
            (mdates.date2num(start_time), zone.low),
            width,
            zone.high - zone.low,
            linewidth=1,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
        )
        ax.add_patch(rect)

    ax.set_title(f"Market Structure Zones - {os.path.basename(data_path)}")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Improve date formatting on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Successfully saved visualization to {output_path}")


if __name__ == "__main__":
    visualize_zones()
