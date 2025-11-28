from pathlib import Path
from typing import Optional

import pandas as pd

# Define base directories for the data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRADES_DIR = DATA_DIR / "trades"
ORDERBOOK_DIR = DATA_DIR / "orderbook"


def load_trades(
    symbol: str, timeframe: str, trades_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Loads historical trade data from a CSV file into a DataFrame.

    The CSV is expected to have columns: timestamp, side, price, size.

    Args:
        symbol: The trading symbol (e.g., 'BTCUSDT').
        timeframe: The timeframe (e.g., '15m').
        trades_dir: The directory to load from. Defaults to 'data/trades'.

    Returns:
        A DataFrame with trade data, or None if the file is not found.
    """
    base_dir = trades_dir or TRADES_DIR
    file_path = base_dir / f"{symbol}_{timeframe}_trades.csv"

    if not file_path.exists():
        print(f"Trade data file not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        return df
    except Exception as e:
        print(f"Error loading trade data from {file_path}: {e}")
        return None


def load_orderbook_snapshots(
    symbol: str, timeframe: str, orderbook_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Loads historical order book snapshot data from a CSV file.

    The CSV is expected to have columns: timestamp, side, level, price, size.

    Args:
        symbol: The trading symbol (e.g., 'BTCUSDT').
        timeframe: The timeframe (e.g., '15m').
        orderbook_dir: The directory to load from. Defaults to 'data/orderbook'.

    Returns:
        A DataFrame with order book data, or None if the file is not found.
    """
    base_dir = orderbook_dir or ORDERBOOK_DIR
    file_path = base_dir / f"{symbol}_{timeframe}_book.csv"

    if not file_path.exists():
        print(f"Order book data file not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        return df
    except Exception as e:
        print(f"Error loading order book data from {file_path}: {e}")
        return None


if __name__ == "__main__":
    # Simple test to verify loaders are working with dummy data
    print("--- Testing Trade Loader ---")
    dummy_trades = load_trades("DUMMY", "15m")
    if dummy_trades is not None:
        print(f"Loaded {len(dummy_trades)} trades.")
        print(dummy_trades.head())
        # Acceptance criteria check:
        assert len(dummy_trades) == 4
        print("Trade loader acceptance criteria met.")

    print("\n--- Testing Order Book Loader ---")
    dummy_book = load_orderbook_snapshots("DUMMY", "15m")
    if dummy_book is not None:
        print(f"Loaded {len(dummy_book)} order book rows.")
        print(dummy_book.head())
        # Acceptance criteria check:
        assert len(dummy_book) == 8
        print("Order book loader acceptance criteria met.")

    print("\n--- Testing with non-existent file ---")
    non_existent = load_trades("NONEXISTENT", "1h")
    assert non_existent is None
    print("Correctly handled non-existent file.")
