from typing import Optional

import pandas as pd


def compute_imbalance_from_df(
    book_df: pd.DataFrame,
    depth_levels: int
) -> Optional[pd.Series]:
    """
    Computes order book imbalance from a DataFrame of order book snapshots.

    Imbalance is calculated as: (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    for a given number of depth levels.

    Args:
        book_df: DataFrame containing order book snapshots with columns
                 ['timestamp', 'side', 'level', 'price', 'size'].
        depth_levels: The number of order book levels to include in the calculation.

    Returns:
        A Series of imbalance values indexed by timestamp, or None if the
        input is invalid.
    """
    if book_df is None or book_df.empty:
        return None

    # Filter for the desired depth
    filtered_book = book_df[book_df["level"] < depth_levels].copy()

    # Calculate total volume for each side at each timestamp
    volume_by_side = filtered_book.groupby(["timestamp", "side"])["size"].sum().unstack()

    # Ensure both bid and ask columns exist, fill missing with 0
    if "bid" not in volume_by_side.columns:
        volume_by_side["bid"] = 0
    if "ask" not in volume_by_side.columns:
        volume_by_side["ask"] = 0
    
    volume_by_side = volume_by_side.fillna(0)

    # Calculate imbalance
    total_volume = volume_by_side["bid"] + volume_by_side["ask"]
    
    # Avoid division by zero
    imbalance = (volume_by_side["bid"] - volume_by_side["ask"]) / (total_volume + 1e-9)
    imbalance.name = "imbalance"

    return imbalance


if __name__ == "__main__":
    from finantradealgo.data_engine.orderbook_loader import (
        load_orderbook_snapshots,
    )

    print("--- Testing Imbalance Calculation ---")
    dummy_book_df = load_orderbook_snapshots("DUMMY", "15m")

    if dummy_book_df is not None:
        # Test with depth = 1 (top of the book)
        imbalance_d1 = compute_imbalance_from_df(dummy_book_df, depth_levels=1)
        print("\nImbalance (depth=1):")
        print(imbalance_d1)
        # Expected for first timestamp: (10 - 12) / (10 + 12) = -2 / 22 = -0.0909...
        assert abs(imbalance_d1.iloc[0] - (-0.090909)) < 1e-5
        # Expected for second timestamp: (11 - 13) / (11 + 13) = -2 / 24 = -0.0833...
        assert abs(imbalance_d1.iloc[1] - (-0.083333)) < 1e-5
        print("Depth=1 calculation is correct.")

        # Test with depth = 2 (all data)
        imbalance_d2 = compute_imbalance_from_df(dummy_book_df, depth_levels=2)
        print("\nImbalance (depth=2):")
        print(imbalance_d2)
        # Expected for first timestamp: (10+15 - 12+18) / (25 + 30) = -5 / 55 = -0.0909...
        assert abs(imbalance_d2.iloc[0] - (-0.090909)) < 1e-5
        # Expected for second timestamp: (11+14 - 13+17) / (25 + 30) = -5 / 55 = -0.0909...
        assert abs(imbalance_d2.iloc[1] - (-0.090909)) < 1e-5
        print("Depth=2 calculation is correct.")

    # Test symmetric book -> imbalance should be 0
    symmetric_data = {
        "timestamp": pd.to_datetime(["2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z"]),
        "side": ["bid", "ask"],
        "level": [0, 0],
        "price": [99, 101],
        "size": [10, 10],
    }
    symmetric_df = pd.DataFrame(symmetric_data)
    imbalance_sym = compute_imbalance_from_df(symmetric_df, depth_levels=1)
    print("\nImbalance (symmetric):")
    print(imbalance_sym)
    assert abs(imbalance_sym.iloc[0]) < 1e-9
    print("Symmetric book test passed.")

    # Test bid-heavy book -> positive imbalance
    bid_heavy_data = {
        "timestamp": pd.to_datetime(["2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z"]),
        "side": ["bid", "ask"],
        "level": [0, 0],
        "price": [99, 101],
        "size": [20, 10],
    }
    bid_heavy_df = pd.DataFrame(bid_heavy_data)
    imbalance_bid_heavy = compute_imbalance_from_df(bid_heavy_df, depth_levels=1)
    print("\nImbalance (bid-heavy):")
    print(imbalance_bid_heavy)
    assert imbalance_bid_heavy.iloc[0] > 0
    print("Bid-heavy book test passed.")
