"""
Functions for computing a simple volume profile based on price and volume data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .types import VolumeProfile, VolumeProfileBin


def compute_price_volume_profile(
    prices: pd.Series, volumes: pd.Series, n_bins: int = 50
) -> VolumeProfile:
    """
    Computes a simple, volume-weighted price histogram.

    Args:
        prices: Series of closing prices.
        volumes: Series of volumes corresponding to the prices.
        n_bins: The number of price bins to create.

    Returns:
        A VolumeProfile object containing the histogram and Point of Control.
    """
    if prices.empty or volumes.empty:
        # Create a dummy profile to avoid errors downstream
        empty_bin = VolumeProfileBin(0, 0, 0)
        return VolumeProfile(bins=[], poc_bin=empty_bin)

    price_range = prices.max() - prices.min()
    bin_size = price_range / n_bins
    
    # If there is no price variation, just create one bin
    if bin_size == 0:
        single_bin = VolumeProfileBin(prices.min(), prices.max(), volumes.sum())
        return VolumeProfile(bins=[single_bin], poc_bin=single_bin)

    # `cut` is a great tool for this, but we need to create the bins manually
    # to handle the volumes correctly.
    bins = np.linspace(prices.min(), prices.max(), n_bins + 1)
    
    # Add a small epsilon to the last bin to include the max price
    bins[-1] = bins[-1] + 1e-6

    # Assign each price to a bin index
    binned_prices = pd.cut(prices, bins=bins, right=False, labels=False)

    # Group volumes by the price bin and sum them up
    volume_by_bin = volumes.groupby(binned_prices).sum()

    profile_bins: List[VolumeProfileBin] = []
    for i in range(n_bins):
        vol = volume_by_bin.get(i, 0.0)
        profile_bin = VolumeProfileBin(
            price_low=bins[i], price_high=bins[i + 1], volume=vol
        )
        profile_bins.append(profile_bin)

    # Find the Point of Control (POC) - the bin with the most volume
    if not profile_bins:
        # Fallback for safety, though it shouldn't happen with the guards above
        empty_bin = VolumeProfileBin(0, 0, 0)
        return VolumeProfile(bins=[], poc_bin=empty_bin)
        
    poc_bin = max(profile_bins, key=lambda b: b.volume)

    return VolumeProfile(bins=profile_bins, poc_bin=poc_bin)
