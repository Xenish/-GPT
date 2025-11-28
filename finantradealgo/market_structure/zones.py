"""
Functions for identifying supply and demand zones by clustering swing points.
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import ZoneConfig
from .types import SwingPoint, Zone
from .volume_profile import compute_price_volume_profile


def build_zones(
    swings: List[SwingPoint],
    cfg: ZoneConfig,
    bars: pd.DataFrame,
    last_ts: Optional[int] = None,
) -> List[Zone]:
    """
    Builds supply and demand zones by clustering recent swing points and
    weighting them by volume.

    Args:
        swings: A list of all detected swing points.
        cfg: The configuration for zone detection.
        bars: The historical OHLCV data as a DataFrame.
        last_ts: The current timestamp (integer index) to anchor the lookback window.
                 If None, it's taken from the last swing point.

    Returns:
        A list of Zone objects representing supply and demand areas.
    """
    if not swings:
        return []

    if last_ts is None:
        last_ts = swings[-1].ts

    # 1. Filter data to the recent window
    window_start_ts = max(0, last_ts - cfg.window_bars)
    recent_swings = [s for s in swings if s.ts >= window_start_ts]
    window_bars = bars.iloc[window_start_ts:last_ts]

    if window_bars.empty:
        return []

    # 2. Compute volume profile for the window
    volume_profile = compute_price_volume_profile(
        window_bars["close"], window_bars["volume"]
    )
    total_volume_in_window = sum(b.volume for b in volume_profile.bins)

    demand_swings = [s for s in recent_swings if s.kind == "low"]
    supply_swings = [s for s in recent_swings if s.kind == "high"]

    zones = []
    zone_id_counter = 0

    # 3. Cluster swings for each type (demand/supply)
    for swing_list, zone_type in [
        (demand_swings, "demand"),
        (supply_swings, "supply"),
    ]:
        if not swing_list:
            continue

        clusters: List[List[SwingPoint]] = []
        sorted_swings = sorted(swing_list, key=lambda s: s.price)

        for swing in sorted_swings:
            placed = False
            for cluster in clusters:
                avg_price = np.mean([s.price for s in cluster])
                if abs(swing.price / avg_price - 1) < cfg.price_proximity_pct:
                    cluster.append(swing)
                    placed = True
                    break
            if not placed:
                clusters.append([swing])

        # 4. Create Zone objects from valid clusters and add volume strength
        for cluster in clusters:
            if len(cluster) >= cfg.min_touches:
                prices = [s.price for s in cluster]
                timestamps = [s.ts for s in cluster]
                zone_low = min(prices)
                zone_high = max(prices)

                # Calculate volume-based strength
                volume_in_zone = sum(
                    b.volume
                    for b in volume_profile.bins
                    if max(b.price_low, zone_low) < min(b.price_high, zone_high)
                )
                
                volume_strength = 0
                if total_volume_in_window > 0:
                    volume_strength = volume_in_zone / total_volume_in_window
                
                # Strength = touches + normalized volume (0 to 1)
                strength = len(cluster) + volume_strength

                zone = Zone(
                    id=zone_id_counter,
                    type=zone_type,  # type: ignore
                    low=zone_low,
                    high=zone_high,
                    strength=strength,
                    first_ts=min(timestamps),
                    last_ts=max(timestamps),
                )
                zones.append(zone)
                zone_id_counter += 1

    return zones