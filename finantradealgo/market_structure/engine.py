"""
The main engine for computing all market structure features.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .breaks import detect_bos_choch
from .config import MarketStructureConfig
from .fvg import detect_fvg_series
from .regime import infer_trend_regime
from .swings import detect_swings
from .zones import build_zones
from .types import Zone


@dataclass
class MarketStructureResult:
    """
    Standardized output from MarketStructureEngine.

    Attributes:
        features: DataFrame with all market structure columns (ms_*)
        zones: List of all active Zone objects (supply/demand)
    """
    features: pd.DataFrame
    zones: List[Zone]


@dataclass
class MarketStructureEngine:
    """
    Orchestrates the calculation of all market structure features.
    """

    cfg: MarketStructureConfig

    def compute_df(self, df: pd.DataFrame) -> MarketStructureResult:
        """
        Computes all market structure features for a given OHLCV DataFrame.

        Args:
            df: Input OHLCV DataFrame, expected to have a DatetimeIndex.

        Returns:
            MarketStructureResult containing:
                - features: DataFrame with all market structure signal columns (ms_*)
                - zones: List of all Zone objects (supply/demand)
        """
        out = pd.DataFrame(index=df.index)

        # --- Sprint 2: Swing and Trend Regime ---
        swings = detect_swings(df["high"], df["low"], self.cfg.swing)
        trend = infer_trend_regime(swings, self.cfg.trend.min_swings)

        sh_indices = [s.ts for s in swings if s.kind == "high"]
        sl_indices = [s.ts for s in swings if s.kind == "low"]
        out["ms_swing_high"] = 0
        out["ms_swing_low"] = 0
        if sh_indices:
            out.iloc[sh_indices, out.columns.get_loc("ms_swing_high")] = 1
        if sl_indices:
            out.iloc[sl_indices, out.columns.get_loc("ms_swing_low")] = 1
        
        out["ms_trend_regime"] = trend

        # --- Sprint 3: FVG, BoS, ChoCh ---
        fvg_up, fvg_down = detect_fvg_series(df, self.cfg.fvg)
        out["ms_fvg_up"] = fvg_up
        out["ms_fvg_down"] = fvg_down

        # Note: detect_bos_choch can be slow on large datasets due to the loop.
        # TODO: Vectorize this calculation if performance becomes an issue.
        bos_up, bos_down, choch = detect_bos_choch(
            df, swings, out["ms_trend_regime"], self.cfg.breaks
        )
        out["ms_bos_up"] = bos_up
        out["ms_bos_down"] = bos_down
        out["ms_choch"] = choch

        # --- Sprint 4: Supply/Demand Zones ---
        # First, build all zones based on the full history of swings
        zones = build_zones(swings, self.cfg.zone, df, last_ts=len(df) - 1)
        
        demand_zones = [z for z in zones if z.type == "demand"]
        supply_zones = [z for z in zones if z.type == "supply"]

        # Create series to hold the strength of the zone at each bar
        # Initialize with 0.0
        demand_strength = pd.Series(0.0, index=df.index)
        supply_strength = pd.Series(0.0, index=df.index)

        # For each zone type, find which bars fall into which zones
        # If a bar falls into multiple zones, we'll keep the one with the highest strength
        for zone_list, strength_series in [
            (demand_zones, demand_strength),
            (supply_zones, supply_strength),
        ]:
            for zone in zone_list:
                # Find all bars where the close price is within the zone
                in_zone_mask = (df["close"] >= zone.low) & (df["close"] <= zone.high)
                
                # Update the series: set strength, but only if the new strength is higher
                current_strength = strength_series[in_zone_mask]
                strength_series[in_zone_mask] = np.maximum(current_strength, zone.strength)

        out["ms_zone_demand"] = demand_strength
        out["ms_zone_supply"] = supply_strength

        return MarketStructureResult(features=out, zones=zones)
