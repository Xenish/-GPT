"""
Debug and visualization helpers for market structure analysis.

Task S1.E3: Provide visual debugging tools for market structure signals.
"""
from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .types import Zone, MarketStructureColumns
from .engine import MarketStructureResult


def plot_market_structure(
    df: pd.DataFrame,
    result: MarketStructureResult,
    title: str = "Market Structure Analysis",
    show_price_smooth: bool = True,
    show_swings: bool = True,
    show_fvg: bool = True,
    show_zones: bool = True,
    show_bos_choch: bool = True,
    show_regime: bool = True,
    figsize: tuple = (16, 10),
) -> None:
    """
    Plot comprehensive market structure visualization.

    Args:
        df: Original OHLCV DataFrame
        result: MarketStructureResult from engine.compute_df()
        title: Plot title
        show_price_smooth: Whether to show smoothed price
        show_swings: Whether to show swing points
        show_fvg: Whether to show Fair Value Gaps
        show_zones: Whether to show supply/demand zones
        show_bos_choch: Whether to show BoS/ChoCh
        show_regime: Whether to show trend and chop regime
        figsize: Figure size (width, height)

    Raises:
        ImportError: If matplotlib is not installed
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    cols = MarketStructureColumns()
    features = result.features

    # Create subplots
    n_subplots = 1
    if show_regime:
        n_subplots += 1

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]

    # Main price chart
    ax_price = axes[0]
    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    # Plot candlesticks (simplified as high-low bars with close markers)
    x = np.arange(len(df))
    ax_price.plot(x, df["close"], label="Close", color="black", linewidth=0.8, alpha=0.7)

    # Plot smoothed price
    if show_price_smooth and cols.price_smooth in features.columns:
        ax_price.plot(
            x,
            features[cols.price_smooth],
            label="Price Smooth",
            color="blue",
            linewidth=1.5,
            alpha=0.8,
        )

    # Plot swing points
    if show_swings:
        if cols.swing_high in features.columns:
            swing_high_idx = features[features[cols.swing_high] == 1].index
            swing_high_pos = [df.index.get_loc(idx) for idx in swing_high_idx]
            swing_high_prices = df.loc[swing_high_idx, "high"].values
            ax_price.scatter(
                swing_high_pos,
                swing_high_prices,
                marker="v",
                color="red",
                s=100,
                label="Swing High",
                zorder=5,
            )

        if cols.swing_low in features.columns:
            swing_low_idx = features[features[cols.swing_low] == 1].index
            swing_low_pos = [df.index.get_loc(idx) for idx in swing_low_idx]
            swing_low_prices = df.loc[swing_low_idx, "low"].values
            ax_price.scatter(
                swing_low_pos,
                swing_low_prices,
                marker="^",
                color="green",
                s=100,
                label="Swing Low",
                zorder=5,
            )

    # Plot FVG (Fair Value Gaps)
    if show_fvg and cols.fvg_up in features.columns and cols.fvg_down in features.columns:
        fvg_up_idx = features[features[cols.fvg_up] == 1].index
        for idx in fvg_up_idx:
            pos = df.index.get_loc(idx)
            # Highlight FVG area (approximate)
            ax_price.axvspan(pos - 0.5, pos + 0.5, alpha=0.2, color="green", label="FVG Up" if idx == fvg_up_idx[0] else "")

        fvg_down_idx = features[features[cols.fvg_down] == 1].index
        for idx in fvg_down_idx:
            pos = df.index.get_loc(idx)
            ax_price.axvspan(pos - 0.5, pos + 0.5, alpha=0.2, color="red", label="FVG Down" if idx == fvg_down_idx[0] else "")

    # Plot supply/demand zones
    if show_zones and result.zones:
        for zone in result.zones:
            start_pos = df.index.get_loc(zone.first_ts) if zone.first_ts in df.index else 0
            end_pos = df.index.get_loc(zone.last_ts) if zone.last_ts in df.index else len(df) - 1

            color = "red" if zone.type == "supply" else "green"
            alpha = 0.15 + (min(zone.strength, 5) * 0.05)  # Stronger zones more opaque

            rect = patches.Rectangle(
                (start_pos, zone.low),
                end_pos - start_pos,
                zone.high - zone.low,
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                label=f"{zone.type.title()} Zone" if zone == result.zones[0] else "",
            )
            ax_price.add_patch(rect)

    # Plot BoS/ChoCh
    if show_bos_choch:
        if cols.bos_up in features.columns:
            bos_up_idx = features[features[cols.bos_up] == 1].index
            for idx in bos_up_idx:
                pos = df.index.get_loc(idx)
                price = df.loc[idx, "high"]
                ax_price.annotate(
                    "BoS↑",
                    xy=(pos, price),
                    xytext=(pos, price * 1.005),
                    fontsize=8,
                    color="green",
                    ha="center",
                )

        if cols.bos_down in features.columns:
            bos_down_idx = features[features[cols.bos_down] == 1].index
            for idx in bos_down_idx:
                pos = df.index.get_loc(idx)
                price = df.loc[idx, "low"]
                ax_price.annotate(
                    "BoS↓",
                    xy=(pos, price),
                    xytext=(pos, price * 0.995),
                    fontsize=8,
                    color="red",
                    ha="center",
                )

        if cols.choch in features.columns:
            choch_idx = features[features[cols.choch] != 0].index
            for idx in choch_idx:
                pos = df.index.get_loc(idx)
                price = df.loc[idx, "close"]
                direction = "↑" if features.loc[idx, cols.choch] > 0 else "↓"
                ax_price.annotate(
                    f"ChoCh{direction}",
                    xy=(pos, price),
                    xytext=(pos, price),
                    fontsize=8,
                    color="orange",
                    ha="center",
                )

    ax_price.legend(loc="upper left", fontsize=8)

    # Regime subplot
    if show_regime and n_subplots > 1:
        ax_regime = axes[1]
        ax_regime.set_ylabel("Regime")
        ax_regime.set_xlabel("Bar Index")
        ax_regime.grid(True, alpha=0.3)

        # Plot trend regime
        if cols.trend_regime in features.columns:
            ax_regime.plot(
                x,
                features[cols.trend_regime],
                label="Trend Regime (1=Up, -1=Down, 0=Neutral)",
                color="blue",
                linewidth=1.5,
            )
            ax_regime.axhline(0, color="gray", linestyle="--", alpha=0.5)

        # Plot chop regime on secondary y-axis
        if cols.chop_regime in features.columns:
            ax_chop = ax_regime.twinx()
            ax_chop.set_ylabel("Chop Score", color="purple")
            ax_chop.plot(
                x,
                features[cols.chop_regime],
                label="Chop Score (0=Trend, 1=Chop)",
                color="purple",
                linewidth=1.5,
                alpha=0.7,
            )
            ax_chop.tick_params(axis="y", labelcolor="purple")
            ax_chop.set_ylim(0, 1)

        ax_regime.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


def summarize_market_structure(
    result: MarketStructureResult,
    df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a text summary of market structure signals.

    Args:
        result: MarketStructureResult from engine.compute_df()
        df: Optional original DataFrame for additional context

    Returns:
        String summary of market structure analysis
    """
    cols = MarketStructureColumns()
    features = result.features

    lines = ["=" * 60]
    lines.append("MARKET STRUCTURE SUMMARY")
    lines.append("=" * 60)

    # Basic stats
    lines.append(f"\nTotal bars: {len(features)}")

    # Swing points
    if cols.swing_high in features.columns:
        n_swing_high = features[cols.swing_high].sum()
        lines.append(f"Swing Highs: {n_swing_high}")

    if cols.swing_low in features.columns:
        n_swing_low = features[cols.swing_low].sum()
        lines.append(f"Swing Lows: {n_swing_low}")

    # Trend regime
    if cols.trend_regime in features.columns:
        current_trend = features[cols.trend_regime].iloc[-1]
        trend_str = "UPTREND" if current_trend > 0 else "DOWNTREND" if current_trend < 0 else "NEUTRAL"
        lines.append(f"\nCurrent Trend: {trend_str} ({current_trend})")

    # Chop regime
    if cols.chop_regime in features.columns:
        current_chop = features[cols.chop_regime].iloc[-1]
        chop_str = "CHOPPY" if current_chop > 0.6 else "TRENDING" if current_chop < 0.4 else "MIXED"
        lines.append(f"Current Chop: {chop_str} ({current_chop:.3f})")

    # FVG stats
    if cols.fvg_up in features.columns and cols.fvg_down in features.columns:
        n_fvg_up = features[cols.fvg_up].sum()
        n_fvg_down = features[cols.fvg_down].sum()
        lines.append(f"\nFVG Up: {n_fvg_up}")
        lines.append(f"FVG Down: {n_fvg_down}")

    # BoS/ChoCh stats
    if cols.bos_up in features.columns and cols.bos_down in features.columns:
        n_bos_up = features[cols.bos_up].sum()
        n_bos_down = features[cols.bos_down].sum()
        lines.append(f"\nBoS Up: {n_bos_up}")
        lines.append(f"BoS Down: {n_bos_down}")

    if cols.choch in features.columns:
        n_choch = (features[cols.choch] != 0).sum()
        lines.append(f"ChoCh: {n_choch}")

    # Zones
    if result.zones:
        supply_zones = [z for z in result.zones if z.type == "supply"]
        demand_zones = [z for z in result.zones if z.type == "demand"]
        lines.append(f"\nSupply Zones: {len(supply_zones)}")
        lines.append(f"Demand Zones: {len(demand_zones)}")

        if supply_zones:
            avg_supply_strength = np.mean([z.strength for z in supply_zones])
            lines.append(f"  Avg Supply Strength: {avg_supply_strength:.2f}")

        if demand_zones:
            avg_demand_strength = np.mean([z.strength for z in demand_zones])
            lines.append(f"  Avg Demand Strength: {avg_demand_strength:.2f}")

    # Price context
    if df is not None and len(df) > 0:
        current_price = df["close"].iloc[-1]
        price_change = df["close"].iloc[-1] - df["close"].iloc[0]
        price_change_pct = (price_change / df["close"].iloc[0]) * 100

        lines.append(f"\nPrice Context:")
        lines.append(f"  Current: {current_price:.2f}")
        lines.append(f"  Change: {price_change:+.2f} ({price_change_pct:+.2f}%)")

    lines.append("=" * 60)

    return "\n".join(lines)


def print_zone_details(zones: List[Zone]) -> None:
    """
    Print detailed information about supply/demand zones.

    Args:
        zones: List of Zone objects
    """
    if not zones:
        print("No zones found.")
        return

    print("=" * 80)
    print("ZONE DETAILS")
    print("=" * 80)
    print(f"{'Type':<8} {'Start':<12} {'End':<12} {'Price Low':<12} {'Price High':<12} {'Strength':<8}")
    print("-" * 80)

    for zone in zones:
        print(
            f"{zone.type.upper():<8} "
            f"{str(zone.first_ts):<12} "
            f"{str(zone.last_ts):<12} "
            f"{zone.low:<12.2f} "
            f"{zone.high:<12.2f} "
            f"{zone.strength:<8.2f}"
        )

    print("=" * 80)
