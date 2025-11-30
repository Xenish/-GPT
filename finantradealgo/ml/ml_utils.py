"""
ML utilities for target configuration and batch processing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def get_ml_targets(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract ML target symbol/timeframe combinations from config.

    The ml.targets configuration allows limiting ML training to specific
    symbol/timeframe combinations instead of processing all combinations.

    Args:
        cfg: System configuration dict

    Returns:
        List of (symbol, timeframe) tuples to process

    Example config:
        ```yaml
        ml:
          targets:
            - symbol: "BTCUSDT"
              timeframe: "15m"
            - symbol: "AIAUSDT"
              timeframe: "15m"
        ```

    Fallback:
        If ml.targets is empty or not specified, returns the first
        symbol and first timeframe from data.symbols and data.timeframes.
    """
    ml_cfg = cfg.get("ml", {}) or {}
    targets = ml_cfg.get("targets", [])

    # If targets explicitly configured, use them
    if targets:
        result = []
        for target in targets:
            symbol = target.get("symbol")
            timeframe = target.get("timeframe")
            if symbol and timeframe:
                result.append((symbol, timeframe))
            else:
                logger.warning(f"Invalid ML target: {target} (missing symbol or timeframe)")

        if result:
            logger.info(f"Using {len(result)} ML targets from config: {result}")
            return result

    # Fallback: use first symbol/timeframe from data config
    data_cfg = cfg.get("data_cfg")
    if data_cfg:
        symbols = data_cfg.symbols
        timeframes = data_cfg.timeframes
    else:
        data_section = cfg.get("data", {}) or {}
        symbols = data_section.get("symbols", [])
        timeframes = data_section.get("timeframes", [])

    # Additional fallback to legacy single symbol/timeframe
    if not symbols:
        symbols = [cfg.get("symbol", "BTCUSDT")]
    if not timeframes:
        timeframes = [cfg.get("timeframe", "15m")]

    default_target = [(symbols[0], timeframes[0])]
    logger.info(
        f"No ML targets configured, using default: {default_target[0]} "
        f"(first symbol/timeframe from config)"
    )
    return default_target


def is_ml_enabled(cfg: Dict[str, Any]) -> bool:
    """
    Check if ML is enabled in configuration.

    Args:
        cfg: System configuration dict

    Returns:
        True if ML is enabled, False otherwise
    """
    ml_cfg = cfg.get("ml", {}) or {}
    return ml_cfg.get("enabled", True)  # Default to enabled for backward compatibility
