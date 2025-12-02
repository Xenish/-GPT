"""
Configuration validation for research mode.

This module provides validation functions to ensure safe research operations
and prevent accidental live trading when running research jobs.
"""
from __future__ import annotations

from typing import Any, Dict

from finantradealgo.system.config_loader import ResearchConfig


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def validate_research_config(sys_cfg: Dict[str, Any]) -> None:
    """
    Validate research configuration for safety and correctness.

    Args:
        sys_cfg: System configuration dictionary

    Raises:
        ConfigValidationError: If validation fails

    Validation checks:
    1. Exchange type must be 'backtest' or 'mock' (not live)
    2. Mode must be 'research'
    3. All strategies in strategy_universe must exist in STRATEGY_REGISTRY
    4. max_parallel_jobs must be positive
    """
    # Check 1: Exchange type must be backtest/mock
    exchange_cfg = sys_cfg.get("exchange", {})
    exchange_type = exchange_cfg.get("type", "").lower()

    if exchange_type not in ("backtest", "mock"):
        raise ConfigValidationError(
            f"Research mode requires exchange.type to be 'backtest' or 'mock', "
            f"got '{exchange_type}'. "
            f"This prevents accidental live trading during research operations."
        )

    # Check 2: Mode must be research
    mode = sys_cfg.get("mode", "").lower()
    if mode != "research":
        raise ConfigValidationError(
            f"Expected mode='research', got mode='{mode}'. "
            f"Use config/system.research.yml for research operations."
        )

    # Check 3: Validate research_cfg
    research_cfg = sys_cfg.get("research_cfg")
    if not isinstance(research_cfg, ResearchConfig):
        raise ConfigValidationError(
            f"research_cfg must be a ResearchConfig instance, got {type(research_cfg)}"
        )

    # Check 4: Validate strategy universe
    if research_cfg.strategy_universe:
        # Import here to avoid circular dependency
        try:
            from finantradealgo.strategies.strategy_engine import STRATEGY_REGISTRY
        except ImportError:
            # If registry not available, skip this check
            pass
        else:
            invalid_strategies = [
                s for s in research_cfg.strategy_universe
                if s.lower() not in STRATEGY_REGISTRY
            ]
            if invalid_strategies:
                available = list(STRATEGY_REGISTRY.keys())
                raise ConfigValidationError(
                    f"Invalid strategies in research.default_strategy_universe: {invalid_strategies}. "
                    f"Available strategies: {available}"
                )

    # Check 5: max_parallel_jobs must be positive
    if research_cfg.max_parallel_jobs <= 0:
        raise ConfigValidationError(
            f"research.max_parallel_jobs must be positive, got {research_cfg.max_parallel_jobs}"
        )

    # Check 6: Warn if exchange is a real exchange name
    exchange_name = exchange_cfg.get("name", "").lower()
    real_exchanges = ["binance", "binance_futures", "bybit", "okx"]
    if any(exch in exchange_name for exch in real_exchanges) and exchange_type != "backtest":
        import warnings
        warnings.warn(
            f"Research mode with real exchange name '{exchange_name}' and type '{exchange_type}'. "
            f"Ensure this is intentional and you're not accidentally using live credentials.",
            UserWarning,
        )


def assert_research_mode(sys_cfg: Dict[str, Any]) -> None:
    """
    Assert that the system is in research mode.

    This is a convenience wrapper around validate_research_config that
    can be called at the start of research operations.

    Args:
        sys_cfg: System configuration dictionary

    Raises:
        ConfigValidationError: If not in valid research mode
    """
    validate_research_config(sys_cfg)


__all__ = [
    "validate_research_config",
    "assert_research_mode",
    "ConfigValidationError",
]
