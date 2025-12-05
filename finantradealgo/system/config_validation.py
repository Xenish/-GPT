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


def _require_non_empty(value: Any, name: str) -> None:
    if value is None:
        raise ConfigValidationError(f"Missing required field: {name}")
    if isinstance(value, dict) and not value:
        raise ConfigValidationError(f"Required map {name} cannot be empty.")
    if isinstance(value, (list, tuple, set)) and len(value) == 0:
        raise ConfigValidationError(f"Required list {name} cannot be empty.")


def _require_range(
    value: Any,
    name: str,
    *,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    if value is None:
        return
    try:
        val = float(value)
    except Exception:
        raise ConfigValidationError(f"{name} must be numeric, got {type(value)}")

    if min_value is not None:
        if min_inclusive:
            if val < min_value:
                raise ConfigValidationError(f"{name} must be >= {min_value}, got {val}")
        else:
            if val <= min_value:
                raise ConfigValidationError(f"{name} must be > {min_value}, got {val}")
    if max_value is not None:
        if max_inclusive:
            if val > max_value:
                raise ConfigValidationError(f"{name} must be <= {max_value}, got {val}")
        else:
            if val >= max_value:
                raise ConfigValidationError(f"{name} must be < {max_value}, got {val}")


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

    # Check 2: Profile/mode must resolve to research (mode is legacy; profile authoritative)
    profile = sys_cfg.get("profile", "").lower()
    mode = sys_cfg.get("mode", "").lower()
    if profile != "research" and mode != "research":
        raise ConfigValidationError(
            f"Expected profile/mode 'research', got profile='{profile}', mode='{mode}'. "
            f"Use config/system.research.yml for research operations."
        )

    # Required data fields
    data_cfg = sys_cfg.get("data", {})
    _require_non_empty(data_cfg.get("symbols"), "data.symbols (research)")
    _require_non_empty(data_cfg.get("timeframes"), "data.timeframes (research)")
    _require_non_empty(data_cfg.get("lookback_days"), "data.lookback_days (research)")

    # Numeric ranges
    risk_limits = sys_cfg.get("risk_limits_cfg")
    if risk_limits:
        _require_range(risk_limits.risk_per_trade_pct, "risk.risk_per_trade_pct", min_value=0, min_inclusive=False, max_value=1, max_inclusive=True)
        _require_range(risk_limits.leverage_ceiling, "risk.leverage_ceiling", min_value=0, min_inclusive=False, max_value=100, max_inclusive=True)
    exchange_risk = sys_cfg.get("exchange_risk_cfg")
    if exchange_risk:
        _require_range(exchange_risk.max_leverage, "exchange.max_leverage", min_value=0, min_inclusive=False, max_value=100, max_inclusive=True)
    kill_switch_cfg = sys_cfg.get("kill_switch_cfg")
    if kill_switch_cfg:
        _require_range(kill_switch_cfg.max_equity_drawdown_pct, "kill_switch.max_equity_drawdown_pct", min_value=0, min_inclusive=True, max_value=100, max_inclusive=True)
        _require_range(kill_switch_cfg.max_exceptions_per_hour, "kill_switch.max_exceptions_per_hour", min_value=0, min_inclusive=False)

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


def validate_live_config(sys_cfg: Dict[str, Any]) -> None:
    """
    Validate live configuration for safety and correctness.

    Checks:
    1. profile/mode must be live
    2. exchange.type must be live
    3. kill_switch and risk blocks must exist
    4. live_cfg must be a LiveConfig with single symbol/timeframe
    5. risk limits must allow at least one open trade
    """
    profile = str(sys_cfg.get("profile", sys_cfg.get("mode", ""))).lower()
    if profile != "live":
        raise ConfigValidationError(f"Expected profile/mode 'live', got '{profile}'.")

    exchange_cfg = sys_cfg.get("exchange", {})
    exchange_type = str(exchange_cfg.get("type", "")).lower()
    if exchange_type != "live":
        raise ConfigValidationError(f"Live profile requires exchange.type='live', got '{exchange_type}'.")
    # Cross-field: API envs must be provided for live exchange
    api_key_env = exchange_cfg.get("api_key_env") or exchange_cfg.get("api_key") or ""
    secret_key_env = exchange_cfg.get("secret_key_env") or exchange_cfg.get("secret_key") or ""
    if not api_key_env or not secret_key_env:
        raise ConfigValidationError("Live exchange requires api_key_env and secret_key_env (or explicit keys).")

    if not sys_cfg.get("kill_switch"):
        raise ConfigValidationError("Live profile must define kill_switch block.")
    kill_cfg = sys_cfg.get("kill_switch", {})
    if not kill_cfg.get("enabled", True):
        raise ConfigValidationError("Live profile requires kill_switch.enabled=True for safety.")
    if not sys_cfg.get("risk"):
        raise ConfigValidationError("Live profile must define risk block.")

    live_cfg = sys_cfg.get("live_cfg")
    try:
        from finantradealgo.system.config_loader import LiveConfig
    except Exception:
        LiveConfig = None  # type: ignore
    if LiveConfig and not isinstance(live_cfg, LiveConfig):
        raise ConfigValidationError(f"live_cfg must be LiveConfig, got {type(live_cfg)}")

    if live_cfg:
        if not live_cfg.symbols or len(live_cfg.symbols) != 1:
            raise ConfigValidationError("Live profile must define exactly one symbol in live.symbols.")
        if not live_cfg.symbol:
            raise ConfigValidationError("Live profile must define live.symbol.")
        if getattr(live_cfg, "max_daily_loss", 0) <= 0:
            raise ConfigValidationError("Live profile must set live.max_daily_loss > 0.")
        _require_range(live_cfg.max_position_notional, "live.max_position_notional", min_value=0, min_inclusive=False)
        _require_range(live_cfg.max_open_trades, "live.max_open_trades", min_value=0, min_inclusive=False)

    # Required data fields (live)
    data_cfg = sys_cfg.get("data", {})
    _require_non_empty(data_cfg.get("symbols"), "data.symbols (live)")
    _require_non_empty(data_cfg.get("timeframes"), "data.timeframes (live)")
    _require_non_empty(data_cfg.get("lookback_days"), "data.lookback_days (live)")

    # Numeric ranges
    risk_limits = sys_cfg.get("risk_limits_cfg")
    if risk_limits:
        _require_range(risk_limits.risk_per_trade_pct, "risk.risk_per_trade_pct", min_value=0, min_inclusive=False, max_value=1, max_inclusive=True)
        _require_range(risk_limits.leverage_ceiling, "risk.leverage_ceiling", min_value=0, min_inclusive=False, max_value=100, max_inclusive=True)
    exchange_risk = sys_cfg.get("exchange_risk_cfg")
    if exchange_risk:
        _require_range(exchange_risk.max_leverage, "exchange.max_leverage", min_value=0, min_inclusive=False, max_value=100, max_inclusive=True)
    kill_switch_cfg = sys_cfg.get("kill_switch_cfg")
    if kill_switch_cfg:
        _require_range(kill_switch_cfg.max_equity_drawdown_pct, "kill_switch.max_equity_drawdown_pct", min_value=0, min_inclusive=True, max_value=100, max_inclusive=True)
        _require_range(kill_switch_cfg.max_exceptions_per_hour, "kill_switch.max_exceptions_per_hour", min_value=0, min_inclusive=False)
        # daily_realized_pnl_limit is expected to be negative (loss threshold)
        if kill_switch_cfg.daily_realized_pnl_limit >= 0:
            raise ConfigValidationError("kill_switch.daily_realized_pnl_limit should be negative for loss threshold.")

    risk_limits = sys_cfg.get("risk_limits_cfg")
    if risk_limits and getattr(risk_limits, "max_open_trades", 0) <= 0:
        raise ConfigValidationError("Live risk config must allow at least one open trade (max_open_trades > 0).")


def validate_config(sys_cfg: Dict[str, Any]) -> None:
    """
    Dispatch validation based on profile/mode.
    """
    profile = str(sys_cfg.get("profile", sys_cfg.get("mode", ""))).lower()
    if profile == "research":
        validate_research_config(sys_cfg)
    elif profile == "live":
        validate_live_config(sys_cfg)
    else:
        raise ConfigValidationError(f"Unknown or missing profile for validation: {profile!r}")


__all__ = [
    "validate_research_config",
    "validate_live_config",
    "validate_config",
    "assert_research_mode",
    "ConfigValidationError",
]
