from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from finantradealgo.backtester.runners import run_backtest_once
from finantradealgo.strategies.param_space import (
    ParamSpace,
    apply_strategy_params_to_cfg,
    sample_params,
)
from finantradealgo.strategies.strategy_engine import get_strategy_meta
from finantradealgo.system.config_loader import load_system_config


def _compute_win_rate(trades: Any) -> Optional[float]:
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return None
    if "pnl" not in trades.columns:
        return None
    closed = trades.dropna(subset=["pnl"])
    if closed.empty:
        return None
    win_rate = (closed["pnl"] > 0).mean()
    return float(win_rate)


def evaluate_strategy_once(
    strategy_name: str,
    params: Optional[Dict[str, Any]] = None,
    sys_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single backtest evaluation for the given strategy/params pair.
    """
    base_cfg = sys_cfg or load_system_config()
    cfg_with_params = apply_strategy_params_to_cfg(base_cfg, strategy_name, params or {})
    symbol = cfg_with_params.get("symbol", base_cfg.get("symbol"))
    timeframe = cfg_with_params.get("timeframe", base_cfg.get("timeframe"))
    if symbol is None or timeframe is None:
        raise ValueError("System config must provide symbol/timeframe for strategy evaluation.")

    result = run_backtest_once(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        cfg=cfg_with_params,
    )
    metrics = result.get("metrics", {}) or {}
    win_rate = _compute_win_rate(result.get("trades"))

    return {
        "params": dict(params or {}),
        "cum_return": metrics.get("cum_return"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": win_rate,
        "trade_count": metrics.get("trade_count"),
    }


def random_search(
    strategy_name: str,
    n_samples: int,
    sys_cfg: Optional[Dict[str, Any]] = None,
    param_space: Optional[ParamSpace] = None,
) -> List[Dict[str, Any]]:
    """
    Sample random parameter sets from the strategy's ParamSpace and evaluate each once.
    """
    if n_samples <= 0:
        return []

    cfg = sys_cfg or load_system_config()
    space = param_space or getattr(get_strategy_meta(strategy_name), "param_space", None)
    if not space:
        raise ValueError(f"Strategy '{strategy_name}' has no ParamSpace defined.")

    results: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        params = sample_params(space)
        results.append(evaluate_strategy_once(strategy_name, params=params, sys_cfg=cfg))
    return results


__all__ = ["evaluate_strategy_once", "random_search"]
