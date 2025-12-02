"""
Ensemble Strategy Backtesting Engine.

Utilities for backtesting ensemble strategies and tracking component performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from finantradealgo.research.ensemble.base import EnsembleStrategy
from finantradealgo.strategies.strategy_engine import create_strategy, get_strategy_meta


@dataclass
class EnsembleBacktestResult:
    """Results from ensemble backtest."""

    ensemble_metrics: Dict[str, float]
    component_metrics: pd.DataFrame
    ensemble_signals: pd.DataFrame
    component_signals: Dict[str, pd.DataFrame]
    bandit_stats: Optional[pd.DataFrame] = None
    weight_history: Optional[pd.DataFrame] = None


def prepare_component_signals(
    df: pd.DataFrame,
    components: List[Dict[str, Any]],
    sys_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate signals for all component strategies.

    Args:
        df: OHLCV DataFrame
        components: List of component configurations
        sys_cfg: System configuration

    Returns:
        DataFrame with all component signals
    """
    result_df = df.copy()

    for comp in components:
        strategy_name = comp.get("strategy_name", comp.get("strategy"))
        params = comp.get("params", comp.get("strategy_params", {}))
        label = comp.get("label", strategy_name)

        # Create strategy
        strategy = create_strategy(strategy_name, sys_cfg, params)

        # Generate signals (using generate_signals if available, otherwise use on_bar)
        if hasattr(strategy, "generate_signals"):
            # Modern pattern
            comp_df = strategy.generate_signals(df.copy())

            # Check for entry/exit columns
            if "long_entry" in comp_df.columns and "long_exit" in comp_df.columns:
                # Convert entry/exit to signal column (0/1)
                signal = []
                position = 0
                for i in range(len(comp_df)):
                    entry = comp_df.iloc[i]["long_entry"]
                    exit_ = comp_df.iloc[i]["long_exit"]

                    if position == 0 and entry:
                        position = 1
                    elif position == 1 and exit_:
                        position = 0

                    signal.append(position)

                signal_col_name = f"ensemble_component_{label}_signal"
                result_df[signal_col_name] = signal

        else:
            # Fallback: use init + on_bar pattern
            strategy.init(result_df)

            # Generate signals via on_bar
            from finantradealgo.core.strategy import StrategyContext

            signals = []
            position = 0

            for i in range(len(result_df)):
                row = result_df.iloc[i]
                ctx = StrategyContext(equity=10000.0, position=None, index=i)

                if position == 1:
                    ctx.position = type('Position', (), {'side': 'LONG', 'qty': 1, 'entry_price': 0})()

                sig = strategy.on_bar(row, ctx)

                # Track position
                if position == 0 and sig == "LONG":
                    position = 1
                elif position == 1 and sig in ("CLOSE", "SHORT"):
                    position = 0

                signals.append(position)

            signal_col_name = f"ensemble_component_{label}_signal"
            result_df[signal_col_name] = signals

    return result_df


def calculate_component_metrics(
    df: pd.DataFrame,
    component_signal_cols: Dict[str, str],
) -> pd.DataFrame:
    """
    Calculate performance metrics for each component.

    Args:
        df: DataFrame with component signals
        component_signal_cols: Mapping of component label to signal column

    Returns:
        DataFrame with metrics per component
    """
    metrics_list = []

    for label, signal_col in component_signal_cols.items():
        if signal_col not in df.columns:
            continue

        signals = df[signal_col].fillna(0)
        returns = df["close"].pct_change().fillna(0)

        # Strategy returns (forward returns when signal = 1)
        strategy_returns = returns * signals.shift(1).fillna(0)

        # Calculate metrics
        cum_return = (1 + strategy_returns).prod() - 1
        sharpe = 0.0
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * (252 * 96) ** 0.5  # Annualized

        # Trade count (transitions from 0 to 1)
        entries = (signals.diff() > 0).sum()

        # Win rate
        active_returns = strategy_returns[signals.shift(1) > 0]
        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0.0

        # Max drawdown
        equity_curve = (1 + strategy_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()

        metrics_list.append({
            "component": label,
            "cum_return": cum_return,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trade_count": entries,
            "win_rate": win_rate,
        })

    return pd.DataFrame(metrics_list)


def run_ensemble_backtest(
    ensemble_strategy: EnsembleStrategy,
    df: pd.DataFrame,
    components: List[Dict[str, Any]],
    sys_cfg: Dict[str, Any],
) -> EnsembleBacktestResult:
    """
    Run backtest for ensemble strategy.

    Args:
        ensemble_strategy: Ensemble strategy instance
        df: OHLCV DataFrame
        components: Component configurations
        sys_cfg: System configuration

    Returns:
        Backtest results
    """
    # Step 1: Generate component signals
    df_with_signals = prepare_component_signals(df, components, sys_cfg)

    # Step 2: Run ensemble strategy
    ensemble_df = ensemble_strategy.generate_signals(df_with_signals.copy())

    # Step 3: Calculate ensemble metrics
    ensemble_returns = ensemble_df["close"].pct_change().fillna(0)

    # Create ensemble signal column if entry/exit exist
    if "long_entry" in ensemble_df.columns:
        ensemble_signal = []
        position = 0
        for i in range(len(ensemble_df)):
            entry = ensemble_df.iloc[i].get("long_entry", False)
            exit_ = ensemble_df.iloc[i].get("long_exit", False)

            if position == 0 and entry:
                position = 1
            elif position == 1 and exit_:
                position = 0

            ensemble_signal.append(position)

        ensemble_df["ensemble_signal"] = ensemble_signal
        ensemble_strat_returns = ensemble_returns * pd.Series(ensemble_signal).shift(1).fillna(0)
    else:
        ensemble_strat_returns = pd.Series([0.0] * len(ensemble_df))

    # Calculate ensemble metrics
    ensemble_cum_return = (1 + ensemble_strat_returns).prod() - 1
    ensemble_sharpe = 0.0
    if ensemble_strat_returns.std() > 0:
        ensemble_sharpe = ensemble_strat_returns.mean() / ensemble_strat_returns.std() * (252 * 96) ** 0.5

    ensemble_entries = (pd.Series(ensemble_signal).diff() > 0).sum() if "long_entry" in ensemble_df.columns else 0

    ensemble_active_returns = ensemble_strat_returns[pd.Series(ensemble_signal).shift(1) > 0]
    ensemble_win_rate = (ensemble_active_returns > 0).mean() if len(ensemble_active_returns) > 0 else 0.0

    ensemble_equity_curve = (1 + ensemble_strat_returns).cumprod()
    ensemble_running_max = ensemble_equity_curve.cummax()
    ensemble_drawdown = (ensemble_equity_curve - ensemble_running_max) / ensemble_running_max
    ensemble_max_dd = ensemble_drawdown.min()

    ensemble_metrics = {
        "cum_return": float(ensemble_cum_return),
        "sharpe": float(ensemble_sharpe),
        "max_dd": float(ensemble_max_dd),
        "trade_count": int(ensemble_entries),
        "win_rate": float(ensemble_win_rate),
    }

    # Step 4: Calculate component metrics
    component_metrics = calculate_component_metrics(
        df_with_signals,
        ensemble_strategy.component_signal_cols,
    )

    # Step 5: Get bandit stats if applicable
    bandit_stats = None
    if hasattr(ensemble_strategy, "get_bandit_stats_df"):
        bandit_stats = ensemble_strategy.get_bandit_stats_df()

    # Step 6: Get weight history if applicable
    weight_history = None
    if hasattr(ensemble_strategy, "current_weights"):
        # For now, just get final weights
        weights_data = []
        for label, weight in ensemble_strategy.current_weights.items():
            weights_data.append({"component": label, "weight": weight})
        weight_history = pd.DataFrame(weights_data)

    return EnsembleBacktestResult(
        ensemble_metrics=ensemble_metrics,
        component_metrics=component_metrics,
        ensemble_signals=ensemble_df,
        component_signals={},  # Placeholder
        bandit_stats=bandit_stats,
        weight_history=weight_history,
    )


def save_ensemble_results(
    result: EnsembleBacktestResult,
    output_dir: Path,
    job_id: str,
) -> None:
    """
    Save ensemble backtest results to disk.

    Args:
        result: Backtest results
        output_dir: Output directory
        job_id: Job identifier
    """
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble metrics
    ensemble_metrics_df = pd.DataFrame([result.ensemble_metrics])
    ensemble_metrics_df.to_csv(job_dir / "ensemble_metrics.csv", index=False)

    # Save component metrics
    result.component_metrics.to_csv(job_dir / "component_metrics.csv", index=False)

    # Save ensemble signals
    result.ensemble_signals.to_parquet(job_dir / "ensemble_signals.parquet", index=False)

    # Save bandit stats if available
    if result.bandit_stats is not None:
        result.bandit_stats.to_csv(job_dir / "bandit_stats.csv", index=False)

    # Save weight history if available
    if result.weight_history is not None:
        result.weight_history.to_csv(job_dir / "weight_history.csv", index=False)

    print(f"[PASS] Ensemble results saved to {job_dir}")
