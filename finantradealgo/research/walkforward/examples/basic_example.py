"""
Basic Walk-Forward Analysis Example.

Demonstrates how to use the WalkForwardEngine for strategy robustness testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from finantradealgo.research.walkforward import (
    WalkForwardEngine,
    WalkForwardConfig,
    WindowType,
    OptimizationMetric,
)


def simple_ma_crossover_backtest(data_df: pd.DataFrame, params: dict):
    """
    Simple moving average crossover strategy for demonstration.

    Args:
        data_df: DataFrame with 'close' column
        params: Dict with 'fast_ma' and 'slow_ma' parameters

    Returns:
        Tuple of (trades_df, metrics_dict)
    """
    fast_period = params.get('fast_ma', 10)
    slow_period = params.get('slow_ma', 50)

    # Calculate moving averages
    data = data_df.copy()
    data['fast_ma'] = data['close'].rolling(fast_period).mean()
    data['slow_ma'] = data['close'].rolling(slow_period).mean()

    # Generate signals
    data['signal'] = 0
    data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
    data['position'] = data['signal'].diff()

    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']

    # Simulate trades
    trades = []
    position = 0
    entry_price = 0
    entry_date = None

    for idx, row in data.iterrows():
        if row['position'] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = row['close']
            entry_date = idx
        elif row['position'] == -1 and position == 1:  # Sell signal
            pnl = (row['close'] - entry_price) / entry_price * 100
            trades.append({
                'entry_date': entry_date,
                'exit_date': idx,
                'entry_price': entry_price,
                'exit_price': row['close'],
                'pnl': pnl,
                'return_pct': pnl,
            })
            position = 0

    if not trades:
        # No trades generated
        return pd.DataFrame(), {
            'total_trades': 0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
        }

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_return = trades_df['pnl'].sum()
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0

    # Calculate Sharpe ratio
    if len(trades_df) > 1:
        sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252 / 30)
    else:
        sharpe = 0.0

    # Calculate max drawdown
    cumulative = trades_df['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    metrics = {
        'total_trades': len(trades_df),
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                            trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                            if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
    }

    return trades_df, metrics


def generate_sample_data(days: int = 1000) -> pd.DataFrame:
    """
    Generate sample price data for testing.

    Args:
        days: Number of days of data

    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D')

    # Generate random walk with trend
    np.random.seed(42)
    returns = np.random.randn(days) * 0.02 + 0.0002  # 2% volatility, slight upward drift
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(days) * 0.005),
        'high': prices * (1 + abs(np.random.randn(days)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(days)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
    })

    df.set_index('date', inplace=True)
    return df


def example_rolling_walk_forward():
    """
    Example: Rolling walk-forward optimization.

    Fixed-size training and test windows that roll forward together.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Rolling Walk-Forward Optimization")
    print("="*80 + "\n")

    # Generate sample data
    print("Generating sample price data...")
    data_df = generate_sample_data(days=1000)
    print(f"Data range: {data_df.index.min()} to {data_df.index.max()}")
    print(f"Total days: {len(data_df)}\n")

    # Configure walk-forward
    config = WalkForwardConfig(
        in_sample_periods=6,      # 6 months training
        out_sample_periods=2,      # 2 months testing
        window_type=WindowType.ROLLING,
        period_unit="M",
        optimization_metric=OptimizationMetric.SHARPE_RATIO,
        min_trades_per_period=5,
    )

    # Define parameter grid
    param_grid = {
        'fast_ma': [5, 10, 20],
        'slow_ma': [30, 50, 100],
    }

    # Create engine and run
    engine = WalkForwardEngine(config)

    print("Starting walk-forward optimization...\n")
    result = engine.run(
        strategy_id="MA_Crossover",
        param_grid=param_grid,
        data_df=data_df,
        backtest_function=simple_ma_crossover_backtest,
        auto_validate=True,
    )

    # Print detailed report
    print("\n" + engine.generate_report(result))

    # Export results
    print("\nExporting results...")
    engine.export_results(result, "walkforward_results/rolling", include_plots=True)

    return result


def example_anchored_walk_forward():
    """
    Example: Anchored (expanding) walk-forward optimization.

    Training window expands from fixed start, test window rolls forward.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Anchored (Expanding) Walk-Forward Optimization")
    print("="*80 + "\n")

    # Generate sample data
    data_df = generate_sample_data(days=1000)

    # Configure walk-forward
    config = WalkForwardConfig(
        in_sample_periods=6,      # Initial 6 months, then expanding
        out_sample_periods=2,      # 2 months testing
        window_type=WindowType.ANCHORED,
        period_unit="M",
        optimization_metric=OptimizationMetric.SHARPE_RATIO,
    )

    # Parameter grid
    param_grid = {
        'fast_ma': [10, 20],
        'slow_ma': [50, 100],
    }

    # Create engine and run
    engine = WalkForwardEngine(config)
    result = engine.run(
        strategy_id="MA_Crossover_Anchored",
        param_grid=param_grid,
        data_df=data_df,
        backtest_function=simple_ma_crossover_backtest,
    )

    # Analyze overfitting
    print("\nOverfitting Analysis:")
    overfitting = engine.detect_overfitting(result)
    print(f"  Risk Score: {overfitting['overfitting_risk_score']}/100")
    print(f"  Risk Level: {overfitting['risk_level'].upper()}")

    # Calculate WFE
    print("\nWalk-Forward Efficiency:")
    wfe = engine.calculate_wfe(result)
    print(f"  Sharpe Efficiency: {wfe.sharpe_efficiency:.2f}")
    print(f"  Return Efficiency: {wfe.return_efficiency:.2f}")

    return result


def example_strategy_comparison():
    """
    Example: Compare multiple strategies using walk-forward.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Strategy Comparison")
    print("="*80 + "\n")

    data_df = generate_sample_data(days=800)

    config = WalkForwardConfig(
        in_sample_periods=4,
        out_sample_periods=2,
        window_type=WindowType.ROLLING,
        period_unit="M",
    )

    engine = WalkForwardEngine(config)

    # Test different parameter sets
    strategies = [
        ("Fast_Trend", {'fast_ma': [5, 10], 'slow_ma': [20, 30]}),
        ("Medium_Trend", {'fast_ma': [10, 20], 'slow_ma': [50, 100]}),
        ("Slow_Trend", {'fast_ma': [20, 30], 'slow_ma': [100, 200]}),
    ]

    results = []
    for strategy_name, param_grid in strategies:
        print(f"\nTesting {strategy_name}...")
        result = engine.run(
            strategy_id=strategy_name,
            param_grid=param_grid,
            data_df=data_df,
            backtest_function=simple_ma_crossover_backtest,
            auto_validate=False,
        )
        results.append(result)

    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    comparison_df = engine.compare_strategies(results)
    print(comparison_df.to_string(index=False))

    # Plot comparison
    engine.plot_comparison(results, "walkforward_results/comparison.html")

    return results


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("WALK-FORWARD ANALYSIS FRAMEWORK - EXAMPLES")
    print("="*80)

    # Example 1: Rolling Walk-Forward
    result1 = example_rolling_walk_forward()

    # Example 2: Anchored Walk-Forward
    result2 = example_anchored_walk_forward()

    # Example 3: Strategy Comparison
    results = example_strategy_comparison()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nResults exported to 'walkforward_results/' directory")
    print("Open HTML files in your browser to view interactive charts.")


if __name__ == "__main__":
    main()
