"""
Basic Monte Carlo Simulation Examples.

Demonstrates how to use Monte Carlo simulation for strategy robustness testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from finantradealgo.research.montecarlo import (
    MonteCarloSimulator,
    MonteCarloAnalyzer,
    MonteCarloConfig,
    ResamplingMethod,
)


def generate_sample_trades(n_trades: int = 100, win_rate: float = 0.55) -> pd.DataFrame:
    """
    Generate sample trade results for demonstration.

    Args:
        n_trades: Number of trades to generate
        win_rate: Win rate (0-1)

    Returns:
        DataFrame with trade results
    """
    np.random.seed(42)

    trades = []
    current_date = datetime.now() - timedelta(days=n_trades)

    for i in range(n_trades):
        # Determine if win or loss
        is_win = np.random.random() < win_rate

        if is_win:
            # Winners: mean +$200, std $100
            pnl = np.random.normal(200, 100)
        else:
            # Losers: mean -$150, std $80
            pnl = np.random.normal(-150, 80)

        trades.append({
            'trade_id': i + 1,
            'entry_date': current_date,
            'exit_date': current_date + timedelta(days=1),
            'pnl': pnl,
            'return_pct': (pnl / 10000) * 100,  # Assume $10k per trade
        })

        current_date += timedelta(days=1)

    return pd.DataFrame(trades)


def generate_market_data(days: int = 500) -> pd.DataFrame:
    """
    Generate sample market data for regime analysis.

    Args:
        days: Number of days

    Returns:
        DataFrame with market data
    """
    np.random.seed(42)

    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

    # Generate returns with different regimes
    returns = []
    regime = 'bull'
    regime_length = 0

    for _ in range(days):
        # Change regime periodically
        regime_length += 1
        if regime_length > 50:  # Change regime every 50 days
            regime_length = 0
            regime = np.random.choice(['bull', 'bear', 'sideways'], p=[0.4, 0.3, 0.3])

        # Generate return based on regime
        if regime == 'bull':
            ret = np.random.normal(0.002, 0.015)  # Positive drift, moderate vol
        elif regime == 'bear':
            ret = np.random.normal(-0.003, 0.025)  # Negative drift, high vol
        else:  # sideways
            ret = np.random.normal(0.0, 0.010)  # No drift, low vol

        returns.append(ret)

    df = pd.DataFrame({
        'date': dates,
        'returns': returns,
    })
    df.set_index('date', inplace=True)

    return df


def example_1_trade_sequence_randomization():
    """
    Example 1: Trade Sequence Randomization (Luck vs. Skill).

    Test if results are statistically significant or due to lucky ordering.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Trade Sequence Randomization (Luck vs. Skill)")
    print("="*80 + "\n")

    # Generate sample trades
    trades_df = generate_sample_trades(n_trades=100, win_rate=0.58)

    print(f"Generated {len(trades_df)} trades")
    print(f"Original Total PnL: ${trades_df['pnl'].sum():.2f}")
    print(f"Original Return: {(trades_df['pnl'].sum() / 10000) * 100:.2f}%\n")

    # Configure Monte Carlo
    config = MonteCarloConfig(
        n_simulations=1000,
        resampling_method=ResamplingMethod.BOOTSTRAP,
        confidence_level=0.95,
    )

    # Create simulator
    simulator = MonteCarloSimulator(config)

    # Test trade sequence randomization
    result = simulator.test_trade_sequence(trades_df, strategy_id="SampleStrategy")

    # Analyze luck vs. skill
    analyzer = MonteCarloAnalyzer()
    luck_analysis = analyzer.analyze_luck_vs_skill(trades_df, result)

    print("\n" + "="*80)
    print("LUCK VS. SKILL ANALYSIS")
    print("="*80)
    print(f"Original Return: {luck_analysis.original_return:.2f}%")
    print(f"MC Mean Return: {luck_analysis.mc_mean_return:.2f}%")
    print(f"Percentile Rank: {luck_analysis.percentile_rank:.1f}")
    print(f"Z-Score: {luck_analysis.z_score:.3f}")
    print(f"P-Value: {luck_analysis.p_value:.4f}")
    print(f"Statistically Significant: {luck_analysis.is_statistically_significant}")
    print(f"\nInterpretation: {luck_analysis.interpretation}")

    return result


def example_2_parameter_perturbation():
    """
    Example 2: Parameter Perturbation Testing.

    Test how sensitive strategy is to parameter changes.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Parameter Perturbation Testing")
    print("="*80 + "\n")

    # Simulate a backtest function
    def simple_backtest(data_df, params):
        """Simple backtest function for demonstration."""
        fast_ma = params.get('fast_ma', 10)
        slow_ma = params.get('slow_ma', 50)

        # Simulate performance based on parameters
        # (In real use, this would run actual backtest)
        base_return = 15.0
        param_quality = 1.0 - abs(fast_ma - 20) / 50 - abs(slow_ma - 50) / 100

        total_return = base_return * param_quality + np.random.normal(0, 3)

        trades = generate_sample_trades(50)

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': total_return / 10 if total_return > 0 else 0,
            'max_drawdown': -10 * (1 + np.random.random()),
            'total_trades': len(trades),
        }

        return trades, metrics

    # Define optimal parameters
    optimal_params = {
        'fast_ma': 20,
        'slow_ma': 50,
    }

    # Generate dummy data
    data_df = pd.DataFrame({'close': np.random.randn(1000)})

    # Configure Monte Carlo
    config = MonteCarloConfig(n_simulations=100)
    simulator = MonteCarloSimulator(config)

    # Test parameter perturbation
    perturbation_result = simulator.test_parameter_perturbation(
        optimal_params=optimal_params,
        data_df=data_df,
        backtest_function=simple_backtest,
        noise_level=0.15,  # ±15% noise
        n_perturbations=50,
    )

    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY RESULTS")
    print("="*80)
    print(f"Robustness Score: {perturbation_result['robustness_score']:.1f}/100")
    print(f"Optimal Return: {perturbation_result['optimal_return']:.2f}%")
    print(f"Mean Perturbed: {perturbation_result['return_stats']['mean']:.2f}%")
    print(f"Std Deviation: {perturbation_result['return_stats']['std']:.2f}%")
    print(f"Return Degradation: {perturbation_result['degradation']['return_degradation']:.1%}")

    return perturbation_result


def example_3_comprehensive_analysis():
    """
    Example 3: Comprehensive Monte Carlo Analysis.

    Full analysis including risk metrics, drawdowns, and stress tests.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Comprehensive Monte Carlo Analysis")
    print("="*80 + "\n")

    # Generate sample trades
    trades_df = generate_sample_trades(n_trades=150, win_rate=0.60)

    # Configure Monte Carlo
    config = MonteCarloConfig(
        n_simulations=1000,
        resampling_method=ResamplingMethod.BOOTSTRAP,
        confidence_level=0.95,
    )

    simulator = MonteCarloSimulator(config)

    # Run comprehensive analysis
    analysis = simulator.run_comprehensive_analysis(
        trades_df=trades_df,
        strategy_id="ComprehensiveTest",
        include_stress_tests=True,
    )

    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)

    mc_summary = analysis['monte_carlo']
    print(f"\nMonte Carlo Summary:")
    print(f"  Mean Return: {mc_summary['mean_return']:.2f}%")
    print(f"  Median Return: {mc_summary['median_return']:.2f}%")
    print(f"  VaR (95%): {mc_summary['var']:.2f}%")
    print(f"  CVaR (95%): {mc_summary['cvar']:.2f}%")
    print(f"  Prob Profit: {mc_summary['prob_profit']:.1%}")

    if 'stress_tests' in analysis and analysis['stress_tests']:
        print(f"\nStress Test Results:")
        for scenario_id, stress_result in analysis['stress_tests'].items():
            print(f"  {stress_result['scenario']}:")
            print(f"    Mean Return: {stress_result['mean_return']:.2f}%")
            print(f"    VaR 95%: {stress_result['var_95']:.2f}%")

    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")

    return analysis


def example_4_regime_analysis():
    """
    Example 4: Regime Randomization Analysis.

    Test strategy performance across different market conditions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Market Regime Analysis")
    print("="*80 + "\n")

    # Generate trades and market data
    trades_df = generate_sample_trades(n_trades=200, win_rate=0.58)
    trades_df.set_index('exit_date', inplace=True)

    market_data = generate_market_data(days=500)

    # Create analyzer
    analyzer = MonteCarloAnalyzer()

    # Analyze regime sensitivity
    regime_analysis = analyzer.analyze_regime_randomization(
        trades_df=trades_df,
        market_data=market_data,
        n_randomizations=500,
    )

    print("\n" + "="*80)
    print("REGIME ANALYSIS RESULTS")
    print("="*80)
    print(f"Bull Market Return: {regime_analysis.bull_market_return:.2f}%")
    print(f"Bear Market Return: {regime_analysis.bear_market_return:.2f}%")
    print(f"Sideways Market Return: {regime_analysis.sideways_market_return:.2f}%")
    print(f"\nRegime Consistency Score: {regime_analysis.regime_consistency_score:.1f}/100")
    print(f"Regime Impact Significant: {regime_analysis.regime_impact_significant}")

    if regime_analysis.regime_consistency_score > 70:
        print("\n✓ Strategy performs consistently across market regimes")
    else:
        print("\n⚠ Strategy shows regime-dependent performance")

    return regime_analysis


def example_5_strategy_comparison():
    """
    Example 5: Compare Multiple Strategies.

    Use Monte Carlo to compare different strategies.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Strategy Comparison")
    print("="*80 + "\n")

    # Generate trades for different strategies
    strategies = {
        "Aggressive": generate_sample_trades(100, win_rate=0.52),
        "Conservative": generate_sample_trades(100, win_rate=0.58),
        "Balanced": generate_sample_trades(100, win_rate=0.55),
    }

    # Configure Monte Carlo
    config = MonteCarloConfig(n_simulations=500)
    simulator = MonteCarloSimulator(config)

    # Compare strategies
    comparison_df = simulator.compare_strategies(strategies)

    print("\nStrategy Comparison Results:")
    print(comparison_df.to_string(index=False))

    # Determine best strategy
    best_strategy = comparison_df.iloc[0]['strategy_id']
    print(f"\n✓ Best Strategy (by mean return): {best_strategy}")

    return comparison_df


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION FRAMEWORK - EXAMPLES")
    print("="*80)

    # Example 1: Trade Sequence Randomization
    example_1_trade_sequence_randomization()

    # Example 2: Parameter Perturbation
    example_2_parameter_perturbation()

    # Example 3: Comprehensive Analysis
    example_3_comprehensive_analysis()

    # Example 4: Regime Analysis
    example_4_regime_analysis()

    # Example 5: Strategy Comparison
    example_5_strategy_comparison()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nMonte Carlo simulations help test:")
    print("  1. Trade sequence randomization (Luck vs. Skill)")
    print("  2. Parameter sensitivity (Robustness)")
    print("  3. Risk metrics distribution (VaR, CVaR)")
    print("  4. Market regime sensitivity")
    print("  5. Strategy comparison\n")


if __name__ == "__main__":
    main()
