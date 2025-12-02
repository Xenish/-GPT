"""
Stress Testing & Scenario Generator.

Generate and test market stress scenarios.
"""

from __future__ import annotations

from typing import Dict, List
import pandas as pd
import numpy as np

from finantradealgo.research.montecarlo.models import (
    StressTestScenario,
    MonteCarloConfig,
    STRESS_SCENARIOS,
)
from finantradealgo.research.montecarlo.resampler import BootstrapResampler


class StressTestEngine:
    """
    Stress testing engine for strategy robustness.

    Tests strategy under extreme market conditions.
    """

    def __init__(self):
        """Initialize stress test engine."""
        self.scenarios = STRESS_SCENARIOS.copy()

    def add_scenario(self, scenario: StressTestScenario):
        """Add custom stress scenario."""
        self.scenarios[scenario.scenario_id] = scenario

    def run_stress_test(
        self,
        strategy_id: str,
        trades_df: pd.DataFrame,
        scenarios: List[str] = None,
        n_simulations: int = 500,
    ) -> Dict[str, any]:
        """
        Run stress tests on strategy.

        Args:
            strategy_id: Strategy identifier
            trades_df: Historical trades
            scenarios: List of scenario IDs (None = all)
            n_simulations: Monte Carlo sims per scenario

        Returns:
            Dictionary with stress test results
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())

        results = {}

        for scenario_id in scenarios:
            if scenario_id not in self.scenarios:
                continue

            scenario = self.scenarios[scenario_id]

            # Apply stress to trades
            stressed_pnl = scenario.apply_to_returns(trades_df['pnl'])
            stressed_df = trades_df.copy()
            stressed_df['pnl'] = stressed_pnl

            # Run Monte Carlo on stressed data
            config = MonteCarloConfig(n_simulations=n_simulations, random_seed=42)
            resampler = BootstrapResampler(config)
            mc_result = resampler.run_monte_carlo(strategy_id, stressed_df)

            results[scenario_id] = {
                "scenario": scenario.name,
                "description": scenario.description,
                "mean_return": mc_result.mean_return,
                "var_95": mc_result.value_at_risk,
                "cvar_95": mc_result.conditional_var,
                "prob_profit": mc_result.prob_profit,
                "worst_case": mc_result.percentile_1,
            }

        return results
