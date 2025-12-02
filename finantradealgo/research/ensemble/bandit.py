"""
Multi-Armed Bandit Ensemble Strategies.

Implements ensemble strategies that use multi-armed bandit algorithms
to dynamically select and weight component strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from finantradealgo.core.strategy import SignalType, StrategyContext
from finantradealgo.research.ensemble.base import (
    ComponentStrategy,
    EnsembleConfig,
    EnsembleStrategy,
)


class BanditAlgorithm(str, Enum):
    """Multi-armed bandit algorithms."""

    EPSILON_GREEDY = "epsilon_greedy"  # Explore with probability epsilon
    UCB1 = "ucb1"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian sampling


@dataclass
class BanditStats:
    """Statistics for a single bandit arm (component strategy)."""

    n_pulls: int = 0  # Number of times selected
    total_reward: float = 0.0  # Cumulative reward
    mean_reward: float = 0.0  # Average reward
    variance: float = 0.0  # Reward variance

    # Thompson Sampling (Beta distribution)
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0  # Failures + 1


@dataclass
class BanditEnsembleConfig(EnsembleConfig):
    """
    Configuration for bandit ensemble strategy.

    Attributes:
        components: List of component strategies
        bandit_algorithm: Bandit algorithm to use
        epsilon: Exploration rate for epsilon-greedy (0-1)
        ucb_c: Exploration parameter for UCB1
        update_period: Bars between bandit updates
        reward_lookback: Bars to use for computing reward
        reward_metric: Metric for reward ("return", "sharpe", "win_rate")
        min_pulls_per_arm: Minimum pulls before algorithm starts
    """

    bandit_algorithm: BanditAlgorithm = BanditAlgorithm.EPSILON_GREEDY
    epsilon: float = 0.1  # For epsilon-greedy
    ucb_c: float = 2.0  # For UCB1
    update_period: int = 20  # Update every N bars
    reward_lookback: int = 50  # Bars for reward calculation
    reward_metric: str = "return"  # "return" | "sharpe" | "win_rate"
    min_pulls_per_arm: int = 10  # Force exploration initially

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditEnsembleConfig":
        """Create config from dictionary."""
        base_config = super().from_dict(data)

        bandit_algorithm_str = data.get("bandit_algorithm", "epsilon_greedy")
        bandit_algorithm = BanditAlgorithm(bandit_algorithm_str)

        return cls(
            components=base_config.components,
            warmup_bars=base_config.warmup_bars,
            use_component_signals=base_config.use_component_signals,
            bandit_algorithm=bandit_algorithm,
            epsilon=data.get("epsilon", 0.1),
            ucb_c=data.get("ucb_c", 2.0),
            update_period=data.get("update_period", 20),
            reward_lookback=data.get("reward_lookback", 50),
            reward_metric=data.get("reward_metric", "return"),
            min_pulls_per_arm=data.get("min_pulls_per_arm", 10),
        )


class BanditEnsembleStrategy(EnsembleStrategy):
    """
    Bandit ensemble strategy that uses multi-armed bandit algorithms
    to dynamically select the best component strategy.

    Unlike weighted ensembles that use all components, bandit ensembles
    select ONE component at a time based on exploration-exploitation tradeoff.
    """

    def __init__(
        self,
        config: BanditEnsembleConfig,
        component_strategies: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, component_strategies)
        self.config: BanditEnsembleConfig = config  # Type hint

        # Bandit statistics for each component
        self.bandit_stats: Dict[str, BanditStats] = {
            comp.label: BanditStats() for comp in config.components
        }

        # Currently selected component
        self.selected_component: Optional[str] = None

        # Track last update bar
        self.last_update_bar: int = 0

        # Track reward history
        self.reward_history: List[float] = []

        # Random number generator
        self.rng = np.random.RandomState(42)

    def init(self, df: pd.DataFrame) -> None:
        """Initialize bandit ensemble with data."""
        super().init(df)

        # Select initial component (random)
        if self.config.components:
            self.selected_component = self.rng.choice(
                [comp.label for comp in self.config.components]
            )

    def _select_arm(self, current_bar: int) -> str:
        """
        Select component strategy (arm) using bandit algorithm.

        Args:
            current_bar: Current bar index

        Returns:
            Label of selected component
        """
        # Force initial exploration
        min_pulls = self.config.min_pulls_per_arm
        for label, stats in self.bandit_stats.items():
            if stats.n_pulls < min_pulls:
                return label

        # Select based on algorithm
        if self.config.bandit_algorithm == BanditAlgorithm.EPSILON_GREEDY:
            return self._epsilon_greedy()
        elif self.config.bandit_algorithm == BanditAlgorithm.UCB1:
            return self._ucb1(current_bar)
        elif self.config.bandit_algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
            return self._thompson_sampling()
        else:
            # Fallback to random
            return self.rng.choice([comp.label for comp in self.config.components])

    def _epsilon_greedy(self) -> str:
        """Epsilon-greedy arm selection."""
        if self.rng.random() < self.config.epsilon:
            # Explore: random arm
            return self.rng.choice([comp.label for comp in self.config.components])
        else:
            # Exploit: best arm
            best_arm = max(
                self.bandit_stats.items(), key=lambda x: x[1].mean_reward
            )[0]
            return best_arm

    def _ucb1(self, current_bar: int) -> str:
        """Upper Confidence Bound (UCB1) arm selection."""
        total_pulls = sum(stats.n_pulls for stats in self.bandit_stats.values())

        if total_pulls == 0:
            # No pulls yet, select random
            return self.rng.choice([comp.label for comp in self.config.components])

        # Compute UCB scores
        ucb_scores = {}
        for label, stats in self.bandit_stats.items():
            if stats.n_pulls == 0:
                # Unplayed arm gets infinite score
                ucb_scores[label] = float("inf")
            else:
                # UCB score = mean + c * sqrt(ln(total_pulls) / n_pulls)
                exploration_bonus = self.config.ucb_c * np.sqrt(
                    np.log(total_pulls) / stats.n_pulls
                )
                ucb_scores[label] = stats.mean_reward + exploration_bonus

        # Select arm with highest UCB score
        best_arm = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_arm

    def _thompson_sampling(self) -> str:
        """Thompson Sampling (Bayesian) arm selection."""
        # Sample from Beta distribution for each arm
        samples = {}
        for label, stats in self.bandit_stats.items():
            # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
            sample = self.rng.beta(stats.alpha, stats.beta)
            samples[label] = sample

        # Select arm with highest sample
        best_arm = max(samples.items(), key=lambda x: x[1])[0]
        return best_arm

    def _compute_reward(self, df: pd.DataFrame, current_bar: int, component_label: str) -> float:
        """
        Compute reward for a component strategy.

        Args:
            df: DataFrame with component signals
            current_bar: Current bar index
            component_label: Component to compute reward for

        Returns:
            Reward value
        """
        # Get lookback window
        start_bar = max(0, current_bar - self.config.reward_lookback)
        if start_bar >= current_bar:
            return 0.0

        lookback_df = df.iloc[start_bar:current_bar]

        signal_col = self.component_signal_cols.get(component_label)
        if not signal_col or signal_col not in lookback_df.columns:
            return 0.0

        # Compute reward based on metric
        signals = lookback_df[signal_col].fillna(0)
        returns = lookback_df["close"].pct_change().fillna(0)
        strategy_returns = returns * signals.shift(1).fillna(0)

        if self.config.reward_metric == "return":
            # Cumulative return
            reward = (1 + strategy_returns).prod() - 1

        elif self.config.reward_metric == "sharpe":
            # Sharpe ratio
            if strategy_returns.std() > 0:
                reward = strategy_returns.mean() / strategy_returns.std()
            else:
                reward = 0.0

        elif self.config.reward_metric == "win_rate":
            # Win rate (% of positive returns)
            active_returns = strategy_returns[signals.shift(1) > 0]
            if len(active_returns) > 0:
                reward = (active_returns > 0).sum() / len(active_returns)
            else:
                reward = 0.0

        else:
            reward = 0.0

        return float(reward)

    def _update_bandit_stats(self, df: pd.DataFrame, current_bar: int) -> None:
        """
        Update bandit statistics for the selected component.

        Args:
            df: DataFrame with component signals
            current_bar: Current bar index
        """
        if self.selected_component is None:
            return

        # Compute reward for selected component
        reward = self._compute_reward(df, current_bar, self.selected_component)

        # Update stats
        stats = self.bandit_stats[self.selected_component]
        stats.n_pulls += 1
        stats.total_reward += reward

        # Update mean reward (incremental)
        stats.mean_reward = stats.total_reward / stats.n_pulls

        # Update Thompson Sampling parameters (Beta distribution)
        if reward > 0:
            stats.alpha += 1  # Success
        else:
            stats.beta += 1  # Failure

        # Track reward history
        self.reward_history.append(reward)

    def _aggregate_signals(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        """
        Use selected component's signal.

        Args:
            row: Current bar
            ctx: Strategy context

        Returns:
            Signal from selected component
        """
        # Check if update is needed
        if (
            self.config.update_period > 0
            and ctx.index - self.last_update_bar >= self.config.update_period
            and self.df is not None
        ):
            # Update stats for previous selection
            self._update_bandit_stats(self.df, ctx.index)

            # Select new component
            self.selected_component = self._select_arm(ctx.index)
            self.last_update_bar = ctx.index

        # Get signal from selected component
        if self.selected_component is None:
            return None

        signal_col = self.component_signal_cols.get(self.selected_component)
        if not signal_col or signal_col not in row.index:
            return None

        component_signal = row[signal_col]

        # Map component signal to trade signal
        current_position = 1 if ctx.position is not None else 0

        if current_position == 0 and component_signal == 1:
            return "LONG"
        elif current_position == 1 and component_signal == 0:
            return "CLOSE"

        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals using bandit algorithm.

        Args:
            df: DataFrame with component signals

        Returns:
            DataFrame with ensemble signals
        """
        self.init(df)

        # Generate long_entry and long_exit signals
        df["long_entry"] = False
        df["long_exit"] = False
        df["short_entry"] = False
        df["short_exit"] = False

        position = 0

        for i in range(len(df)):
            row = df.iloc[i]
            ctx = StrategyContext(equity=10000.0, position=None, index=i)

            if position == 1:
                ctx.position = type('Position', (), {'side': 'LONG', 'qty': 1, 'entry_price': 0})()

            # Get signal
            signal = self.on_bar(row, ctx)

            # Update position and signals
            if position == 0 and signal == "LONG":
                df.loc[df.index[i], "long_entry"] = True
                position = 1
            elif position == 1 and signal == "CLOSE":
                df.loc[df.index[i], "long_exit"] = True
                position = 0

        return df

    def get_bandit_stats_df(self) -> pd.DataFrame:
        """
        Get bandit statistics as DataFrame.

        Returns:
            DataFrame with bandit statistics per component
        """
        stats_list = []
        for label, stats in self.bandit_stats.items():
            stats_list.append({
                "component": label,
                "n_pulls": stats.n_pulls,
                "total_reward": stats.total_reward,
                "mean_reward": stats.mean_reward,
                "alpha": stats.alpha,
                "beta": stats.beta,
            })

        return pd.DataFrame(stats_list)
