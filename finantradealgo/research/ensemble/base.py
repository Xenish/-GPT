"""
Base Ensemble Strategy Infrastructure.

Provides the foundation for combining multiple strategies into a meta-strategy.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class ComponentStrategy:
    """
    Configuration for a component strategy within an ensemble.

    Attributes:
        strategy_name: Name of the strategy (e.g., "rule", "trend_continuation")
        strategy_params: Parameters for the strategy
        weight: Weight/importance of this strategy (interpretation depends on ensemble type)
        label: Optional human-readable label for this component
    """
    strategy_name: str
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.strategy_name


@dataclass
class EnsembleConfig:
    """
    Base configuration for ensemble strategies.

    Attributes:
        components: List of component strategies
        warmup_bars: Number of bars before ensemble starts trading
        use_component_signals: If True, aggregate component signals; if False, use component recommendations
    """
    components: List[ComponentStrategy]
    warmup_bars: int = 100
    use_component_signals: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleConfig":
        """Create config from dictionary."""
        components_data = data.get("components", [])
        components = [
            ComponentStrategy(
                strategy_name=c.get("strategy_name", c.get("strategy")),
                strategy_params=c.get("params", c.get("strategy_params", {})),
                weight=c.get("weight", 1.0),
                label=c.get("label"),
            )
            for c in components_data
        ]

        return cls(
            components=components,
            warmup_bars=data.get("warmup_bars", 100),
            use_component_signals=data.get("use_component_signals", True),
        )


class EnsembleStrategy(BaseStrategy):
    """
    Base class for ensemble strategies.

    An ensemble strategy combines multiple component strategies and makes
    trading decisions based on their aggregated signals.

    Subclasses must implement:
    - _aggregate_signals(): Combine component signals into ensemble signal
    - generate_signals(): Generate ensemble signals from component signals
    """

    def __init__(self, config: EnsembleConfig, component_strategies: Optional[Dict[str, BaseStrategy]] = None):
        """
        Initialize ensemble strategy.

        Args:
            config: Ensemble configuration
            component_strategies: Pre-initialized component strategies (optional)
        """
        self.config = config
        self.component_strategies = component_strategies or {}
        self.df: Optional[pd.DataFrame] = None

        # Component signal columns will be stored here
        self.component_signal_cols: Dict[str, str] = {}

    def init(self, df: pd.DataFrame) -> None:
        """
        Initialize ensemble with data.

        This calls init() on all component strategies and stores their signals.
        """
        self.df = df.copy()

        # Initialize each component strategy
        for comp in self.config.components:
            strategy_name = comp.strategy_name

            # Get or create component strategy instance
            if strategy_name not in self.component_strategies:
                # Strategy will be created externally and passed in
                # For now, just note the component
                pass

            # Each component will have its own signal column
            signal_col_name = f"ensemble_component_{comp.label}_signal"
            self.component_signal_cols[comp.label] = signal_col_name

    @abstractmethod
    def _aggregate_signals(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        """
        Aggregate component signals into a single ensemble signal.

        Args:
            row: Current bar data (includes component signal columns)
            ctx: Strategy context

        Returns:
            Ensemble signal (LONG/SHORT/CLOSE/None)
        """
        pass

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        """
        Generate signal for current bar based on component signals.

        Args:
            row: Current bar data
            ctx: Strategy context

        Returns:
            Trading signal
        """
        # Warmup period
        if ctx.index < self.config.warmup_bars:
            return None

        # Aggregate component signals
        return self._aggregate_signals(row, ctx)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals from component signals.

        This method should:
        1. Ensure all component signals are available
        2. Apply ensemble logic to combine signals
        3. Return DataFrame with ensemble signal column(s)

        Args:
            df: DataFrame with OHLCV data and component signals

        Returns:
            DataFrame with ensemble signals
        """
        pass

    def get_component_weights(self) -> Dict[str, float]:
        """
        Get current weights for each component.

        Returns:
            Dictionary mapping component label to weight
        """
        return {comp.label: comp.weight for comp in self.config.components}

    def get_component_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for each component strategy.

        Args:
            df: DataFrame with component signals

        Returns:
            DataFrame with performance metrics per component
        """
        # This is a placeholder - actual implementation would calculate
        # metrics like sharpe, return, win rate for each component
        metrics = []

        for comp in self.config.components:
            signal_col = self.component_signal_cols.get(comp.label)
            if signal_col and signal_col in df.columns:
                # Calculate basic metrics (placeholder)
                metrics.append({
                    "component": comp.label,
                    "strategy": comp.strategy_name,
                    "weight": comp.weight,
                    "n_signals": int(df[signal_col].sum()),
                })

        return pd.DataFrame(metrics) if metrics else pd.DataFrame()
