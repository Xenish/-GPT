"""
Weighted Ensemble Strategies.

Implements ensemble strategies that combine component strategies using
weighted aggregation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _normalize_with_bounds(raw_weights: np.ndarray,
                           min_weight: float | None,
                           max_weight: float | None) -> np.ndarray:
    """
    Normalize weights so that:
    - sum(weights) == 1
    - if min_weight is not None: w_i >= min_weight
    - if max_weight is not None: w_i <= max_weight
    - raw_weights'in göreli büyüklüklerini olabildiğince korur.
    """
    w_raw = np.asarray(raw_weights, dtype=float)
    n = len(w_raw)

    if n == 0:
        return w_raw

    # Negatifleri sıfıra çek
    w_raw = np.maximum(w_raw, 0.0)

    # Default bound'lar
    if min_weight is None:
        min_weight = 0.0
    if max_weight is None:
        max_weight = 1.0

    # Feasibility check
    if n * min_weight > 1.0 + 1e-12:
        # Min'ler bile toplam1'i aşıyor → fallback: uniform
        return np.full(n, 1.0 / n, dtype=float)

    if n * max_weight < 1.0 - 1e-12:
        # Max'lerle bile toplam1'e ulaşamıyoruz → uniform + clip
        base = np.full(n, 1.0 / n, dtype=float)
        return np.clip(base, min_weight, max_weight)

    # Basit normalize et
    total = w_raw.sum()
    if total <= 0:
        # Tüm raw weight'ler sıfır veya negatif → uniform distribution
        return np.full(n, 1.0 / n, dtype=float)

    # İlk normalize
    w = w_raw / total

    # Iteratif projection: clip ve error'u movable elemanlara dağıt
    max_iterations = 100
    for iteration in range(max_iterations):
        # Clip to bounds
        w_clipped = np.clip(w, min_weight, max_weight)

        # Calculate how far we are from sum=1
        error = 1.0 - w_clipped.sum()

        if abs(error) < 1e-9:
            # Converged
            return w_clipped

        # Find elements that CAN move in the direction we need
        if error > 0:
            # Need to add weight - find elements that can increase (below max)
            can_move = (w < max_weight - 1e-9)
        else:
            # Need to remove weight - find elements that can decrease (above min)
            can_move = (w > min_weight + 1e-9)

        if can_move.sum() == 0:
            # No elements can move in the needed direction
            # Just normalize and return
            return w_clipped / w_clipped.sum()

        # Redistribute error to movable elements proportionally
        w_movable_sum = w[can_move].sum()
        if w_movable_sum > 1e-12:
            # Add error proportionally to current weights
            w[can_move] = w[can_move] + error * (w[can_move] / w_movable_sum)
        else:
            # Distribute error equally
            w[can_move] = w[can_move] + error / can_move.sum()

        # Clip all elements after redistribution
        w = np.clip(w, min_weight, max_weight)

    # If didn't converge, just return best effort
    return np.clip(w, min_weight, max_weight)


class WeightingMethod(str, Enum):
    """Weighting methods for ensemble aggregation."""

    EQUAL = "equal"  # Equal weight for all components
    SHARPE = "sharpe"  # Weight by historical Sharpe ratio
    INVERSE_VOL = "inverse_vol"  # Weight by inverse volatility
    RETURN = "return"  # Weight by historical return
    CUSTOM = "custom"  # Use custom weights from component config


@dataclass
class WeightedEnsembleConfig(EnsembleConfig):
    """
    Configuration for weighted ensemble strategy.

    Attributes:
        components: List of component strategies
        weighting_method: Method for computing component weights
        reweight_period: Bars between weight recalculations (0 = static)
        lookback_bars: Bars to use for computing weights
        min_weight: Minimum allowed weight per component
        max_weight: Maximum allowed weight per component
        normalize_weights: Whether to normalize weights to sum to 1
        signal_threshold: Minimum weighted signal strength to trade (0-1)
    """

    weighting_method: WeightingMethod = WeightingMethod.EQUAL
    reweight_period: int = 0  # 0 = static weights
    lookback_bars: int = 100
    min_weight: float | None = None
    max_weight: float | None = None
    normalize_weights: bool = True
    signal_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightedEnsembleConfig":
        """Create config from dictionary."""
        base_config = super().from_dict(data)

        weighting_method_str = data.get("weighting_method", "equal")
        weighting_method = WeightingMethod(weighting_method_str)

        return cls(
            components=base_config.components,
            warmup_bars=base_config.warmup_bars,
            use_component_signals=base_config.use_component_signals,
            weighting_method=weighting_method,
            reweight_period=data.get("reweight_period", 0),
            lookback_bars=data.get("lookback_bars", 100),
            min_weight=data.get("min_weight", None),
            max_weight=data.get("max_weight", None),
            normalize_weights=data.get("normalize_weights", True),
            signal_threshold=data.get("signal_threshold", 0.5),
        )


class WeightedEnsembleStrategy(EnsembleStrategy):
    """
    Weighted ensemble strategy that aggregates component signals using weights.

    Signal Aggregation:
    - Each component emits long_entry/long_exit signals (or signal column)
    - Weighted sum of signals determines ensemble decision
    - If weighted_signal > threshold → LONG
    - If weighted_signal < -threshold → SHORT/CLOSE
    - Otherwise → None (no signal)
    """

    def __init__(
        self,
        config: WeightedEnsembleConfig,
        component_strategies: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, component_strategies)
        self.config: WeightedEnsembleConfig = config  # Type hint

        # Track current weights
        self.current_weights: Dict[str, float] = {}

        # Track last reweight bar
        self.last_reweight_bar: int = 0

        # Component performance tracking
        self.component_returns: Dict[str, List[float]] = {
            comp.label: [] for comp in config.components
        }

    def init(self, df: pd.DataFrame) -> None:
        """Initialize ensemble with data."""
        super().init(df)

        # Initialize weights based on weighting method
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize current_weights according to config and weighting method."""
        components = self.config.components
        n = len(components)

        if n == 0:
            self.current_weights = {}
            return

        method = self.config.weighting_method
        min_w = getattr(self.config, "min_weight", None)
        max_w = getattr(self.config, "max_weight", None)
        normalize = getattr(self.config, "normalize_weights", True)

        labels = [(c.label or c.strategy_name) for c in components]

        # --- FAST PATH 1: EQUAL weighting, hiçbir bound yok ---
        if method == WeightingMethod.EQUAL and min_w is None and max_w is None:
            # Tüm component'ler eşit ve toplam = 1
            w = np.full(n, 1.0 / n, dtype=float)
            self.current_weights = {label: float(x) for label, x in zip(labels, w)}
            return

        # --- FAST PATH 2: CUSTOM, normalize=True, bound yok ---
        if (
            method == WeightingMethod.CUSTOM
            and normalize
            and min_w is None
            and max_w is None
        ):
            # ComponentStrategy.weight değerlerini oransal normalize et
            raw = np.array(
                [
                    (c.weight if getattr(c, "weight", None) is not None else 1.0)
                    for c in components
                ],
                dtype=float,
            )
            total = float(raw.sum())
            if total <= 0:
                w = np.full(n, 1.0 / n, dtype=float)
            else:
                w = raw / total

            self.current_weights = {label: float(x) for label, x in zip(labels, w)}
            return

        # ---------- 1) RAW WEIGHTS OLUŞTUR ----------
        # WeightingMethod'e göre raw weight vektörü üret
        if self.config.weighting_method == WeightingMethod.EQUAL:
            # Tüm komponentler eşit
            raw = np.ones(n, dtype=float)
        elif self.config.weighting_method == WeightingMethod.CUSTOM:
            # ComponentStrategy.weight değerlerini kullan
            raw_list: list[float] = []
            for comp in components:
                # None ise 1.0 say (eşit pay)
                w = comp.weight if comp.weight is not None else 1.0
                raw_list.append(float(w))
            raw = np.asarray(raw_list, dtype=float)
        else:
            # Diğer weighting method'ların (SHARPE vs.) kendi raw hesabı varsa
            # onu burada yapmalısın. Eğer şimdilik yoksa, CUSTOM gibi davran.
            raw_list = []
            for comp in components:
                w = comp.weight if getattr(comp, "weight", None) is not None else 1.0
                raw_list.append(float(w))
            raw = np.asarray(raw_list, dtype=float)

        # ---------- 2) BOUND VE NORMALIZATION AYARLARI ----------
        min_w = self.config.min_weight
        max_w = self.config.max_weight

        if self.config.normalize_weights:
            # normalize_weights = True ise iki durum var:
            #  a) min/max yok → basit normalize
            #  b) min/max var → hem sum=1 hem bound'lar korunmalı
            if min_w is None and max_w is None:
                total = float(raw.sum())
                if total <= 0:
                    # Fallback: uniform distribution
                    weights = np.full(n, 1.0 / n, dtype=float)
                else:
                    weights = raw / total
            else:
                weights = _normalize_with_bounds(raw, min_w, max_w)
        else:
            # normalize_weights = False ise, sadece raw kullan,
            # varsa min/max ile clip et ama normalize etme
            weights = raw.copy()
            if min_w is not None or max_w is not None:
                lo = min_w if min_w is not None else -np.inf
                hi = max_w if max_w is not None else np.inf
                weights = np.clip(weights, lo, hi)

        # ---------- 3) SÖZLÜĞE AKTAR ----------
        labels = [(c.label or c.strategy_name) for c in components]
        self.current_weights = {
            label: float(w) for label, w in zip(labels, weights)
        }

    def _constrain_weights(self) -> None:
        """Apply min/max weight constraints and normalize using projection."""
        
        n = len(self.config.components)
        if n == 0:
            return

        # Extract raw weights from dictionary into an ordered array
        labels = [c.label or c.strategy_name for c in self.config.components]
        raw = np.array([self.current_weights.get(l, 0.0) for l in labels], dtype=float)

        min_w = self.config.min_weight
        max_w = self.config.max_weight

        if self.config.normalize_weights:
            # Logic for when normalization is needed
            if min_w is None and max_w is None:
                # Simple normalization, no constraints
                total = raw.sum()
                if total > 0:
                    weights = raw / total
                else:
                    weights = np.full(n, 1.0 / n)
            else:
                # Both normalize and apply min/max constraints
                weights = _normalize_with_bounds(raw, min_w, max_w)
        else:
            # Logic for when normalization is false, just clip
            if min_w is not None or max_w is not None:
                lo = min_w if min_w is not None else -np.inf
                hi = max_w if max_w is not None else np.inf
                weights = np.clip(raw, lo, hi)
            else:
                weights = raw

        # Update the dictionary with constrained weights
        self.current_weights = {
            label: float(w)
            for label, w in zip(labels, weights)
        }

    def _recompute_weights(self, df: pd.DataFrame, current_bar: int) -> None:
        """
        Recompute weights based on historical performance.

        Args:
            df: DataFrame with component signals
            current_bar: Current bar index
        """
        if self.config.weighting_method == WeightingMethod.CUSTOM:
            # Custom weights don't change
            return

        if self.config.weighting_method == WeightingMethod.EQUAL:
            # Equal weights don't change
            return

        # Need historical data to compute weights
        start_bar = max(0, current_bar - self.config.lookback_bars)
        if start_bar >= current_bar:
            return

        lookback_df = df.iloc[start_bar:current_bar]

        # Compute weights based on method
        if self.config.weighting_method == WeightingMethod.SHARPE:
            self._compute_sharpe_weights(lookback_df)
        elif self.config.weighting_method == WeightingMethod.INVERSE_VOL:
            self._compute_inverse_vol_weights(lookback_df)
        elif self.config.weighting_method == WeightingMethod.RETURN:
            self._compute_return_weights(lookback_df)

        self._constrain_weights()

    def _compute_sharpe_weights(self, df: pd.DataFrame) -> None:
        """Compute weights based on Sharpe ratios."""
        sharpe_ratios = {}

        for comp in self.config.components:
            signal_col = self.component_signal_cols.get(comp.label)
            if not signal_col or signal_col not in df.columns:
                sharpe_ratios[comp.label] = 0.0
                continue

            # Compute returns when signal is active
            signals = df[signal_col].fillna(0)
            returns = df["close"].pct_change().fillna(0)

            # Strategy returns (simplified - just use forward returns when signal = 1)
            strategy_returns = returns * signals.shift(1).fillna(0)

            # Sharpe ratio
            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 96)  # Annualized for 15m bars
                sharpe_ratios[comp.label] = max(0, sharpe)  # Only positive Sharpe
            else:
                sharpe_ratios[comp.label] = 0.0

        # Weight by Sharpe (higher Sharpe = higher weight)
        total_sharpe = sum(sharpe_ratios.values())
        if total_sharpe > 0:
            self.current_weights = {
                label: sharpe / total_sharpe for label, sharpe in sharpe_ratios.items()
            }
        else:
            # Fallback to equal weights
            n = len(self.config.components)
            self.current_weights = {label: 1.0 / n for label in sharpe_ratios}

    def _compute_inverse_vol_weights(self, df: pd.DataFrame) -> None:
        """Compute weights based on inverse volatility."""
        volatilities = {}

        for comp in self.config.components:
            signal_col = self.component_signal_cols.get(comp.label)
            if not signal_col or signal_col not in df.columns:
                volatilities[comp.label] = 1.0
                continue

            # Compute volatility when signal is active
            signals = df[signal_col].fillna(0)
            returns = df["close"].pct_change().fillna(0)
            strategy_returns = returns * signals.shift(1).fillna(0)

            vol = strategy_returns.std()
            volatilities[comp.label] = max(vol, 1e-8)  # Avoid division by zero

        # Weight by inverse volatility
        inverse_vols = {label: 1.0 / vol for label, vol in volatilities.items()}
        total_inv_vol = sum(inverse_vols.values())

        if total_inv_vol > 0:
            self.current_weights = {
                label: inv_vol / total_inv_vol for label, inv_vol in inverse_vols.items()
            }

    def _compute_return_weights(self, df: pd.DataFrame) -> None:
        """Compute weights based on cumulative returns."""
        cum_returns = {}

        for comp in self.config.components:
            signal_col = self.component_signal_cols.get(comp.label)
            if not signal_col or signal_col not in df.columns:
                cum_returns[comp.label] = 0.0
                continue

            # Compute returns when signal is active
            signals = df[signal_col].fillna(0)
            returns = df["close"].pct_change().fillna(0)
            strategy_returns = returns * signals.shift(1).fillna(0)

            cum_ret = (1 + strategy_returns).prod() - 1
            cum_returns[comp.label] = max(0, cum_ret)  # Only positive returns

        # Weight by return
        total_return = sum(cum_returns.values())
        if total_return > 0:
            self.current_weights = {
                label: ret / total_return for label, ret in cum_returns.items()
            }
        else:
            # Fallback to equal weights
            n = len(self.config.components)
            self.current_weights = {label: 1.0 / n for label in cum_returns}

    def _aggregate_signals(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        """
        Aggregate component signals using weighted sum.

        Args:
            row: Current bar
            ctx: Strategy context

        Returns:
            Ensemble signal
        """
        # Check if reweighting is needed
        if (
            self.config.reweight_period > 0
            and ctx.index - self.last_reweight_bar >= self.config.reweight_period
            and self.df is not None
        ):
            self._recompute_weights(self.df, ctx.index)
            self.last_reweight_bar = ctx.index

        # Aggregate component signals
        weighted_signal = 0.0

        for comp in self.config.components:
            signal_col = self.component_signal_cols.get(comp.label)
            if not signal_col or signal_col not in row.index:
                continue

            component_signal = row[signal_col]
            weight = self.current_weights.get(comp.label, 0.0)

            # Component signal is 0/1, treat 1 as bullish vote
            weighted_signal += component_signal * weight

        # Decision based on weighted signal
        current_position = 1 if ctx.position is not None else 0

        if current_position == 0:
            # Not in position - check for entry
            if weighted_signal >= self.config.signal_threshold:
                return "LONG"
        else:
            # In position - check for exit
            if weighted_signal < self.config.signal_threshold / 2:  # Hysteresis
                return "CLOSE"

        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals from component signals.

        Expects df to have component signal columns already generated.

        Args:
            df: DataFrame with component signals

        Returns:
            DataFrame with ensemble signal columns
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
