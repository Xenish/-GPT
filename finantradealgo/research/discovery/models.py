"""
Data models for strategy discovery and mining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import numpy as np


class RuleOperator(str, Enum):
    """Trading rule operators."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUALS = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    AND = "and"
    OR = "or"


@dataclass
class TradingRule:
    """
    Trading rule representation.

    Can be simple (e.g., "SMA(10) > SMA(50)") or composite (multiple conditions).
    """

    rule_id: str
    expression: str  # Human-readable rule
    operator: RuleOperator
    operands: List[Any]  # Can be indicators, values, or nested rules

    # Performance metrics
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # Complexity metrics
    depth: int = 1  # Tree depth for composite rules
    num_conditions: int = 1

    def evaluate(self, market_data: Dict[str, Any]) -> bool:
        """Evaluate rule on market data."""
        # This would be implemented to actually evaluate the rule
        # For now, placeholder
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'expression': self.expression,
            'operator': self.operator.value,
            'fitness': self.fitness,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'depth': self.depth,
            'num_conditions': self.num_conditions,
        }


@dataclass
class Pattern:
    """
    Discovered market pattern.
    """

    pattern_id: str
    pattern_type: str  # 'sequential', 'association', 'temporal'
    description: str

    # Pattern definition
    conditions: List[Dict[str, Any]]
    support: float  # Frequency of occurrence
    confidence: float  # Reliability when it occurs

    # Performance when traded
    avg_return: float = 0.0
    win_rate: float = 0.0
    occurrences: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'support': self.support,
            'confidence': self.confidence,
            'avg_return': self.avg_return,
            'win_rate': self.win_rate,
            'occurrences': self.occurrences,
        }


@dataclass
class FeatureImportance:
    """
    Feature importance metrics.
    """

    feature_name: str

    # Different importance measures
    permutation_importance: float = 0.0
    shap_value: float = 0.0
    correlation_with_returns: float = 0.0

    # Interaction effects
    top_interactions: List[tuple[str, float]] = field(default_factory=list)

    # Statistical significance
    p_value: float = 1.0
    is_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'permutation_importance': self.permutation_importance,
            'shap_value': self.shap_value,
            'correlation_with_returns': self.correlation_with_returns,
            'top_interactions': self.top_interactions,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
        }


@dataclass
class GeneticPopulation:
    """
    Population of trading rules in genetic algorithm.
    """

    generation: int
    individuals: List[TradingRule]
    population_size: int

    # Population statistics
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 0.0

    def get_best_individual(self) -> Optional[TradingRule]:
        """Get best performing rule."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.fitness)

    def get_top_n(self, n: int) -> List[TradingRule]:
        """Get top N rules."""
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:n]


@dataclass
class DiscoveryResult:
    """
    Complete strategy discovery result.
    """

    method: str  # 'genetic_programming', 'pattern_mining', 'feature_mining'

    # Discovered strategies/patterns
    rules: List[TradingRule] = field(default_factory=list)
    patterns: List[Pattern] = field(default_factory=list)
    features: List[FeatureImportance] = field(default_factory=list)

    # Metadata
    total_candidates_evaluated: int = 0
    best_fitness: float = 0.0
    convergence_generation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'rules': [r.to_dict() for r in self.rules],
            'patterns': [p.to_dict() for p in self.patterns],
            'features': [f.to_dict() for f in self.features],
            'total_candidates_evaluated': self.total_candidates_evaluated,
            'best_fitness': self.best_fitness,
            'convergence_generation': self.convergence_generation,
        }
