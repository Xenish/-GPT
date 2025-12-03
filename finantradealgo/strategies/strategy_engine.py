from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from finantradealgo.core.strategy import BaseStrategy
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.param_space import ParamSpace
from finantradealgo.strategies.rule_param_space import RULE_PARAM_SPACE
from finantradealgo.strategies.sweep_param_space import SWEEP_PARAM_SPACE
from finantradealgo.strategies.trend_param_space import TREND_PARAM_SPACE
from finantradealgo.strategies.volatility_param_space import VOLATILITY_PARAM_SPACE
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.strategies.sweep_reversal import (
    SweepReversalConfig,
    SweepReversalStrategy,
)
from finantradealgo.strategies.trend_continuation import (
    TrendContinuationConfig,
    TrendContinuationStrategy,
)
from finantradealgo.strategies.volatility_breakout import (
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)


StrategyConfigType = Type[Any]


@dataclass
class StrategyMeta:
    """Metadata for strategy registration and discovery.

    Attributes:
        name: Strategy name (identifier)
        description: Human-readable description of the strategy
        family: Strategy family (trend, range, microstructure, volatility, rule, ml)
        status: Development status (experimental, candidate, baseline, live)
        uses_ml: Whether strategy uses ML features
        uses_microstructure: Whether strategy uses microstructure features
        uses_market_structure: Whether strategy uses market structure features
        default_label_preset: Default ML label preset (if applicable)
        default_feature_preset: Default feature preset
        param_space: Parameter space for strategy_search (if searchable)
        is_searchable: Whether strategy is searchable via strategy_search
    """
    name: str
    description: str
    family: str  # "trend" | "range" | "microstructure" | "volatility" | "rule" | "ml"
    status: str = "experimental"  # "experimental" | "candidate" | "baseline" | "live"
    uses_ml: bool = False
    uses_microstructure: bool = False
    uses_market_structure: bool = False
    default_label_preset: Optional[str] = None
    default_feature_preset: Optional[str] = None
    param_space: Optional[ParamSpace] = None
    is_searchable: bool = False  # True if param_space is defined

    def __post_init__(self):
        """Auto-set is_searchable based on param_space."""
        if self.param_space is not None and not self.is_searchable:
            self.is_searchable = True


@dataclass
class StrategySpec:
    name: str
    strategy_cls: Type[BaseStrategy]
    config_cls: StrategyConfigType
    config_extractor: Callable[[Dict[str, Any]], Dict[str, Any]]
    meta: StrategyMeta


def _extract_from_strategy_block(strategy_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    strategy_section = cfg.get("strategy", {}) or {}
    return dict(strategy_section.get(strategy_name, {}) or {})


def _extract_rule_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = _extract_from_strategy_block("rule", cfg)
    if not block:
        block = cfg.get("rule", {}) or {}
    return dict(block)


def _extract_ml_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = _extract_from_strategy_block("ml", cfg)
    if not block:
        block = cfg.get("ml", {}).get("backtest", {}) if cfg.get("ml") else {}
    return dict(block or {})


def _default_extractor(name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _inner(cfg: Dict[str, Any]) -> Dict[str, Any]:
        block = _extract_from_strategy_block(name, cfg)
        return dict(block)

    return _inner


STRATEGY_SPECS: Dict[str, StrategySpec] = {
    "rule": StrategySpec(
        name="rule",
        strategy_cls=RuleSignalStrategy,
        config_cls=RuleStrategyConfig,
        config_extractor=_extract_rule_cfg,
        meta=StrategyMeta(
            name="rule",
            description="Multi-indicator rule-based strategy with market structure and microstructure filters",
            family="rule",  # Rule-based strategy
            status="candidate",  # Well-tested, production-ready
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset=None,
            default_feature_preset="extended",
            param_space=RULE_PARAM_SPACE,
        ),
    ),
    "ml": StrategySpec(
        name="ml",
        strategy_cls=MLSignalStrategy,
        config_cls=MLStrategyConfig,
        config_extractor=_extract_ml_cfg,
        meta=StrategyMeta(
            name="ml",
            description="Machine learning strategy using RandomForest with microstructure and market structure features",
            family="ml",  # Machine learning strategy
            status="experimental",  # Still under development
            uses_ml=True,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset="fixed_horizon",
            default_feature_preset="extended",
        ),
    ),
    "trend_continuation": StrategySpec(
        name="trend_continuation",
        strategy_cls=TrendContinuationStrategy,
        config_cls=TrendContinuationConfig,
        config_extractor=_default_extractor("trend_continuation"),
        meta=StrategyMeta(
            name="trend_continuation",
            description="Trend-following strategy with EMA crossovers and regime filters",
            family="trend",  # Trend-following strategy
            status="candidate",  # Well-tested
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset=None,
            default_feature_preset="extended",
            param_space=TREND_PARAM_SPACE,  # Searchable
        ),
    ),
    "sweep_reversal": StrategySpec(
        name="sweep_reversal",
        strategy_cls=SweepReversalStrategy,
        config_cls=SweepReversalConfig,
        config_extractor=_default_extractor("sweep_reversal"),
        meta=StrategyMeta(
            name="sweep_reversal",
            description="Liquidity sweep detection with mean reversion entries",
            family="microstructure",  # Microstructure-based strategy
            status="candidate",  # Well-tested
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset=None,
            default_feature_preset="extended",
            param_space=SWEEP_PARAM_SPACE,  # Searchable
        ),
    ),
    "volatility_breakout": StrategySpec(
        name="volatility_breakout",
        strategy_cls=VolatilityBreakoutStrategy,
        config_cls=VolatilityBreakoutConfig,
        config_extractor=_default_extractor("volatility_breakout"),
        meta=StrategyMeta(
            name="volatility_breakout",
            description="Range breakout strategy with volatility expansion filters",
            family="volatility",  # Volatility-based strategy
            status="experimental",  # Needs more testing
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=False,
            default_label_preset=None,
            default_feature_preset="extended",
            param_space=VOLATILITY_PARAM_SPACE,  # Now searchable
        ),
    ),
}


def create_strategy(
    strategy_name: str,
    system_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> BaseStrategy:
    """
    Factory that wires YAML config -> dataclasses -> concrete strategy instances.
    """
    name = strategy_name.lower()
    if name not in STRATEGY_SPECS:
        raise ValueError(f"Unsupported strategy '{strategy_name}'. Available: {list(STRATEGY_SPECS)}")

    spec = STRATEGY_SPECS[name]
    base_config = spec.config_extractor(system_cfg) or {}
    if overrides:
        merged = dict(base_config)
        merged.update(overrides)
    else:
        merged = base_config

    config_obj = spec.config_cls.from_dict(merged)
    return spec.strategy_cls(config_obj)


STRATEGY_REGISTRY: Dict[str, StrategyMeta] = {
    name: spec.meta for name, spec in STRATEGY_SPECS.items()
}


def get_strategy_meta(strategy_name: str) -> StrategyMeta:
    name = strategy_name.lower()
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown strategy '{strategy_name}'")


def get_strategies_by_family(family: str) -> Dict[str, StrategyMeta]:
    """Get all strategies belonging to a specific family.

    Args:
        family: Strategy family ("trend", "range", "microstructure", "volatility", "rule", "ml")

    Returns:
        Dictionary mapping strategy name to StrategyMeta for all strategies in the family

    Example:
        >>> trend_strategies = get_strategies_by_family("trend")
        >>> print(list(trend_strategies.keys()))
        ['trend_continuation']
    """
    return {
        name: meta
        for name, meta in STRATEGY_REGISTRY.items()
        if meta.family == family
    }


def get_searchable_strategies() -> Dict[str, StrategyMeta]:
    """Get all strategies that are searchable (have param_space defined).

    Returns:
        Dictionary mapping strategy name to StrategyMeta for searchable strategies

    Example:
        >>> searchable = get_searchable_strategies()
        >>> print(list(searchable.keys()))
        ['rule']  # Only strategies with param_space
    """
    return {
        name: meta
        for name, meta in STRATEGY_REGISTRY.items()
        if meta.is_searchable
    }


def list_strategies(
    family: Optional[str] = None,
    status: Optional[str] = None,
    searchable_only: bool = False,
) -> Dict[str, StrategyMeta]:
    """
    Query strategies with filters.

    Args:
        family: Filter by strategy family (trend, range, microstructure, volatility, rule, ml)
        status: Filter by development status (experimental, candidate, baseline, live)
        searchable_only: Only return strategies with param_space defined

    Returns:
        Dictionary mapping strategy name to StrategyMeta for matching strategies

    Examples:
        >>> # Get all candidate strategies
        >>> candidates = list_strategies(status="candidate")

        >>> # Get searchable trend strategies
        >>> searchable_trend = list_strategies(family="trend", searchable_only=True)

        >>> # Get all experimental strategies
        >>> experimental = list_strategies(status="experimental")
    """
    metas = dict(STRATEGY_REGISTRY)

    # Apply filters
    if family is not None:
        metas = {k: v for k, v in metas.items() if v.family == family}

    if status is not None:
        metas = {k: v for k, v in metas.items() if v.status == status}

    if searchable_only:
        metas = {k: v for k, v in metas.items() if v.is_searchable}

    return metas


__all__ = [
    "StrategyMeta",
    "create_strategy",
    "get_strategy_meta",
    "get_strategies_by_family",
    "get_searchable_strategies",
    "list_strategies",
    "STRATEGY_REGISTRY",
    "STRATEGY_SPECS",
]
