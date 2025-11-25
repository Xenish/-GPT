from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

from finantradealgo.core.strategy import BaseStrategy
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
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
    name: str
    kind: str
    uses_ml: bool
    uses_microstructure: bool
    uses_market_structure: bool
    default_label_preset: Optional[str] = None
    default_feature_preset: Optional[str] = None


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


STRATEGY_REGISTRY: Dict[str, StrategySpec] = {
    "rule": StrategySpec(
        name="rule",
        strategy_cls=RuleSignalStrategy,
        config_cls=RuleStrategyConfig,
        config_extractor=_extract_rule_cfg,
        meta=StrategyMeta(
            name="rule_signals",
            kind="rule",
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset=None,
            default_feature_preset="extended",
        ),
    ),
    "ml": StrategySpec(
        name="ml",
        strategy_cls=MLSignalStrategy,
        config_cls=MLStrategyConfig,
        config_extractor=_extract_ml_cfg,
        meta=StrategyMeta(
            name="ml_signals",
            kind="ml",
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
            kind="price_action",
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset=None,
            default_feature_preset="extended",
        ),
    ),
    "sweep_reversal": StrategySpec(
        name="sweep_reversal",
        strategy_cls=SweepReversalStrategy,
        config_cls=SweepReversalConfig,
        config_extractor=_default_extractor("sweep_reversal"),
        meta=StrategyMeta(
            name="sweep_reversal",
            kind="price_action",
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=True,
            default_label_preset=None,
            default_feature_preset="extended",
        ),
    ),
    "volatility_breakout": StrategySpec(
        name="volatility_breakout",
        strategy_cls=VolatilityBreakoutStrategy,
        config_cls=VolatilityBreakoutConfig,
        config_extractor=_default_extractor("volatility_breakout"),
        meta=StrategyMeta(
            name="volatility_breakout",
            kind="volatility",
            uses_ml=False,
            uses_microstructure=True,
            uses_market_structure=False,
            default_label_preset=None,
            default_feature_preset="extended",
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
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY)}")

    spec = STRATEGY_REGISTRY[name]
    base_config = spec.config_extractor(system_cfg) or {}
    if overrides:
        merged = dict(base_config)
        merged.update(overrides)
    else:
        merged = base_config

    config_obj = spec.config_cls.from_dict(merged)
    return spec.strategy_cls(config_obj)


def get_strategy_meta(strategy_name: str) -> StrategyMeta:
    name = strategy_name.lower()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy for meta lookup: {strategy_name}")
    return STRATEGY_REGISTRY[name].meta


__all__ = ["StrategyMeta", "create_strategy", "get_strategy_meta", "STRATEGY_REGISTRY"]
