from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from copy import deepcopy

import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.backtester.runners import run_backtest_once
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.strategies.strategy_engine import create_strategy


@dataclass
class ScenarioConfig:
    name: str
    strategy_name: str
    strategy_params: Optional[Dict[str, Any]] = None
    risk_params: Optional[Dict[str, Any]] = None
    feature_preset: Optional[str] = None
    train_mode: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None


@dataclass
class Scenario:
    symbol: str
    timeframe: str
    strategy: str
    params: Dict[str, Any]
    label: Optional[str] = None
    feature_preset: Optional[str] = None
    risk_params: Optional[Dict[str, Any]] = None


def run_scenarios(
    base_cfg: Dict[str, Any],
    scenarios: List[Scenario],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, sc in enumerate(scenarios):
        cfg_local = deepcopy(base_cfg)
        cfg_local["symbol"] = sc.symbol
        cfg_local["timeframe"] = sc.timeframe

        if sc.feature_preset:
            feature_section = cfg_local.get("features", {}) or {}
            feature_section["feature_preset"] = sc.feature_preset
            cfg_local["features"] = feature_section

        if sc.risk_params:
            risk_section = dict(cfg_local.get("risk", {}) or {})
            risk_section.update(sc.risk_params)
            cfg_local["risk"] = risk_section

        result = run_backtest_once(
            symbol=sc.symbol,
            timeframe=sc.timeframe,
            strategy_name=sc.strategy,
            cfg=cfg_local,
            strategy_params=sc.params,
        )
        metrics = result.get("metrics", {}) or {}
        scenario_id = sc.label or f"{sc.strategy}-{idx}"
        rows.append(
            {
                "scenario_id": scenario_id,
                "label": sc.label or scenario_id,
                "symbol": sc.symbol,
                "timeframe": sc.timeframe,
                "strategy": sc.strategy,
                "params": sc.params,
                "cum_return": float(metrics.get("cum_return", 0.0)),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "trade_count": int(metrics.get("trade_count", 0)),
            }
        )
    return pd.DataFrame(rows)


class ScenarioEngine:
    def __init__(self, base_cfg: Dict[str, Any]):
        self.base_cfg = base_cfg

    def _build_strategy_cfg(self, scenario: ScenarioConfig) -> Dict[str, Any]:
        cfg = dict(self.base_cfg)
        if scenario.feature_preset:
            feature_section = cfg.get("features", {}) or {}
            feature_section["feature_preset"] = scenario.feature_preset
            cfg["features"] = feature_section
        return cfg

    def run_scenarios(
        self,
        scenarios: List[ScenarioConfig],
        df_features: pd.DataFrame,
    ) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for sc in scenarios:
            cfg = self._build_strategy_cfg(sc)
            strategy = create_strategy(sc.strategy_name, cfg, overrides=sc.strategy_params or {})

            risk_cfg_base = dict(cfg.get("risk", {}) or {})
            if sc.risk_params:
                risk_cfg_base.update(sc.risk_params)
            risk_engine = RiskEngine(RiskConfig.from_dict(risk_cfg_base))

            engine = BacktestEngine(
                strategy=strategy,
                risk_engine=risk_engine,
                price_col="close",
                timestamp_col="timestamp",
            )
            result = engine.run(df_features)
            metrics = result["metrics"]
            trades = result["trades"]
            trade_count = metrics.get("trade_count")
            if trade_count is None and isinstance(trades, pd.DataFrame):
                trade_count = len(trades)

            win_rate = 0.0
            if isinstance(trades, pd.DataFrame) and not trades.empty and "pnl" in trades:
                wins = trades["pnl"] > 0
                win_rate = float(wins.mean())

            risk_stats = result.get("risk_stats", {}) or {}
            blocked_entries = risk_stats.get("blocked_entries")
            if isinstance(blocked_entries, dict):
                blocked_total = sum(blocked_entries.values())
            else:
                blocked_total = blocked_entries or 0

            records.append(
                {
                    "scenario_name": sc.name,
                    "strategy_name": sc.strategy_name,
                    "cum_return": metrics.get("cum_return"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "sharpe": metrics.get("sharpe"),
                    "final_equity": metrics.get("final_equity"),
                    "trade_count": trade_count,
                    "win_rate": win_rate,
                    "blocked_entries": blocked_total,
                }
            )

        return pd.DataFrame.from_records(records)


def load_scenarios_from_config(cfg: Dict[str, Any], preset_name: str) -> List[ScenarioConfig]:
    scenario_section = cfg.get("scenario", {}) or {}
    presets = scenario_section.get("presets", {}) or {}
    entries = presets.get(preset_name)
    if not entries:
        raise KeyError(preset_name)
    scenarios: list[ScenarioConfig] = []
    for entry in entries:
        scenarios.append(
            ScenarioConfig(
                name=entry["name"],
                strategy_name=entry["strategy_name"],
                strategy_params=entry.get("strategy_params"),
                risk_params=entry.get("risk_params"),
                feature_preset=entry.get("feature_preset"),
                train_mode=entry.get("train_mode"),
                symbol=entry.get("symbol"),
                timeframe=entry.get("timeframe"),
            )
        )
    return scenarios


def run_scenario_preset(cfg: Dict[str, Any], preset_name: str) -> pd.DataFrame:
    cfg_local = deepcopy(cfg)
    configs = load_scenarios_from_config(cfg_local, preset_name)
    scenarios: List[Scenario] = []
    for sc in configs:
        symbol = getattr(sc, "symbol", None) or cfg_local.get("symbol", "BTCUSDT")
        timeframe = getattr(sc, "timeframe", None) or cfg_local.get("timeframe", "15m")
        scenarios.append(
            Scenario(
                symbol=symbol,
                timeframe=timeframe,
                strategy=sc.strategy_name,
                params=sc.strategy_params or {},
                label=sc.name,
                feature_preset=sc.feature_preset,
                risk_params=sc.risk_params,
            )
        )
    return run_scenarios(cfg_local, scenarios)


__all__ = [
    "ScenarioConfig",
    "ScenarioEngine",
    "Scenario",
    "run_scenario_preset",
    "load_scenarios_from_config",
    "run_scenarios",
]
