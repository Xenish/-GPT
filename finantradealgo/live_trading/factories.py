from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple, Optional

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy
from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, FileReplayDataSource
from finantradealgo.data_engine.binance_ws_source import BinanceWsDataSource
from finantradealgo.execution.execution_client import create_execution_client
from finantradealgo.features.feature_pipeline_15m import build_feature_pipeline_from_system_config
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import LiveConfig, KillSwitchConfig, NotificationsConfig
from finantradealgo.system.kill_switch import KillSwitch
from finantradealgo.system.notifications import create_notification_manager


def build_strategy(cfg: Dict[str, any], df: pd.DataFrame) -> Tuple[str, BaseStrategy]:
    strategy_name = cfg.get("strategy", {}).get("default", "rule").lower()
    if strategy_name == "ml":
        strat_cfg = MLStrategyConfig.from_dict(cfg.get("ml", {}).get("backtest", {}))
        strategy = MLSignalStrategy(strat_cfg)
    else:
        strat_cfg = RuleStrategyConfig.from_dict(cfg.get("rule", {}))
        strategy = RuleSignalStrategy(strat_cfg)
        strategy_name = "rule"
    strategy.init(df)
    return strategy_name, strategy


def create_data_source(
    cfg: Dict[str, any],
    live_cfg: LiveConfig,
    df_features: pd.DataFrame,
) -> AbstractLiveDataSource:
    source = (live_cfg.data_source or "replay").lower()
    if source in {"replay", "file"}:
        return FileReplayDataSource(
            df_features,
            bars_limit=live_cfg.replay.bars_limit,
            start_index=live_cfg.replay.start_index,
            start_timestamp=live_cfg.replay.start_timestamp,
        )
    if source == "binance_ws":
        symbols = live_cfg.symbols or [live_cfg.symbol]
        return BinanceWsDataSource(cfg, symbols)
    raise ValueError(f"Unsupported live data source: {live_cfg.data_source}")


def create_live_engine(
    system_cfg: Dict[str, any],
    *,
    run_id: str,
    mode: Optional[str] = None,
    logger=None,
) -> Tuple[LiveEngine, str]:
    cfg_local = deepcopy(system_cfg)
    live_section = deepcopy(cfg_local.get("live", {}) or {})
    if mode:
        live_section["mode"] = mode
    cfg_local["live"] = live_section

    live_cfg = cfg_local.get("live_cfg")
    if not isinstance(live_cfg, LiveConfig):
        live_cfg = LiveConfig.from_dict(
            live_section,
            default_symbol=cfg_local.get("symbol"),
            default_timeframe=cfg_local.get("timeframe"),
        )
    cfg_local["live_cfg"] = live_cfg
    kill_cfg = cfg_local.get("kill_switch_cfg")
    if not isinstance(kill_cfg, KillSwitchConfig):
        kill_cfg = KillSwitchConfig.from_dict(cfg_local.get("kill_switch", {}))
    cfg_local["kill_switch_cfg"] = kill_cfg
    kill_switch = KillSwitch(kill_cfg) if kill_cfg.enabled else None
    notifications_cfg = cfg_local.get("notifications_cfg")
    if not isinstance(notifications_cfg, NotificationsConfig):
        notifications_cfg = NotificationsConfig.from_dict(cfg_local.get("notifications", {}))
    cfg_local["notifications_cfg"] = notifications_cfg
    notifier = create_notification_manager(notifications_cfg)

    df_features, meta = build_feature_pipeline_from_system_config(cfg_local)
    strategy_name, strategy = build_strategy(cfg_local, df_features)
    risk_engine = RiskEngine(RiskConfig.from_dict(cfg_local.get("risk", {})))
    data_source = create_data_source(cfg_local, live_cfg, df_features)
    execution_client = create_execution_client(cfg_local, run_id=run_id)

    engine = LiveEngine(
        system_cfg=cfg_local,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=execution_client,
        data_source=data_source,
        run_id=run_id,
        logger=logger,
        pipeline_meta=meta,
        strategy_name=strategy_name,
        kill_switch=kill_switch,
        notifier=notifier,
    )
    return engine, strategy_name


__all__ = ["create_live_engine", "create_data_source", "build_strategy"]
