from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, FileReplayDataSource
from finantradealgo.data_engine.binance_ws_source import BinanceWsDataSource
from finantradealgo.execution.execution_client import create_execution_client
from finantradealgo.features.feature_pipeline_15m import build_feature_pipeline_from_system_config
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.ml_strategy import MLSignalStrategy, MLStrategyConfig
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import LiveConfig, load_system_config
from finantradealgo.system.logger import init_logger


def build_strategy(cfg: dict, df: pd.DataFrame) -> Tuple[str, object]:
    strategy_type = cfg.get("strategy", {}).get("default", "rule").lower()
    if strategy_type == "ml":
        strat_cfg = MLStrategyConfig.from_dict(cfg.get("ml", {}).get("backtest", {}))
        strategy = MLSignalStrategy(strat_cfg)
    else:
        strat_cfg = RuleStrategyConfig.from_dict(cfg.get("rule", {}))
        strategy = RuleSignalStrategy(strat_cfg)
        strategy_type = "rule"
    strategy.init(df)
    return strategy_type, strategy


def build_run_id(symbol: str, timeframe: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{timeframe}_{ts}"


def create_data_source(
    cfg_local: dict,
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
        return BinanceWsDataSource(cfg_local, symbols)
    raise ValueError(f"Unsupported live data source: {live_cfg.data_source}")


def create_live_engine(
    cfg_local: dict,
    *,
    run_id: str,
    logger=None,
) -> tuple[LiveEngine, str]:
    live_cfg = cfg_local.get("live_cfg")
    if not isinstance(live_cfg, LiveConfig):
        live_cfg = LiveConfig.from_dict(
            cfg_local.get("live"),
            default_symbol=cfg_local.get("symbol"),
            default_timeframe=cfg_local.get("timeframe"),
        )
        cfg_local["live_cfg"] = live_cfg

    df_features, meta = build_feature_pipeline_from_system_config(cfg_local)
    strategy_type, strategy = build_strategy(cfg_local, df_features)
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
        strategy_name=strategy_type,
    )
    return engine, strategy_type


def main(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
    cfg = load_system_config()
    cfg_local = dict(cfg)
    live_section = dict(cfg_local.get("live", {}) or {})
    if symbol:
        cfg_local["symbol"] = symbol
        live_section["symbol"] = symbol
    if timeframe:
        cfg_local["timeframe"] = timeframe
        live_section["timeframe"] = timeframe
    cfg_local["live"] = live_section

    live_cfg = LiveConfig.from_dict(
        live_section,
        default_symbol=cfg_local.get("symbol"),
        default_timeframe=cfg_local.get("timeframe"),
    )
    cfg_local["live_cfg"] = live_cfg

    run_id = build_run_id(live_cfg.symbol, live_cfg.timeframe)
    logger = init_logger(run_id, live_cfg.log_dir, level=live_cfg.log_level)
    log_path = getattr(logger, "log_path", None)

    engine, strategy_type = create_live_engine(cfg_local, run_id=run_id, logger=logger)
    engine.run_loop()
    engine.shutdown()

    outputs = engine.export_results()
    execution_client = engine.execution_client
    portfolio = execution_client.get_portfolio() or {}
    trade_count = len(execution_client.get_trade_log())

    print("\n=== Live Paper Replay Summary ===")
    print(f"Run ID        : {run_id}")
    print(f"Strategy      : {strategy_type}")
    print(f"Symbol        : {live_cfg.symbol}")
    print(f"Timeframe     : {live_cfg.timeframe}")
    print(f"Bars consumed : {engine.iteration}")
    print(f"Final equity  : {portfolio['equity']:.2f}")
    print(f"Trade count   : {trade_count}")
    if log_path:
        print(f"Log file      : {log_path}")
    print(f"Snapshot      : {engine.state_path}")
    print(f"Latest state  : {engine.latest_state_path}")
    if "equity" in outputs:
        print(f"Equity CSV    : {outputs['equity']}")
    if "trades" in outputs:
        print(f"Trades CSV    : {outputs['trades']}")
    print(f"State JSON    : {engine.state_path}")


if __name__ == "__main__":
    main()
