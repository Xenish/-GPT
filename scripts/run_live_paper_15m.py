from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.data_engine.live_data_source import FileReplayDataSource
from finantradealgo.execution.paper_client import PaperExecutionClient
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


def main() -> None:
    cfg = load_system_config()
    live_cfg = LiveConfig.from_dict(
        cfg.get("live"),
        default_symbol=cfg.get("symbol"),
        default_timeframe=cfg.get("timeframe"),
    )

    df_features, meta = build_feature_pipeline_from_system_config(cfg)
    strategy_type, strategy = build_strategy(cfg, df_features)
    risk_engine = RiskEngine(RiskConfig.from_dict(cfg.get("risk", {})))

    data_source = FileReplayDataSource(
        df_features,
        bars_limit=live_cfg.replay.bars_limit,
        start_index=live_cfg.replay.start_index,
        start_timestamp=live_cfg.replay.start_timestamp,
    )

    paper_client = PaperExecutionClient(
        initial_cash=live_cfg.paper.initial_cash,
        output_dir=live_cfg.paper.output_dir,
        state_path=live_cfg.paper.state_path,
    )

    run_id = build_run_id(live_cfg.symbol, live_cfg.timeframe)
    logger = init_logger(run_id, live_cfg.log_dir, level=live_cfg.log_level)
    log_path = getattr(logger, "log_path", None)

    engine = LiveEngine(
        config=live_cfg,
        data_source=data_source,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=paper_client,
        logger=logger,
        run_id=run_id,
        pipeline_meta=meta,
    )
    engine.run_loop()
    engine.shutdown()

    outputs = engine.export_results()
    portfolio = paper_client.get_portfolio()
    trade_count = len(paper_client.get_trade_log())

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
    print(f"Equity CSV    : {outputs['equity']}")
    print(f"Trades CSV    : {outputs['trades']}")
    print(f"State JSON    : {engine.state_path}")


if __name__ == "__main__":
    main()
