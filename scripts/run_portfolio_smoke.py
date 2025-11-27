from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.backtester.portfolio_engine import PortfolioBacktestEngine
from finantradealgo.features.feature_pipeline_15m import build_feature_pipeline_from_system_config
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.strategy_engine import create_strategy
from finantradealgo.system.config_loader import PortfolioConfig, load_system_config


def main() -> None:
    cfg = load_system_config()
    portfolio_cfg = PortfolioConfig.from_dict(cfg.get("portfolio", {}))

    engines = {}
    data = {}

    for symbol in portfolio_cfg.symbols:
        cfg_local = dict(cfg)
        cfg_local["symbol"] = symbol
        cfg_local["timeframe"] = portfolio_cfg.timeframe

        df_features, _ = build_feature_pipeline_from_system_config(cfg_local)
        df_features = df_features.tail(300)
        strategy = create_strategy(portfolio_cfg.strategy, cfg_local)
        risk_engine = RiskEngine(RiskConfig.from_dict(cfg_local.get("risk", {})))
        engine = BacktestEngine(strategy=strategy, risk_engine=risk_engine, price_col="close", timestamp_col="timestamp")
        engines[(symbol, portfolio_cfg.strategy)] = engine
        data[(symbol, portfolio_cfg.strategy)] = df_features

    pb = PortfolioBacktestEngine(
        engines=engines,
        data=data,
        capital_allocation=portfolio_cfg.weights,
        initial_capital=portfolio_cfg.initial_capital,
    )

    result = pb.run()
    print("[SMOKE] Portfolio metrics:", result.metrics)
    print("[SMOKE] Trades:", len(result.trades))
    print("[SMOKE] Equity last value:", result.portfolio_equity.iloc[-1] if not result.portfolio_equity.empty else None)


if __name__ == "__main__":
    main()
