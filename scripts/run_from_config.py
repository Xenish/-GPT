from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.system.config_loader import load_config


def load_strategy_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy config not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid strategy config structure in {path}")
    if "class_path" not in cfg:
        raise ValueError(f"Strategy config {path} missing required field 'class_path'")
    cfg.setdefault("params", {})
    return cfg


def instantiate_strategy(strategy_cfg: Dict[str, Any]):
    class_path = strategy_cfg["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    strategy_cls = getattr(module, class_name)
    return strategy_cls(**(strategy_cfg.get("params") or {}))


def resolve_symbol_timeframe(sys_cfg: Dict[str, Any], symbol: str | None, timeframe: str | None) -> Tuple[str, str]:
    live_cfg = sys_cfg.get("live_cfg")
    data_cfg = sys_cfg.get("data_cfg")

    resolved_symbol = symbol or sys_cfg.get("symbol") or getattr(live_cfg, "symbol", None)
    if not resolved_symbol and data_cfg and getattr(data_cfg, "symbols", None):
        resolved_symbol = data_cfg.symbols[0]

    resolved_timeframe = timeframe or sys_cfg.get("timeframe") or getattr(live_cfg, "timeframe", None)
    if not resolved_timeframe and data_cfg and getattr(data_cfg, "timeframes", None):
        resolved_timeframe = data_cfg.timeframes[0]

    if not resolved_symbol or not resolved_timeframe:
        raise RuntimeError("Symbol/timeframe could not be resolved from system config.")
    return resolved_symbol, resolved_timeframe


def build_backtest_config(sys_cfg: Dict[str, Any]) -> BacktestConfig:
    bt_section = sys_cfg.get("backtest", {}) or {}
    data_section = sys_cfg.get("data", {}) or {}
    bars_mode = (data_section.get("bars") or {}).get("mode", "time")
    return BacktestConfig(
        initial_cash=float(bt_section.get("initial_cash", BacktestConfig.initial_cash)),
        fee_pct=float(bt_section.get("fee_pct", BacktestConfig.fee_pct)),
        slippage_pct=float(bt_section.get("slippage_pct", BacktestConfig.slippage_pct)),
        bar_mode=bars_mode,
    )


def run_from_configs(*, strategy_config: str, profile: str, symbol: str | None = None, timeframe: str | None = None):
    sys_cfg = load_config(profile)
    resolved_symbol, resolved_timeframe = resolve_symbol_timeframe(sys_cfg, symbol, timeframe)

    # Build features using official system config pipeline
    df_feat, pipeline_meta = build_feature_pipeline_from_system_config(
        sys_cfg,
        symbol=resolved_symbol,
        timeframe=resolved_timeframe,
    )

    # Strategy wiring via strategy config
    strategy_cfg = load_strategy_config(strategy_config)
    strategy = instantiate_strategy(strategy_cfg)

    # Risk/backtest configs
    risk_engine = RiskEngine(RiskConfig.from_dict(sys_cfg.get("risk")))
    backtest_cfg = build_backtest_config(sys_cfg)
    backtester = Backtester(strategy=strategy, risk_engine=risk_engine, config=backtest_cfg)

    result = backtester.run(df_feat)

    report_cfg = ReportConfig(regime_columns=[])
    report = generate_report(backtest_result=result, df=df_feat, config=report_cfg)

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print("\n=== Backtest Report ===")
    print("Equity:")
    print(f"  Initial cash : {eq['initial_cash']}")
    print(f"  Final equity : {eq['final_equity']}")
    print(f"  Cumulative R.: {eq['cum_return']}")
    print(f"  Max drawdown : {eq['max_drawdown']}")
    print(f"  Sharpe       : {eq['sharpe']}")

    print("\nTrades:")
    print(f"  Count        : {ts['trade_count']}")
    print(f"  Win rate     : {ts['win_rate']}")
    print(f"  Avg PnL      : {ts['avg_pnl']}")
    print(f"  Avg Win      : {ts['avg_win']}")
    print(f"  Avg Loss     : {ts['avg_loss']}")
    print(f"  ProfitFactor : {ts['profit_factor']}")
    print(f"  Median hold  : {ts['median_hold_time']}")

    print("\n[INFO] Feature pipeline:")
    print(f"  symbol      : {pipeline_meta.get('symbol', resolved_symbol)}")
    print(f"  timeframe   : {pipeline_meta.get('timeframe', resolved_timeframe)}")
    print(f"  preset      : {pipeline_meta.get('feature_preset')}")
    print(f"  version     : {pipeline_meta.get('pipeline_version')}")
    print(f"  feature_cnt : {len(df_feat.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinanTradeAlgo from system + strategy config.")
    parser.add_argument(
        "--strategy-config",
        type=str,
        default="config/strategies/ema_example.yml",
        help="Path to strategy config (name/class_path/params).",
    )
    parser.add_argument(
        "--profile",
        choices=["research", "live"],
        default="research",
        help="System config profile to load.",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Override symbol from system config.")
    parser.add_argument("--timeframe", type=str, default=None, help="Override timeframe from system config.")
    args = parser.parse_args()

    run_from_configs(
        strategy_config=args.strategy_config,
        profile=args.profile,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )


if __name__ == "__main__":
    main()
