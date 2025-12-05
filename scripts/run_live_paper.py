from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.system.config_loader import LiveConfig, load_config, load_config_from_env
from finantradealgo.system.logger import init_logger


def build_run_id(symbol: str, timeframe: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{timeframe}_{ts}"


def main(symbol: Optional[str] = None, timeframe: Optional[str] = None, profile: Optional[str] = None) -> None:
    # Load config with profile support
    cfg = load_config(profile) if profile else load_config_from_env()

    # SAFETY: Assert live/paper mode for live trading
    cfg_mode = cfg.get("mode", "unknown")
    if cfg_mode not in ("live", "paper"):
        raise RuntimeError(
            f"Live trading must run with mode='live' or mode='paper'. Got mode='{cfg_mode}'. "
            f"Set FINANTRADE_PROFILE=live or pass --profile live."
        )
    cfg_local = dict(cfg)
    live_section = dict(cfg_local.get("live", {}) or {})
    # Force paper mode unless explicitly set
    live_section["mode"] = "paper"
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
    if live_cfg.mode != "paper":
        raise RuntimeError("run_live_paper enforces live.mode='paper'. Update config or use run_live_exchange.")
    cfg_local["live_cfg"] = live_cfg

    run_id = build_run_id(live_cfg.symbol, live_cfg.timeframe)
    logger = init_logger(run_id, live_cfg.log_dir, level=live_cfg.log_level)
    log_path = getattr(logger, "log_path", None)

    engine, strategy_type = create_live_engine(cfg_local, run_id=run_id, logger=logger)
    engine.run()
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
    parser = argparse.ArgumentParser(description="Run live paper trading")
    parser.add_argument(
        "--profile",
        choices=["live", "research"],
        default=None,
        help="Config profile to load; if omitted uses FINANTRADE_PROFILE or 'research'",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Override symbol")
    parser.add_argument("--timeframe", type=str, default=None, help="Override timeframe")
    args = parser.parse_args()

    config_path = args.config or None
    main(symbol=args.symbol, timeframe=args.timeframe, profile=args.profile)
