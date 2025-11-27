from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.system.config_loader import LiveConfig, load_system_config
from finantradealgo.system.logger import init_logger


def build_run_id(symbol: str, timeframe: str, prefix: str = "live_exchange") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{symbol}_{timeframe}_{ts}"


def main() -> None:
    cfg = load_system_config()
    live_cfg = cfg.get("live_cfg")
    if not isinstance(live_cfg, LiveConfig):
        live_cfg = LiveConfig.from_dict(
            cfg.get("live"),
            default_symbol=cfg.get("symbol"),
            default_timeframe=cfg.get("timeframe"),
        )
        cfg["live_cfg"] = live_cfg

    run_id = build_run_id(live_cfg.symbol, live_cfg.timeframe)
    logger = init_logger(run_id, live_cfg.log_dir, level=live_cfg.log_level)
    log_path = getattr(logger, "log_path", None)

    engine, strategy_name = create_live_engine(cfg, mode="exchange", run_id=run_id, logger=logger)

    testnet_flag = getattr(cfg.get("exchange_cfg"), "testnet", True)
    dry_run_flag = getattr(cfg.get("exchange_cfg"), "dry_run", True)
    print(f"Starting live exchange run: {run_id} | strategy={strategy_name} | testnet={testnet_flag} | dry_run={dry_run_flag}")
    if dry_run_flag:
        print("[INFO] exchange.dry_run=true › orders should be treated as simulation/testnet. Verify before deploying real capital.")

    try:
        engine.run()
    except KeyboardInterrupt:
        print("Interrupted by user; stopping...")
    finally:
        engine.stop()
        if log_path:
            print(f"Log file: {log_path}")
        print(f"Latest snapshot: {engine.latest_state_path}")


if __name__ == "__main__":
    main()
