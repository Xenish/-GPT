from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.live_trading.factories import create_live_engine
from finantradealgo.system.config_loader import LiveConfig, load_config
from finantradealgo.system.logger import init_logger


def build_run_id(symbol: str, timeframe: str, prefix: str = "live_exchange") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{symbol}_{timeframe}_{ts}"


def main(config_path: str | None = None, profile: str = "live") -> None:
    # Load config with profile support
    if config_path:
        raise RuntimeError(
            "Explicit config path is no longer supported. Use load_config(profile=...) with system.live.yml."
        )
    cfg = load_config(profile)

    # SAFETY: Assert live/paper mode for live trading
    cfg_mode = cfg.get("mode", "unknown")
    if cfg_mode not in ("live", "paper"):
        raise RuntimeError(
            f"Live trading must run with mode='live' or mode='paper' config. Got mode='{cfg_mode}'. "
            f"Use --config config/system.live.yml or set FT_CONFIG_PATH=config/system.live.yml"
        )
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
        print("[INFO] exchange.dry_run=true � orders should be treated as simulation/testnet. Verify before deploying real capital.")

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
    parser = argparse.ArgumentParser(description="Run live exchange trading")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit config path (overrides profile)",
    )
    parser.add_argument(
        "--profile",
        choices=["live", "research"],
        default="live",
        help="Config profile to load when --config is not provided",
    )
    args = parser.parse_args()

    main(config_path=args.config, profile=args.profile)
