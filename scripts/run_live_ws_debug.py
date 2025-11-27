from __future__ import annotations

import argparse
import sys
from typing import List

from finantradealgo.system.config_loader import load_system_config
from finantradealgo.data_engine.binance_ws_source import BinanceWsDataSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to Binance WS data source and print a few aggregated bars."
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="Symbol(s) to stream (default: live.symbol or config.symbol). "
        "Can be provided multiple times.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of aggregated bars to print before exiting (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_system_config()
    live_cfg = cfg.get("live", {}) or {}

    default_symbol = None
    if args.symbol:
        symbols: List[str] = [s.upper() for s in args.symbol]
    else:
        default_symbol = (
            live_cfg.get("symbol")
            or cfg.get("symbol")
            or next(iter(cfg.get("symbols", []) or []), None)
        )
        symbols = [default_symbol or "AIAUSDT"]

    print(f"[WS DEBUG] connecting for symbols={symbols}")
    data_source = BinanceWsDataSource(cfg, symbols)
    data_source.connect()

    printed = 0
    try:
        while printed < args.count:
            bar = data_source.next_bar()
            if bar is None:
                continue
            print(
                f"{bar.symbol:<10} {bar.timeframe:<5} {bar.close_time} "
                f"O={bar.open:.4f} H={bar.high:.4f} L={bar.low:.4f} "
                f"C={bar.close:.4f} V={bar.volume:.4f}"
            )
            printed += 1
    except KeyboardInterrupt:
        print("\n[WS DEBUG] interrupted by user, shutting down...", file=sys.stderr)
    finally:
        data_source.close()
        print("[WS DEBUG] closed.")


if __name__ == "__main__":
    main()
