from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple

from finantradealgo.system.config_loader import load_config


def add_symbol_timeframe_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--symbol", default=None, help="Symbol override, e.g. AIAUSDT.")
    parser.add_argument("--timeframe", default=None, help="Timeframe override, e.g. 15m.")
    return parser


def parse_symbol_timeframe_args(description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_symbol_timeframe_args(parser)
    return parser.parse_args()


def load_config_with_overrides(
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config("research")
    cfg_local = dict(cfg)
    if symbol:
        cfg_local["symbol"] = symbol
        if "live" in cfg_local and isinstance(cfg_local["live"], dict):
            cfg_local["live"]["symbol"] = symbol
    if timeframe:
        cfg_local["timeframe"] = timeframe
        if "live" in cfg_local and isinstance(cfg_local["live"], dict):
            cfg_local["live"]["timeframe"] = timeframe
    return cfg_local


def apply_symbol_timeframe_overrides(
    cfg: Dict[str, Any],
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    cfg_local = dict(cfg)
    if symbol:
        cfg_local["symbol"] = symbol
        if "live" in cfg_local and isinstance(cfg_local["live"], dict):
            cfg_local["live"]["symbol"] = symbol
    if timeframe:
        cfg_local["timeframe"] = timeframe
        if "live" in cfg_local and isinstance(cfg_local["live"], dict):
            cfg_local["live"]["timeframe"] = timeframe
    return cfg_local


def resolve_symbol_timeframe(
    cfg: Dict[str, Any],
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Tuple[str, str]:
    if symbol is None:
        symbol = cfg.get("symbol") or getattr(cfg.get("live_cfg"), "symbol", None)
    if timeframe is None:
        timeframe = cfg.get("timeframe") or getattr(cfg.get("live_cfg"), "timeframe", None)
    if not symbol or not timeframe:
        raise ValueError("Symbol/timeframe must be provided via config or arguments.")
    return symbol, timeframe
