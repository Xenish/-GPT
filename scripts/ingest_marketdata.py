"""
Unified ingestion entrypoint for historical backfill, live WS ingest, and gap repair.

Examples:
    python scripts/ingest_marketdata.py historical --config config/system.research.yml --symbols BTCUSDT --timeframes 1m 15m --lookback-days 30
    python scripts/ingest_marketdata.py live --config config/system.live.yml --symbols BTCUSDT --max-messages 500
    python scripts/ingest_marketdata.py repair-gaps --config config/system.research.yml --symbols BTCUSDT --timeframes 15m --lookback-hours 12
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Iterable

import click
import pandas as pd

from finantradealgo.data_engine.ingestion.ohlcv import (
    BinanceRESTCandleSource,
    HistoricalOHLCVIngestor,
    LiveOHLCVIngestor,
    timeframe_to_seconds,
)
from finantradealgo.data_engine.ingestion.writer import TimescaleWarehouse
from finantradealgo.system.config_loader import (
    WarehouseConfig,
    load_config,
    load_exchange_credentials,
)
from finantradealgo.data_engine.binance_ws_source import BinanceWsDataSource

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _resolve_symbols(cfg: dict, cli_symbols: Iterable[str] | None) -> list[str]:
    if cli_symbols:
        return [s.upper() for s in cli_symbols]
    data_cfg = cfg.get("data_cfg")
    if data_cfg and getattr(data_cfg, "symbols", None):
        return [s.upper() for s in data_cfg.symbols]
    sym = cfg.get("symbol") or cfg.get("live_cfg", {}).symbol
    return [str(sym).upper()] if sym else []


def _resolve_timeframes(cfg: dict, cli_timeframes: Iterable[str] | None) -> list[str]:
    if cli_timeframes:
        return [tf for tf in cli_timeframes]
    data_cfg = cfg.get("data_cfg")
    if data_cfg and getattr(data_cfg, "timeframes", None):
        return list(data_cfg.timeframes)
    tf = cfg.get("timeframe") or cfg.get("live_cfg", {}).timeframe
    return [str(tf)] if tf else []


def _build_warehouse(cfg: dict) -> TimescaleWarehouse:
    wh_cfg: WarehouseConfig = cfg["warehouse_cfg"]
    dsn = wh_cfg.get_dsn()
    table_map = {
        "ohlcv": wh_cfg.ohlcv_table,
        "funding": wh_cfg.funding_table,
        "open_interest": wh_cfg.open_interest_table,
        "flow": wh_cfg.flow_table,
        "sentiment": wh_cfg.sentiment_table,
    }
    return TimescaleWarehouse(dsn, table_map=table_map, batch_size=wh_cfg.live_batch_size)


@click.group()
@click.option("--profile", default="research", type=click.Choice(["research", "live"]), help="Config profile to load.")
@click.option("--config", default=None, help="Explicit config path (overrides profile).")
@click.pass_context
def cli(ctx: click.Context, profile: str, config: str | None):
    # Ensure optional FCM env placeholder is satisfied for config parsing
    os.environ.setdefault("FCM_SERVER_KEY", "dummy_ingest_key")
    if config:
        raise RuntimeError(
            "Explicit config path is no longer supported. Use load_config(profile=...) with system.research.yml or system.live.yml."
        )
    cfg = load_config(profile)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg
    ctx.obj["warehouse"] = _build_warehouse(cfg)
    exch_cfg = cfg["exchange_cfg"]
    api_key, secret = load_exchange_credentials(exch_cfg)
    ctx.obj["rest_source"] = BinanceRESTCandleSource(exch_cfg, api_key=api_key, secret=secret)


@cli.command("historical")
@click.option("--symbols", multiple=True, help="Symbols to ingest (defaults to config data.symbols).")
@click.option("--timeframes", multiple=True, help="Timeframes to ingest (defaults to config data.timeframes).")
@click.option("--start", type=click.DateTime(), help="Start datetime (UTC). Defaults to lookback from now.")
@click.option("--end", type=click.DateTime(), help="End datetime (UTC). Defaults to now.")
@click.option("--lookback-days", type=int, default=365, show_default=True, help="Lookback window when start not provided.")
@click.pass_context
def historical(ctx: click.Context, symbols, timeframes, start, end, lookback_days: int):
    cfg = ctx.obj["cfg"]
    warehouse: TimescaleWarehouse = ctx.obj["warehouse"]
    source: BinanceRESTCandleSource = ctx.obj["rest_source"]
    ingestor = HistoricalOHLCVIngestor(source, warehouse)

    symbols_list = _resolve_symbols(cfg, symbols)
    tfs_list = _resolve_timeframes(cfg, timeframes)
    if not symbols_list or not tfs_list:
        raise click.UsageError("No symbols/timeframes resolved. Provide via --symbols/--timeframes or system config.")

    end_ts = end or pd.Timestamp.utcnow().tz_localize("UTC")
    start_ts = start or end_ts - timedelta(days=lookback_days)

    for sym in symbols_list:
        for tf in tfs_list:
            logger.info("Historical ingest %s %s [%s -> %s]", sym, tf, start_ts, end_ts)
            written = ingestor.backfill_range(sym, tf, start_ts, end_ts)
            logger.info("Written %s rows for %s %s", written, sym, tf)


@cli.command("repair-gaps")
@click.option("--symbols", multiple=True, help="Symbols to repair.")
@click.option("--timeframes", multiple=True, help="Timeframes to repair.")
@click.option("--lookback-hours", type=int, default=24, show_default=True)
@click.pass_context
def repair_gaps(ctx: click.Context, symbols, timeframes, lookback_hours: int):
    cfg = ctx.obj["cfg"]
    warehouse: TimescaleWarehouse = ctx.obj["warehouse"]
    source: BinanceRESTCandleSource = ctx.obj["rest_source"]
    ingestor = HistoricalOHLCVIngestor(source, warehouse)

    symbols_list = _resolve_symbols(cfg, symbols)
    tfs_list = _resolve_timeframes(cfg, timeframes)
    for sym in symbols_list:
        for tf in tfs_list:
            repaired = ingestor.repair_recent_gaps(sym, tf, lookback_hours=lookback_hours)
            logger.info("Repaired %s missing candles for %s %s", repaired, sym, tf)


@cli.command("live")
@click.option("--symbols", multiple=True, help="Symbols to stream (defaults to config).")
@click.option("--max-messages", type=int, default=None, help="Process limited messages (useful for smoke-tests).")
@click.pass_context
def live(ctx: click.Context, symbols, max_messages: int | None):
    cfg = ctx.obj["cfg"]
    warehouse: TimescaleWarehouse = ctx.obj["warehouse"]
    rest_source: BinanceRESTCandleSource = ctx.obj["rest_source"]
    live_cfg = cfg["live_cfg"]
    symbols_list = _resolve_symbols(cfg, symbols)
    ws_source = BinanceWsDataSource(cfg, symbols=symbols_list)
    ingestor = LiveOHLCVIngestor(
        ws_source=ws_source,
        rest_source=rest_source,
        warehouse=warehouse,
        live_cfg=live_cfg,
        flush_every=cfg["warehouse_cfg"].live_batch_size,
    )
    logger.info("Starting live ingest for %s on %s", symbols_list, live_cfg.timeframe)
    ingestor.run(max_messages=max_messages)


if __name__ == "__main__":
    cli()
