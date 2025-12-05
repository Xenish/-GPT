"""
Feature build runner supporting batch and incremental modes.

Examples:
    FINANTRADE_PROFILE=research python scripts/run_feature_builder.py batch --symbols BTCUSDT --timeframes 15m
    FINANTRADE_PROFILE=research python scripts/run_feature_builder.py incremental --symbols BTCUSDT --timeframes 15m
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import click
import pandas as pd

from finantradealgo.features.feature_builder import FeatureBuilderService, FeatureSinkConfig
from finantradealgo.data_engine.ingestion.state import IngestionStateStore, init_state_store
from finantradealgo.system.config_loader import load_config, load_config_from_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _resolve_symbols(cfg: dict, cli_symbols: Iterable[str] | None) -> list[str]:
    if cli_symbols:
        return [s.upper() for s in cli_symbols]
    data_cfg = cfg.get("data_cfg")
    if data_cfg and getattr(data_cfg, "symbols", None):
        return [s.upper() for s in data_cfg.symbols]
    sym = cfg.get("symbol")
    return [str(sym).upper()] if sym else []


def _resolve_timeframes(cfg: dict, cli_timeframes: Iterable[str] | None) -> list[str]:
    if cli_timeframes:
        return [tf for tf in cli_timeframes]
    data_cfg = cfg.get("data_cfg")
    if data_cfg and getattr(data_cfg, "timeframes", None):
        return list(data_cfg.timeframes)
    tf = cfg.get("timeframe")
    return [str(tf)] if tf else []


def _build_sink(kind: str, output_dir: Path, duckdb_path: Path | None, duckdb_table: str) -> FeatureSinkConfig:
    cfg = FeatureSinkConfig(
        kind=kind,
        output_dir=output_dir,
        duckdb_path=duckdb_path,
        duckdb_table=duckdb_table,
    )
    return cfg


@click.group()
@click.option("--profile", default=None, type=click.Choice(["research", "live"]), help="Config profile to load (defaults to FINANTRADE_PROFILE or research).")
@click.option("--output-dir", type=Path, default=None, help="Parquet output dir (defaults to data.features_dir).")
@click.option("--sink", type=click.Choice(["parquet", "duckdb"]), default="parquet")
@click.option("--duckdb-path", type=Path, default=None, help="DuckDB file path for duckdb sink.")
@click.option("--duckdb-table", type=str, default="features", help="DuckDB base table name.")
@click.pass_context
def cli(ctx: click.Context, profile: str | None, output_dir: Path | None, sink: str, duckdb_path: Path | None, duckdb_table: str):
    os.environ.setdefault("FCM_SERVER_KEY", "dummy_feature_builder_key")
    sys_cfg = load_config(profile) if profile else load_config_from_env()
    data_cfg = sys_cfg["data_cfg"]
    sink_cfg = _build_sink(
        sink,
        output_dir or Path(data_cfg.features_dir),
        duckdb_path,
        duckdb_table,
    )
    service = FeatureBuilderService(sys_cfg=sys_cfg, sink_cfg=sink_cfg)
    ctx.ensure_object(dict)
    ctx.obj["service"] = service
    ctx.obj["cfg"] = sys_cfg


@cli.command("batch")
@click.option("--symbols", multiple=True, help="Symbols to build.")
@click.option("--timeframes", multiple=True, help="Timeframes to build.")
@click.option("--start", type=click.DateTime(), default=None, help="Start datetime UTC")
@click.option("--end", type=click.DateTime(), default=None, help="End datetime UTC (default now)")
@click.pass_context
def batch(ctx: click.Context, symbols, timeframes, start, end):
    svc: FeatureBuilderService = ctx.obj["service"]
    cfg = ctx.obj["cfg"]
    symbols_list = _resolve_symbols(cfg, symbols)
    tfs_list = _resolve_timeframes(cfg, timeframes)
    for sym in symbols_list:
        for tf in tfs_list:
            logger.info("Batch feature build %s %s", sym, tf)
            df, meta = svc.build_batch(
                symbol=sym,
                timeframe=tf,
                start_ts=pd.to_datetime(start) if start else None,
                end_ts=pd.to_datetime(end) if end else None,
            )
            logger.info("Built %s rows (%s features) -> %s", len(df), len(df.columns), meta.get("sink_path"))


@cli.command("incremental")
@click.option("--symbols", multiple=True, help="Symbols to build.")
@click.option("--timeframes", multiple=True, help="Timeframes to build.")
@click.option("--job-name", default="feature_incremental", help="Watermark job name.")
@click.option("--context-bars", type=int, default=500, help="Bars of context to include before watermark.")
@click.option("--dsn", default=None, help="State store DSN (overrides warehouse.dsn_env).")
@click.pass_context
def incremental(ctx: click.Context, symbols, timeframes, job_name: str, context_bars: int, dsn: str | None):
    svc: FeatureBuilderService = ctx.obj["service"]
    cfg = ctx.obj["cfg"]
    wh_cfg = cfg["warehouse_cfg"]
    if dsn:
        dsn_resolved = dsn
    else:
        dsn_resolved = wh_cfg.get_dsn()
    state_store = init_state_store(dsn_resolved)
    symbols_list = _resolve_symbols(cfg, symbols)
    tfs_list = _resolve_timeframes(cfg, timeframes)
    for sym in symbols_list:
        for tf in tfs_list:
            logger.info("Incremental feature build %s %s", sym, tf)
            res = svc.build_incremental(
                symbol=sym,
                timeframe=tf,
                state_store=state_store,
                job_name=job_name,
                context_bars=context_bars,
            )
            logger.info("Incremental result: %s", res)


if __name__ == "__main__":
    cli()
