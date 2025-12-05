"""
Lightweight scheduler for ingestion + feature jobs.

Jobs:
    - historical_backfill: catch up candles to now (idempotent via DB upserts).
    - live_poll: per-minute REST catch-up (covers WS downtime).
    - feature_build: rebuild features per timeframe/symbol.
    - archive_cleanup: placeholder for retention/compaction hooks.

State:
    - ingestion_runs and ingestion_watermarks tables (see migrations/0003_ingestion_state.py).
    - run_id + watermarks tracked for idempotence.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import timedelta
from typing import Iterable

import click
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from prometheus_client import CollectorRegistry, Gauge, start_http_server

from finantradealgo.data_engine.ingestion import (
    BinanceRESTCandleSource,
    HistoricalOHLCVIngestor,
    timeframe_to_seconds,
)
from finantradealgo.data_engine.ingestion.state import init_state_store
from finantradealgo.data_engine.ingestion.writer import TimescaleWarehouse
from finantradealgo.system.config_loader import (
    WarehouseConfig,
    load_config,
    load_config_from_env,
    load_exchange_credentials,
)

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


def _build_wh_and_sources(cfg: dict):
    wh_cfg: WarehouseConfig = cfg["warehouse_cfg"]
    dsn = wh_cfg.get_dsn()
    table_map = {
        "ohlcv": wh_cfg.ohlcv_table,
        "funding": wh_cfg.funding_table,
        "open_interest": wh_cfg.open_interest_table,
        "flow": wh_cfg.flow_table,
        "sentiment": wh_cfg.sentiment_table,
    }
    warehouse = TimescaleWarehouse(dsn, table_map=table_map, batch_size=wh_cfg.live_batch_size)
    exch_cfg = cfg["exchange_cfg"]
    api_key, secret = load_exchange_credentials(exch_cfg)
    rest_source = BinanceRESTCandleSource(exch_cfg, api_key=api_key, secret=secret)
    return warehouse, rest_source


def historical_backfill_job(
    symbols: list[str],
    timeframes: list[str],
    ingestor: HistoricalOHLCVIngestor,
    state,
) -> None:
    run = state.start_run("historical_backfill")
    try:
        for sym in symbols:
            for tf in timeframes:
                logger.info("[backfill] %s %s catch-up", sym, tf)
                written = ingestor.catch_up_from_latest(sym, tf, lookback_bars=5000)
                logger.info("[backfill] wrote %s rows for %s %s", written, sym, tf)
                latest = ingestor.warehouse.get_latest_ts(sym, tf)
                if latest is not None:
                    state.upsert_watermark("historical_backfill", f"{sym}:{tf}", latest)
        state.finish_run(run.run_id, status="succeeded")
    except Exception as exc:
        logger.exception("historical_backfill failed")
        state.finish_run(run.run_id, status="failed", error=str(exc))
        raise


def live_poll_job(
    symbols: list[str],
    timeframes: list[str],
    ingestor: HistoricalOHLCVIngestor,
    state,
    min_interval_seconds: int = 60,
) -> None:
    run = state.start_run("live_poll")
    try:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        for sym in symbols:
            for tf in timeframes:
                scope = f"{sym}:{tf}"
                last = state.get_watermark("live_poll", scope)
                if last is not None:
                    elapsed = (now - last).total_seconds()
                    if elapsed < min_interval_seconds - 1:
                        logger.info("[live_poll] skip %s, last %ss ago", scope, elapsed)
                        continue
                written = ingestor.catch_up_from_latest(sym, tf, lookback_bars=300)
                logger.info("[live_poll] wrote %s rows for %s %s", written, sym, tf)
                latest = ingestor.warehouse.get_latest_ts(sym, tf)
                if latest is not None:
                    state.upsert_watermark("live_poll", scope, latest)
        state.finish_run(run.run_id, status="succeeded")
    except Exception as exc:
        logger.exception("live_poll failed")
        state.finish_run(run.run_id, status="failed", error=str(exc))
        raise


def feature_build_job(
    symbols: list[str],
    timeframes: list[str],
    state,
    *,
    profile: str,
    cooldown_minutes: int = 5,
) -> None:
    run = state.start_run("feature_build")
    try:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        for tf in timeframes:
            scope = f"features:{tf}"
            last = state.get_watermark("feature_build", scope)
            if last and (now - last) < timedelta(minutes=cooldown_minutes):
                logger.info("[feature_build] skip %s (cooldown)", scope)
                continue
            cmd = [
                sys.executable,
                "scripts/build_features_batch.py",
                "--profile",
                profile,
            ]
            if symbols:
                cmd.extend(["--symbols", *symbols])
            cmd.extend(["--timeframes", tf])
            logger.info("[feature_build] running %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
            state.upsert_watermark("feature_build", scope, now)
        state.finish_run(run.run_id, status="succeeded")
    except Exception as exc:
        logger.exception("feature_build failed")
        state.finish_run(run.run_id, status="failed", error=str(exc))
        raise


def archive_cleanup_job(state) -> None:
    run = state.start_run("archive_cleanup")
    try:
        # Placeholder for retention/compaction policies driven by Timescale
        state.upsert_watermark("archive_cleanup", "retention", pd.Timestamp.utcnow().tz_localize("UTC"))
        state.finish_run(run.run_id, status="succeeded")
    except Exception as exc:
        logger.exception("archive_cleanup failed")
        state.finish_run(run.run_id, status="failed", error=str(exc))
        raise


@click.command()
@click.option("--profile", default=None, type=click.Choice(["research", "live"]), help="Config profile to load (defaults to FINANTRADE_PROFILE or research).")
@click.option("--symbols", multiple=True, help="Symbols to manage.")
@click.option("--timeframes", multiple=True, help="Timeframes to manage.")
@click.option("--run-once", is_flag=True, help="Run each job once and exit (no scheduler loop).")
def main(profile: str | None, symbols, timeframes, run_once: bool):
    os.environ.setdefault("FCM_SERVER_KEY", "dummy_scheduler_key")
    cfg = load_config(profile) if profile else load_config_from_env()
    profile_name = cfg.get("profile", "research")
    wh_cfg: WarehouseConfig = cfg["warehouse_cfg"]
    warehouse, rest_source = _build_wh_and_sources(cfg)
    state = init_state_store(wh_cfg.get_dsn())
    ingestor = HistoricalOHLCVIngestor(rest_source, warehouse)

    symbols_list = _resolve_symbols(cfg, symbols)
    tfs_list = _resolve_timeframes(cfg, timeframes)
    if not symbols_list or not tfs_list:
        raise click.UsageError("No symbols/timeframes resolved. Provide via --symbols/--timeframes or system config.")

    if run_once:
        historical_backfill_job(symbols_list, tfs_list, ingestor, state)
        live_poll_job(symbols_list, tfs_list, ingestor, state)
        feature_build_job(symbols_list, tfs_list, state, profile=profile_name)
        archive_cleanup_job(state)
        return

    registry = CollectorRegistry()
    lag_gauge = Gauge("fta_watermark_lag_seconds", "Watermark lag per scope", ["job", "scope"], registry=registry)
    status_gauge = Gauge("fta_job_status", "Last run status (1=success,0=failed)", ["job"], registry=registry)

    def _update_metrics():
        try:
            # DB-backed store
            if hasattr(state, "_conn"):
                with state._conn.cursor() as cur:  # type: ignore[attr-defined]
                    cur.execute("SELECT job_name, scope, watermark_ts FROM ingestion_watermarks")
                    for job, scope, wm in cur.fetchall():
                        if wm:
                            lag = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.to_datetime(wm, utc=True)).total_seconds()
                            lag_gauge.labels(job=job, scope=scope).set(lag)
                with state._conn.cursor() as cur:  # type: ignore[attr-defined]
                    cur.execute(
                        """
                        SELECT job_name, status FROM (
                            SELECT job_name, status, ROW_NUMBER() OVER (PARTITION BY job_name ORDER BY started_at DESC) AS rn
                            FROM ingestion_runs
                        ) t WHERE rn = 1;
                        """
                    )
                    for job, status in cur.fetchall():
                        status_gauge.labels(job=job).set(1 if status == "succeeded" else 0)
            else:
                wm_state = getattr(state, "_state", {}).get("watermarks", {})  # type: ignore[attr-defined]
                for job, scopes in wm_state.items():
                    for scope, val in scopes.items():
                        lag = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.to_datetime(val, utc=True)).total_seconds()
                        lag_gauge.labels(job=job, scope=scope).set(lag)
                runs_state = getattr(state, "_state", {}).get("runs", {})  # type: ignore[attr-defined]
                latest_per_job: dict[str, str] = {}
                for rid, info in runs_state.items():
                    job_name = info.get("job_name")
                    if not job_name:
                        continue
                    latest_per_job[job_name] = info.get("status", "failed")
                for job, status in latest_per_job.items():
                    status_gauge.labels(job=job).set(1 if status == "succeeded" else 0)
        except Exception:
            logger.exception("Failed to update Prom metrics")

    start_http_server(9200, registry=registry)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        historical_backfill_job,
        trigger="cron",
        minute=0,
        hour="*/4",
        args=[symbols_list, tfs_list, ingestor, state],
        id="historical_backfill",
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        live_poll_job,
        trigger="interval",
        seconds=60,
        args=[symbols_list, tfs_list, ingestor, state],
        id="live_poll",
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        feature_build_job,
        trigger="cron",
        minute="*/5",
        args=[symbols_list, tfs_list, state],
        kwargs={"profile": profile_name},
        id="feature_build",
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        archive_cleanup_job,
        trigger="cron",
        hour="*/6",
        args=[state],
        id="archive_cleanup",
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        _update_metrics,
        trigger="interval",
        seconds=60,
        id="metrics_update",
        max_instances=1,
        replace_existing=True,
    )

    logger.info("Scheduler started with symbols=%s timeframes=%s", symbols_list, tfs_list)
    scheduler.start()


if __name__ == "__main__":
    main()
