from __future__ import annotations

from typing import Optional

import click

from finantradealgo.backtester.runners import run_backtest_once
from finantradealgo.system.config_loader import load_config, load_config_from_env
from scripts.run_build_features import main as build_features_main
from scripts.run_live_paper import main as live_paper_main
from scripts.run_ml_train import main as ml_train_main


def _override_symbol_tf(
    cfg: dict,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> dict:
    cfg_local = dict(cfg)
    if symbol:
        cfg_local["symbol"] = symbol
    if timeframe:
        cfg_local["timeframe"] = timeframe
    return cfg_local


@click.group()
def cli() -> None:
    """FinanTradeAlgo command line interface."""


@cli.command("build-features")
@click.option("--symbol", default=None, help="Symbol to process, e.g. AIAUSDT")
@click.option("--tf", "timeframe", default=None, help="Timeframe, e.g. 15m")
def build_features(symbol: Optional[str], timeframe: Optional[str]) -> None:
    """Build the configured feature pipeline."""
    build_features_main(symbol=symbol, timeframe=timeframe)


@cli.command()
@click.option("--strategy", required=True, help="Strategy name, e.g. rule or ml")
@click.option("--symbol", default=None, help="Override symbol")
@click.option("--tf", "timeframe", default=None, help="Override timeframe")
def backtest(strategy: str, symbol: Optional[str], timeframe: Optional[str]) -> None:
    """Run a single backtest using the configured pipelines."""
    cfg = load_config_from_env()
    if cfg.get("profile") != "research":
        raise RuntimeError(
            f"Backtest CLI requires the 'research' profile. Set FINANTRADE_PROFILE=research (current: {cfg.get('profile')!r})."
        )
    cfg_local = _override_symbol_tf(cfg, symbol=symbol, timeframe=timeframe)
    resolved_symbol = cfg_local.get("symbol", cfg.get("symbol", "BTCUSDT"))
    resolved_timeframe = cfg_local.get("timeframe", cfg.get("timeframe", "15m"))

    result = run_backtest_once(
        symbol=resolved_symbol,
        timeframe=resolved_timeframe,
        strategy_name=strategy,
        cfg=cfg_local,
    )

    click.echo(f"Run ID      : {result.get('run_id')}")
    click.echo(f"Symbol/TF   : {result.get('symbol')}/{result.get('timeframe')}")
    click.echo(f"Strategy    : {result.get('strategy')}")
    metrics = result.get("metrics", {}) or {}
    if metrics:
        click.echo("Metrics:")
        for key, value in metrics.items():
            click.echo(f"  {key}: {value}")
    click.echo(f"Trade count : {result.get('trade_count', 0)}")


@cli.command(name="ml-train")
@click.option("--symbol", default=None, help="Override symbol")
@click.option("--tf", "timeframe", default=None, help="Override timeframe")
@click.option("--preset", default=None, help="Feature preset override, e.g. extended")
def ml_train(symbol: Optional[str], timeframe: Optional[str], preset: Optional[str]) -> None:
    """Train the ML model pipeline on 15m data."""
    ml_train_main(symbol=symbol, timeframe=timeframe, preset=preset)


@cli.command(name="live-paper")
@click.option("--symbol", default=None, help="Override symbol")
@click.option("--tf", "timeframe", default=None, help="Override timeframe")
def live_paper(symbol: Optional[str], timeframe: Optional[str]) -> None:
    """Start the paper trading live/replay loop."""
    live_paper_main(symbol=symbol, timeframe=timeframe)


def main() -> None:
    cli()
