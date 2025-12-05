from __future__ import annotations

import subprocess
import sys

import pytest
from click.testing import CliRunner

from finantradealgo.system import cli as system_cli
import scripts.ingest_marketdata as ingest_cli
import scripts.run_feature_builder as feature_cli


def test_system_cli_backtest_uses_env_loader(monkeypatch):
    called = {}

    def fake_load_config_from_env():
        called["env"] = True
        return {"profile": "research", "symbol": "BTCUSDT", "timeframe": "15m"}

    def fake_run_backtest_once(symbol, timeframe, strategy_name, cfg):
        called["run"] = (symbol, timeframe, strategy_name, cfg.get("profile"))
        return {
            "run_id": "test-run",
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy_name,
            "metrics": {},
            "trade_count": 0,
        }

    monkeypatch.setattr(system_cli, "load_config_from_env", fake_load_config_from_env)
    monkeypatch.setattr(system_cli, "run_backtest_once", fake_run_backtest_once)

    runner = CliRunner()
    result = runner.invoke(system_cli.cli, ["backtest", "--strategy", "rule"])
    assert result.exit_code == 0, result.output
    assert called.get("env") is True
    assert called.get("run") == ("BTCUSDT", "15m", "rule", "research")


@pytest.mark.parametrize(
    "cli_entry,args",
    [
        (ingest_cli.cli, ["--config", "foo", "historical"]),
        (feature_cli.cli, ["--config", "foo", "batch"]),
    ],
)
def test_click_scripts_reject_config_option(cli_entry, args):
    runner = CliRunner()
    result = runner.invoke(cli_entry, args)
    assert result.exit_code != 0
    assert "No such option: --config" in result.output


def test_schedule_ingestion_rejects_config_option():
    try:
        import scripts.schedule_ingestion as schedule_cli
    except ImportError:
        pytest.skip("apscheduler not installed; skip schedule_ingestion config check")
    runner = CliRunner()
    result = runner.invoke(schedule_cli.main, ["--config", "foo"])
    assert result.exit_code != 0
    assert "No such option: --config" in result.output


def test_run_live_exchange_rejects_config_option():
    proc = subprocess.run(
        [sys.executable, "scripts/run_live_exchange.py", "--config", "foo"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "unrecognized arguments: --config" in proc.stderr
