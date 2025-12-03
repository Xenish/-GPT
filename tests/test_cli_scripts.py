import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode


def test_run_backtest_help():
    rc = run_module("scripts.run_backtest", ["--help"])
    assert rc == 0


def test_run_strategy_search_help():
    rc = run_module("scripts.run_strategy_search", ["--help"])
    assert rc == 0
