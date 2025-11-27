from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from finantradealgo.backtester.scenario_engine import Scenario, run_scenarios
from finantradealgo.system.config_loader import load_system_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-strategy scenario grid on 15m data.")
    parser.add_argument("--symbol", help="Override symbol from config.")
    parser.add_argument("--timeframe", help="Override timeframe from config.")
    args = parser.parse_args()

    cfg = load_system_config()
    symbol = args.symbol or cfg.get("symbol", "AIAUSDT")
    timeframe = args.timeframe or cfg.get("timeframe", "15m")

    rule_param_grid = {
        "rsi_entry_min": [20, 25, 30],
        "rsi_entry_max": [60, 70],
        "ms_trend_min": [0.2, 0.4],
        "ms_trend_max": [0.8],
        "tp_atr_mult": [2.0, 3.0],
        "sl_atr_mult": [1.0, 1.5],
    }

    ml_param_grid = {
        "entry_threshold": [0.55, 0.6, 0.65],
        "exit_threshold": [0.45, 0.4],
    }

    scenarios: list[Scenario] = []

    rule_keys = list(rule_param_grid.keys())
    for values in itertools.product(*(rule_param_grid[k] for k in rule_keys)):
        params = dict(zip(rule_keys, values))
        label = (
            f"rule_rsi_{params['rsi_entry_min']}-{params['rsi_entry_max']}_"
            f"tp{params['tp_atr_mult']}_sl{params['sl_atr_mult']}"
        )
        scenarios.append(
            Scenario(
                symbol=symbol,
                timeframe=timeframe,
                strategy="rule",
                params=params,
                label=label,
            )
        )

    ml_keys = list(ml_param_grid.keys())
    for values in itertools.product(*(ml_param_grid[k] for k in ml_keys)):
        params = dict(zip(ml_keys, values))
        label = f"ml_th_{params['entry_threshold']}_{params['exit_threshold']}"
        scenarios.append(
            Scenario(
                symbol=symbol,
                timeframe=timeframe,
                strategy="ml",
                params=params,
                label=label,
            )
        )

    df_results = run_scenarios(cfg, scenarios)
    if df_results.empty:
        print("No scenario results generated.")
        return

    df_results["params_json"] = df_results["params"].apply(lambda d: json.dumps(d, sort_keys=True))
    df_results = df_results.drop(columns=["params"])

    out_dir = Path("outputs/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scenario_grid_{symbol}_{timeframe}.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Wrote {len(df_results)} scenarios to {out_path}")


if __name__ == "__main__":
    main()
