from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.features.feature_pipeline import PIPELINE_VERSION
from finantradealgo.system.config_loader import load_config
from scripts.cli_utils import parse_symbol_timeframe_args, resolve_symbol_timeframe
from scripts.run_rule_backtest import log_run_header, run_rule_backtest


def main(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
    sys_cfg = load_config("research")
    resolved_symbol, resolved_timeframe = resolve_symbol_timeframe(sys_cfg, symbol=symbol, timeframe=timeframe)
    sweep_values = [0.01, 0.02, 0.03, 0.05]
    rows = []

    for loss_limit in sweep_values:
        cfg = copy.deepcopy(sys_cfg)
        cfg.setdefault("risk", {})
        cfg["risk"]["max_daily_loss_pct"] = loss_limit
        report, _, meta = run_rule_backtest(sys_cfg=cfg, symbol=resolved_symbol, timeframe=resolved_timeframe)
        symbol = meta.get("symbol", cfg.get("symbol", "BTCUSDT"))
        timeframe = meta.get("timeframe", cfg.get("timeframe"))
        preset = meta.get("feature_preset", cfg.get("features", {}).get("feature_preset", "extended"))
        pipeline_version = meta.get("pipeline_version", PIPELINE_VERSION)
        log_run_header(symbol, timeframe, preset, pipeline_version, extra=f"max_daily_loss_pct={loss_limit}")

        eq = report["equity_metrics"]
        ts = report["trade_stats"]
        risk_stats = report.get("risk_stats", {}) or {}
        blocked_total = sum(risk_stats.get("blocked_entries", {}).values())
        rows.append(
            {
                "max_daily_loss_pct": loss_limit,
                "final_equity": eq["final_equity"],
                "cum_return": eq["cum_return"],
                "max_drawdown": eq["max_drawdown"],
                "trade_count": ts["trade_count"],
                "win_rate": ts["win_rate"],
                "blocked_entries": blocked_total,
            }
        )

    df_summary = pd.DataFrame(rows)
    out_dir = Path("outputs") / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"risk_overlay_sweep_{resolved_symbol}_{resolved_timeframe}.csv"
    df_summary.to_csv(out_path, index=False)

    print("\n=== Risk Overlay Sweep Summary ===")
    print(df_summary)
    print(f"\n[INFO] Saved sweep -> {out_path}")


if __name__ == "__main__":
    args = parse_symbol_timeframe_args("Sweep risk overlays for the rule strategy.")
    main(symbol=args.symbol, timeframe=args.timeframe)
