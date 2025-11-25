from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.features.feature_pipeline_15m import PIPELINE_VERSION_15M
from finantradealgo.system.config_loader import load_system_config
from scripts.run_rule_backtest_15m import log_run_header, run_rule_backtest


def main() -> None:
    sys_cfg = load_system_config()
    sweep_values = [0.01, 0.02, 0.03, 0.05]
    rows = []

    for loss_limit in sweep_values:
        cfg = copy.deepcopy(sys_cfg)
        cfg.setdefault("risk", {})
        cfg["risk"]["max_daily_loss_pct"] = loss_limit
        report, _, meta = run_rule_backtest(sys_cfg=cfg)
        symbol = meta.get("symbol", cfg.get("symbol", "BTCUSDT"))
        timeframe = meta.get("timeframe", cfg.get("timeframe", "15m"))
        preset = meta.get("feature_preset", cfg.get("features", {}).get("feature_preset", "extended"))
        pipeline_version = meta.get("pipeline_version", PIPELINE_VERSION_15M)
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
    out_path = out_dir / "risk_overlay_sweep_15m.csv"
    df_summary.to_csv(out_path, index=False)

    print("\n=== Risk Overlay Sweep Summary ===")
    print(df_summary)
    print(f"\n[INFO] Saved sweep -> {out_path}")


if __name__ == "__main__":
    main()
