from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestConfig, Backtester
from finantradealgo.features.feature_pipeline import (
    PIPELINE_VERSION,
    build_feature_pipeline_from_system_config,
)
from finantradealgo.core.report import ReportConfig, generate_report
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.rule_signals import RuleSignalStrategy, RuleStrategyConfig
from finantradealgo.system.config_loader import load_system_config
from scripts.cli_utils import (
    apply_symbol_timeframe_overrides,
    parse_symbol_timeframe_args,
    resolve_symbol_timeframe,
)


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str, extra: str | None = None) -> None:
    msg = (
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )
    if extra:
        msg += f" {extra}"
    print(msg)


def run_rule_backtest(
    sys_cfg: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> tuple[dict, pd.DataFrame, dict]:
    sys_cfg = sys_cfg or load_system_config()
    cfg_local = apply_symbol_timeframe_overrides(sys_cfg, symbol=symbol, timeframe=timeframe)
    resolved_symbol, resolved_timeframe = resolve_symbol_timeframe(cfg_local, symbol=symbol, timeframe=timeframe)

    if df is None:
        df, pipeline_meta = build_feature_pipeline_from_system_config(
            cfg_local,
            symbol=resolved_symbol,
            timeframe=resolved_timeframe,
        )
    else:
        df = df.copy()
        pipeline_meta = {
            "feature_preset": cfg_local.get("features", {}).get("feature_preset", "extended"),
            "pipeline_version": PIPELINE_VERSION,
            "symbol": resolved_symbol,
            "timeframe": resolved_timeframe,
        }

    strat_cfg = RuleStrategyConfig.from_dict(cfg_local.get("risk"))
    strategy = RuleSignalStrategy(strat_cfg)

    risk_engine = RiskEngine(RiskConfig.from_dict(cfg_local.get("risk")))

    bt_config = BacktestConfig(
        initial_cash=10_000.0,
        fee_pct=0.0004,
        slippage_pct=0.0005,
    )

    backtester = Backtester(
        strategy=strategy,
        risk_engine=risk_engine,
        config=bt_config,
    )

    result = backtester.run(df)

    report_cfg = ReportConfig(regime_columns=["regime_trend", "regime_vol"])
    report = generate_report(result, df=df, config=report_cfg)
    report["risk_stats"] = result.get("risk_stats", {})
    return report, df, pipeline_meta


def _ensure_output_dirs() -> tuple[Path, Path]:
    bt_dir = Path("outputs") / "backtests"
    tr_dir = Path("outputs") / "trades"
    bt_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)
    return bt_dir, tr_dir


def main(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
    report, df, meta = run_rule_backtest(symbol=symbol, timeframe=timeframe)
    symbol = meta.get("symbol", "UNKNOWN")
    timeframe = meta.get("timeframe", "UNKNOWN")
    preset = meta.get("feature_preset", "extended")
    pipeline_version = meta.get("pipeline_version", PIPELINE_VERSION)
    log_run_header(symbol, timeframe, preset, pipeline_version, extra="mode=rule_backtest")
    print(f"[INFO] Prepared DF shape: {df.shape}")

    eq = report["equity_metrics"]
    ts = report["trade_stats"]

    print(f"\n=== Rule-based strategy backtest ({timeframe}) ===")
    print("Equity:")
    print(f"  Initial cash : {eq['initial_cash']}")
    print(f"  Final equity : {eq['final_equity']}")
    print(f"  Cumulative R.: {eq['cum_return']}")
    print(f"  Max drawdown : {eq['max_drawdown']}")
    print(f"  Sharpe       : {eq['sharpe']}")

    print("\nTrades:")
    print(f"  Count        : {ts['trade_count']}")
    print(f"  Win rate     : {ts['win_rate']}")
    print(f"  Avg PnL      : {ts['avg_pnl']}")
    print(f"  Avg Win      : {ts['avg_win']}")
    print(f"  Avg Loss     : {ts['avg_loss']}")
    print(f"  ProfitFactor : {ts['profit_factor']}")
    print(f"  Median hold  : {ts['median_hold_time']}")

    risk_stats = report.get("risk_stats", {}) or {}
    blocked_total = sum(risk_stats.get("blocked_entries", {}).values())
    print(f"[RISK] blocked_entries_total={blocked_total}")

    bt_dir, tr_dir = _ensure_output_dirs()
    eq_path = bt_dir / f"rule_equity_{symbol}_{timeframe}.csv"
    trades_path = tr_dir / f"rule_trades_{symbol}_{timeframe}.csv"
    report["equity_curve"].to_csv(eq_path, header=True)
    report["trades"].to_csv(trades_path, index=False)
    print(f"\n[INFO] Saved {eq_path} and {trades_path}")


if __name__ == "__main__":
    args = parse_symbol_timeframe_args("Run rule-based backtest.")
    main(symbol=args.symbol, timeframe=args.timeframe)
