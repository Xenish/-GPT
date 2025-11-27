from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.system.config_loader import load_system_config


def _load_funding(symbol: str, timeframe: str, external_dir: Path) -> pd.DataFrame:
    funding_path = external_dir / "funding" / f"{symbol}_{timeframe}_funding.csv"
    if not funding_path.is_file():
        raise FileNotFoundError(f"Funding CSV not found at {funding_path}")
    df = pd.read_csv(funding_path)
    if "timestamp" not in df.columns:
        raise ValueError("Funding CSV must include timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return df


def build_flow_dataframe(
    symbol: str,
    timeframe: str,
    *,
    ohlcv_dir: Path,
    external_dir: Path,
    output_dir: Path,
) -> Path:
    ohlcv_path = ohlcv_dir / f"{symbol}_{timeframe}.csv"
    df_price = load_ohlcv_csv(str(ohlcv_path))
    df_funding = _load_funding(symbol, timeframe, external_dir)

    df = pd.merge_asof(
        df_price[["timestamp", "close"]],
        df_funding.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0

    df["perp_premium"] = df["funding_rate"].fillna(0.0) * 10_000.0
    df["basis"] = df["close"].rolling(window=48, min_periods=1).mean() - df["close"]
    flow_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "perp_premium": df["perp_premium"],
            "basis": df["basis"],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_{timeframe}_flow.csv"
    flow_df.to_csv(out_path, index=False)
    return out_path


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate mock flow data (perp premium / basis) for 15m.")
    parser.add_argument("--symbol", help="Symbol override, defaults to config value.")
    parser.add_argument("--timeframe", help="Timeframe override, defaults to config value (15m).")
    parsed = parser.parse_args(args=args)

    cfg = load_system_config()
    symbol = parsed.symbol or cfg.get("symbol", "BTCUSDT")
    timeframe = parsed.timeframe or cfg.get("timeframe", "15m")
    data_cfg = cfg.get("data", {}) or {}
    ohlcv_dir = Path(data_cfg.get("ohlcv_dir", "data/ohlcv"))
    external_dir = Path(data_cfg.get("external_dir", "data/external"))
    flow_dir = Path(data_cfg.get("flow_dir", "data/flow"))

    path = build_flow_dataframe(
        symbol,
        timeframe,
        ohlcv_dir=ohlcv_dir,
        external_dir=external_dir,
        output_dir=flow_dir,
    )
    print(f"[flow] wrote {path}")


if __name__ == "__main__":
    main()
