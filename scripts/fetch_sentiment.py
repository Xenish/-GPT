from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.system.config_loader import load_config


def build_sentiment_dataframe(
    symbol: str,
    timeframe: str,
    *,
    ohlcv_dir: Path,
) -> pd.DataFrame:
    df_price = load_ohlcv_csv(str(ohlcv_dir / f"{symbol}_{timeframe}.csv"))
    n = len(df_price)
    if n == 0:
        raise ValueError("Cannot generate sentiment for empty OHLCV data.")
    x = np.linspace(0.0, 10.0 * math.pi, n)
    sentiment = np.sin(x) + 0.1 * np.random.randn(n)
    sentiment = np.clip(sentiment, -1.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp": df_price["timestamp"],
            "sentiment_score": sentiment,
        }
    )


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate mock sentiment series aligned to OHLCV timestamps.")
    parser.add_argument("--symbol", help="Override symbol (default from config).")
    parser.add_argument("--timeframe", help="Override timeframe (default from config).")
    parsed = parser.parse_args(args=args)

    cfg = load_config("research")
    symbol = parsed.symbol or cfg.get("symbol", "BTCUSDT")
    timeframe = parsed.timeframe or cfg.get("timeframe", "15m")
    data_cfg = cfg["data_cfg"]
    ohlcv_dir = Path(data_cfg.ohlcv_dir)
    sentiment_dir = Path(data_cfg.sentiment_dir)

    df = build_sentiment_dataframe(symbol, timeframe, ohlcv_dir=ohlcv_dir)
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    out_path = sentiment_dir / f"{symbol}_{timeframe}_sentiment.csv"
    df.to_csv(out_path, index=False)
    print(f"[sentiment] wrote {out_path}")


if __name__ == "__main__":
    main()
