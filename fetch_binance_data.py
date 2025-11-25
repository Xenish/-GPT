from __future__ import annotations

from pathlib import Path

from finantradealgo.data_sources import BinanceKlinesConfig, fetch_klines_series


def main() -> None:
    total_candles = 10_000
    config = BinanceKlinesConfig(symbol="BTCUSDT", interval="15m", limit=1000)
    df = fetch_klines_series(config, total_limit=total_candles)

    output_path = Path("data/BTCUSDT_P_15m.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"Saved {len(df)} candles for {config.symbol} ({config.interval}) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
