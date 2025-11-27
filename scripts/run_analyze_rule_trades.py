from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import pandas as pd


def main() -> None:
    # 1) CSV'yi oku
    trades = pd.read_csv("rule_trades_15m.csv")

    print("=== Columns ===")
    print(list(trades.columns))

    # 2) PnL kolonu: bizde zaten 'pnl'
    pnl_col = "pnl"
    if pnl_col not in trades.columns:
        raise ValueError(f"'{pnl_col}' kolonu yok, lütfen file'a bak.")

    print("\n=== Basic PnL stats ===")
    print(trades[[pnl_col]].describe())

    # 3) Entry time: 'timestamp'
    entry_col = "timestamp"
    if entry_col not in trades.columns:
        print("\n[WARN] 'timestamp' kolonu yok, saat/gün analizi atlandı.")
        return

    # Datetime'e çevir
    trades[entry_col] = pd.to_datetime(trades[entry_col])

    # 4) Saat ve hafta günü çıkar
    trades["entry_hour"] = trades[entry_col].dt.hour
    trades["weekday"] = trades[entry_col].dt.dayofweek  # 0=Mon, 6=Sun

    # 5) Saat bazlı performans
    print("\n=== PnL by entry hour ===")
    print(
        trades.groupby("entry_hour")[pnl_col]
        .agg(["count", "mean", "sum"])
        .sort_values("sum", ascending=False)
    )

    # 6) Gün bazlı performans
    print("\n=== PnL by weekday (0=Mon, 6=Sun) ===")
    print(
        trades.groupby("weekday")[pnl_col]
        .agg(["count", "mean", "sum"])
        .sort_values("sum", ascending=False)
    )

    # 7) Exit timestamp varsa, hold time çıkar
    if "timestamp_exit" in trades.columns:
        trades["timestamp_exit"] = pd.to_datetime(trades["timestamp_exit"])
        trades["hold_minutes"] = (
            trades["timestamp_exit"] - trades["timestamp"]
        ).dt.total_seconds() / 60.0

        print("\n=== Hold time (minutes) stats ===")
        print(trades["hold_minutes"].describe())

    # 8) Win / loss oranı
    trades["is_win"] = trades[pnl_col] > 0
    print("\n=== Win/Loss basic ===")
    print(trades["is_win"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
