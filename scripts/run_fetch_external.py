from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import time
from typing import List

import pandas as pd
import requests

from finantradealgo.system.config_loader import load_config

SYS_CFG = load_config("research")
DATA_CFG = SYS_CFG["data_cfg"]
SYMBOL = SYS_CFG.get("symbol", "BTCUSDT")
TIMEFRAME = SYS_CFG.get("timeframe", "15m")
OHLCV_CSV = Path(DATA_CFG.ohlcv_dir) / f"{SYMBOL}_{TIMEFRAME}.csv"

FUNDING_OUT_CSV = Path(DATA_CFG.external_dir) / "funding" / f"{SYMBOL}_funding_{TIMEFRAME}.csv"
OI_OUT_CSV = Path(DATA_CFG.external_dir) / "open_interest" / f"{SYMBOL}_oi_{TIMEFRAME}.csv"


def _load_time_range_from_ohlcv() -> tuple[int, int]:
    df = pd.read_csv(OHLCV_CSV, parse_dates=["timestamp"])
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()

    start_ms = int(t_min.timestamp() * 1000)
    end_ms = int(t_max.timestamp() * 1000)
    print(f"[INFO] OHLCV range: {t_min}  ->  {t_max}")
    return start_ms, end_ms


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"

    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    print(f"[INFO] Fetching funding rate for {symbol} ...")
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        print("[WARN] Funding response empty")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)

    df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
    print(f"[INFO] Funding rows: {len(df)}")
    return df


def fetch_oi_15m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Binance Futures OI history.
    Endpoint: /futures/data/openInterestHist
    period=15m

    Binance kuralı: startTime & endTime aynı anda gönderilirse,
    aradaki fark 30 günden (ms cinsinden) büyük OLAMAZ.
    O yüzden 30 günlük pencereler halinde çekiyoruz.
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"

    all_rows: List[dict] = []

    # 30 gün (ms cinsinden)
    window_ms = 30 * 24 * 60 * 60 * 1000
    cur_start = start_ms

    print(f"[INFO] Fetching OI 15m for {symbol} in 30d windows ...")

    while cur_start < end_ms:
        # Bu pencerenin endTime'i (30 günü geçmesin, genel end_ms'i aşmasın)
        cur_end = min(cur_start + window_ms - 1, end_ms)

        params = {
            "symbol": symbol,
            "period": "15m",
            "limit": 500,
            "startTime": cur_start,
            "endTime": cur_end,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[WARN] OI request failed ({exc}); stopping OI collection.")
            break

        data = resp.json()

        if not data:
            # Bu pencere boş geldi → sonraki pencereye geç
            print(f"[INFO] Empty OI chunk for range [{cur_start}–{cur_end}], skipping...")
            cur_start = cur_end + 1
            continue

        all_rows.extend(data)

        # Son kaydın timestamp'i
        last_time = int(data[-1]["timestamp"])
        print(
            f"[INFO] OI chunk: {len(data)} rows, "
            f"from {data[0]['timestamp']} to {data[-1]['timestamp']}"
        )

        # Son kaydının hemen sonrasından devam et
        cur_start = last_time + 1

        # API'yı boğmamak için ufak delay
        time.sleep(0.2)

    if not all_rows:
        print("[WARN] OI response empty (no data collected)")
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    df = pd.DataFrame(all_rows)

    # json formatında:
    #  - "timestamp": ms
    #  - "sumOpenInterest": string
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open_interest"] = df["sumOpenInterest"].astype(float)

    df = df[["timestamp", "open_interest"]].sort_values("timestamp").reset_index(drop=True)
    print(f"[INFO] OI rows total: {len(df)}")
    return df



def main() -> None:
    start_ms, end_ms = _load_time_range_from_ohlcv()

    df_funding = fetch_funding(SYMBOL, start_ms, end_ms)
    if not df_funding.empty:
        FUNDING_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_funding.to_csv(FUNDING_OUT_CSV, index=False)
        print(f"[INFO] Saved funding CSV -> {FUNDING_OUT_CSV}")

    df_oi = fetch_oi_15m(SYMBOL, start_ms, end_ms)
    if not df_oi.empty:
        OI_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_oi.to_csv(OI_OUT_CSV, index=False)
        print(f"[INFO] Saved OI CSV -> {OI_OUT_CSV}")


if __name__ == "__main__":
    main()
