from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _load_external_csv(path: str, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in {path}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def add_external_features(
    df: pd.DataFrame,
    csv_funding_path: Optional[str] = None,
    csv_oi_path: Optional[str] = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{timestamp_col}' column.")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Funding
    if csv_funding_path and Path(csv_funding_path).exists():
        try:
            df_funding = _load_external_csv(csv_funding_path, timestamp_col)
        except Exception as exc:
            print(f"[WARN] Unable to load funding CSV ({csv_funding_path}): {exc}")
        else:
            if "funding_rate" not in df_funding.columns:
                raise ValueError(f"'funding_rate' column missing in {csv_funding_path}")

            df = pd.merge_asof(
                df,
                df_funding[[timestamp_col, "funding_rate"]],
                on=timestamp_col,
                direction="backward",
            )
            df["funding_rate_raw"] = df["funding_rate"]
            df["funding_rate_abs"] = df["funding_rate_raw"].abs()
            df["funding_rate_sign"] = df["funding_rate_raw"].apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )

            roll_mean = df["funding_rate_raw"].rolling(96).mean()
            roll_std = df["funding_rate_raw"].rolling(96).std()
            df["funding_rate_zscore"] = (df["funding_rate_raw"] - roll_mean) / roll_std
            df = df.drop(columns=["funding_rate"])
    else:
        print("[WARN] Funding CSV missing; skipping funding merge.")

    # Open Interest
    if csv_oi_path and Path(csv_oi_path).exists():
        try:
            df_oi = _load_external_csv(csv_oi_path, timestamp_col)
        except Exception as exc:
            print(f"[WARN] Unable to load OI CSV ({csv_oi_path}): {exc}")
        else:
            if "open_interest" not in df_oi.columns:
                raise ValueError(f"'open_interest' column missing in {csv_oi_path}")

            df = pd.merge_asof(
                df,
                df_oi[[timestamp_col, "open_interest"]],
                on=timestamp_col,
                direction="backward",
            )
            df["oi_raw"] = df["open_interest"]
            df["oi_change_1"] = df["oi_raw"].diff(4)
            df["oi_change_4"] = df["oi_raw"].diff(16)
            df["oi_change_16"] = df["oi_raw"].diff(64)

            base = df["oi_raw"].shift(4).replace(0, pd.NA)
            for col in ["oi_change_1", "oi_change_4", "oi_change_16"]:
                df[f"{col}_pct"] = df[col] / base
            df = df.drop(columns=["open_interest"])
    else:
        print("[WARN] OI CSV missing; skipping OI merge.")

    return df
