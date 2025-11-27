from __future__ import annotations

import pandas as pd


def add_flow_features(
    df_ohlcv: pd.DataFrame,
    df_flow: pd.DataFrame,
    tolerance: str = "1h",
    zscore_window: int = 96,
) -> pd.DataFrame:
    if df_flow is None or df_flow.empty:
        return df_ohlcv

    df_ohlcv = df_ohlcv.sort_values("timestamp")
    df_flow = df_flow.sort_values("timestamp")

    merged = pd.merge_asof(
        df_ohlcv,
        df_flow,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )

    rename_map = {}
    for col in df_flow.columns:
        if col == "timestamp":
            continue
        if col.startswith("flow_"):
            continue
        rename_map[col] = f"flow_{col}"
    merged = merged.rename(columns=rename_map)

    if "flow_perp_premium" in merged.columns and zscore_window > 0:
        series = merged["flow_perp_premium"]
        roll = series.rolling(zscore_window, min_periods=max(1, zscore_window // 4))
        mean = roll.mean()
        std = roll.std(ddof=0)
        merged["flow_perp_premium_z"] = (series - mean) / std.replace(0, pd.NA)

    if "flow_basis" in merged.columns and zscore_window > 0:
        series = merged["flow_basis"]
        roll = series.rolling(zscore_window, min_periods=max(1, zscore_window // 4))
        mean = roll.mean()
        std = roll.std(ddof=0)
        merged["flow_basis_z"] = (series - mean) / std.replace(0, pd.NA)

    return merged
