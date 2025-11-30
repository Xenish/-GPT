from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from finantradealgo.data_engine.event_bars import build_event_bars
from finantradealgo.system.config_loader import EventBarConfig, DataConfig

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

FLOW_REQUIRED_BASE = ["perp_premium", "basis"]
FLOW_OPTIONAL_COLUMNS = ["oi", "oi_change", "liq_up", "liq_down"]
FLOW_OUTLIER_LIMITS: dict[str, Tuple[float, float]] = {
    "oi": (-5e9, 5e9),
    "oi_change": (-5e7, 5e7),
    "perp_premium": (-5.0, 5.0),
    "basis": (-10.0, 10.0),
    "liq_up": (-5e9, 5e9),
    "liq_down": (-5e9, 5e9),
}
FLOW_COMBINED_FILENAMES = (
    "flow_{symbol}_{timeframe}.csv",
    "{symbol}_{timeframe}_flow.csv",
)

SENTIMENT_REQUIRED_COLUMNS = ["sentiment_score"]
SENTIMENT_OPTIONAL_DEFAULTS = {"volume": 0.0, "source": "unknown"}
SENTIMENT_OUTLIER_LIMITS: dict[str, Tuple[float, float]] = {
    "sentiment_score": (-1.0, 1.0),
}
SENTIMENT_COMBINED_FILENAMES = (
    "sentiment_{symbol}_{timeframe}.csv",
    "{symbol}_{timeframe}_sentiment.csv",
)


def load_ohlcv_csv(
    path: str,
    config: DataConfig | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV with optional lookback filtering.

    Args:
        path: Path to CSV file
        config: DataConfig for event bars and other settings
        lookback_days: If provided, filter data to last N days from now

    Returns:
        DataFrame with OHLCV data
    """
    if config is None:
        config = DataConfig()

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # Use timestamp_col if available, otherwise default to "timestamp"
    timestamp_col = getattr(config, 'timestamp_col', 'timestamp')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Apply lookback filter if specified
    if lookback_days is not None:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        df = df[df[timestamp_col] >= cutoff].reset_index(drop=True)
        logger.info(f"Applied lookback filter: {lookback_days} days, {len(df)} bars retained")

    # Apply event bars if configured
    if config and hasattr(config, 'bars') and config.bars:
        # Validate timeframe for non-time event bars
        if config.bars.mode in ("volume", "dollar", "tick"):
            if config.bars.source_timeframe != "1m":
                raise ValueError(
                    f"Event bars currently only supported from 1m data; "
                    f"got source_timeframe={config.bars.source_timeframe!r}. "
                    f"Set timeframe='1m' in your config when using event bars."
                )

        df = build_event_bars(df, config.bars)

    return df


def load_ohlcv_for_symbol_tf(
    symbol: str,
    timeframe: str,
    data_cfg: DataConfig,
) -> pd.DataFrame:
    """
    Load OHLCV data for a specific symbol and timeframe using DataConfig.

    This helper automatically:
    - Resolves the file path using ohlcv_path_template
    - Applies the appropriate lookback_days filter based on timeframe
    - Applies event bar configuration if specified

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "15m", "1h")
        data_cfg: DataConfig containing paths and lookback settings

    Returns:
        DataFrame with OHLCV data (filtered and processed)

    Example:
        >>> from finantradealgo.system.config_loader import load_system_config
        >>> cfg = load_system_config()
        >>> df = load_ohlcv_for_symbol_tf("BTCUSDT", "15m", cfg["data_cfg"])
    """
    # Resolve file path from template
    path = data_cfg.ohlcv_path_template.format(symbol=symbol, timeframe=timeframe)

    # Get lookback days for this timeframe
    lookback = data_cfg.lookback_days.get(timeframe, data_cfg.default_lookback_days)

    logger.info(
        f"Loading {symbol} {timeframe} from {path} (lookback: {lookback} days)"
    )

    return load_ohlcv_csv(path, config=data_cfg, lookback_days=lookback)


def _load_timeseries_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Timeseries file not found at {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV at {path} missing timestamp column")
    df["timestamp"] = _normalize_timestamp_series(df["timestamp"], source=str(path))
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _normalize_timestamp_series(series: pd.Series, source: str) -> pd.Series:
    ser = series
    if pd.api.types.is_datetime64_any_dtype(ser):
        return pd.to_datetime(ser, utc=True)
    if pd.api.types.is_numeric_dtype(ser):
        ser = pd.to_numeric(ser, errors="coerce")
        max_abs = ser.dropna().abs().max()
        if pd.isna(max_abs):
            unit = "s"
        elif max_abs > 1e15:
            unit = "ns"
        elif max_abs > 1e12:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(ser, utc=True, unit=unit, errors="coerce")
    try:
        return pd.to_datetime(ser, utc=True, errors="coerce")
    except Exception:
        logger.warning("Failed to parse timestamp column for %s; returning NaT.", source)
        return pd.to_datetime(pd.Series([pd.NaT] * len(series)))


def _validate_monotonic(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if not df["timestamp"].is_monotonic_increasing:
        logger.warning("%s timestamps not sorted. Re-sorting.", label)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _remove_outliers(df: pd.DataFrame, limits: dict[str, Tuple[float, float]], label: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for col, (low, high) in limits.items():
        if col in df.columns:
            mask &= df[col].between(low, high)
    removed = len(df) - int(mask.sum())
    if removed > 0:
        logger.warning("%s outlier filter removed %s rows.", label, removed)
    return df[mask].reset_index(drop=True)


def _prepare_flow_df(df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = _validate_monotonic(df, "Flow features")
    df = _remove_outliers(df, FLOW_OUTLIER_LIMITS, "Flow features")
    missing = [c for c in FLOW_REQUIRED_BASE if c not in df.columns]
    if missing:
        logger.error("[FLOW] Missing required columns %s in %s", missing, source)
        return None
    df = df.dropna(subset=FLOW_REQUIRED_BASE)
    if df.empty:
        logger.warning("[FLOW] All rows dropped due to NaN in required columns for %s", source)
        return None
    keep_cols = ["timestamp"] + FLOW_REQUIRED_BASE + FLOW_OPTIONAL_COLUMNS
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def _prepare_sentiment_df(df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = _validate_monotonic(df, "Sentiment features")
    df = _remove_outliers(df, SENTIMENT_OUTLIER_LIMITS, "Sentiment features")
    if "sentiment_score" not in df.columns:
        logger.error("[SENTIMENT] Missing required 'sentiment_score' column in %s", source)
        return None
    df = df.dropna(subset=["sentiment_score"])
    if df.empty:
        logger.warning("[SENTIMENT] All rows dropped due to NaN in %s", source)
        return None
    for col, default in SENTIMENT_OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    columns = ["timestamp", "sentiment_score"] + list(SENTIMENT_OPTIONAL_DEFAULTS.keys())
    columns.extend(c for c in df.columns if c not in columns)
    return df[columns].reset_index(drop=True)


def _load_flow_dataframe(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = _load_timeseries_csv(path)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        logger.warning("Invalid flow features file at %s: %s", path, exc)
        return None
    return _prepare_flow_df(df, str(path))


def _load_sentiment_dataframe(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = _load_timeseries_csv(path)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        logger.warning("Invalid sentiment features file at %s: %s", path, exc)
        return None
    return _prepare_sentiment_df(df, str(path))


def load_flow_features(
    symbol: str,
    timeframe: str,
    *,
    flow_dir: str | Path | None = None,
    base_dir: str | Path = "data",
    data_cfg: Optional[DataConfig] = None,
) -> Optional[pd.DataFrame]:
    base_path = Path(flow_dir) if flow_dir else Path(base_dir) / "flow"
    # Try combined files first
    for pattern in FLOW_COMBINED_FILENAMES:
        path = base_path / pattern.format(symbol=symbol, timeframe=timeframe)
        df = _load_flow_dataframe(path)
        if df is not None:
            return df

    df_components = _load_flow_metrics_from_dirs(base_path, symbol, timeframe)
    if df_components is not None:
        return _prepare_flow_df(df_components, str(base_path))

    logger.warning(
        "[FLOW] Flow file not found for %s %s at %s",
        symbol,
        timeframe,
        base_path,
    )
    return None


def _load_flow_metrics_from_dirs(base_path: Path, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for metric in FLOW_REQUIRED_BASE + FLOW_OPTIONAL_COLUMNS:
        metric_dir = base_path / metric
        path = metric_dir / f"{symbol}_{timeframe}.csv"
        if not path.is_file():
            continue
        try:
            df_metric = _load_timeseries_csv(path)
        except Exception as exc:
            logger.warning("[FLOW] Failed to load %s metric from %s: %s", metric, path, exc)
            continue
        value_cols = [c for c in df_metric.columns if c != "timestamp"]
        if not value_cols:
            continue
        value_col = metric if metric in df_metric.columns else value_cols[0]
        frames.append(
            df_metric[["timestamp", value_col]].rename(columns={value_col: metric})
        )

    if not frames:
        return None
    merged = frames[0]
    for extra in frames[1:]:
        merged = pd.merge(merged, extra, on="timestamp", how="outer")
    return merged


def load_sentiment_features(
    symbol: str,
    timeframe: str,
    *,
    sentiment_dir: str | Path | None = None,
    base_dir: str | Path = "data",
    data_cfg: Optional[DataConfig] = None,
) -> Optional[pd.DataFrame]:
    base_path = Path(sentiment_dir) if sentiment_dir else Path(base_dir) / "sentiment"
    for pattern in SENTIMENT_COMBINED_FILENAMES:
        path = base_path / pattern.format(symbol=symbol, timeframe=timeframe)
        df = _load_sentiment_dataframe(path)
        if df is not None:
            return df

    logger.warning(
        "[SENTIMENT] file not found for %s %s under %s",
        symbol,
        timeframe,
        base_path,
    )
    return None
