from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataBackend(ABC):
    """
    Abstract data backend to read OHLCV and external features from different sources
    (CSV, Timescale/Postgres, DuckDB/Parquet).
    """

    @abstractmethod
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_days: Optional[int] = None,
        start_ts: Optional[pd.Timestamp] = None,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def load_flow(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        ...

    @abstractmethod
    def load_sentiment(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        ...


@dataclass
class CsvBackend(DataBackend):
    """
    CSV-backed reader using the existing loader utilities.
    """

    ohlcv_path_template: str
    flow_dir: str = "data/flow"
    sentiment_dir: str = "data/sentiment"
    data_cfg: Any = None  # DataConfig; kept loose to avoid circular import typing

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_days: Optional[int] = None,
        start_ts: Optional[pd.Timestamp] = None,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        from finantradealgo.data_engine.loader import load_ohlcv_csv  # local import to avoid cycle

        path = self.ohlcv_path_template.format(symbol=symbol, timeframe=timeframe)
        df = load_ohlcv_csv(path, config=self.data_cfg, lookback_days=lookback_days)
        if start_ts is not None:
            df = df[df["timestamp"] >= pd.to_datetime(start_ts, utc=True)]
        if end_ts is not None:
            df = df[df["timestamp"] <= pd.to_datetime(end_ts, utc=True)]
        return df.reset_index(drop=True)

    def load_flow(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        from finantradealgo.data_engine.loader import load_flow_features

        return load_flow_features(
            symbol,
            timeframe,
            flow_dir=self.flow_dir,
            data_cfg=self.data_cfg,
        )

    def load_sentiment(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        from finantradealgo.data_engine.loader import load_sentiment_features

        return load_sentiment_features(
            symbol,
            timeframe,
            sentiment_dir=self.sentiment_dir,
            data_cfg=self.data_cfg,
        )


@dataclass
class TimescaleBackend(DataBackend):
    """
    Postgres/Timescale-backed reader that pulls from raw_* tables.
    """

    dsn: str
    tables: Dict[str, str]

    def __post_init__(self) -> None:
        try:
            import psycopg2  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            logger.error("psycopg2 is required for TimescaleBackend: %s", exc)
            raise
        self._psycopg2 = psycopg2

    def _fetch_df(self, query: str, params: tuple) -> pd.DataFrame:
        with self._psycopg2.connect(self.dsn) as conn:
            return pd.read_sql_query(query, conn, params=params, parse_dates=["ts"])

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_days: Optional[int] = None,
        start_ts: Optional[pd.Timestamp] = None,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        table = self.tables.get("ohlcv", "raw_ohlcv")
        where = ["symbol = %s", "timeframe = %s"]
        params: list[Any] = [symbol, timeframe]
        if lookback_days is not None:
            where.append("ts >= NOW() - (%s * INTERVAL '1 day')")
            params.append(int(lookback_days))
        if start_ts is not None:
            where.append("ts >= %s")
            params.append(pd.to_datetime(start_ts, utc=True))
        if end_ts is not None:
            where.append("ts <= %s")
            params.append(pd.to_datetime(end_ts, utc=True))
        sql = f"""
        SELECT ts as timestamp, open, high, low, close, volume, vwap
        FROM {table}
        WHERE {' AND '.join(where)}
        ORDER BY ts ASC;
        """
        df = self._fetch_df(sql, tuple(params))
        return df

    def load_flow(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        table = self.tables.get("flow", "raw_flow")
        sql = f"""
        SELECT ts as timestamp, perp_premium, basis, oi, oi_change, liq_up, liq_down
        FROM {table}
        WHERE symbol = %s AND timeframe = %s
        ORDER BY ts ASC;
        """
        df = self._fetch_df(sql, (symbol, timeframe))
        return df if not df.empty else None

    def load_sentiment(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        table = self.tables.get("sentiment", "raw_sentiment")
        sql = f"""
        SELECT ts as timestamp, sentiment_score, volume, source
        FROM {table}
        WHERE symbol = %s AND timeframe = %s
        ORDER BY ts ASC;
        """
        df = self._fetch_df(sql, (symbol, timeframe))
        return df if not df.empty else None


@dataclass
class DuckDBBackend(DataBackend):
    """
    DuckDB/Parquet backend placeholder. Expects a parquet catalog path.
    """

    database: str
    ohlcv_table: str = "raw_ohlcv"
    flow_table: str = "raw_flow"
    sentiment_table: str = "raw_sentiment"

    def __post_init__(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover
            logger.error("duckdb is required for DuckDBBackend: %s", exc)
            raise
        self._duckdb = duckdb.connect(self.database, read_only=True)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_days: Optional[int] = None,
        start_ts: Optional[pd.Timestamp] = None,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        where = ["symbol = ?", "timeframe = ?"]
        params: list[Any] = [symbol, timeframe]
        if lookback_days is not None:
            where.append("ts >= now() - INTERVAL ? DAY")
            params.append(lookback_days)
        if start_ts is not None:
            where.append("ts >= ?")
            params.append(pd.to_datetime(start_ts, utc=True))
        if end_ts is not None:
            where.append("ts <= ?")
            params.append(pd.to_datetime(end_ts, utc=True))
        sql = f"""
        SELECT ts as timestamp, open, high, low, close, volume, vwap
        FROM {self.ohlcv_table}
        WHERE {' AND '.join(where)}
        ORDER BY ts ASC
        """
        return self._duckdb.execute(sql, params).fetch_df()

    def load_flow(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        sql = f"""
        SELECT ts as timestamp, perp_premium, basis, oi, oi_change, liq_up, liq_down
        FROM {self.flow_table}
        WHERE symbol = ? AND timeframe = ?
        ORDER BY ts ASC
        """
        df = self._duckdb.execute(sql, [symbol, timeframe]).fetch_df()
        return df if not df.empty else None

    def load_sentiment(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        sql = f"""
        SELECT ts as timestamp, sentiment_score, volume, source
        FROM {self.sentiment_table}
        WHERE symbol = ? AND timeframe = ?
        ORDER BY ts ASC
        """
        df = self._duckdb.execute(sql, [symbol, timeframe]).fetch_df()
        return df if not df.empty else None


def build_backend(data_cfg) -> DataBackend:
    """
    Factory that returns the backend specified in DataConfig or WarehouseConfig.

    Priority:
    - data_cfg.backend controls the choice.
    - DSN for DB backends is sourced from backend_params or WarehouseConfig if present.
    """
    backend = getattr(data_cfg, "backend", "csv") or "csv"
    params = getattr(data_cfg, "backend_params", {}) or {}
    warehouse_cfg = getattr(data_cfg, "warehouse_cfg", None)

    if backend == "csv":
        return CsvBackend(
            ohlcv_path_template=data_cfg.ohlcv_path_template,
            flow_dir=data_cfg.flow_dir,
            sentiment_dir=data_cfg.sentiment_dir,
            data_cfg=data_cfg,
        )
    if backend in ("timescale", "postgres"):
        dsn = params.get("dsn")
        if not dsn and warehouse_cfg is not None:
            try:
                dsn = warehouse_cfg.get_dsn(allow_missing=False)
            except Exception:
                dsn = warehouse_cfg.get_dsn(allow_missing=True)
        if not dsn:
            # In environments without DB creds, allow caller to handle.
            raise ValueError("DataConfig.backend_params.dsn must be set for timescale/postgres backend.")
        table_map = {
            "ohlcv": params.get("ohlcv_table", "raw_ohlcv"),
            "flow": params.get("flow_table", "raw_flow"),
            "sentiment": params.get("sentiment_table", "raw_sentiment"),
        }
        return TimescaleBackend(dsn=dsn, tables=table_map)
    if backend == "duckdb":
        db_path = params.get("database") or params.get("path")
        if not db_path:
            raise ValueError("DataConfig.backend_params.database must be set for duckdb backend.")
        return DuckDBBackend(
            database=db_path,
            ohlcv_table=params.get("ohlcv_table", "raw_ohlcv"),
            flow_table=params.get("flow_table", "raw_flow"),
            sentiment_table=params.get("sentiment_table", "raw_sentiment"),
        )
    raise ValueError(f"Unsupported data backend: {backend}")
