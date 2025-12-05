from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable, Mapping, Sequence, Optional

import logging

import pandas as pd

from finantradealgo.system.config_loader import WarehouseConfig

logger = logging.getLogger(__name__)


class TimeSeriesBackend(Enum):
    """
    Supported time-series storage backends.

    - TIMESCALE:
        Uses PostgreSQL + TimescaleDB extension.
    - INFLUX:
        Uses InfluxDB (to be implemented).
    - MOCK:
        In-memory mock, useful for tests and development.
    """

    TIMESCALE = auto()
    INFLUX = auto()
    MOCK = auto()


@dataclass
class TimeSeriesDBConfig:
    """
    Configuration for time-series database.

    For TIMESCALE:
        - dsn: standard PostgreSQL DSN (postgres://user:pass@host:port/dbname)
    For INFLUX:
        - url, token, org, bucket (stored in options).

    Attributes:
        backend:
            Which backend to use (TIMESCALE, INFLUX, MOCK).
        dsn:
            Connection string for PostgreSQL/Timescale (if used).
        options:
            Extra backend-specific options (e.g. influx url, token, etc.).
        ohlcv_table:
            Table/measurement name for OHLCV data.
        metrics_table:
            Table/measurement name for metrics.
    """

    backend: TimeSeriesBackend = TimeSeriesBackend.MOCK
    dsn: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    ohlcv_table: str = "ohlcv"
    metrics_table: str = "metrics"


class TimeSeriesDBClient:
    """
    High-level client for time-series data.

    Responsibilities:
    - Write OHLCV candles to a time-series backend.
    - Query OHLCV ranges efficiently.
    - Store/query time-series metrics (e.g. PnL, latency).

    This class hides backend-specific details (Timescale vs Influx).
    """

    def __init__(self, config: TimeSeriesDBConfig) -> None:
        self.config = config

        if self.config.backend is TimeSeriesBackend.TIMESCALE:
            self._backend = _TimescaleBackend(config)
        elif self.config.backend is TimeSeriesBackend.INFLUX:
            self._backend = _InfluxBackend(config)
        elif self.config.backend is TimeSeriesBackend.MOCK:
            self._backend = _MockBackend(config)
        else:
            raise ValueError(f"Unsupported time-series backend: {self.config.backend}")

    # ---- OHLCV API ---------------------------------------------------------

    def write_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """
        Persist OHLCV candles for a given symbol/timeframe.

        df:
            DataFrame indexed by timestamp (DatetimeIndex) with at least:
            - open, high, low, close, volume columns.
        """
        self._backend.write_ohlcv(symbol=symbol, timeframe=timeframe, df=df)

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Any | None = None,
        end_ts: Any | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Query OHLCV candles for symbol/timeframe in [start_ts, end_ts).

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns.
        """
        return self._backend.query_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
        )

    # ---- Metrics API -------------------------------------------------------

    def write_metric(
        self,
        name: str,
        timestamp: Any,
        value: float,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Write a single metrics datapoint (e.g. latency, pnl, drawdown).
        """
        self._backend.write_metric(
            name=name,
            timestamp=timestamp,
            value=value,
            tags=tags or {},
        )

    def query_metric_series(
        self,
        name: str,
        start_ts: Any,
        end_ts: Any | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Query a metric time-series between start_ts and end_ts.
        """
        return self._backend.query_metric_series(
            name=name,
            start_ts=start_ts,
            end_ts=end_ts,
            tags=tags or {},
        )


def build_timeseries_client_from_warehouse(cfg: WarehouseConfig, allow_missing_dsn: bool = True) -> TimeSeriesDBClient:
    """
    Helper to create a TimeSeriesDBClient from WarehouseConfig.

    - backend 'none' returns a mock client.
    - backend 'duckdb' returns a mock for now (can be extended).
    - backend 'timescale'/'postgres' expects a DSN from env.
    """
    backend = cfg.backend.lower()
    if backend in ("none", "csv", "duckdb"):
        ts_cfg = TimeSeriesDBConfig(backend=TimeSeriesBackend.MOCK)
        return TimeSeriesDBClient(ts_cfg)

    if backend in ("timescale", "postgres"):
        dsn = cfg.get_dsn(allow_missing=allow_missing_dsn)
        if not dsn:
            raise RuntimeError(f"Warehouse backend '{backend}' requires DSN env {cfg.dsn_env}")
        ts_cfg = TimeSeriesDBConfig(
            backend=TimeSeriesBackend.TIMESCALE,
            dsn=dsn,
            ohlcv_table=cfg.ohlcv_table,
            metrics_table="metrics",
        )
        return TimeSeriesDBClient(ts_cfg)

    raise ValueError(f"Unsupported warehouse backend for timeseries client: {backend}")


class _TimescaleBackend:
    """
    TimescaleDB-based implementation.

    Uses psycopg2 or asyncpg under the hood.
    For now, we use a simple psycopg2-based blocking implementation.
    """

    def __init__(self, config: TimeSeriesDBConfig) -> None:
        self.config = config
        if not self.config.dsn:
            raise ValueError("TimeSeriesDBConfig.dsn must be set for TIMESCALE backend")

        try:
            import psycopg2  # type: ignore
        except Exception as exc:
            logger.error("psycopg2 is required for Timescale backend: %s", exc)
            raise

        self._psycopg2 = psycopg2
        self._psycopg2_extras = psycopg2.extras
        self._conn = self._psycopg2.connect(self.config.dsn)
        self._conn.autocommit = True

        # We assume schema/tables are created via migrations.

    def _cursor(self):
        return self._conn.cursor()

    def write_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        if df.empty:
            return

        table = self.config.ohlcv_table
        # We assume df.index is DatetimeIndex
        records = []
        for ts, row in df.iterrows():
            records.append(
                (
                    ts.to_pydatetime(),
                    symbol,
                    timeframe,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0)),
                )
            )

        sql = f"""
        INSERT INTO {table} (
            ts, symbol, timeframe,
            open, high, low, close, volume
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        with self._cursor() as cur:
            self._psycopg2_extras.execute_batch(cur, sql, records)

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Any | None = None,
        end_ts: Any | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        table = self.config.ohlcv_table
        params: list[Any] = [symbol, timeframe]
        where = "symbol = %s AND timeframe = %s"
        if start_ts is not None:
            where += " AND ts >= %s"
            params.append(start_ts)
        if end_ts is not None:
            where += " AND ts < %s"
            params.append(end_ts)

        order = "ORDER BY ts ASC"
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT %s"
            params.append(limit)

        sql = f"""
        SELECT ts, open, high, low, close, volume
        FROM {table}
        WHERE {where}
        {order}
        {limit_clause};
        """

        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
            )

        df = pd.DataFrame(
            rows,
            columns=["ts", "open", "high", "low", "close", "volume"],
        ).set_index("ts")
        df.index = pd.to_datetime(df.index)
        return df

    def write_metric(
        self,
        name: str,
        timestamp: Any,
        value: float,
        tags: Mapping[str, Any],
    ) -> None:
        table = self.config.metrics_table
        # For simplicity, we store tags as JSONB
        import json

        sql = f"""
        INSERT INTO {table} (ts, name, value, tags)
        VALUES (%s, %s, %s, %s)
        """
        payload = json.dumps(dict(tags)) if tags else "{}"

        with self._cursor() as cur:
            cur.execute(sql, (timestamp, name, float(value), payload))

    def query_metric_series(
        self,
        name: str,
        start_ts: Any,
        end_ts: Any | None,
        tags: Mapping[str, Any],
    ) -> pd.DataFrame:
        table = self.config.metrics_table
        params: list[Any] = [name, start_ts]
        where = "name = %s AND ts >= %s"

        if end_ts is not None:
            where += " AND ts < %s"
            params.append(end_ts)

        # Tag filtering can be enhanced later; for now, ignore tags or add a TODO.
        sql = f"""
        SELECT ts, value
        FROM {table}
        WHERE {where}
        ORDER BY ts ASC;
        """

        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=["value"])

        df = pd.DataFrame(rows, columns=["ts", "value"]).set_index("ts")
        df.index = pd.to_datetime(df.index)
        return df


class _InfluxBackend:
    """
    InfluxDB-based implementation.

    Currently a placeholder; real implementation should use influxdb_client.
    """

    def __init__(self, config: TimeSeriesDBConfig) -> None:
        self.config = config
        # TODO: implement Influx client initialization.
        logger.warning("Influx backend is not yet implemented; using no-op methods")

    def write_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        logger.debug("write_ohlcv no-op for Influx backend")

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Any,
        end_ts: Any | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        logger.debug("query_ohlcv no-op for Influx backend")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def write_metric(
        self,
        name: str,
        timestamp: Any,
        value: float,
        tags: Mapping[str, Any],
    ) -> None:
        logger.debug("write_metric no-op for Influx backend")

    def query_metric_series(
        self,
        name: str,
        start_ts: Any,
        end_ts: Any | None,
        tags: Mapping[str, Any],
    ) -> pd.DataFrame:
        logger.debug("query_metric_series no-op for Influx backend")
        return pd.DataFrame(columns=["value"])


class _MockBackend:
    """
    In-memory backend for tests and local development.
    """

    def __init__(self, config: TimeSeriesDBConfig) -> None:
        self.config = config
        self._ohlcv: dict[tuple[str, str], pd.DataFrame] = {}
        self._metrics: list[dict[str, Any]] = []

    def write_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        key = (symbol, timeframe)
        existing = self._ohlcv.get(key)
        if existing is None:
            self._ohlcv[key] = df.copy()
        else:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            self._ohlcv[key] = combined.sort_index()

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Any,
        end_ts: Any | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        key = (symbol, timeframe)
        df = self._ohlcv.get(key)
        if df is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = df.loc[start_ts:end_ts] if end_ts is not None else df.loc[start_ts:]
        if limit is not None:
            result = result.iloc[:limit]
        return result.copy()

    def write_metric(
        self,
        name: str,
        timestamp: Any,
        value: float,
        tags: Mapping[str, Any],
    ) -> None:
        self._metrics.append(
            {
                "ts": pd.to_datetime(timestamp),
                "name": name,
                "value": float(value),
                "tags": dict(tags),
            }
        )

    def query_metric_series(
        self,
        name: str,
        start_ts: Any,
        end_ts: Any | None,
        tags: Mapping[str, Any],
    ) -> pd.DataFrame:
        rows = [
            m
            for m in self._metrics
            if m["name"] == name and m["ts"] >= pd.to_datetime(start_ts)
        ]
        if end_ts is not None:
            end_ts_dt = pd.to_datetime(end_ts)
            rows = [m for m in rows if m["ts"] < end_ts_dt]

        if not rows:
            return pd.DataFrame(columns=["value"])

        df = pd.DataFrame(rows).set_index("ts")
        return df[["value"]]
