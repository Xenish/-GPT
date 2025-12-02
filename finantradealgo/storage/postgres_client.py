from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass
class DBConnectionConfig:
    """
    Connection configuration for PostgreSQL.

    Attributes:
        dsn:
            Full PostgreSQL DSN (e.g. postgres://user:pass@host:5432/dbname).
        minconn:
            Min connections in pool (if using pooling).
        maxconn:
            Max connections in pool.
        options:
            Extra options (application_name, sslmode, etc.).
    """

    dsn: str
    minconn: int = 1
    maxconn: int = 10
    options: dict[str, Any] = field(default_factory=dict)


class PostgresClient:
    """
    Thin wrapper around psycopg2 connection pool.

    Responsibilities:
    - Provide a central place for DB access.
    - Basic query/execute helpers.
    - High-level methods for common domain operations (trades, strategies, configs).
    """

    def __init__(self, conn_config: DBConnectionConfig) -> None:
        self.conn_config = conn_config

        try:
            import psycopg2  # type: ignore
            from psycopg2 import pool as pg_pool  # type: ignore
        except Exception as exc:
            logger.error("psycopg2 is required for PostgresClient: %s", exc)
            raise

        self._psycopg2 = psycopg2
        self._pool = pg_pool.SimpleConnectionPool(
            minconn=self.conn_config.minconn,
            maxconn=self.conn_config.maxconn,
            dsn=self.conn_config.dsn,
        )

        logger.info("PostgresClient initialized with DSN=%s", self.conn_config.dsn)

    def _get_conn(self):
        return self._pool.getconn()

    def _put_conn(self, conn) -> None:
        self._pool.putconn(conn)

    def close(self) -> None:
        """
        Close all connections in the pool.
        """
        self._pool.closeall()

    def execute(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> None:
        """
        Execute a write/update statement (INSERT/UPDATE/DELETE).
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
            conn.commit()
        finally:
            self._put_conn(conn)

    def fetchone(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        """
        Execute a SELECT query expecting a single row.
        Returns a dict-like row or None.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                row = cur.fetchone()
                if row is None:
                    return None
                desc = [c.name for c in cur.description]
                return dict(zip(desc, row))
        finally:
            self._put_conn(conn)

    def fetchall(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> list[Mapping[str, Any]]:
        """
        Execute a SELECT query returning multiple rows.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                rows = cur.fetchall()
                desc = [c.name for c in cur.description]
                return [dict(zip(desc, r)) for r in rows]
        finally:
            self._put_conn(conn)

    # ---- Trade history -----------------------------------------------------

    def insert_trade(self, trade: Mapping[str, Any]) -> None:
        """
        Insert a single trade into the trades table.

        Expected keys in `trade` (adapt to your schema):
        - trade_id, account_id, strategy_id, symbol, side, qty, price,
          pnl, opened_at, closed_at, metadata (JSON).
        """
        import json

        sql = """
        INSERT INTO trades (
            trade_id,
            account_id,
            strategy_id,
            symbol,
            side,
            qty,
            price,
            pnl,
            opened_at,
            closed_at,
            metadata
        ) VALUES (
            %(trade_id)s,
            %(account_id)s,
            %(strategy_id)s,
            %(symbol)s,
            %(side)s,
            %(qty)s,
            %(price)s,
            %(pnl)s,
            %(opened_at)s,
            %(closed_at)s,
            %(metadata)s
        )
        ON CONFLICT (trade_id) DO NOTHING;
        """
        payload = dict(trade)
        payload["metadata"] = json.dumps(payload.get("metadata", {}))

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
            conn.commit()
        finally:
            self._put_conn(conn)

    def load_trades_for_account(
        self,
        account_id: str,
        *,
        limit: int = 1000,
    ) -> list[Mapping[str, Any]]:
        sql = """
        SELECT
            trade_id,
            account_id,
            strategy_id,
            symbol,
            side,
            qty,
            price,
            pnl,
            opened_at,
            closed_at,
            metadata
        FROM trades
        WHERE account_id = %s
        ORDER BY opened_at DESC
        LIMIT %s;
        """
        return self.fetchall(sql, [account_id, limit])

    # ---- Strategy metadata -------------------------------------------------

    def upsert_strategy_metadata(self, meta: Mapping[str, Any]) -> None:
        """
        Upsert strategy metadata (name, description, params, etc.).
        """
        import json

        sql = """
        INSERT INTO strategy_metadata (
            strategy_id,
            name,
            description,
            params
        ) VALUES (
            %(strategy_id)s,
            %(name)s,
            %(description)s,
            %(params)s
        )
        ON CONFLICT (strategy_id) DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            params = EXCLUDED.params;
        """
        payload = dict(meta)
        payload["params"] = json.dumps(payload.get("params", {}))

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
            conn.commit()
        finally:
            self._put_conn(conn)

    def get_strategy_metadata(self, strategy_id: str) -> Mapping[str, Any] | None:
        sql = """
        SELECT strategy_id, name, description, params
        FROM strategy_metadata
        WHERE strategy_id = %s;
        """
        row = self.fetchone(sql, [strategy_id])
        return row

    # ---- Configuration storage --------------------------------------------

    def save_config(
        self,
        key: str,
        config: Mapping[str, Any],
        *,
        profile: str | None = None,
    ) -> None:
        """
        Save a configuration under a key/profile.
        """
        import json

        sql = """
        INSERT INTO configs (key, profile, payload)
        VALUES (%s, %s, %s)
        ON CONFLICT (key, profile) DO UPDATE SET
            payload = EXCLUDED.payload;
        """
        payload = json.dumps(dict(config))
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (key, profile, payload))
            conn.commit()
        finally:
            self._put_conn(conn)

    def load_config(
        self,
        key: str,
        *,
        profile: str | None = None,
    ) -> Mapping[str, Any] | None:
        """
        Load a config blob by key/profile.
        """
        import json

        sql = """
        SELECT payload
        FROM configs
        WHERE key = %s AND profile IS NOT DISTINCT FROM %s;
        """
        row = self.fetchone(sql, [key, profile])
        if row is None:
            return None
        return json.loads(row["payload"])
