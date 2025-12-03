from __future__ import annotations

import logging
from datetime import timedelta
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from finantradealgo.data_engine.ingestion.models import (
    FlowSnapshot,
    FundingRate,
    IngestCandle,
    OpenInterestSnapshot,
    SentimentSignal,
    ensure_timestamp,
)

logger = logging.getLogger(__name__)


class TimescaleWarehouse:
    """
    Thin Timescale writer optimized for ingestion jobs.

    Provides:
    - Upserts for OHLCV and external metrics (funding/OI/flow/sentiment)
    - Gap detection helpers for catch-up/backfill flows
    """

    def __init__(
        self,
        dsn: str,
        *,
        table_map: Optional[Mapping[str, str]] = None,
        batch_size: int = 1000,
    ) -> None:
        self.dsn = dsn
        self.batch_size = batch_size
        self.tables = {
            "ohlcv": "raw_ohlcv",
            "funding": "raw_funding_rates",
            "open_interest": "raw_open_interest",
            "flow": "raw_flow",
            "sentiment": "raw_sentiment",
        }
        if table_map:
            self.tables.update(table_map)

        try:
            import psycopg2  # type: ignore
            import psycopg2.extras  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            logger.error("psycopg2 is required for TimescaleWarehouse: %s", exc)
            raise

        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self._conn = self._psycopg2.connect(self.dsn)
        self._conn.autocommit = True

    # ------------------ write paths ------------------ #

    def upsert_ohlcv(self, candles: Sequence[IngestCandle]) -> int:
        if not candles:
            return 0
        rows = [
            (
                c.ts.to_pydatetime(),
                c.symbol,
                c.timeframe,
                float(c.open),
                float(c.high),
                float(c.low),
                float(c.close),
                float(c.volume) if c.volume is not None else None,
                float(c.vwap) if c.vwap is not None else None,
            )
            for c in candles
        ]
        sql = f"""
        INSERT INTO {self.tables["ohlcv"]} (
            ts, symbol, timeframe,
            open, high, low, close, volume, vwap
        ) VALUES %s
        ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = COALESCE(EXCLUDED.vwap, {self.tables["ohlcv"]}.vwap);
        """
        self._execute_values(sql, rows)
        return len(rows)

    def upsert_funding(self, rows: Sequence[FundingRate]) -> int:
        if not rows:
            return 0
        payload = [
            (
                r.ts.to_pydatetime(),
                r.symbol,
                r.timeframe,
                float(r.funding_rate),
                float(r.mark_price) if r.mark_price is not None else None,
                float(r.index_price) if r.index_price is not None else None,
                float(r.open_interest) if r.open_interest is not None else None,
            )
            for r in rows
        ]
        sql = f"""
        INSERT INTO {self.tables["funding"]} (
            ts, symbol, timeframe, funding_rate, mark_price, index_price, open_interest
        ) VALUES %s
        ON CONFLICT (symbol, ts) DO UPDATE SET
            funding_rate = EXCLUDED.funding_rate,
            mark_price = COALESCE(EXCLUDED.mark_price, {self.tables["funding"]}.mark_price),
            index_price = COALESCE(EXCLUDED.index_price, {self.tables["funding"]}.index_price),
            open_interest = COALESCE(EXCLUDED.open_interest, {self.tables["funding"]}.open_interest);
        """
        self._execute_values(sql, payload)
        return len(payload)

    def upsert_open_interest(self, rows: Sequence[OpenInterestSnapshot]) -> int:
        if not rows:
            return 0
        payload = [
            (
                r.ts.to_pydatetime(),
                r.symbol,
                r.timeframe,
                float(r.open_interest),
                float(r.volume) if r.volume is not None else None,
                float(r.turnover) if r.turnover is not None else None,
            )
            for r in rows
        ]
        sql = f"""
        INSERT INTO {self.tables["open_interest"]} (
            ts, symbol, timeframe, open_interest, volume, turnover
        ) VALUES %s
        ON CONFLICT (symbol, ts) DO UPDATE SET
            open_interest = EXCLUDED.open_interest,
            volume = COALESCE(EXCLUDED.volume, {self.tables["open_interest"]}.volume),
            turnover = COALESCE(EXCLUDED.turnover, {self.tables["open_interest"]}.turnover);
        """
        self._execute_values(sql, payload)
        return len(payload)

    def upsert_flow(self, rows: Sequence[FlowSnapshot]) -> int:
        if not rows:
            return 0
        payload = [
            (
                r.ts.to_pydatetime(),
                r.symbol,
                r.timeframe,
                r.perp_premium,
                r.basis,
                r.oi,
                r.oi_change,
                r.liq_up,
                r.liq_down,
            )
            for r in rows
        ]
        sql = f"""
        INSERT INTO {self.tables["flow"]} (
            ts, symbol, timeframe, perp_premium, basis, oi, oi_change, liq_up, liq_down
        ) VALUES %s
        ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
            perp_premium = COALESCE(EXCLUDED.perp_premium, {self.tables["flow"]}.perp_premium),
            basis = COALESCE(EXCLUDED.basis, {self.tables["flow"]}.basis),
            oi = COALESCE(EXCLUDED.oi, {self.tables["flow"]}.oi),
            oi_change = COALESCE(EXCLUDED.oi_change, {self.tables["flow"]}.oi_change),
            liq_up = COALESCE(EXCLUDED.liq_up, {self.tables["flow"]}.liq_up),
            liq_down = COALESCE(EXCLUDED.liq_down, {self.tables["flow"]}.liq_down);
        """
        self._execute_values(sql, payload)
        return len(payload)

    def upsert_sentiment(self, rows: Sequence[SentimentSignal]) -> int:
        if not rows:
            return 0
        payload = [
            (
                r.ts.to_pydatetime(),
                r.symbol,
                r.timeframe,
                float(r.sentiment_score),
                float(r.volume) if r.volume is not None else None,
                r.source,
            )
            for r in rows
        ]
        sql = f"""
        INSERT INTO {self.tables["sentiment"]} (
            ts, symbol, timeframe, sentiment_score, volume, source
        ) VALUES %s
        ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
            sentiment_score = EXCLUDED.sentiment_score,
            volume = COALESCE(EXCLUDED.volume, {self.tables["sentiment"]}.volume),
            source = COALESCE(EXCLUDED.source, {self.tables["sentiment"]}.source);
        """
        self._execute_values(sql, payload)
        return len(payload)

    # ------------------ gap detection ------------------ #

    def get_latest_ts(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        sql = f"""
        SELECT MAX(ts) FROM {self.tables["ohlcv"]}
        WHERE symbol = %s AND timeframe = %s;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (symbol, timeframe))
            row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return ensure_timestamp(row[0])

    def get_span(self, symbol: str, timeframe: str) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], int]:
        sql = f"""
        SELECT MIN(ts), MAX(ts), COUNT(*) FROM {self.tables["ohlcv"]}
        WHERE symbol = %s AND timeframe = %s;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (symbol, timeframe))
            row = cur.fetchone()
        if not row:
            return None, None, 0
        min_ts = ensure_timestamp(row[0]) if row[0] else None
        max_ts = ensure_timestamp(row[1]) if row[1] else None
        count = int(row[2] or 0)
        return min_ts, max_ts, count

    def detect_gaps(
        self,
        symbol: str,
        timeframe: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        step_seconds: int,
        *,
        max_gaps: int = 50,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Identify missing contiguous ranges between start_ts and end_ts (inclusive).
        """
        start_ts = ensure_timestamp(start_ts)
        end_ts = ensure_timestamp(end_ts)
        step = timedelta(seconds=step_seconds)

        sql = f"""
        SELECT ts FROM {self.tables["ohlcv"]}
        WHERE symbol = %s AND timeframe = %s AND ts >= %s AND ts <= %s
        ORDER BY ts ASC;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (symbol, timeframe, start_ts, end_ts))
            rows = cur.fetchall()

        existing = [ensure_timestamp(r[0]) for r in rows if r and r[0] is not None]
        if not existing:
            return [(start_ts, end_ts)]

        gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        prev = start_ts - step
        for ts in existing:
            expected = prev + step
            if ts > expected:
                gaps.append((expected, ts - step))
            prev = ts
            if len(gaps) >= max_gaps:
                break

        tail_start = prev + step
        if tail_start <= end_ts and len(gaps) < max_gaps:
            gaps.append((tail_start, end_ts))
        return gaps

    # ------------------ internal helpers ------------------ #

    def _execute_values(self, sql: str, rows: Iterable[tuple]) -> None:
        with self._conn.cursor() as cur:
            self._extras.execute_values(cur, sql, rows, page_size=self.batch_size)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
