from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Iterable, Protocol, Sequence

import pandas as pd

from finantradealgo.data_engine.ingestion.models import (
    FlowSnapshot,
    FundingRate,
    OpenInterestSnapshot,
    SentimentSignal,
    ensure_timestamp,
)
from finantradealgo.data_engine.ingestion.writer import TimescaleWarehouse

logger = logging.getLogger(__name__)


class ExternalFetcher(Protocol):
    def __call__(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp | None = None,
    ) -> Sequence:
        ...


def _normalize_time(value: datetime | str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return ensure_timestamp(value)


class FundingIngestJob:
    """
    Generic funding-rate ingestion job.

    Provide a fetch_fn returning Sequence[FundingRate].
    """

    def __init__(self, fetch_fn: ExternalFetcher, warehouse: TimescaleWarehouse) -> None:
        self.fetch_fn = fetch_fn
        self.warehouse = warehouse

    def run(self, symbol: str, start_ts, end_ts=None) -> int:
        start = _normalize_time(start_ts)
        end = _normalize_time(end_ts)
        rows = self.fetch_fn(symbol, start, end)
        if not rows:
            return 0
        written = self.warehouse.upsert_funding(rows)  # type: ignore[arg-type]
        logger.info("Funding ingest wrote %s rows for %s", written, symbol)
        return written


class OpenInterestIngestJob:
    """
    Generic open-interest ingestion job.
    """

    def __init__(self, fetch_fn: ExternalFetcher, warehouse: TimescaleWarehouse) -> None:
        self.fetch_fn = fetch_fn
        self.warehouse = warehouse

    def run(self, symbol: str, start_ts, end_ts=None) -> int:
        start = _normalize_time(start_ts)
        end = _normalize_time(end_ts)
        rows = self.fetch_fn(symbol, start, end)
        if not rows:
            return 0
        written = self.warehouse.upsert_open_interest(rows)  # type: ignore[arg-type]
        logger.info("Open interest ingest wrote %s rows for %s", written, symbol)
        return written


class FlowIngestJob:
    """
    Generic flow metrics ingestion job.
    """

    def __init__(self, fetch_fn: ExternalFetcher, warehouse: TimescaleWarehouse) -> None:
        self.fetch_fn = fetch_fn
        self.warehouse = warehouse

    def run(self, symbol: str, start_ts, end_ts=None) -> int:
        start = _normalize_time(start_ts)
        end = _normalize_time(end_ts)
        rows = self.fetch_fn(symbol, start, end)
        if not rows:
            return 0
        written = self.warehouse.upsert_flow(rows)  # type: ignore[arg-type]
        logger.info("Flow ingest wrote %s rows for %s", written, symbol)
        return written


class SentimentIngestJob:
    """
    Generic sentiment ingestion job.
    """

    def __init__(self, fetch_fn: ExternalFetcher, warehouse: TimescaleWarehouse) -> None:
        self.fetch_fn = fetch_fn
        self.warehouse = warehouse

    def run(self, symbol: str, start_ts, end_ts=None) -> int:
        start = _normalize_time(start_ts)
        end = _normalize_time(end_ts)
        rows = self.fetch_fn(symbol, start, end)
        if not rows:
            return 0
        written = self.warehouse.upsert_sentiment(rows)  # type: ignore[arg-type]
        logger.info("Sentiment ingest wrote %s rows for %s", written, symbol)
        return written
