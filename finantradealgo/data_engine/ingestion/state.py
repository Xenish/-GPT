from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class JobRun:
    run_id: str
    job_name: str
    status: str
    watermark: dict[str, Any] | None
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class BaseStateStore:
    def start_run(self, job_name: str, *, watermark: dict[str, Any] | None = None) -> JobRun:
        raise NotImplementedError

    def finish_run(self, run_id: str, *, status: str = "succeeded", error: str | None = None, watermark=None) -> None:
        raise NotImplementedError

    def get_watermark(self, job_name: str, scope: str) -> Optional[pd.Timestamp]:
        raise NotImplementedError

    def upsert_watermark(self, job_name: str, scope: str, ts: pd.Timestamp) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class IngestionStateStore(BaseStateStore):
    """
    Stores ingestion/scheduler run metadata and watermarks in Postgres/Timescale.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            logger.error("psycopg2 is required for IngestionStateStore: %s", exc)
            raise
        self._psycopg2 = psycopg2
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True

    def start_run(self, job_name: str, *, watermark: dict[str, Any] | None = None) -> JobRun:
        run_id = uuid.uuid4().hex
        sql = """
        INSERT INTO ingestion_runs (run_id, job_name, status, watermark)
        VALUES (%s, %s, %s, %s)
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (run_id, job_name, "running", watermark))
        return JobRun(
            run_id=run_id,
            job_name=job_name,
            status="running",
            watermark=watermark,
            started_at=datetime.utcnow(),
        )

    def finish_run(self, run_id: str, *, status: str = "succeeded", error: str | None = None, watermark=None) -> None:
        sql = """
        UPDATE ingestion_runs
        SET status = %s, finished_at = NOW(), error = %s, watermark = COALESCE(%s, watermark)
        WHERE run_id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (status, error, watermark, run_id))

    def get_watermark(self, job_name: str, scope: str) -> Optional[pd.Timestamp]:
        sql = """
        SELECT watermark_ts
        FROM ingestion_watermarks
        WHERE job_name = %s AND scope = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (job_name, scope))
            row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return pd.to_datetime(row[0], utc=True)

    def upsert_watermark(self, job_name: str, scope: str, ts: pd.Timestamp) -> None:
        sql = """
        INSERT INTO ingestion_watermarks (job_name, scope, watermark_ts)
        VALUES (%s, %s, %s)
        ON CONFLICT (job_name, scope) DO UPDATE SET
            watermark_ts = EXCLUDED.watermark_ts
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (job_name, scope, ts.to_pydatetime()))

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class FileStateStore(BaseStateStore):
    """
    Lightweight file-based state store for CSV/DuckDB workflows.
    """

    def __init__(self, path: str | Path = "outputs/state/ingestion_state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = {"runs": {}, "watermarks": {}}
        if self.path.exists():
            try:
                import json

                self._state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._state = {"runs": {}, "watermarks": {}}

    def _persist(self) -> None:
        import json

        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, default=str, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def start_run(self, job_name: str, *, watermark: dict[str, Any] | None = None) -> JobRun:
        run_id = uuid.uuid4().hex
        now = datetime.utcnow()
        self._state.setdefault("runs", {})[run_id] = {
            "job_name": job_name,
            "status": "running",
            "watermark": watermark,
            "started_at": now.isoformat(),
            "finished_at": None,
            "error": None,
        }
        self._persist()
        return JobRun(run_id=run_id, job_name=job_name, status="running", watermark=watermark, started_at=now)

    def finish_run(self, run_id: str, *, status: str = "succeeded", error: str | None = None, watermark=None) -> None:
        run = self._state.get("runs", {}).get(run_id)
        if run is None:
            return
        run["status"] = status
        run["error"] = error
        run["watermark"] = watermark or run.get("watermark")
        run["finished_at"] = datetime.utcnow().isoformat()
        self._persist()

    def get_watermark(self, job_name: str, scope: str) -> Optional[pd.Timestamp]:
        wm = self._state.get("watermarks", {}).get(job_name, {}).get(scope)
        if wm is None:
            return None
        return pd.to_datetime(wm, utc=True)

    def upsert_watermark(self, job_name: str, scope: str, ts: pd.Timestamp) -> None:
        self._state.setdefault("watermarks", {}).setdefault(job_name, {})[scope] = ts.isoformat()
        self._persist()


def init_state_store(dsn: Optional[str]) -> BaseStateStore:
    if dsn:
        try:
            return IngestionStateStore(dsn)
        except Exception as exc:
            logger.warning("Falling back to file state store, DB unavailable: %s", exc)
    return FileStateStore()
