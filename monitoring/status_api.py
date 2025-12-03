from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from finantradealgo.data_engine.ingestion.state import IngestionStateStore, init_state_store, FileStateStore

logger = logging.getLogger(__name__)
app = FastAPI(title="FinanTrade Status", version="0.1.0")


def _try_store(dsn: Optional[str]) -> Optional[IngestionStateStore]:
    try:
        return init_state_store(dsn)
    except Exception as exc:  # pragma: no cover - connectivity issues
        logger.warning("Failed to init state store: %s", exc)
        return None


@app.get("/health")
def health(dsn: Optional[str] = None):
    store = _try_store(dsn)
    status = {"status": "ok", "store_connected": not isinstance(store, type(None))}
    return status


@app.get("/watermarks")
def watermarks(dsn: Optional[str] = None, job: Optional[str] = None, scope: Optional[str] = None):
    store = _try_store(dsn)
    if store is None:
        raise HTTPException(status_code=400, detail="No state store configured/reachable")
    if job and scope:
        wm = store.get_watermark(job, scope)
        return {"job": job, "scope": scope, "watermark": wm.isoformat() if wm is not None else None}
    if isinstance(store, IngestionStateStore) and hasattr(store, "_conn"):
        query = """
            SELECT job_name, scope, watermark_ts
            FROM ingestion_watermarks
            WHERE (%s IS NULL OR job_name = %s)
        """
        with store._conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(query, (job, job))
            rows = cur.fetchall()
        return [
            {"job": r[0], "scope": r[1], "watermark": pd.to_datetime(r[2]).isoformat() if r[2] else None}
            for r in rows
        ]
    if isinstance(store, FileStateStore):
        results = []
        for jname, scopes in store._state.get("watermarks", {}).items():  # type: ignore[attr-defined]
            if job and jname != job:
                continue
            for sc, val in scopes.items():
                results.append({"job": jname, "scope": sc, "watermark": val})
        return results
    return []


@app.get("/runs")
def runs(dsn: Optional[str] = None, job: Optional[str] = None, limit: int = 50):
    store = _try_store(dsn)
    if store is None:
        raise HTTPException(status_code=400, detail="No state store configured/reachable")
    if isinstance(store, IngestionStateStore) and hasattr(store, "_conn"):
        sql = """
            SELECT run_id, job_name, status, started_at, finished_at, error
            FROM ingestion_runs
            WHERE (%s IS NULL OR job_name = %s)
            ORDER BY started_at DESC
            LIMIT %s
        """
        with store._conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(sql, (job, job, limit))
            rows = cur.fetchall()
        return [
            {
                "run_id": r[0],
                "job": r[1],
                "status": r[2],
                "started_at": r[3].isoformat() if r[3] else None,
                "finished_at": r[4].isoformat() if r[4] else None,
                "error": r[5],
            }
            for r in rows
        ]
    if isinstance(store, FileStateStore):
        rows = []
        for rid, info in store._state.get("runs", {}).items():  # type: ignore[attr-defined]
            if job and info.get("job_name") != job:
                continue
            rows.append(
                {
                    "run_id": rid,
                    "job": info.get("job_name"),
                    "status": info.get("status"),
                    "started_at": info.get("started_at"),
                    "finished_at": info.get("finished_at"),
                    "error": info.get("error"),
                }
            )
        rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
        return rows[:limit]
    return []
