"""
Run a lightweight status API (FastAPI) exposing ingestion watermarks and run history.

Usage:
    uvicorn scripts.status_api:app --reload --port 8001
    uvicorn scripts.status_api:app --reload --port 8001 --host 0.0.0.0

DSN is passed as query param (?dsn=postgres://...), or it will rely on warehouse.dsn
if the FastAPI app is imported and initialized with environment-provided DSN.
"""

from monitoring.status_api import app  # re-export for uvicorn

__all__ = ["app"]
