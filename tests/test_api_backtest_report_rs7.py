import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.research_service.reporting_api import router


def _setup_job(tmp_path: Path):
    job_id = "job_api"
    base_dir = tmp_path / "backtests"
    job_dir = base_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "equity_metrics": {"cum_return": 0.15, "sharpe": 1.2, "max_drawdown": -0.08},
        "trade_stats": {"trade_count": 4, "win_rate": 0.5},
    }
    (job_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    pd.DataFrame({"equity": [100, 102, 105]}).to_csv(job_dir / "equity_curve.csv", index=False)
    pd.DataFrame({"pnl": [5, -2, 3, -1], "side": ["long", "short", "long", "short"]}).to_csv(
        job_dir / "trades.csv", index=False
    )
    return job_id, base_dir


def test_api_backtest_report_returns_content(tmp_path, monkeypatch):
    job_id, base_dir = _setup_job(tmp_path)
    monkeypatch.setenv("BACKTEST_REPORT_BASE_DIR", str(base_dir))

    app = FastAPI()
    app.include_router(router, prefix="/api/research/reports")
    client = TestClient(app)

    resp_json = client.get(f"/api/research/reports/backtests/{job_id}/report?format=json")
    assert resp_json.status_code == 200
    payload = resp_json.json()
    assert payload["format"] == "json"
    assert payload["content"]["metrics"]["sharpe"] == 1.2

    resp_html = client.get(f"/api/research/reports/backtests/{job_id}/report?format=html")
    assert resp_html.status_code == 200
    payload_html = resp_html.json()
    assert payload_html["format"] == "html"
    assert "<html" in payload_html["content"]
