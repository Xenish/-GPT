import json
from pathlib import Path

from fastapi.testclient import TestClient

from services.research_service.main import app

client = TestClient(app)


def _write_dummy_job(job_dir: Path):
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "results.parquet").write_text("")  # minimal to fail gracefully? but need valid parquet -> use csv
    import pandas as pd

    df = pd.DataFrame(
        [
            {"param_a": 1, "param_b": 2, "sharpe": 1.0, "cum_return": 0.1, "max_drawdown": -0.1, "trade_count": 10, "status": "ok", "error_message": None},
            {"param_a": 2, "param_b": 3, "sharpe": 0.5, "cum_return": 0.05, "max_drawdown": -0.2, "trade_count": 8, "status": "ok", "error_message": None},
        ]
    )
    df.to_parquet(job_dir / "results.parquet", index=False)
    meta = {
        "job_id": job_dir.name,
        "strategy": "rule",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "n_samples": 2,
        "search_type": "random",
        "profile": "research",
    }
    (job_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_get_job_report_html(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STRATEGY_SEARCH_OUTPUT_DIR", str(tmp_path / "strategy_search"))
    base_dir = Path(tmp_path / "strategy_search")
    job_dir = base_dir / "job_report"
    _write_dummy_job(job_dir)

    resp = client.get(f"/api/research/strategy-search/jobs/{job_dir.name}/report?format=html")
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "html"
    assert "<html>" in data["content"]


def test_get_job_report_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STRATEGY_SEARCH_OUTPUT_DIR", str(tmp_path / "strategy_search"))
    resp = client.get("/api/research/strategy-search/jobs/missing_job/report?format=html")
    assert resp.status_code == 404
