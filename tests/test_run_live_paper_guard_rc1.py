import types

import pytest

import scripts.run_live_paper as rlp
from finantradealgo.system.config_loader import LiveConfig


def test_run_live_paper_forces_paper_mode(monkeypatch):
    captured_cfg = {}

    def fake_load_config(profile):
        return {
            "mode": "live",
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "live": {"mode": "exchange", "symbol": "BTCUSDT", "timeframe": "15m"},
            "live_cfg": LiveConfig.from_dict({"mode": "exchange", "symbol": "BTCUSDT", "timeframe": "15m"}),
        }

    class FakeExecClient:
        def get_portfolio(self):
            return {"equity": 1000.0}

        def get_trade_log(self):
            return []

    class FakeEngine:
        def __init__(self):
            self.iteration = 0
            self.execution_client = FakeExecClient()
            self.state_path = "state.json"
            self.latest_state_path = "latest.json"

        def run(self):
            self.iteration = 1

        def shutdown(self):
            return

        def export_results(self):
            return {}

    def fake_create_live_engine(cfg, **kwargs):
        captured_cfg.update(cfg)
        return FakeEngine(), "rule"

    def fake_init_logger(*args, **kwargs):
        return types.SimpleNamespace(log_path=None, info=lambda *a, **k: None)

    monkeypatch.setattr(rlp, "load_config", fake_load_config)
    monkeypatch.setattr(rlp, "create_live_engine", fake_create_live_engine)
    monkeypatch.setattr(rlp, "init_logger", fake_init_logger)

    rlp.main(symbol="BTCUSDT", timeframe="15m", profile="live")

    assert captured_cfg["live"]["mode"] == "paper"
    assert captured_cfg["live_cfg"].mode == "paper"


def test_run_live_paper_requires_profile(monkeypatch):
    def fake_load_config(profile):
        return {}
    monkeypatch.setattr(rlp, "load_config", fake_load_config)
    # Passing None profile should still work (defaults), but not allow config_path
    with pytest.raises(RuntimeError):
        rlp.main(profile=None)
