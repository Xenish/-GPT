from finantradealgo.system.config_loader import load_config, MLConfig
from finantradealgo.ml.ml_utils import get_ml_targets, is_ml_enabled


def test_ml_config_from_research_profile():
    cfg = load_config("research")
    ml_cfg = cfg.get("ml_cfg")
    assert isinstance(ml_cfg, MLConfig)
    assert ml_cfg.enabled is True
    assert ml_cfg.model_dir
    targets = get_ml_targets(cfg)
    assert len(targets) >= 1
    assert all(len(t) == 2 for t in targets)


def test_ml_enabled_flag_respects_config(monkeypatch):
    cfg = {"ml_cfg": MLConfig(enabled=False)}
    assert is_ml_enabled(cfg) is False
    cfg2 = {"ml": {"enabled": True}}
    assert is_ml_enabled(cfg2) is True
