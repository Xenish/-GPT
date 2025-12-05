from finantradealgo.system.config_loader import load_config, MLConfig
from finantradealgo.ml.ml_utils import get_ml_targets


def test_ml_config_contract_research():
    cfg = load_config("research")
    ml_cfg = cfg.get("ml_cfg")
    assert isinstance(ml_cfg, MLConfig)
    assert ml_cfg.proba_column == "ml_long_proba"
    assert ml_cfg.signal_column == "ml_long_signal"
    assert ml_cfg.model_dir
    targets = get_ml_targets(cfg)
    assert targets  # at least one target
    # Targets should match configured ml.targets when present
    expected_targets = [(t["symbol"], t["timeframe"]) for t in ml_cfg.targets]
    if expected_targets:
        assert targets == expected_targets
