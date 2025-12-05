from __future__ import annotations

import numpy as np
import pytest

from finantradealgo.risk.var_calculator import VaRCalculator, RiskMetricConfig


def test_var_calculator_blocks_on_heavy_tail():
    pnl_samples = np.concatenate([np.random.normal(0, 1, 1000), np.array([-20, -15, -10])])
    cfg = RiskMetricConfig(confidence_level=0.95, horizon_days=1)
    calc = VaRCalculator(cfg)
    var_result = calc.historical_var(pnl_samples)
    # VaR is reported as a positive loss threshold; heavy left tail should yield a positive number
    assert var_result.var_value > 0.0
    assert var_result.var_value > 0.01


def test_var_calculator_small_variance_series():
    pnl_samples = np.random.normal(1, 0.2, 1000)
    cfg = RiskMetricConfig(confidence_level=0.95, horizon_days=1)
    calc = VaRCalculator(cfg)
    var_result = calc.historical_var(pnl_samples)
    # For a tight positive distribution, VaR should not be an extreme loss
    assert 0.0 <= var_result.var_value < 1.0
