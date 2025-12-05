from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finantradealgo.risk import RiskMetricConfig
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine


def test_tail_risk_guard_blocks_when_var_exceeds_risk():
    # Construct returns with large losses to trigger VaR guard
    returns = pd.Series(np.concatenate([np.full(50, -0.05), np.full(50, 0.001)]))
    tail_cfg = RiskMetricConfig(confidence_level=0.99, horizon_days=1, use_log_returns=False)
    risk_cfg = RiskConfig(capital_risk_pct_per_trade=0.01, max_daily_loss_pct=0.02)
    engine = RiskEngine(risk_cfg)

    allowed = engine.can_open_new_trade(
        current_date=pd.Timestamp("2024-01-01"),
        equity_start_of_day=1000.0,
        realized_pnl_today=0.0,
        row=None,
        open_positions=[],
        max_open_trades=5,
        tail_guard_returns=returns,
        tail_guard_config=tail_cfg,
    )
    assert allowed is False
