# Testing Guide

## Running The Suite
- `python -m pytest` runs all tests, including the slow integration/regression ones.
- `python -m pytest tests/test_risk.py` focuses on the risk unit tests during development.
- `pytest -m "not slow"` skips the heavier integration/regression checks by using the slow marker.

## What Each Test Covers
- `tests/test_risk.py` → position sizing, leverage guardrails, and daily loss enforcement.
- `tests/test_feature_pipeline.py` → feature pipeline metadata + column presets.
- `tests/test_model_registry.py` → registry CRUD and metadata bookkeeping.
- `tests/test_backtester_integration.py` → Rule strategy + RiskEngine run end-to-end inside the backtester.
- `tests/test_rule_vs_ml_integration.py` (`@slow`) → rule and ML strategies sharing the same feature pipeline, model training, and backtester wiring.
- `tests/test_regression_backtests.py` (`@slow`) → regression safety net comparing rule/ML equity + trade counts to golden values.

## Golden Output Refresh
When you intentionally change configs or strategy behavior that affects regression outputs:
1. Run the existing CLI entry-points (e.g. `python scripts/run_rule_backtest_15m.py` and `python scripts/run_ml_backtest_15m.py`) or replicate what the regression test does to gather fresh metrics.
2. Update `tests/golden/regression_rule_ml_15m.json` with the new `final_equity` and `trade_count` values for both rule and ML runs.
3. Commit the updated golden file alongside the code/config change and mention the refresh reason in the PR or changelog.
