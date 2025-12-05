# Strategy & Backtest Contract (Quick Reference)

This project enforces a thin strategy/backtest contract to avoid silent regressions.

## Strategy interface
- Implement `BaseStrategy` with two core methods:
  - `init(df: pd.DataFrame)`: Prepare any indicators/columns on the input DataFrame.
  - `on_bar(row: pd.Series, ctx: StrategyContext) -> SignalType`: Return `LONG`, `SHORT`, `CLOSE`, or `None`.
- Context fields:
  - `ctx.index`: Current integer index in the loop (DataFrame is iterated row-by-row).
  - `ctx.position`: Current open position (or `None`).
  - `ctx.equity`: Current equity.

## Param space expectations
- Each searchable strategy must expose a `param_space` (see `finantradealgo/strategies/param_space.py` and `tests/test_strategy_param_space_contract.py`).
- Rules enforced by `validate_param_space`:
  - Numeric params (`int`/`float`) must have `low` and `high`, with `low < high`.
  - Categorical params must declare non-empty `choices`.
  - `name` in `ParamSpec` must match the dict key.
- New strategies: add a `StrategyMeta` entry in `strategy_engine.py` with `param_space` set; CI will fail if missing/invalid.

## Backtester contract
- Backtests run via `BacktestEngine`/`Backtester` with minimal required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- Metrics returned include `final_equity`, `cum_return`, `max_drawdown`, `sharpe`, and `trade_count` (see `tests/test_backtest_basic_run.py`).
- Event/bar pipeline invariants are guarded by `tests/test_event_bars_validation.py` (monotonic timestamps, non-negative volume).

## Golden regression
- Golden fixtures live under `tests/golden/`. A deterministic EMA Cross fixture is provided (`ema_cross_synthetic.json`).
- Regeneration script: `python scripts/generate_backtest_golden.py` writes golden outputs. Tests compare against goldens with `pytest.approx` (`tests/test_backtest_golden_regression.py`).
- When updating strategies/backtester behavior intentionally, regenerate goldens and commit the updated files.

## Adding a new strategy
1) Implement the strategy subclass with `init`/`on_bar`.
2) Define `param_space` (if searchable) and register in `strategy_engine.py` with a `StrategyMeta`.
3) Add/extend tests if needed; CI contract tests will enforce registry + param space validity and basic backtest behavior.
