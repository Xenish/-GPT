# Live / Paper Trading Guide

## Configure
- Update `config/system.live.yml` → `live` block. Key options:
  - `mode`: `"replay"`, `"paper"`, or `"live"` (currently replay/paper supported).
  - `symbol`, `timeframe`, `exchange`, `log_level`, `log_dir`.
  - `replay`: `bars_limit`, `start_index`, `start_timestamp` for slicing the feature DataFrame.
  - `paper`: `initial_cash`, `save_state_every_n_bars`, `state_path`, `output_dir`.
  - Strategy selection still comes from `strategy.default` (rule vs ML).

## Run
```
python scripts/run_live_paper_15m.py
```
- Feature pipeline is built from `system.yml`, then the configured strategy, risk engine, replay data source, and paper execution client are wired together.
- Each run gets a `run_id` (`<symbol>_<timeframe>_<UTC timestamp>`) and logs to `outputs/live_logs/<run_id>.log`.

## Outputs
- Equity curve CSV: `<paper.output_dir>/live_paper_equity_<timeframe>.csv`
- Trades CSV: `<paper.output_dir>/live_paper_trades_<timeframe>.csv`
- State snapshot: `paper.state_path` (JSON with cash, positions, closed trades, etc.)
- Logger console & file entries show entries/exits, risk blocks, iterations.

Use `docs/dev/testing.md` for the pytest matrix; to rerun live-mode replay tests quickly: `python -m pytest tests/test_live_replay.py`.
