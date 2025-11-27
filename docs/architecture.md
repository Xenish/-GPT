# Architecture Overview

```
Raw Data  -->  data_engine  -->  features  -->  strategies + risk  -->  backtester / live_engine  -->  api  -->  frontend
                     ^                |                 |
        flow/sentiment/OHLCV          |                 v
                                      |           portfolio/scenario/ml
```

## System Overview

### data_engine
- CSV/Parquet loader, Binance klines, funding, open interest, liquidation map, sentiment/flow mocks.
- Normalizes timestamps (UTC), handles resampling and missing-bar detection.

### features
- 15m feature pipeline: TA indicators, candlestick stats, oscillators, higher-timeframe merges, market structure and microstructure metrics.
- Adds external blocks: funding/OI, flow (perp premium/basis), sentiment (score + z-score).
- Rule signal generation (entry/exit columns) lives here as well.

### strategies
- Strategy classes (rule, ML, trend follow, reversion, etc.) and configs.
- StrategyEngine factory chooses strategy by name and injects overrides.

### risk
- RiskConfig + RiskEngine: per-trade risk, daily loss limit, leverage multiplier, tail-risk guard.
- Blocks entries when limits exceeded; feeds stats back to report.

### backtester
- BacktestEngine (single instrument), PortfolioBacktestEngine (multi symbol/weight), ScenarioEngine (grid/preset), walk-forward helpers.
- Produces equity curve, trade log, risk stats, scenario result tables.

### ml
- Labels (forward returns, horizon thresholds), model wrappers (Sklearn + XGBoost), registry (save/load), hyperparameter search (RFC grid), feature importance.
- Live registry validation helpers and CLI utilities (`run_list_models.py`, `run_ml_rf_grid_15m.py`).

### live_trading
- LiveEngine coordinates data source (FileReplayDataSource), strategy, RiskEngine, PaperExecutionClient.
- Maintains snapshot JSON (state, positions, requested_action) for API control (pause/resume/flatten/stop).

### api
- FastAPI app exposing:
  - `/api/backtests/run`, `/api/scenarios/run`, `/api/meta`, `/api/portfolio/*`
  - ML registry endpoints `/api/ml/models/...`, `/api/ml/models/{id}/importance`
  - Live control `/api/live/control`, `/api/live/status`
- Uses `run_backtest_once`, ScenarioEngine, registry helpers, etc.

### frontend
- Next.js + Zustand + lightweight-charts.
- Tabs: 
  - *Single*: chart + trades + overlays + advanced rule params
  - *Portfolio*: run selector, chart stub, metrics
  - *Strategy Lab*: scenario grid viewer
  - *ML Lab*: model registry list + feature importance
  - *Live*: control panel with status polling and commands

## Data Flow
1. **Raw data** ingested via `data_engine` (OHLCV, funding, flow, sentiment).
2. **Feature pipeline** merges all signals, attaches rule/ML columns.
3. **Strategies & Risk** read the feature frame; RiskEngine enforces limits.
4. **Backtester / LiveEngine** execute strategies, producing equity/trade logs.
5. **ML layer** trains models, logs importance, registers artifacts.
6. **API** exposes orchestration endpoints and registry/lab views.
7. **Frontend** consumes API for charting, labs, live control.

## Notes
- Portfolio config (`config/system.yml`) controls multi-symbol backtests, equal/custom weights.
- Scenario presets live under `scenario.presets`; CLI `scripts/run_scenario_grid_15m.py` enumerates grids.
- CLI wrapper (`finantrade ...`) ties everything together for devops and CI use.
