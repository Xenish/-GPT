A quick project map with each file’s purpose and notable bits:

run_backtest.py: Loads 15m OHLCV CSV, adds basic features, runs EMA crossover backtest, prints report, writes equity/trades CSV.
run_ml_backtest.py: Loads 15m OHLCV, builds features/labels, generates walk‑forward ML probabilities/signals, prints per‑block metrics and classification report, backtests signal column, writes equity/trades CSV.
run_fetch_external_15m.py: Reads OHLCV range, pulls Binance funding and 15m open interest for that window, saves to CSVs.
scripts/runml_compare_models.py: One-pass feature/label prep, runs walk‑forward/backtest for multiple model configs (GBM, LogReg, RF, XGB), prints classification/backtest metrics, writes comparison CSV.
Core package:

finantradealgo/core/backtest.py: Backtester with long/short, slippage/fees, stop/TP, flip-on-opposite-signal; builds metrics and trade log; BacktestConfig.
core/data.py: CSV loader with required OHLCV schema, timestamp parsing/sorting.
core/features.py: Basic returns/vol/EMA trend + regime labeling.
core/portfolio.py: Portfolio/Position (side-aware) and equity tracking.
core/risk.py: RiskEngine position sizing (risk-per-trade, stop-based).
core/strategy.py: Strategy interface, SignalType, StrategyContext.
core/report.py: Equity/trade/regime stats aggregation.
core/pipeline.py: Unified 15m feature pipeline, toggles TA/candle/osc/MTF/external/rule blocks and returns feature list.
(Other referenced core modules: ta_features, candle_features, osc_features, multi_tf_features, external_features, rule_signals assumed present.)
Data sources:

data_sources/binance.py: Binance futures klines fetch (single/multi batches), config, save helpers.
data_sources/init.py: Re-exports Binance helpers.
Strategies:

strategies/ema_cross.py: EMA crossover LONG/CLOSE signals.
strategies/ml_signal.py: Uses model proba to emit LONG/CLOSE with hysteresis.
strategies/rsi_reversion.py: RSI-based LONG/SHORT/CLOSE reversion test.
strategies/signal_column.py: Simple long-only executor using a signal column.
strategies/rule_signals.py: Generates rule-based signal from entry/exit columns; strategy opens/closes based on that signal.
ML:

ml/model.py: Configurable classifier wrapper (GBM/LogReg/RF/XGB), fit/predict_proba/evaluate, save/load.
ml/labels.py: Forward-return long-only labels.
ml/walkforward.py: Walk-forward training/prediction, writes ml_proba_long/ml_signal_long, per-block precision/recall/accuracy/f1 logging.
Configs:

config/ema_example.yml, config/ml_example.yml, config/rsi_example.yml: Sample YAMLs for runner.
run_from_config.py: YAML-driven runner (data load, features/labels, optional model, strategy selection including RSI/EMA/ML), reporting.
Data:

data/ holds fetched CSVs (15m OHLCV, funding, OI).
