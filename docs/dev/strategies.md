# Strategies & Scenarios

## Strategy Types
- **rule** (`RuleSignalStrategy`): deterministic rules using HTF filters, ATR TP/SL.
- **ml** (`MLSignalStrategy`): probability-based signals from the ML pipeline.
- **trend_continuation**: rides EMA/RSi aligned trends with optional MS filters.
- **sweep_reversal**: liquidity sweep/fair-value-gap confluence entries.
- **volatility_breakout**: Bollinger contractions + ATR/HV expansion breakouts.

## Config From YAML
```
strategy:
  default: "rule"
  rule:
    use_ms_trend_filter: true
  trend_continuation:
    min_trend_score: 0.1
```
- Legacy top-level blocks (`rule`, `ml`, …) still work, but putting overrides under `strategy.<name>` keeps everything together.
- Call `create_strategy(name, system_cfg)` to map YAML → dataclass → strategy class.

## Scenario Engine
```
scenario:
  presets:
    core_15m:
      - name: "rule_ms_on"
        strategy_name: "rule"
        strategy_params:
          use_ms_trend_filter: true
        risk_params: {}
      - name: "vol_breakout_aggr"
        strategy_name: "volatility_breakout"
        strategy_params:
          min_contraction: 0.008
        risk_params:
          capital_risk_pct_per_trade: 0.02
```
- Run via `python scripts/run_scenario_grid_15m.py --preset core_15m`.
- Output CSV lands in `outputs/backtests/scenario_grid_<preset>.csv`; console prints top Sharpe scenarios.

## Adding A New Strategy
1. Create `<NewStrategy>Config` + `<NewStrategy>Strategy` that inherit `BaseStrategy`.
2. Add the pair to `finantradealgo/strategies/strategy_engine.py` registry (with `StrategyMeta`).
3. Wire any YAML defaults under `strategy.<name>` in `config/system.research.yml` / `config/system.live.yml`.
4. (Optional) Append entries to `scenario.presets` for quick experimentation & mention the strategy in this doc.
