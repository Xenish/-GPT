# Playbook: Ensemble Strategy Development

**Objective**: Build and evaluate an ensemble meta-strategy that combines multiple base strategies.

**Duration**: 1-3 hours

**Output**: Ensemble configuration, performance report, component analysis

---

## Prerequisites

### Base Strategies
- ✅ At least 2-3 component strategies validated individually
- ✅ Component strategies have uncorrelated signals (ideally)
- ✅ Each component has positive expected value

### Data & Config
- ✅ OHLCV data for testing
- ✅ Research mode config (`mode: research`)

### Knowledge
- ✅ Understand difference between weighted and bandit ensembles
- ✅ Familiar with exploration-exploitation tradeoff

---

## Decision: Weighted vs Bandit Ensemble

### Use Weighted Ensemble When:
- You want to combine ALL component signals
- Components are complementary (low correlation)
- Market regime is stable
- You have confidence in all components

### Use Bandit Ensemble When:
- You want to SELECT best component dynamically
- Components are substitutes (high correlation)
- Market regime shifts frequently
- Uncertain which component will perform best

**For this playbook, we'll show both approaches.**

---

## Step 1: Select Component Strategies

### Criteria for Component Selection

**Diversity**:
- [ ] Different strategy families (trend, mean-reversion, volatility)
- [ ] Different parameter sets
- [ ] Different lookback periods

**Individual Performance**:
- [ ] Each component has Sharpe > 0.3
- [ ] Trade count > 20
- [ ] Max DD < 30%

**Complementarity**:
```python
# Check signal correlation
from finantradealgo.data.ohlcv_loader import load_ohlcv
from finantradealgo.strategies.strategy_engine import create_strategy

strategies = ["rule", "trend_continuation", "sweep_reversal"]
df = load_ohlcv("AIAUSDT", "15m")

signals = {}
for strat_name in strategies:
    strat = create_strategy(strat_name, sys_cfg)
    result = strat.generate_signals(df.copy())
    signals[strat_name] = result["long_entry"].astype(int)

# Correlation matrix
import pandas as pd
signal_df = pd.DataFrame(signals)
print(signal_df.corr())
```

**Target**: Correlation < 0.7 between components (lower is better for diversification)

---

## Step 2: Configure Weighted Ensemble

### Option A: Equal Weight (Baseline)

```python
from finantradealgo.research.ensemble import (
    WeightedEnsembleConfig,
    WeightingMethod,
    ComponentStrategy,
)

components = [
    ComponentStrategy("rule", params={}, weight=1.0, label="rule"),
    ComponentStrategy("trend_continuation", params={"trend_min": 1.0}, weight=1.0, label="trend"),
    ComponentStrategy("sweep_reversal", params={}, weight=1.0, label="sweep"),
]

config = WeightedEnsembleConfig(
    components=components,
    weighting_method=WeightingMethod.EQUAL,
    warmup_bars=100,
    signal_threshold=0.5,  # 50% of components must agree
)
```

### Option B: Sharpe-Weighted (Adaptive)

```python
config = WeightedEnsembleConfig(
    components=components,
    weighting_method=WeightingMethod.SHARPE,
    reweight_period=50,  # Reweight every 50 bars
    lookback_bars=100,   # Use 100 bars for Sharpe calculation
    signal_threshold=0.5,
)
```

**Weighting Methods**:
- `EQUAL`: Simple average (1/N each)
- `SHARPE`: Weight by Sharpe ratio (reward good performers)
- `INVERSE_VOL`: Weight by inverse volatility (risk parity)
- `RETURN`: Weight by cumulative return
- `CUSTOM`: Specify weights manually

---

## Step 3: Configure Bandit Ensemble

### Option A: Epsilon-Greedy (Simple)

```python
from finantradealgo.research.ensemble import (
    BanditEnsembleConfig,
    BanditAlgorithm,
)

config = BanditEnsembleConfig(
    components=components,
    bandit_algorithm=BanditAlgorithm.EPSILON_GREEDY,
    epsilon=0.1,  # 10% exploration
    update_period=20,  # Switch/update every 20 bars
    reward_metric="sharpe",
)
```

### Option B: UCB1 (Optimistic)

```python
config = BanditEnsembleConfig(
    components=components,
    bandit_algorithm=BanditAlgorithm.UCB1,
    ucb_c=2.0,  # Exploration bonus (higher = more exploration)
    update_period=20,
    reward_metric="sharpe",
)
```

### Option C: Thompson Sampling (Bayesian)

```python
config = BanditEnsembleConfig(
    components=components,
    bandit_algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
    update_period=20,
    reward_metric="return",  # Can use "sharpe", "win_rate"
)
```

---

## Step 4: Run Backtest

### Via API (Recommended)

```python
import requests

payload = {
    "ensemble_type": "weighted",
    "symbol": "AIAUSDT",
    "timeframe": "15m",
    "components": [
        {"strategy_name": "rule", "weight": 1.0},
        {"strategy_name": "trend_continuation", "weight": 1.5},
        {"strategy_name": "sweep_reversal", "weight": 1.0},
    ],
    "weighting_method": "sharpe",
    "reweight_period": 50,
}

response = requests.post(
    "http://localhost:8001/api/research/ensemble/run",
    json=payload
)

results = response.json()
```

### Via Python SDK

```python
from finantradealgo.research.ensemble.backtest import run_ensemble_backtest
from finantradealgo.research.ensemble import WeightedEnsembleStrategy
from finantradealgo.data.ohlcv_loader import load_ohlcv

df = load_ohlcv("AIAUSDT", "15m")
ensemble = WeightedEnsembleStrategy(config)

result = run_ensemble_backtest(
    ensemble_strategy=ensemble,
    df=df,
    components=components,
    sys_cfg=sys_cfg,
)
```

---

## Step 5: Analyze Results

### Key Metrics to Check

**Ensemble vs Best Component**:
```python
ensemble_sharpe = result.ensemble_metrics["sharpe"]
best_component_sharpe = result.component_metrics["sharpe"].max()

improvement = ensemble_sharpe - best_component_sharpe
print(f"Ensemble improvement: {improvement:.4f} Sharpe points")
```

**Target**: Ensemble Sharpe > best individual component

**Component Contributions**:
```python
# For weighted ensemble
print("Component Weights:")
print(result.weight_history)

# For bandit ensemble
print("Bandit Statistics:")
print(result.bandit_stats)
```

**Diversification Benefit**:
- Lower drawdown than components
- More stable equity curve
- Reduced single-strategy risk

---

## Step 6: Generate Report

```python
from finantradealgo.research.reporting import EnsembleReportGenerator

generator = EnsembleReportGenerator()
report = generator.generate(
    backtest_result=result,
    ensemble_type="weighted",  # or "bandit"
    symbol="AIAUSDT",
    timeframe="15m",
)

report.save("reports/ensemble_weighted_AIAUSDT_15m.html")
```

Review the HTML report for:
- [ ] Ensemble performance summary
- [ ] Component comparison table
- [ ] Weight evolution (weighted) or selection history (bandit)
- [ ] Recommendations

---

## Step 7: Compare Ensemble Types

Run both weighted and bandit ensembles with same components:

```python
# Test both
weighted_result = run_ensemble_backtest(weighted_ensemble, df, components, sys_cfg)
bandit_result = run_ensemble_backtest(bandit_ensemble, df, components, sys_cfg)

# Compare
comparison = pd.DataFrame({
    "Weighted": weighted_result.ensemble_metrics,
    "Bandit": bandit_result.ensemble_metrics,
}).T

print(comparison)
```

**Decision Criteria**:
- If weighted > bandit: Components are complementary → use weighted
- If bandit > weighted: Components are substitutes → use bandit
- If similar: Try hybrid or use simpler (weighted)

---

## Step 8: Optimize Ensemble Parameters

### For Weighted Ensemble

Test different:
- **Signal thresholds**: 0.3, 0.5, 0.7
- **Reweight periods**: 20, 50, 100 bars
- **Weighting methods**: EQUAL, SHARPE, INVERSE_VOL

### For Bandit Ensemble

Test different:
- **Algorithms**: EPSILON_GREEDY, UCB1, THOMPSON_SAMPLING
- **Epsilon values** (epsilon-greedy): 0.05, 0.1, 0.2
- **UCB c values** (UCB1): 1.0, 2.0, 3.0
- **Update periods**: 10, 20, 50 bars

**Run mini parameter search**:
```python
results = []
for threshold in [0.3, 0.5, 0.7]:
    config.signal_threshold = threshold
    ensemble = WeightedEnsembleStrategy(config)
    result = run_ensemble_backtest(ensemble, df, components, sys_cfg)
    results.append({
        "threshold": threshold,
        "sharpe": result.ensemble_metrics["sharpe"],
    })

best = max(results, key=lambda x: x["sharpe"])
```

---

## Step 9: Validate Robustness

### Cross-Symbol Validation

```python
symbols = ["AIAUSDT", "BTCUSDT", "ETHUSDT"]

for symbol in symbols:
    df = load_ohlcv(symbol, "15m")
    result = run_ensemble_backtest(ensemble, df, components, sys_cfg)
    print(f"{symbol}: Sharpe = {result.ensemble_metrics['sharpe']:.4f}")
```

**Success Criteria**:
- [ ] Positive Sharpe on all symbols
- [ ] Performance degradation < 40% vs best symbol

### Timeframe Validation

```python
timeframes = ["15m", "1h", "4h"]

for tf in timeframes:
    df = load_ohlcv("AIAUSDT", tf)
    result = run_ensemble_backtest(ensemble, df, components, sys_cfg)
    print(f"{tf}: Sharpe = {result.ensemble_metrics['sharpe']:.4f}")
```

---

## Quality Checklist

### Component Selection
- [ ] 2-5 components selected (sweet spot: 3-4)
- [ ] Each component has positive individual performance
- [ ] Signal correlation < 0.7

### Ensemble Design
- [ ] Ensemble type chosen based on component characteristics
- [ ] Parameters tuned (threshold, epsilon, update period)
- [ ] Warmup period sufficient (> 100 bars)

### Performance
- [ ] Ensemble Sharpe > best individual component
- [ ] Ensemble max DD < worst component
- [ ] Trade count reasonable (not too high/low)

### Validation
- [ ] Tested on multiple symbols
- [ ] Tested on different timeframes
- [ ] Out-of-sample period validated

### Documentation
- [ ] Report generated and saved
- [ ] Component selection rationale documented
- [ ] Ensemble configuration saved

---

## Common Pitfalls

### ❌ Too Many Components
**Problem**: 10+ components, each contributing 10%, high complexity.

**Solution**:
- Keep ensemble size small (3-5 components)
- Remove underperforming or highly correlated components

### ❌ Highly Correlated Components
**Problem**: All components have 0.9+ correlation, no diversification benefit.

**Solution**:
- Check signal correlation before building ensemble
- Select diverse strategies (different families/params)

### ❌ Not Accounting for Transaction Costs
**Problem**: Ensemble switches components frequently (bandits), high costs.

**Solution**:
- Increase update period
- Add transaction cost buffer
- Prefer weighted ensemble for high-frequency switching

### ❌ Overfitting Ensemble Parameters
**Problem**: Optimizing 10 ensemble parameters on same data as component optimization.

**Solution**:
- Use simple defaults (equal weight, 0.5 threshold)
- Validate on out-of-sample data
- Prefer interpretable configurations

---

## Next Steps

After completing this playbook:

1. **If ensemble outperforms**: Proceed to [Robustness Testing](robustness_testing.md)
2. **If components weak**: Go back to [Strategy Parameter Search](strategy_param_search.md)
3. **If ready for production**: Paper trade ensemble for 1-2 weeks

---

**Remember**: Ensembles reduce risk but don't eliminate it. Monitor performance and component contributions over time.
