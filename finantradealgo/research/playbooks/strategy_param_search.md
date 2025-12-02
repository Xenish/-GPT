# Playbook: Strategy Parameter Search

**Objective**: Find optimal parameters for a single strategy through systematic search.

**Duration**: 30 minutes - 2 hours (depending on search size)

**Output**: Parameter recommendations, performance report, reproducible job

---

## Prerequisites

### Data
- ✅ OHLCV data for symbol/timeframe in `data/ohlcv/`
- ✅ Data quality validated (no gaps, outliers checked)

### Config
- ✅ `config/system.research.yml` configured with `mode: research`
- ✅ Exchange type set to `backtest` or `mock`

### Strategy
- ✅ Strategy registered in `finantradealgo/strategies/strategy_engine.py`
- ✅ `ParamSpace` defined for the strategy
- ✅ Strategy passes unit tests

---

## Step 1: Define Research Question

**Clarify your goal**:
- [ ] What parameter(s) are you optimizing?
- [ ] What is your primary metric? (Sharpe, return, max DD, etc.)
- [ ] What is the acceptable trade-off? (e.g., Sharpe > X, trades > Y)

**Example Questions**:
- "What ATR multiplier gives best risk-adjusted returns for rule strategy?"
- "What trend threshold minimizes drawdown while maintaining returns?"

**Document** your hypothesis and success criteria.

---

## Step 2: Review Parameter Space

Check the strategy's parameter space definition:

```python
from finantradealgo.strategies.strategy_engine import get_strategy_meta

meta = get_strategy_meta("rule")  # Replace with your strategy
print("Parameter Space:")
for param_name, spec in meta.param_space.items():
    print(f"  {param_name}: {spec.type} [{spec.low}, {spec.high}]")
```

**Questions to ask**:
- [ ] Are ranges realistic? (not too wide or narrow)
- [ ] Are step sizes appropriate? (granular enough but not excessive)
- [ ] Are there constraints between parameters?

**Adjust if needed** by modifying the param space file.

---

## Step 3: Configure Search Job

Create a job configuration or use CLI:

### Option A: CLI (Quick)

```bash
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol AIAUSDT \
  --timeframe 15m \
  --n-samples 100 \
  --seed 42 \
  --notes "First parameter search for rule strategy"
```

### Option B: YAML Config (Reproducible)

Create `jobs/my_search.yml`:

```yaml
job_name: "rule_param_search_v1"
strategy_name: "rule"
symbol: "AIAUSDT"
timeframe: "15m"
mode: "random"
n_samples: 100
random_seed: 42
notes: "Initial parameter search for rule strategy"
```

Run:
```bash
python -m scripts.run_strategy_search_from_yaml jobs/my_search.yml
```

**Recommendations**:
- Start with **50-100 samples** for initial exploration
- Use **random seed** for reproducibility
- Document **notes** for future reference

---

## Step 4: Run Search

Execute the search:

```bash
python -m scripts.run_strategy_search \
  --strategy rule \
  --symbol AIAUSDT \
  --timeframe 15m \
  --n-samples 100 \
  --seed 42
```

**Monitor Progress**:
- Watch for errors in console output
- Check CPU/memory usage (shouldn't spike excessively)
- Estimated time: ~1-5 minutes per 100 samples (depends on strategy complexity)

**Expected Output**:
```
outputs/strategy_search/rule_AIAUSDT_15m_20251202_143022/
├── results.parquet        # Search results
├── meta.json             # Job metadata
└── config_snapshot.yml   # Config used
```

---

## Step 5: Analyze Results

### Quick Analysis (Command Line)

```python
import pandas as pd

# Load results
df = pd.read_parquet("outputs/strategy_search/{job_id}/results.parquet")

# Top 10 by Sharpe
print(df.nlargest(10, "sharpe")[["sharpe", "cum_return", "max_drawdown", "trade_count"]])

# Parameter correlations
param_cols = [c for c in df.columns if c.startswith("param_")]
for param in param_cols:
    corr = df[param].corr(df["sharpe"])
    print(f"{param}: {corr:.3f}")
```

### Generate Report

```python
from finantradealgo.research.reporting import StrategySearchReportGenerator
from pathlib import Path

generator = StrategySearchReportGenerator()
report = generator.generate_and_save(
    job_dir=Path("outputs/strategy_search/{job_id}"),
    output_path=Path("reports/{job_id}.html"),
    top_n=10,
)
```

Open `reports/{job_id}.html` in browser to review.

---

## Step 6: Interpret Results

### Key Questions

**1. Did we find good parameters?**
- [ ] Best Sharpe > baseline (rule of thumb: > 0.5 for crypto)
- [ ] Trade count reasonable (> 20 trades for statistical significance)
- [ ] Drawdown acceptable (< 30% for aggressive, < 15% for conservative)

**2. Is there parameter sensitivity?**
- [ ] Check correlation table: which params matter most?
- [ ] Are top performers clustered in parameter space?
- [ ] Is performance stable across nearby parameter values?

**3. Are results robust?**
- [ ] Do top 10 parameter sets have similar performance?
- [ ] Or is there one clear winner (potential overfit)?
- [ ] Are there multiple "regimes" of good parameters?

### Red Flags ⚠️

- **Single outlier**: One config vastly outperforms others → likely overfit
- **High variance**: Top 10 have widely different metrics → unstable
- **Low trade count**: < 20 trades → insufficient data
- **No clear pattern**: Parameters uncorrelated with performance → random results

---

## Step 7: Validate Best Parameters

### Out-of-Sample Test

Test best parameters on different symbol/timeframe:

```python
from finantradealgo.research.scenario_analysis import run_scenarios

best_params = {...}  # From search results

scenarios = [
    {"symbol": "BTCUSDT", "timeframe": "15m", "strategy": "rule", "params": best_params},
    {"symbol": "AIAUSDT", "timeframe": "1h", "strategy": "rule", "params": best_params},
]

results = run_scenarios(scenarios)
```

**Success Criteria**:
- [ ] Performance degrades < 30% on out-of-sample data
- [ ] Sharpe remains positive
- [ ] No catastrophic drawdowns

---

## Step 8: Document and Save

### Create Summary Document

Save `docs/research/rule_param_search_YYYYMMDD.md`:

```markdown
# Rule Strategy Parameter Search - 2025-12-02

## Objective
Find optimal ATR multipliers for rule strategy on AIAUSDT/15m.

## Method
- Random search, 100 samples
- Seed: 42
- Data: 2024-01-01 to 2024-12-01

## Results
Best parameters:
- param_atr_mult_tp: 2.5
- param_atr_mult_sl: 1.0
- Sharpe: 1.2
- Cum Return: 45%
- Max DD: -12%

## Validation
Tested on BTCUSDT/15m:
- Sharpe: 0.9 (25% degradation, acceptable)

## Decision
✅ Approved for paper trading with best params
```

### Save Artifacts

```bash
# Copy results to research archive
mkdir -p research_archive/rule_param_search_20251202/
cp -r outputs/strategy_search/{job_id}/* research_archive/rule_param_search_20251202/
cp reports/{job_id}.html research_archive/rule_param_search_20251202/report.html
```

---

## Quality Checklist

Before concluding, verify:

### Data Quality
- [ ] Used correct symbol/timeframe
- [ ] Data period covers multiple market regimes
- [ ] No data gaps or anomalies

### Methodology
- [ ] Random seed set for reproducibility
- [ ] Sufficient sample size (50-100 minimum)
- [ ] Parameter ranges realistic

### Results
- [ ] Report generated and reviewed
- [ ] Best parameters identified
- [ ] Out-of-sample validation performed

### Documentation
- [ ] Research question documented
- [ ] Results saved with git SHA
- [ ] Decision recorded (approve/reject/iterate)

### Reproducibility
- [ ] Job metadata saved (meta.json)
- [ ] Config snapshot saved
- [ ] Can re-run search with same seed

---

## Common Pitfalls

### ❌ Overfitting
**Problem**: Selecting parameters that work perfectly on one dataset but fail elsewhere.

**Solution**:
- Use multiple symbols/timeframes for validation
- Prefer stable parameter regions over single outliers
- Test on recent out-of-sample data

### ❌ Insufficient Samples
**Problem**: 10-20 samples, missing optimal regions.

**Solution**:
- Use at least 50 samples for initial search
- Run 200-500 samples for production strategies
- Consider grid search for low-dimensional spaces

### ❌ Ignoring Trade Count
**Problem**: Optimizing on 5-10 trades (statistically insignificant).

**Solution**:
- Filter results by trade_count > 20
- Extend data period if needed
- Consider multiple symbols for aggregated stats

### ❌ No Out-of-Sample Test
**Problem**: Deploying directly from in-sample optimization.

**Solution**:
- Always validate on different data
- Use walk-forward analysis
- Paper trade before live

---

## Next Steps

After completing this playbook:

1. **If results good**: Proceed to [Multi-Strategy Comparison](multi_strategy_comparison.md)
2. **If results poor**: Iterate on strategy logic or [Regime Analysis](regime_analysis.md)
3. **If ready for production**: Follow [Robustness Testing](robustness_testing.md)

---

**Remember**: Parameter search is the beginning, not the end. Always validate and monitor performance.
