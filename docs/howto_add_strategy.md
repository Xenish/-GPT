# How to Add a New Strategy

## 1. Create the Strategy Class
1. Add a new module under `finantradealgo/strategies/`, e.g. `my_new_strategy.py`.
2. Implement `MyNewStrategy(BaseStrategy)` (or inherit from an existing helper) and optionally a config dataclass.
   ```python
   @dataclass
   class MyNewStrategyConfig:
       fast: int = 14
       slow: int = 42

       @classmethod
       def from_dict(cls, data: dict | None) -> "MyNewStrategyConfig":
           data = data or {}
           return cls(fast=int(data.get("fast", cls.fast)), slow=int(data.get("slow", cls.slow)))


   class MyNewStrategy(BaseStrategy):
       def __init__(self, config: MyNewStrategyConfig):
           self.config = config

       def init(self, df: pd.DataFrame) -> None:
           ...

       def on_bar(self, ctx: StrategyContext) -> None:
           ...
   ```

## 2. Expose in config/system.yml
```yaml
strategy:
  selected: my_new_strategy
  available: ["rule", "ml", "my_new_strategy"]
  my_new_strategy:
    fast: 10
    slow: 40
```

## 3. Register in Strategy Engine
Add to `finantradealgo/strategies/strategy_engine.py` (or equivalent factory):
```python
from .my_new_strategy import MyNewStrategy, MyNewStrategyConfig

def create_strategy(name: str, cfg: dict, overrides: dict | None = None):
    ...
    if name == "my_new_strategy":
        block = cfg.get("strategy", {}).get("my_new_strategy", {})
        if overrides:
            block = {**block, **overrides}
        return MyNewStrategy(MyNewStrategyConfig.from_dict(block))
```

## 4. Run a Backtest
CLI:
```bash
finantrade backtest --strategy my_new_strategy --symbol AIAUSDT --tf 15m
```
or python:
```bash
python scripts/run_backtest.py --your-flags  # if argument parsing exists
```

## 5. Scenario / Strategy Lab (optional)
In `scripts/run_scenario_grid_15m.py`, add a new grid loop for your strategy to generate parameter sweeps:
```python
my_param_grid = {
    "fast": [8, 12, 16],
    "slow": [32, 48],
}
for values in itertools.product(...):
    scenarios.append(
        Scenario(
            symbol=symbol,
            timeframe=timeframe,
            strategy="my_new_strategy",
            params=dict(zip(keys, values)),
            label=f"mystr_fast{...}",
        )
    )
```
Results will appear in `outputs/backtests/scenario_grid_*.csv` and the Strategy Lab tab via `/api/scenarios`.

## 6. Tests
Create a regression test (e.g., `tests/test_strategies.py`) that loads dummy data and ensures:
- Signals are generated without raising.
- Backtest metrics make sense (non-zero trades, etc.).
- Corner cases (missing columns, invalid params) raise informative errors.

## Done
Your strategy now works via CLI (`finantrade backtest`), API (`/api/backtests/run` with `strategy`), and Strategy/ML Labs once you add it to relevant grids or UI selectors.
