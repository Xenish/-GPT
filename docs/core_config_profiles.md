# Config Profiles - Research & Live Separation

## Overview

The trading system uses **config profiles** to strictly separate research/backtest operations from live/paper trading. This prevents accidental execution of backtests with live configs or vice versa.

## File Structure

```
config/
├── system.base.yml        # Base config (shared settings)
├── system.research.yml    # Research/backtest profile
└── system.live.yml        # Live/paper trading profile
```

### Inheritance Hierarchy

```
DEFAULT_SYSTEM_CONFIG (code)
    ↓
system.base.yml (shared: symbols, features, risk, strategies)
    ↓
    ├─→ system.research.yml (research mode)
    └─→ system.live.yml     (live/paper mode)
```

## Config Profiles

### 1. `system.base.yml` - Base Configuration

**Purpose:** Shared settings for both research and live

**Critical Blocks:**
- `data`: Symbols, timeframes, lookback_days
- `features`: Feature engineering toggles
- `risk`: Risk management parameters
- `rule`, `ml`: Strategy parameters
- `strategy`: Strategy definitions
- `scenario`: Scenario presets
- `portfolio`: Portfolio configuration

**Example:**
```yaml
mode: "base"

data:
  symbols:
    - "AIAUSDT"
    - "BTCUSDT"
  timeframes:
    - "15m"
    - "1h"

features:
  use_microstructure: true
  use_market_structure: true

risk:
  capital_risk_pct_per_trade: 0.01
  max_leverage: 5.0
```

### 2. `system.research.yml` - Research Profile

**Purpose:** Backtest and strategy research

**Critical Settings:**
- `mode: "research"` ← **MANDATORY**
- `exchange.type: "backtest"`
- `exchange.dry_run: true`
- No real API keys required
- Live/kill_switch/notifications disabled or minimal

**Example:**
```yaml
mode: "research"  # CRITICAL: Enforced by backtest scripts

exchange:
  type: "backtest"
  testnet: true
  dry_run: true
  api_key: ""  # Empty for research
  secret_key: ""

backtest:
  warmup_bars: 200
  slippage_pct: 0.001
  commission_pct: 0.0004

live:
  mode: "disabled"  # No live trading in research

kill_switch:
  enabled: false

notifications:
  enabled: false
```

**Usage:**
```bash
# CLI
python scripts/run_backtest.py --config config/system.research.yml

# Environment Variable
export FT_CONFIG_PATH=config/system.research.yml
python scripts/run_backtest.py

# API Server
FT_CONFIG_PATH=config/system.research.yml python scripts/run_api.py
```

### 3. `system.live.yml` - Live/Paper Profile

**Purpose:** Live or paper trading

**Critical Settings:**
- `mode: "paper" | "live"` ← **MANDATORY**
- `exchange.type: "live"`
- Real API keys (from env vars)
- `live.mode: "paper" | "exchange"`
- `kill_switch.enabled: true`
- `notifications.enabled: true` (optional)

**Example:**
```yaml
mode: "paper"  # CRITICAL: Enforced by live trading scripts

exchange:
  type: "live"
  testnet: true  # Use testnet for paper trading
  api_key: "${BINANCE_FUTURES_API_KEY}"
  secret_key: "${BINANCE_FUTURES_API_SECRET}"

live:
  mode: "paper"  # "paper" | "exchange"
  symbol: "AIAUSDT"  # Single symbol (narrow universe)
  timeframe: "15m"
  max_position_notional: 100.0
  max_daily_loss: 20.0

kill_switch:
  enabled: true
  daily_realized_pnl_limit: -20.0
  max_equity_drawdown_pct: 30.0

notifications:
  enabled: false  # Enable in production
```

**Usage:**
```bash
# CLI
python scripts/run_live_paper.py --config config/system.live.yml

# Environment Variable
export FT_CONFIG_PATH=config/system.live.yml
python scripts/run_live_paper.py

# API Server
FT_CONFIG_PATH=config/system.live.yml python scripts/run_api.py
```

## How It Works

### Config Loading

`load_config(profile="research"|"live")` automatically:

1. Resolves profile to `config/system.research.yml` or `config/system.live.yml`
2. Loads `system.base.yml` first (if present)
3. Deep merges: `DEFAULT → base → profile`
4. Adds `_config_meta` for debugging (path/mode/is_profile)

### Safety Checks (Mode Assertions)

**Backtest Scripts** (`run_backtest.py`, etc.): use `load_config("research")` and assert mode is `research`.

**Live Scripts** (`run_live_paper.py`, `run_live_exchange.py`): use `load_config("live")` and assert mode is `live` or `paper`.

## CLI Usage

### All Scripts Support `--config`

```bash
# Backtest
python scripts/run_backtest.py --config config/system.research.yml

# Paper trading
python scripts/run_live_paper.py --config config/system.live.yml

# Live exchange (DANGER!)
python scripts/run_live_exchange.py --config config/system.live.yml
```

### Environment Variable (Recommended for API)

```bash
# Research API
export FT_CONFIG_PATH=config/system.research.yml
python scripts/run_api.py

# Live API
export FT_CONFIG_PATH=config/system.live.yml
python scripts/run_api.py
```

## Best Practices

### ✅ DO

- Always use `system.research.yml` for backtests
- Always use `system.live.yml` for live/paper trading
- Keep `system.base.yml` for shared settings
- Use `mode` field to identify config purpose
- Test config loading with:
  ```python
  from finantradealgo.system.config_loader import load_config
  cfg = load_config('research')
  print(f"Mode: {cfg['mode']}")
  print(f"Meta: {cfg['_config_meta']}")
  ```

### ❌ DON'T

- Don't run backtest with `system.live.yml`
- Don't run live trading with `system.research.yml`
- Don't hardcode API keys in config files (use env vars)
- Don't skip mode assertions in scripts
- Don't create monolithic `system.yml` (use profile separation)

## Migration from Old `system.yml`

If you have an existing `system.yml`:

1. **Identify shared settings** → Move to `system.base.yml`
2. **Identify backtest-specific settings** → Move to `system.research.yml`
3. **Identify live-specific settings** → Move to `system.live.yml`
4. **Update scripts** to use `--config` or `FT_CONFIG_PATH`

## Troubleshooting

### "Backtest must run with mode='research' config"

**Problem:** Running backtest with wrong config

**Solution:**
```bash
python scripts/run_backtest.py --config config/system.research.yml
# OR
export FT_CONFIG_PATH=config/system.research.yml
python scripts/run_backtest.py
```

### "Live trading must run with mode='live' or mode='paper' config"

**Problem:** Running live trader with research config

**Solution:**
```bash
python scripts/run_live_paper.py --config config/system.live.yml
# OR
export FT_CONFIG_PATH=config/system.live.yml
python scripts/run_live_paper.py
```

### Config not found

**Problem:** `FileNotFoundError: System config not found at ...`

**Solution:** Check path and ensure file exists:
```bash
ls -la config/
# Should show: system.base.yml, system.research.yml, system.live.yml
```

### "Profile config found but system.base.yml missing"

**Problem:** Profile exists but base config missing

**Solution:** Create `system.base.yml` or disable profile (not recommended):
```bash
# Quick fix: copy base from research
cp config/system.research.yml config/system.base.yml
# Then remove research-specific overrides from base
```

## Summary Table

| Profile | Mode | Exchange Type | API Keys | Use Case | Kill Switch | Notifications |
|---------|------|---------------|----------|----------|-------------|---------------|
| `system.base.yml` | `base` | - | - | Shared settings | - | - |
| `system.research.yml` | `research` | `backtest` | Empty | Backtest, strategy research | Disabled | Disabled |
| `system.live.yml` | `paper` / `live` | `live` | Required (env) | Live/paper trading | **Enabled** | Optional |

## Next Steps

After setting up config profiles, you can:

1. Run backtests safely with `system.research.yml`
2. Run paper trading with `system.live.yml`
3. Gradually enable live trading (change `mode: "live"` in `system.live.yml`)
4. Monitor with kill switches and notifications

---

**Last Updated:** 2025-11-30
**Version:** 1.0
