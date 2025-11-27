from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


@dataclass
class ReplayConfig:
    bars_limit: Optional[int] = 500
    start_index: int = 0
    start_timestamp: Optional[str] = None


@dataclass
class PaperConfig:
    initial_cash: float = 10_000.0
    save_state_every_n_bars: int = 25
    state_path: str = "outputs/live_state/paper_state.json"
    output_dir: str = "outputs/live_paper"


@dataclass
class LiveConfig:
    mode: str = "replay"
    exchange: str = "binance_futures"
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    ws_enabled: bool = False
    rest_poll_sec: int = 10
    max_concurrent_positions: int = 1
    log_dir: str = "outputs/live_logs"
    log_level: str = "INFO"
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]] = None,
        *,
        default_symbol: Optional[str] = None,
        default_timeframe: Optional[str] = None,
    ) -> "LiveConfig":
        data = data or {}
        symbol = data.get("symbol", default_symbol or cls.symbol)
        timeframe = data.get("timeframe", default_timeframe or cls.timeframe)

        replay_section = data.get("replay", {}) or {}
        replay_cfg = ReplayConfig(
            bars_limit=replay_section.get(
                "bars_limit",
                ReplayConfig.bars_limit,
            ),
            start_index=int(replay_section.get("start_index", ReplayConfig.start_index)),
            start_timestamp=replay_section.get("start_timestamp"),
        )

        paper_section = data.get("paper", {}) or {}
        paper_cfg = PaperConfig(
            initial_cash=float(paper_section.get("initial_cash", PaperConfig.initial_cash)),
            save_state_every_n_bars=int(
                paper_section.get(
                    "save_state_every_n_bars",
                    PaperConfig.save_state_every_n_bars,
                )
            ),
            state_path=paper_section.get("state_path", PaperConfig.state_path),
            output_dir=paper_section.get("output_dir", PaperConfig.output_dir),
        )

        return cls(
            mode=data.get("mode", cls.mode),
            exchange=data.get("exchange", cls.exchange),
            symbol=symbol,
            timeframe=timeframe,
            ws_enabled=bool(data.get("ws_enabled", cls.ws_enabled)),
            rest_poll_sec=int(data.get("rest_poll_sec", cls.rest_poll_sec)),
            max_concurrent_positions=int(
                data.get("max_concurrent_positions", cls.max_concurrent_positions)
            ),
            log_dir=data.get("log_dir", cls.log_dir),
            log_level=str(data.get("log_level", cls.log_level)).upper(),
            replay=replay_cfg,
            paper=paper_cfg,
        )


@dataclass
class PortfolioConfig:
    symbols: List[str]
    timeframe: str
    strategy: str
    initial_capital: float
    allocation_type: str
    weights: Dict[str, float]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioConfig":
        symbols = d.get("symbols", [])
        timeframe = d.get("timeframe", "15m")
        strategy = d.get("strategy", "rule")
        initial_capital = float(d.get("initial_capital", 1000.0))

        alloc = d.get("capital_allocation", {}) or {}
        alloc_type = alloc.get("type", "equal_weight")
        weights = alloc.get("weights", {}) or {}

        if alloc_type == "equal_weight" and symbols:
            w = 1.0 / len(symbols)
            weights = {s: w for s in symbols}

        return cls(
            symbols=symbols,
            timeframe=timeframe,
            strategy=strategy,
            initial_capital=initial_capital,
            allocation_type=alloc_type,
            weights=weights,
        )


DEFAULT_SYSTEM_CONFIG: Dict[str, Any] = {
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "data": {
        "ohlcv_dir": "data/ohlcv",
        "external_dir": "data/external",
        "features_dir": "data/features",
    },
    "features": {
        "use_base": True,
        "use_ta": True,
        "use_candles": True,
        "use_osc": True,
        "use_htf": True,
        "use_external": True,
        "use_microstructure": False,
        "use_market_structure": False,
        "drop_na": True,
        "feature_preset": "core",
    },
    "rule": {
        "allowed_hours": list(range(8, 18)),
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "htf_trend_min": -0.05,
        "htf_rsi_min": 40.0,
        "htf_rsi_max": 75.0,
        "atr_pct_min": 0.002,
        "atr_pct_max": 0.08,
        "rsi_entry_min": 45.0,
        "rsi_entry_max": 75.0,
        "macd_entry_min": -0.001,
        "stoch_k_entry_min": 20.0,
        "stoch_k_entry_max": 85.0,
        "min_body_pct": 0.08,
        "trend_exit_max": -0.10,
        "rsi_exit_max": 45.0,
        "use_ms_trend_filter": False,
        "ms_trend_min": -0.5,
        "ms_trend_max": 1.5,
        "use_ms_chop_filter": False,
        "allow_chop": True,
        "use_fvg_filter": False,
    },
    "risk": {
        "capital_risk_pct_per_trade": 0.01,
        "max_leverage": 5.0,
        "max_notional_per_symbol": 2000.0,
        "max_daily_loss_pct": 0.03,
        "daily_loss_lookback_days": 1,
        "min_hold_bars": 2,
        "use_tail_risk_guard": True,
        "tail_risk_hv_threshold": 0.25,
        "tail_risk_max_leverage_in_crash": 1.0,
        "atr_period": 14,
        "atr_mult_tp": 1.5,
        "atr_mult_sl": 1.0,
        "warmup_bars": 50,
    },
    "ml": {
        "label": {
            "method": "fixed_horizon",
            "horizon_bars": 8,
            "up_threshold": 0.004,
            "down_threshold": -0.004,
        },
        "model": {
            "type": "RandomForest",
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 50,
            "random_state": 42,
        },
        "backtest": {
            "proba_column": "ml_proba_long",
            "proba_threshold": 0.55,
            "proba_exit_threshold": 0.5,
            "warmup_bars": 200,
            "side": "long_only",
            "allow_pipeline_mismatch": False,
            "use_saved_model": False,
            "model_id": None,
        },
        "persistence": {
            "save_model": False,
            "model_dir": "outputs/ml_models",
            "use_registry": True,
            "max_models_per_symbol_tf": 10,
        },
        "registry": {
            "use_latest": True,
            "selected_id": None,
        },
    },
    "strategy": {
        "default": "rule",
    },
    "portfolio": {
        "symbols": [],
        "timeframe": "15m",
        "strategy": "rule",
        "initial_capital": 1000.0,
        "capital_allocation": {
            "type": "equal_weight",
            "weights": {},
        },
    },
    "live": {
        "mode": "replay",
        "exchange": "binance_futures",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "ws_enabled": False,
        "rest_poll_sec": 10,
        "max_concurrent_positions": 1,
        "log_dir": "outputs/live_logs",
        "log_level": "INFO",
        "replay": {
            "bars_limit": 500,
            "start_index": 0,
            "start_timestamp": None,
        },
        "paper": {
            "initial_cash": 10_000.0,
            "save_state_every_n_bars": 25,
            "state_path": "outputs/live_state/paper_state.json",
            "output_dir": "outputs/live_paper",
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = deepcopy(val)
    return result


def load_system_config(path: str = "config/system.yml") -> Dict[str, Any]:
    """
    Load the project-level system configuration and fill any missing fields
    with sensible defaults so downstream modules can depend on them.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"System config not found at {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULT_SYSTEM_CONFIG, user_cfg)
    merged["portfolio_cfg"] = PortfolioConfig.from_dict(merged.get("portfolio", {}))
    return merged


__all__ = ["load_system_config", "LiveConfig", "ReplayConfig", "PaperConfig", "PortfolioConfig"]
