from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List, Literal
import os
import yaml

def resolve_env_placeholders(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        actual = os.getenv(env_var)
        if not actual:
            raise RuntimeError(f"Env var {env_var} not set for sensitive config.")
        return actual
    return value

@dataclass
class EventBarConfig:
    """Configuration for building event-based bars (e.g., volume, dollar)."""
    mode: Literal["time", "volume", "dollar", "tick"] = "time"
    target_volume: Optional[float] = None
    target_notional: Optional[float] = None
    target_ticks: Optional[int] = None
    source_timeframe: Optional[str] = None  # Expected source data timeframe (e.g., "1m")
    keep_partial_last_bar: bool = False  # Whether to keep incomplete final bar

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EventBarConfig":
        data = data or {}
        return cls(
            mode=data.get("mode", cls.mode),
            target_volume=data.get("target_volume"),
            target_notional=data.get("target_notional"),
            target_ticks=data.get("target_ticks"),
            source_timeframe=data.get("source_timeframe"),
            keep_partial_last_bar=bool(data.get("keep_partial_last_bar", cls.keep_partial_last_bar)),
        )

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
class DataConfig:
    """Configuration for data sources and processing."""
    ohlcv_dir: str = "data/ohlcv"
    external_dir: str = "data/external"
    features_dir: str = "data/features"
    flow_dir: str = "data/flow"
    sentiment_dir: str = "data/sentiment"
    base_dir: str = "data"
    ohlcv_path_template: str = "data/ohlcv/{symbol}_{timeframe}.csv"
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    lookback_days: Dict[str, int] = field(default_factory=dict)
    default_lookback_days: int = 365
    bars: EventBarConfig = field(default_factory=EventBarConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DataConfig":
        data = data or {}
        bars_cfg = EventBarConfig.from_dict(data.get("bars"))
        return cls(
            ohlcv_dir=data.get("ohlcv_dir", cls.ohlcv_dir),
            external_dir=data.get("external_dir", cls.external_dir),
            features_dir=data.get("features_dir", cls.features_dir),
            flow_dir=data.get("flow_dir", cls.flow_dir),
            sentiment_dir=data.get("sentiment_dir", cls.sentiment_dir),
            base_dir=data.get("base_dir", cls.base_dir),
            ohlcv_path_template=data.get("ohlcv_path_template", cls.ohlcv_path_template),
            symbols=data.get("symbols", []),
            timeframes=data.get("timeframes", []),
            lookback_days=data.get("lookback_days", {}),
            default_lookback_days=data.get("default_lookback_days", cls.default_lookback_days),
            bars=bars_cfg,
        )


@dataclass
class LiveConfig:
    mode: str = "replay"
    data_source: str = "replay"
    exchange: str = "binance_futures"
    symbol: str = "BTCUSDT"
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "15m"
    ws_enabled: bool = False
    rest_poll_sec: int = 10
    max_concurrent_positions: int = 1
    max_position_notional: float = 100.0
    max_daily_loss: float = 20.0
    max_open_trades: int = 1
    order_retry_limit: int = 3
    order_timeout_seconds: int = 5
    log_dir: str = "outputs/live_logs"
    log_level: str = "INFO"
    state_dir: str = "outputs/live"
    state_path: Optional[str] = None
    latest_state_path: Optional[str] = "outputs/live/live_state.json"
    heartbeat_path: Optional[str] = None
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    ws_use_1m_stream: bool = True
    ws_aggregate_to_tf: str = "15m"
    ws_resync_lookback_bars: int = 200
    ws_max_stale_seconds: int = 30
    ws_max_ws_reconnects: int = 10
    execution_allow_market_orders: bool = True
    execution_allow_limit_orders: bool = True

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
        raw_symbols = data.get("symbols")
        symbols = [str(s) for s in raw_symbols] if raw_symbols else []
        if not symbols:
            symbols = [symbol]
        if symbol not in symbols:
            symbol = symbols[0]
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

        ws_section = data.get("ws", {}) or {}
        exec_section = data.get("execution", {}) or {}

        return cls(
            mode=data.get("mode", cls.mode),
            data_source=data.get("data_source", cls.data_source),
            exchange=data.get("exchange", cls.exchange),
            symbol=symbol,
            symbols=symbols,
            timeframe=timeframe,
            ws_enabled=bool(data.get("ws_enabled", cls.ws_enabled)),
            rest_poll_sec=int(data.get("rest_poll_sec", cls.rest_poll_sec)),
            max_concurrent_positions=int(
                data.get("max_concurrent_positions", cls.max_concurrent_positions)
            ),
            max_position_notional=float(
                data.get("max_position_notional", cls.max_position_notional)
            ),
            max_daily_loss=float(data.get("max_daily_loss", cls.max_daily_loss)),
            max_open_trades=int(data.get("max_open_trades", cls.max_open_trades)),
            order_retry_limit=int(
                data.get("order_retry_limit", cls.order_retry_limit)
            ),
            order_timeout_seconds=int(
                data.get("order_timeout_seconds", cls.order_timeout_seconds)
            ),
            log_dir=data.get("log_dir", cls.log_dir),
            log_level=str(data.get("log_level", cls.log_level)).upper(),
            state_dir=data.get("state_dir", cls.state_dir),
            state_path=data.get("state_path", cls.state_path),
            latest_state_path=data.get("latest_state_path", cls.latest_state_path),
            heartbeat_path=data.get("heartbeat_path", cls.heartbeat_path),
            replay=replay_cfg,
            paper=paper_cfg,
            ws_use_1m_stream=bool(ws_section.get("use_1m_stream", cls.ws_use_1m_stream)),
            ws_aggregate_to_tf=ws_section.get("aggregate_to_tf", cls.ws_aggregate_to_tf),
            ws_resync_lookback_bars=int(
                ws_section.get("resync_lookback_bars", cls.ws_resync_lookback_bars)
            ),
            ws_max_stale_seconds=int(
                ws_section.get("max_stale_seconds", cls.ws_max_stale_seconds)
            ),
            ws_max_ws_reconnects=int(
                ws_section.get("max_ws_reconnects", cls.ws_max_ws_reconnects)
            ),
            execution_allow_market_orders=bool(
                exec_section.get(
                    "allow_market_orders", cls.execution_allow_market_orders
                )
            ),
            execution_allow_limit_orders=bool(
                exec_section.get("allow_limit_orders", cls.execution_allow_limit_orders)
            ),
        )


@dataclass
class KillSwitchConfig:
    enabled: bool = True
    daily_realized_pnl_limit: float = -20.0
    max_equity_drawdown_pct: float = 30.0
    max_exceptions_per_hour: int = 5
    min_equity: float = 0.0
    evaluation_interval_bars: int = 1

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KillSwitchConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            daily_realized_pnl_limit=float(
                data.get("daily_realized_pnl_limit", cls.daily_realized_pnl_limit)
            ),
            max_equity_drawdown_pct=float(
                data.get("max_equity_drawdown_pct", cls.max_equity_drawdown_pct)
            ),
            max_exceptions_per_hour=int(
                data.get("max_exceptions_per_hour", cls.max_exceptions_per_hour)
            ),
            min_equity=float(data.get("min_equity", cls.min_equity)),
            evaluation_interval_bars=int(
                data.get("evaluation_interval_bars", cls.evaluation_interval_bars)
            ),
        )


@dataclass
class ExchangeRiskConfig:
    max_leverage: int = 3
    max_position_notional: float = 0.0
    max_position_contracts: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExchangeRiskConfig":
        data = data or {}
        return cls(
            max_leverage=int(data.get("max_leverage", cls.max_leverage)),
            max_position_notional=float(
                data.get("max_position_notional", cls.max_position_notional)
            ),
            max_position_contracts=float(
                data.get("max_position_contracts", cls.max_position_contracts)
            ),
        )


@dataclass
class FCMConfig:
    enabled: bool = False
    server_key: str = ""
    topic: str = "finantrade"
    min_level: str = "info"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FCMConfig":
        data = data or {}
        server_key_raw = data.get("server_key", "")
        server_key = ""
        if data.get("enabled"):
            if server_key_raw:
                if not str(server_key_raw).startswith("${"):
                    raise RuntimeError("FCM server_key must use ${ENV_VAR} placeholder.")
                server_key = resolve_env_placeholders(server_key_raw)
            else:
                raise RuntimeError("FCM server_key must be provided via env placeholder.")
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            server_key=server_key or "",
            topic=data.get("topic", cls.topic),
            min_level=str(data.get("min_level", cls.min_level)).lower(),
        )


@dataclass
class NotificationsConfig:
    enabled: bool = False
    fcm: FCMConfig = field(default_factory=FCMConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NotificationsConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            fcm=FCMConfig.from_dict(data.get("fcm")),
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
    "exchange": {
        "name": "binance_futures",
        "testnet": True,
        "dry_run": True,
        "base_url_rest": "https://fapi.binance.com",
        "base_url_rest_testnet": "https://testnet.binancefuture.com",
        "base_url_ws": "wss://fstream.binance.com",
        "base_url_ws_testnet": "wss://stream.binancefuture.com",
        "api_key_env": "BINANCE_FUTURES_API_KEY",
        "secret_key_env": "BINANCE_FUTURES_API_SECRET",
        "api_key": "${BINANCE_FUTURES_API_KEY}",
        "secret_key": "${BINANCE_FUTURES_API_SECRET}",
        "recv_window_ms": 5000,
        "time_sync": True,
        "max_time_skew_ms": 1000,
        "symbol_mapping": {},
        "default_leverage": 5,
        "position_mode": "one_way",
        "max_leverage": 3,
        "max_position_notional": 0.0,
        "max_position_contracts": 0.0,
    },
    "data": {
        "ohlcv_dir": "data/ohlcv",
        "external_dir": "data/external",
        "features_dir": "data/features",
        "symbols": ["BTCUSDT"],
        "timeframe": "15m",
        "ohlcv_path_template": "data/ohlcv/{symbol}_{timeframe}.csv",
        "flow_dir": "data/flow",
        "sentiment_dir": "data/sentiment",
        "base_dir": "data",
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
        "data_source": "replay",
        "exchange": "binance_futures",
        "symbol": "BTCUSDT",
        "symbols": ["BTCUSDT"],
        "timeframe": "15m",
        "ws_enabled": False,
        "rest_poll_sec": 10,
        "max_concurrent_positions": 1,
        "max_position_notional": 100.0,
        "max_daily_loss": 20.0,
        "max_open_trades": 1,
        "order_retry_limit": 3,
        "order_timeout_seconds": 5,
        "log_dir": "outputs/live_logs",
        "log_level": "INFO",
        "state_dir": "outputs/live",
        "state_path": None,
        "latest_state_path": "outputs/live/live_state.json",
        "heartbeat_path": None,
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
        "execution": {
            "allow_market_orders": True,
            "allow_limit_orders": True,
        },
        "ws": {
            "use_1m_stream": True,
            "aggregate_to_tf": "15m",
            "resync_lookback_bars": 200,
            "max_stale_seconds": 30,
            "max_ws_reconnects": 10,
        },
    },
    "kill_switch": {
        "enabled": True,
        "daily_realized_pnl_limit": -20.0,
        "max_equity_drawdown_pct": 30.0,
        "max_exceptions_per_hour": 5,
        "min_equity": 0.0,
        "evaluation_interval_bars": 1,
    },
    "notifications": {
        "enabled": False,
        "fcm": {
            "enabled": False,
            "server_key": "${FCM_SERVER_KEY}",
            "topic": "finantrade",
            "min_level": "info",
        },
    },
}


@dataclass
class ExchangeConfig:
    name: str = "binance_futures"
    testnet: bool = True
    base_url_rest: str = "https://fapi.binance.com"
    base_url_rest_testnet: str = "https://testnet.binancefuture.com"
    base_url_ws: str = "wss://fstream.binance.com"
    base_url_ws_testnet: str = "wss://stream.binancefuture.com"
    api_key_env: str = "BINANCE_FUTURES_API_KEY"
    secret_key_env: str = "BINANCE_FUTURES_API_SECRET"
    recv_window_ms: int = 5000
    time_sync: bool = True
    max_time_skew_ms: int = 1000
    symbol_mapping: Dict[str, str] = field(default_factory=dict)
    default_leverage: int = 5
    position_mode: str = "one_way"
    dry_run: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExchangeConfig":
        data = data or {}
        mapping = data.get("symbol_mapping", {}) or {}
        raw_api_key = data.get("api_key", "${BINANCE_FUTURES_API_KEY}")
        raw_secret_key = data.get("secret_key", "${BINANCE_FUTURES_API_SECRET}")
        if raw_api_key and not str(raw_api_key).startswith("${"):
            raise RuntimeError(
                "API key must not be stored directly in system.yml. Use ${ENV_VAR} placeholder."
            )
        if raw_secret_key and not str(raw_secret_key).startswith("${"):
            raise RuntimeError(
                "API secret must not be stored directly in system.yml. Use ${ENV_VAR} placeholder."
            )
        return cls(
            name=data.get("name", cls.name),
            testnet=bool(data.get("testnet", cls.testnet)),
            base_url_rest=data.get("base_url_rest", cls.base_url_rest),
            base_url_rest_testnet=data.get(
                "base_url_rest_testnet", cls.base_url_rest_testnet
            ),
            base_url_ws=data.get("base_url_ws", cls.base_url_ws),
            base_url_ws_testnet=data.get(
                "base_url_ws_testnet", cls.base_url_ws_testnet
            ),
            api_key_env=data.get("api_key_env", cls.api_key_env),
            secret_key_env=data.get("secret_key_env", cls.secret_key_env),
            recv_window_ms=int(data.get("recv_window_ms", cls.recv_window_ms)),
            time_sync=bool(data.get("time_sync", cls.time_sync)),
            max_time_skew_ms=int(
                data.get("max_time_skew_ms", cls.max_time_skew_ms)
            ),
            symbol_mapping={str(k): str(v) for k, v in mapping.items()},
            default_leverage=int(
                data.get("default_leverage", cls.default_leverage)
            ),
            position_mode=data.get("position_mode", cls.position_mode),
            dry_run=bool(data.get("dry_run", cls.dry_run)),
        )


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


def _propagate_source_timeframe(data_cfg: DataConfig, global_timeframe: str) -> DataConfig:
    """
    Propagate global timeframe to EventBarConfig.source_timeframe if not explicitly set.

    For event-based bars (volume, dollar, tick), the source_timeframe should match
    the global timeframe unless explicitly overridden by the user.

    Raises:
        ValueError: If event bars are configured with a non-1m timeframe
    """
    bars_cfg = data_cfg.bars
    if bars_cfg and bars_cfg.mode in ("volume", "dollar", "tick"):
        # If user hasn't specified source_timeframe, use global timeframe
        if bars_cfg.source_timeframe is None:
            bars_cfg.source_timeframe = global_timeframe

        # Validate that event bars only use 1m data
        if bars_cfg.source_timeframe != "1m":
            raise ValueError(
                f"Event bars (mode={bars_cfg.mode!r}) currently only supported from 1m data. "
                f"Got source_timeframe={bars_cfg.source_timeframe!r}. "
                f"Please set timeframe='1m' in your system config when using event bars, "
                f"or set data.bars.mode='time' for higher timeframes."
            )
    return data_cfg


def load_system_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the project-level system configuration with profile support.

    Profiles (system.research.yml / system.live.yml) automatically inherit from
    system.base.yml if it exists.

    Args:
        path: Config file path. If None, reads from FT_CONFIG_PATH env var,
              defaults to "config/system.yml"

    Returns:
        Merged config dictionary with all required fields

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    # Determine config path from parameter or environment
    if path is None:
        path = os.getenv("FT_CONFIG_PATH", "config/system.yml")

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"System config not found at {cfg_path}")

    # Check if this is a profile config (research/live)
    is_profile = cfg_path.stem in ("system.research", "system.live")

    # Load base config if this is a profile
    base_cfg = {}
    if is_profile:
        base_path = cfg_path.parent / "system.base.yml"
        if base_path.exists():
            with base_path.open("r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}
        else:
            # Profile without base is allowed but warn
            import warnings
            warnings.warn(
                f"Profile config {cfg_path.name} found but system.base.yml missing. "
                f"Consider creating system.base.yml for shared settings."
            )

    # Load profile/main config
    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    # Merge: DEFAULT -> base (if profile) -> user
    if is_profile and base_cfg:
        temp = _deep_merge(DEFAULT_SYSTEM_CONFIG, base_cfg)
        merged = _deep_merge(temp, user_cfg)
    else:
        merged = _deep_merge(DEFAULT_SYSTEM_CONFIG, user_cfg)

    # Ensure 'mode' field exists (critical for safety checks)
    if "mode" not in merged:
        merged["mode"] = "unknown"

    # Add config metadata for debugging and validation
    merged["_config_meta"] = {
        "config_path": str(cfg_path),
        "is_profile": is_profile,
        "has_base": bool(base_cfg),
        "mode": merged.get("mode", "unknown"),
    }

    merged["portfolio_cfg"] = PortfolioConfig.from_dict(merged.get("portfolio", {}))
    merged["exchange_cfg"] = ExchangeConfig.from_dict(merged.get("exchange", {}))
    merged["exchange_risk_cfg"] = ExchangeRiskConfig.from_dict(merged.get("exchange", {}))
    merged["kill_switch_cfg"] = KillSwitchConfig.from_dict(merged.get("kill_switch", {}))
    merged["live_cfg"] = LiveConfig.from_dict(
        merged.get("live"),
        default_symbol=merged.get("symbol"),
        default_timeframe=merged.get("timeframe"),
    )
    merged["notifications_cfg"] = NotificationsConfig.from_dict(merged.get("notifications", {}))

    # Create DataConfig and propagate source_timeframe for event bars
    merged["data_cfg"] = DataConfig.from_dict(merged.get("data", {}))
    merged["data_cfg"] = _propagate_source_timeframe(
        merged["data_cfg"],
        merged.get("timeframe", "15m")
    )

    return merged


def load_exchange_credentials(cfg: ExchangeConfig) -> tuple[str, str]:
    raw_api = getattr(cfg, "api_key", None)
    raw_secret = getattr(cfg, "secret_key", None)
    if raw_api:
        api_key = resolve_env_placeholders(raw_api).strip()
    else:
        api_key = os.getenv(cfg.api_key_env, "").strip()
    if raw_secret:
        secret_key = resolve_env_placeholders(raw_secret).strip()
    else:
        secret_key = os.getenv(cfg.secret_key_env, "").strip()
    if not api_key or not secret_key:
        raise RuntimeError(
            f"Exchange API keys not found in environment variables "
            f"{cfg.api_key_env}/{cfg.secret_key_env}"
        )
    return api_key, secret_key


__all__ = [
    "load_system_config",
    "DataConfig",
    "EventBarConfig",
    "LiveConfig",
    "ReplayConfig",
    "PaperConfig",
    "PortfolioConfig",
    "ExchangeConfig",
    "ExchangeRiskConfig",
    "KillSwitchConfig",
    "NotificationsConfig",
    "FCMConfig",
    "load_exchange_credentials",
]
