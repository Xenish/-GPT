from __future__ import annotations

import json
from pathlib import Path
import glob
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from copy import deepcopy

from finantradealgo.features.feature_pipeline import PIPELINE_VERSION
from finantradealgo.system.config_loader import load_system_config, PortfolioConfig, DataConfig
from finantradealgo.backtester.scenario_engine import run_scenario_preset
from finantradealgo.ml.model_registry import load_registry, validate_registry_entry
from finantradealgo.ml.ml_utils import get_ml_targets
from finantradealgo.backtester.portfolio_engine import PortfolioBacktestEngine
from finantradealgo.backtester.runners import run_backtest_once

SCENARIO_GRID_DIR = Path("outputs") / "backtests"

class BarPoint(BaseModel):
    time: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    rule_entry: int | None = None
    rule_exit: int | None = None
    ml_long_proba: float | None = None
    ms_trend: float | None = None
    ms_chop: float | None = None
    ms_hh_ll_trend: float | None = None
    fvg_up: int | None = None
    fvg_down: int | None = None
    trade_entries: Optional[List[str]] = None
    trade_exits: Optional[List[str]] = None


class ChartResponse(BaseModel):
    symbol: str
    timeframe: str
    bars: List[BarPoint]
    meta: Dict[str, Optional[str]]


class BacktestRunInfo(BaseModel):
    run_id: str
    strategy: str
    symbol: str
    timeframe: str
    start: Optional[str] = None
    end: Optional[str] = None
    metrics: Dict[str, Optional[float]]


class TradeRow(BaseModel):
    trade_id: str
    entry_time: str
    exit_time: Optional[str] = None
    side: str
    qty: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None


class SummaryResponse(BaseModel):
    symbol: str
    timeframe: str
    final_equity: float | None = None
    cum_return: float | None = None
    trade_count: int = 0
    win_rate: float | None = None


class LivePosition(BaseModel):
    id: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    current_price: float | None = None
    pnl: float | None = None
    entry_time: Optional[str] = None


class LiveStatus(BaseModel):
    run_id: str
    symbol: str
    timeframe: str
    strategy: str
    start_time: Optional[str] = None
    last_bar_time: Optional[str] = None
    last_bar_time_ts: Optional[float] = None
    mode: Optional[str] = None
    equity: float
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None
    daily_realized_pnl: float | None = None
    daily_unrealized_pnl: float | None = None
    open_positions: List[LivePosition] = []
    risk_stats: Dict[str, Any] = {}
    data_source: Optional[str] = None
    stale_data_seconds: Optional[float] = None
    ws_reconnect_count: Optional[int] = None
    last_orders: List[Dict[str, Any]] = []
    timestamp: Optional[float] = None

class RunBacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    strategy: str  # "rule", "ml", "trend_continuation" vs.
    strategy_params: Optional[Dict[str, Any]] = None


class RunBacktestResponse(BaseModel):
    run_id: str
    symbol: str
    timeframe: str
    strategy: str
    metrics: Dict[str, float]
    trade_count: int


class MLTarget(BaseModel):
    symbol: str
    timeframe: str


class MetaResponse(BaseModel):
    symbols: List[str]
    timeframes: List[str]
    strategies: List[str]
    scenario_presets: List[str] = []
    lookback_days: Optional[Dict[str, int]] = None
    default_lookback_days: Optional[int] = None
    ml_targets: Optional[List[MLTarget]] = None


class ScenarioRunRequest(BaseModel):
    symbol: str
    timeframe: str
    preset_name: str


class ScenarioResultRow(BaseModel):
    label: str
    strategy: str
    cum_return: float | None = None
    sharpe: float | None = None
    trade_count: int | None = None


class ScenarioRunResponse(BaseModel):
    preset_name: str
    rows: List[ScenarioResultRow]


class ScenarioResult(BaseModel):
    scenario_id: str
    symbol: str
    timeframe: str
    strategy: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    label: Optional[str] = None


class ModelInfo(BaseModel):
    model_id: str
    symbol: str
    timeframe: str
    model_type: str
    created_at: str
    metrics: Dict[str, float] = {}


class PortfolioBacktestInfo(BaseModel):
    run_id: str
    symbols: List[str]
    timeframe: str
    start: Optional[str]
    end: Optional[str]
    metrics: Dict[str, Optional[float]]


class PortfolioEquityPoint(BaseModel):
    time: float
    portfolio_equity: float


class LiveControlRequest(BaseModel):
    command: str  # "pause", "resume", "stop", "flatten"
    run_id: Optional[str] = None

def _find_feature_file(symbol: str, timeframe: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "data" / "features" / f"{symbol}_{timeframe}_features.csv",
        root / "data" / "features" / f"{symbol}_features_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Feature CSV not found for {symbol} {timeframe}. Tried: {candidates}"
    )


def _read_feature_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    fea_path = _find_feature_file(symbol, timeframe)
    df = pd.read_csv(fea_path)
    if "timestamp" not in df.columns:
        raise ValueError("Feature CSV must include 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinanTrade API",
        version="0.1.0",
    )
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    cfg = load_system_config()
    ml_proba_col = (
        cfg.get("ml", {}).get("backtest", {}).get("proba_column", "ml_proba_long")
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def _derive_run_id(stem: str) -> str:
        return stem

    def _compute_equity_metrics(eq_series: pd.Series) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = {}
        if eq_series.empty:
            return metrics
        final = float(eq_series.iloc[-1])
        start = float(eq_series.iloc[0])
        metrics["final_equity"] = final
        metrics["cum_return"] = final / start - 1 if start else None
        drawdown = (eq_series / eq_series.cummax()) - 1
        metrics["max_drawdown"] = float(drawdown.min())
        returns = eq_series.pct_change().dropna()
        metrics["sharpe"] = (
            float(returns.mean() / returns.std() * (252**0.5))
            if not returns.empty and returns.std() != 0
            else 0.0
        )
        return metrics

    def _list_backtest_runs(
        symbol: str,
        timeframe: str,
        *,
        strategy_filter: Optional[str] = None,
        limit: int = 20,
    ) -> List[BacktestRunInfo]:
        eq_dir = Path("outputs") / "backtests"
        if not eq_dir.exists():
            return []
        runs: List[BacktestRunInfo] = []
        files = sorted(eq_dir.glob(f"*equity*{timeframe}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for eq_path in files:
            stem = eq_path.stem
            run_id = _derive_run_id(stem)
            strategy = run_id.split("_")[0]
            if strategy_filter and strategy != strategy_filter:
                continue
            try:
                df_eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
            except Exception:
                continue
            if df_eq.empty:
                continue
            eq_series = df_eq.iloc[:, 0]
            start = eq_series.index.min()
            end = eq_series.index.max()
            metrics = _compute_equity_metrics(eq_series)
            runs.append(
                BacktestRunInfo(
                    run_id=run_id,
                    strategy=strategy,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start.isoformat() if isinstance(start, pd.Timestamp) else None,
                    end=end.isoformat() if isinstance(end, pd.Timestamp) else None,
                    metrics=metrics,
                )
            )
            if len(runs) >= limit:
                break
        return runs

    def _resolve_trade_path(run_id: str) -> Optional[Path]:
        trades_dir = Path("outputs") / "trades"
        if not trades_dir.exists():
            return None
        candidates = []
        if "_equity" in run_id:
            candidates.append(trades_dir / f"{run_id.replace('_equity', '_trades')}.csv")
        candidates.append(trades_dir / f"{run_id}_trades.csv")
        candidates.extend(trades_dir.glob(f"{run_id}*trades*.csv"))
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_trades(run_id: str) -> Optional[pd.DataFrame]:
        trade_path = _resolve_trade_path(run_id)
        if trade_path is None:
            return None
        df = pd.read_csv(trade_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "timestamp_exit" in df.columns:
            df["timestamp_exit"] = pd.to_datetime(df["timestamp_exit"])
        return df

    @app.get("/api/chart/{symbol}/{timeframe}", response_model=ChartResponse)
    def get_chart(
        symbol: str,
        timeframe: str = "15m",
        run_id: Optional[str] = Query(default=None),
    ) -> ChartResponse:
        try:
            df = _read_feature_csv(symbol, timeframe)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error loading data: {exc}")

        df = df.sort_values("timestamp").tail(500)

        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing column '{col}' in feature CSV",
                )

        if "volume" not in df.columns:
            df["volume"] = 0.0

        available_cols = set(df.columns)

        trades_df: Optional[pd.DataFrame] = None
        trade_entries: Dict[pd.Timestamp, List[str]] = {}
        trade_exits: Dict[pd.Timestamp, List[str]] = {}
        if run_id:
            trades_df = _load_trades(run_id)
            if trades_df is None:
                raise HTTPException(status_code=404, detail="Trades for run not found.")
            for idx, trade in trades_df.iterrows():
                trade_key = f"{run_id}_{idx}"
                entry_ts = pd.to_datetime(trade.get("timestamp"))
                exit_ts = (
                    pd.to_datetime(trade.get("timestamp_exit"))
                    if "timestamp_exit" in trade
                    else None
                )
                if pd.notna(entry_ts):
                    trade_entries.setdefault(entry_ts, []).append(trade_key)
                if exit_ts is not None and pd.notna(exit_ts):
                    trade_exits.setdefault(exit_ts, []).append(trade_key)

        bars: List[BarPoint] = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            epoch = float(ts.timestamp())
            bar_kwargs = {
                "time": epoch,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }

            if "rule_long_entry" in available_cols:
                bar_kwargs["rule_entry"] = int(row.get("rule_long_entry", 0))
            if "rule_long_exit" in available_cols:
                bar_kwargs["rule_exit"] = int(row.get("rule_long_exit", 0))
            if ml_proba_col in available_cols:
                ml_val = row.get(ml_proba_col)
                bar_kwargs["ml_long_proba"] = (
                    float(ml_val) if pd.notna(ml_val) else None
                )
            if "ms_trend_label" in available_cols:
                bar_kwargs["ms_trend"] = row.get("ms_trend_label")
            if "ms_chop_label" in available_cols:
                bar_kwargs["ms_chop"] = row.get("ms_chop_label")
            if "ms_hh_ll_trend" in available_cols:
                bar_kwargs["ms_hh_ll_trend"] = row.get("ms_hh_ll_trend")
            if "ms_fvg_up" in available_cols:
                bar_kwargs["fvg_up"] = row.get("ms_fvg_up")
            if "ms_fvg_down" in available_cols:
                bar_kwargs["fvg_down"] = row.get("ms_fvg_down")

            entry_marks = trade_entries.get(ts)
            exit_marks = trade_exits.get(ts)
            if entry_marks:
                bar_kwargs["trade_entries"] = entry_marks
            if exit_marks:
                bar_kwargs["trade_exits"] = exit_marks

            bars.append(BarPoint(**bar_kwargs))

        meta = {
            "pipeline_version": PIPELINE_VERSION,
            "features_preset": cfg.get("features", {}).get("feature_preset"),
            "last_updated": pd.Timestamp.utcnow().isoformat(),
            "run_id": run_id,
            "strategy": run_id.split("_")[0] if run_id else None,
        }

        return ChartResponse(symbol=symbol, timeframe=timeframe, bars=bars, meta=meta)

    def _load_equity_stats(
        symbol: str, timeframe: str
    ) -> tuple[Optional[float], Optional[float]]:
        eq_path = Path("outputs") / "backtests" / f"rule_equity_{timeframe}.csv"
        if not eq_path.exists():
            return None, None
        df_eq = pd.read_csv(eq_path)
        if df_eq.empty:
            return None, None
        val_col = df_eq.columns[-1]
        equity = df_eq[val_col].astype(float)
        final_equity = float(equity.iloc[-1])
        cum_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] else None
        return final_equity, cum_return

    def _load_trade_stats(timeframe: str) -> tuple[int, Optional[float]]:
        trades_path = Path("outputs") / "trades" / f"rule_trades_{timeframe}.csv"
        if not trades_path.exists():
            return 0, None
        df_trades = pd.read_csv(trades_path)
        if df_trades.empty or "pnl" not in df_trades.columns:
            return len(df_trades), None
        pnl = df_trades["pnl"].dropna()
        win_rate = float((pnl > 0).mean()) if not pnl.empty else None
        return len(df_trades), win_rate

    @app.get("/api/summary/{symbol}/{timeframe}", response_model=SummaryResponse)
    def get_summary(symbol: str, timeframe: str = "15m") -> SummaryResponse:
        final_equity, cum_return = _load_equity_stats(symbol, timeframe)
        trade_count, win_rate = _load_trade_stats(timeframe)
        if final_equity is None and trade_count == 0:
            raise HTTPException(status_code=404, detail="Summary data not available.")
        return SummaryResponse(
            symbol=symbol,
            timeframe=timeframe,
            final_equity=final_equity,
            cum_return=cum_return,
            trade_count=trade_count,
            win_rate=win_rate,
        )

    @app.get(
        "/api/backtests/{symbol}/{timeframe}",
        response_model=List[BacktestRunInfo],
    )
    def list_backtests(
        symbol: str,
        timeframe: str = "15m",
        strategy: Optional[str] = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> List[BacktestRunInfo]:
        return _list_backtest_runs(
            symbol,
            timeframe,
            strategy_filter=strategy,
            limit=limit,
        )

    @app.get("/api/trades/{run_id}", response_model=List[TradeRow])
    def list_trades(run_id: str) -> List[TradeRow]:
        trades_df = _load_trades(run_id)
        if trades_df is None:
            raise HTTPException(status_code=404, detail="Trades not found for run.")
        trades: List[TradeRow] = []
        for idx, row in trades_df.iterrows():
            trade_id = f"{run_id}_{idx}"
            trades.append(
                TradeRow(
                    trade_id=trade_id,
                    entry_time=str(row.get("timestamp")),
                    exit_time=str(row.get("timestamp_exit"))
                    if "timestamp_exit" in row and pd.notna(row["timestamp_exit"])
                    else None,
                    side=str(row.get("side", "")),
                    qty=float(row.get("qty", 0.0)),
                    entry_price=float(row.get("entry_price", 0.0)),
                    exit_price=float(row.get("exit_price"))
                    if pd.notna(row.get("exit_price"))
                    else None,
                    pnl=float(row.get("pnl")) if pd.notna(row.get("pnl")) else None,
                    reason=row.get("reason"),
                )
            )
        return trades

    live_cfg = cfg.get("live", {}) or {}
    live_dir = Path(live_cfg.get("state_dir", "outputs/live"))
    default_latest = live_cfg.get("latest_state_path") or live_dir / "live_state.json"
    default_state_path = live_cfg.get("state_path")

    def _load_live_snapshot(run_id: Optional[str]) -> Dict[str, Any]:
        if run_id:
            path = Path(default_state_path) if default_state_path else live_dir / f"live_state_{run_id}.json"
        else:
            path = Path(default_latest)
        if not Path(path).exists():
            raise FileNotFoundError("Live snapshot not found.")
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if run_id and data.get("run_id") != run_id:
            raise ValueError("Snapshot run_id mismatch.")
        return data

    @app.get("/api/live/status", response_model=LiveStatus)
    def live_status(run_id: Optional[str] = Query(default=None)) -> LiveStatus:
        try:
            payload = _load_live_snapshot(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No live snapshot found.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read snapshot: {exc}")
        return LiveStatus(**payload)

    @app.post("/api/backtests/run", response_model=RunBacktestResponse)
    async def api_run_backtest(req: RunBacktestRequest):
        """
        Frontend'teki 'Run backtest' butonu buraya POST atıyor.
        """
        cfg = load_system_config()
        try:
            result = run_backtest_once(
                symbol=req.symbol,
                timeframe=req.timeframe,
                strategy_name=req.strategy,
                cfg=cfg,
                strategy_params=req.strategy_params,
            )
        except ValueError as e:
            # Konfig / model / pipeline mismatch gibi beklenen hatalar
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Beklenmeyen her şey: logla ve 500 dön
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

        metrics = {k: float(v) for k, v in result.get("metrics", {}).items()}

        return RunBacktestResponse(
            run_id=result["run_id"],
            symbol=req.symbol,
            timeframe=req.timeframe,
            strategy=req.strategy,
            metrics=metrics,
            trade_count=int(result.get("trade_count", 0)),
        )

    @app.get("/api/meta", response_model=MetaResponse)
    def get_meta():
        """
        Get system metadata including symbols, timeframes, and lookback configuration.

        Returns multi-timeframe configuration with per-timeframe lookback days
        for data filtering and resource management.
        """
        cfg = load_system_config()

        # Get data configuration with multi-TF support
        data_cfg = cfg.get("data_cfg")
        if data_cfg:
            symbols = data_cfg.symbols
            timeframes = data_cfg.timeframes
            lookback_days = data_cfg.lookback_days if data_cfg.lookback_days else None
            default_lookback_days = data_cfg.default_lookback_days
        else:
            # Fallback to legacy config structure
            data_section = cfg.get("data", {}) or {}
            symbols = data_section.get("symbols") or [cfg.get("symbol", "AIAUSDT")]
            timeframes = data_section.get("timeframes") or [cfg.get("timeframe", "15m")]
            lookback_days = data_section.get("lookback_days")
            default_lookback_days = data_section.get("default_lookback_days")

        # Get available strategies
        strategies = cfg.get("strategy", {}).get("available") or ["rule", "ml"]

        # Get scenario presets
        scenario_presets = list((cfg.get("scenario", {}) or {}).get("presets", {}).keys())

        # Get ML targets (symbol/timeframe combinations for ML training)
        ml_targets_list = get_ml_targets(cfg)
        ml_targets = [MLTarget(symbol=sym, timeframe=tf) for sym, tf in ml_targets_list]

        return MetaResponse(
            symbols=list(symbols) if symbols else [],
            timeframes=list(timeframes) if timeframes else [],
            strategies=list(strategies),
            scenario_presets=scenario_presets,
            lookback_days=lookback_days,
            default_lookback_days=default_lookback_days,
            ml_targets=ml_targets if ml_targets else None,
        )

    @app.post("/api/scenarios/run", response_model=ScenarioRunResponse)
    def run_scenarios(req: ScenarioRunRequest):
        cfg = load_system_config()
        cfg_local = deepcopy(cfg)
        cfg_local["symbol"] = req.symbol
        cfg_local["timeframe"] = req.timeframe
        try:
            df_results = run_scenario_preset(cfg_local, req.preset_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Preset not found: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scenario run failed: {exc}")

        rows: List[ScenarioResultRow] = []
        for _, r in df_results.iterrows():
            rows.append(
                ScenarioResultRow(
                    label=str(r.get("label", r.get("scenario_name", ""))),
                    strategy=str(r.get("strategy", r.get("strategy_name", ""))),
                    cum_return=float(r.get("cum_return"))
                    if pd.notna(r.get("cum_return"))
                    else None,
                    sharpe=float(r.get("sharpe"))
                    if pd.notna(r.get("sharpe"))
                    else None,
                    trade_count=int(r.get("trade_count"))
                    if not pd.isna(r.get("trade_count"))
                    else None,
                )
            )

        return ScenarioRunResponse(preset_name=req.preset_name, rows=rows)

    @app.get("/api/scenarios/{symbol}/{timeframe}", response_model=List[ScenarioResult])
    def list_scenario_results(symbol: str, timeframe: str):
        pattern = SCENARIO_GRID_DIR / f"scenario_grid_{symbol}_{timeframe}*.csv"
        files = sorted(glob.glob(str(pattern)))
        if not files:
            raise HTTPException(status_code=404, detail="No scenario grids found for given symbol/timeframe")
        path = Path(files[-1])
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read scenario grid: {exc}")
        if df.empty:
            return []
        results: List[ScenarioResult] = []
        for _, row in df.iterrows():
            raw_params = row.get("params_json")
            if raw_params is None:
                raw_params = row.get("params", "{}")
            if isinstance(raw_params, str):
                try:
                    params = json.loads(raw_params)
                except json.JSONDecodeError:
                    params = {}
            elif isinstance(raw_params, dict):
                params = raw_params
            else:
                params = {}

            metrics: Dict[str, float] = {}
            for key in ("cum_return", "sharpe", "max_drawdown", "trade_count"):
                if key in df.columns:
                    val = row.get(key)
                    if pd.notna(val):
                        metrics[key] = float(val)

            results.append(
                ScenarioResult(
                    scenario_id=str(row.get("scenario_id", "")),
                    label=row.get("label"),
                    symbol=str(row.get("symbol", symbol)),
                    timeframe=str(row.get("timeframe", timeframe)),
                    strategy=str(row.get("strategy", "")),
                    params=params,
                    metrics=metrics,
                )
            )
        return results

    @app.get("/api/ml/models/{model_id}/importance")
    def get_feature_importance(model_id: str) -> Dict[str, float]:
        cfg_local = load_system_config()
        ml_cfg = cfg_local.get("ml", {}) or {}
        persistence_cfg = ml_cfg.get("persistence", {}) or {}
        base_dir = Path(persistence_cfg.get("model_dir", "outputs/ml_models"))
        model_path = base_dir / model_id
        meta_path = model_path / "meta.json"

        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Model metadata not found")

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:  # pragma: no cover - unlikely
            raise HTTPException(status_code=500, detail=f"Failed to read metadata: {exc}")

        feature_importances = meta.get("feature_importances")
        if isinstance(feature_importances, dict) and feature_importances:
            return {str(k): float(v) for k, v in feature_importances.items()}

        csv_path = model_path / "feature_importances.csv"
        if csv_path.exists():
            try:
                df_imp = pd.read_csv(csv_path)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to read feature importance CSV: {exc}")
            if {"feature", "importance"}.issubset(df_imp.columns):
                return {
                    str(row["feature"]): float(row["importance"])
                    for _, row in df_imp.iterrows()
                }

        raise HTTPException(status_code=404, detail="Feature importance not found for this model")

    @app.get("/api/ml/models/{symbol}/{timeframe}", response_model=List[ModelInfo])
    def list_ml_models(symbol: str, timeframe: str) -> List[ModelInfo]:
        cfg_local = load_system_config()
        ml_cfg = cfg_local.get("ml", {}) or {}
        persistence_cfg = ml_cfg.get("persistence", {}) or {}
        base_dir = persistence_cfg.get("model_dir", "outputs/ml_models")

        registry = load_registry(base_dir)
        rows: List[ModelInfo] = []
        for entry in registry.entries:
            if entry.symbol != symbol or entry.timeframe != timeframe:
                continue
            if entry.status != "success":
                continue
            if not validate_registry_entry(base_dir, entry):
                continue
            metrics_dict: Dict[str, float] = {}
            if entry.cum_return is not None:
                metrics_dict["cum_return"] = float(entry.cum_return)
            if entry.sharpe is not None:
                metrics_dict["sharpe"] = float(entry.sharpe)
            rows.append(
                ModelInfo(
                    model_id=entry.model_id,
                    symbol=entry.symbol,
                    timeframe=entry.timeframe,
                    model_type=entry.model_type,
                    created_at=entry.created_at,
                    metrics=metrics_dict,
                )
            )
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return rows

    def _compute_basic_metrics(equity: pd.Series) -> Dict[str, Optional[float]]:
        if equity.empty:
            return {}
        returns = equity.pct_change().dropna()
        start = float(equity.iloc[0])
        final = float(equity.iloc[-1])
        cum_return = final / start - 1 if start else None
        max_equity = equity.cummax()
        dd = (equity - max_equity) / max_equity
        max_drawdown = float(dd.min()) if not dd.empty else None
        sharpe = float(returns.mean() / returns.std() * (252 ** 0.5)) if not returns.empty and returns.std() != 0 else 0.0
        return {
            "final_equity": final,
            "cum_return": cum_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
        }

    def _parse_run_id(stem: str) -> str:
        return stem.replace("_equity", "")

    @app.get("/api/portfolio/backtests", response_model=List[PortfolioBacktestInfo])
    def list_portfolio_backtests():
        bt_dir = Path("outputs") / "backtests"
        if not bt_dir.exists():
            return []
        infos: List[PortfolioBacktestInfo] = []
        for path in bt_dir.glob("portfolio_*_equity.csv"):
            run_id = path.stem
            try:
                df = pd.read_csv(path, parse_dates=["time"])
            except Exception:
                continue
            if df.empty or "portfolio_equity" not in df.columns:
                continue
            equity = pd.Series(df["portfolio_equity"].values, index=df["time"])
            metrics = _compute_basic_metrics(equity)
            timeframe = run_id.split("_")[1] if len(run_id.split("_")) > 1 else ""
            symbols: List[str] = []
            trades_path = Path("outputs") / "trades" / f"{run_id}_trades.csv"
            if trades_path.exists():
                try:
                    trades_df = pd.read_csv(trades_path)
                    if "symbol" in trades_df.columns:
                        symbols = sorted(trades_df["symbol"].dropna().unique().tolist())
                except Exception:
                    symbols = []
            infos.append(
                PortfolioBacktestInfo(
                    run_id=run_id,
                    symbols=symbols,
                    timeframe=timeframe,
                    start=df["time"].iloc[0].isoformat() if not df.empty else None,
                    end=df["time"].iloc[-1].isoformat() if not df.empty else None,
                    metrics=metrics,
                )
            )
        return infos

    @app.get("/api/portfolio/equity/{run_id}", response_model=List[PortfolioEquityPoint])
    def get_portfolio_equity(run_id: str):
        bt_dir = Path("outputs") / "backtests"
        path = bt_dir / f"{run_id}.csv"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Run not found")
        df = pd.read_csv(path, parse_dates=["time"])
        if df.empty or "portfolio_equity" not in df.columns:
            return []
        points: List[PortfolioEquityPoint] = []
        for _, row in df.iterrows():
            t = pd.to_datetime(row["time"]).timestamp()
            points.append(
                PortfolioEquityPoint(
                    time=t,
                    portfolio_equity=float(row["portfolio_equity"]),
                )
            )
        return points

    @app.post("/api/live/control")
    def live_control(req: LiveControlRequest):
        valid = {"pause", "resume", "stop", "flatten"}
        if req.command not in valid:
            raise HTTPException(status_code=400, detail="Invalid live command")

        cfg = load_system_config()
        live_cfg = cfg.get("live", {}) or {}
        state_path = Path(
            live_cfg.get(
                "latest_state_path",
                live_cfg.get("state_path", live_cfg.get("paper", {}).get("state_path", "outputs/live/live_state.json")),
            )
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {}
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        state["requested_action"] = req.command
        if req.run_id is not None:
            state["run_id"] = req.run_id

        state_path.write_text(json.dumps(state), encoding="utf-8")
        return {"status": "ok", "requested_action": req.command}

    return app
