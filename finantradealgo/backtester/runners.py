from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine
from finantradealgo.features.feature_pipeline_15m import (
    PIPELINE_VERSION_15M,
    build_feature_pipeline_from_system_config,
)
from finantradealgo.ml.model_registry import get_latest_model, load_model_by_id
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.strategies.strategy_engine import create_strategy
from finantradealgo.strategies.ml_strategy import MLSignalStrategy
from finantradealgo.system.config_loader import load_system_config


def _generate_run_id(strategy: str, symbol: str, timeframe: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{strategy}_{symbol}_{timeframe}_{timestamp}"


def _serialize_dataframe(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    records = df.copy()
    for col in records.columns:
        if pd.api.types.is_datetime64_any_dtype(records[col]):
            records[col] = records[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return records.to_dict(orient="records")


def _inject_ml_proba_from_registry(
    df_features: pd.DataFrame,
    cfg: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    ml_cfg = cfg.get("ml", {}) or {}
    backtest_cfg = ml_cfg.get("backtest", {}) or {}
    persistence_cfg = ml_cfg.get("persistence", {}) or {}

    proba_col = backtest_cfg.get("proba_column", "ml_proba_long")
    model_dir = Path(persistence_cfg.get("model_dir", "outputs/ml_models"))

    model_type = ml_cfg.get("model", {}).get("type")
    entry = get_latest_model(
        str(model_dir),
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
    )
    if entry is None:
        raise ValueError(
            f"No valid ML model found in registry for {symbol}/{timeframe}. "
            "Train or register a model before running ML backtests."
        )

    try:
        model, meta = load_model_by_id(str(model_dir), entry.model_id)
    except FileNotFoundError as exc:
        raise ValueError(
            f"ML model artifacts missing for {entry.model_id} under {model_dir}. "
            "Run the appropriate ML training script again or clean the registry."
        ) from exc

    feature_cols = getattr(meta, "feature_cols", None)
    if not feature_cols:
        raise ValueError(
            f"Model {entry.model_id} has no feature_cols metadata. "
            "Retrain the model with a newer pipeline that stores feature_cols."
        )

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise ValueError(
            f"Feature mismatch for model {entry.model_id}. "
            f"Missing columns in df_features: {missing}. "
            "Likely the feature pipeline or config changed after training."
        )

    df_features = df_features.copy()
    X = df_features[feature_cols].to_numpy()
    proba = model.predict_proba(X)
    if proba.shape[1] < 2:
        raise ValueError("Loaded model does not provide binary probabilities.")
    df_features[proba_col] = proba[:, 1].astype(float)
    if df_features[proba_col].isna().all():
        raise ValueError(
            f"Model {entry.model_id} produced only NaN probabilities for {symbol}/{timeframe}."
        )
    return df_features


def run_backtest_once(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    cfg: Optional[Dict[str, Any]] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_system_config()
    cfg_local = deepcopy(cfg)
    cfg_local["symbol"] = symbol
    cfg_local["timeframe"] = timeframe
    if strategy_params:
        strategy_cfg = cfg_local.setdefault("strategy", {}).setdefault(strategy_name, {})
        strategy_cfg.update(strategy_params)

    df_features, pipeline_meta = build_feature_pipeline_from_system_config(cfg_local)

    strategy = create_strategy(strategy_name, cfg_local)

    if isinstance(strategy, MLSignalStrategy):
        df_features = _inject_ml_proba_from_registry(
            df_features=df_features,
            cfg=cfg_local,
            symbol=symbol,
            timeframe=timeframe,
        )

    risk_cfg = RiskConfig.from_dict(cfg_local.get("risk", {}))
    risk_engine = RiskEngine(risk_cfg)

    engine = BacktestEngine(
        strategy=strategy,
        risk_engine=risk_engine,
        price_col="close",
        timestamp_col="timestamp",
    )
    result = engine.run(df_features)
    run_id = _generate_run_id(strategy_name, symbol, timeframe)

    output_dir = Path("outputs") / "backtests"
    trades_dir = Path("outputs") / "trades"
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)

    equity_curve = result.get("equity_curve")
    if isinstance(equity_curve, pd.Series):
        eq_df = equity_curve.to_frame(name="equity")
        eq_path = output_dir / f"{run_id}_equity.csv"
        eq_df.to_csv(eq_path)
    else:
        eq_path = None

    trades_df = result.get("trades")
    if isinstance(trades_df, pd.DataFrame):
        trade_count = len(trades_df)
        trades_path = trades_dir / f"{run_id}_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        trades_serialized = _serialize_dataframe(trades_df)
    else:
        trade_count = 0
        trades_path = None
        trades_serialized = []

    payload = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "metrics": result.get("metrics", {}),
        "risk_stats": result.get("risk_stats", {}),
        "trade_count": trade_count,
        "equity_csv": str(eq_path) if eq_path else None,
        "trades_csv": str(trades_path) if trades_path else None,
        "trades": trades_serialized,
    }
    return payload


def run_rule_backtest(symbol: str, timeframe: str) -> Dict[str, Any]:
    return run_backtest_once(symbol, timeframe, "rule")


def run_ml_backtest(symbol: str, timeframe: str) -> Dict[str, Any]:
    return run_backtest_once(symbol, timeframe, "ml")
