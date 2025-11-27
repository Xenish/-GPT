from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.feature_pipeline_15m import (
    PIPELINE_VERSION_15M,
    build_feature_pipeline_from_system_config,
)
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import (
    SklearnLongModel,
    SklearnModelConfig,
    save_sklearn_model,
)
from finantradealgo.ml.model_registry import register_model
from finantradealgo.system.config_loader import load_system_config
import pandas as pd


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str, extra: str | None = None) -> None:
    msg = (
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )
    if extra:
        msg += f" {extra}"
    print(msg)


def main(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    preset: Optional[str] = None,
) -> None:
    sys_cfg = load_system_config()
    cfg = dict(sys_cfg)
    if symbol:
        cfg["symbol"] = symbol
    if timeframe:
        cfg["timeframe"] = timeframe
    if preset:
        features_section = dict(cfg.get("features", {}) or {})
        features_section["feature_preset"] = preset
        cfg["features"] = features_section

    df, pipeline_meta = build_feature_pipeline_from_system_config(cfg)
    symbol = pipeline_meta.get("symbol", cfg.get("symbol", "BTCUSDT"))
    timeframe = pipeline_meta.get("timeframe", cfg.get("timeframe", "15m"))
    preset = pipeline_meta.get("feature_preset", cfg.get("features", {}).get("feature_preset", "extended"))
    pipeline_version = pipeline_meta.get("pipeline_version", PIPELINE_VERSION_15M)
    log_run_header(symbol, timeframe, preset, pipeline_version, extra="mode=train-only")

    ml_cfg = cfg.get("ml", {})
    persistence_cfg = ml_cfg.get("persistence", {}) or {}
    feature_cols = pipeline_meta.get("feature_cols")
    if not feature_cols:
        raise ValueError("Pipeline metadata missing feature_cols for training.")

    label_cfg = LabelConfig.from_dict(ml_cfg.get("label"))
    df_lab = add_long_only_labels(df, label_cfg)
    target_col = "label_long"
    if target_col not in df_lab.columns:
        raise ValueError(f"{target_col} column missing after labeling.")

    df_train = df_lab.dropna(subset=[target_col]).reset_index(drop=True)
    if df_train.empty:
        raise ValueError("No rows available after labeling for training.")

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy(dtype=int)

    model_cfg = SklearnModelConfig.from_dict(ml_cfg.get("model"))
    model = SklearnLongModel(model_cfg)
    model.fit(X_train, y_train)

    if not persistence_cfg.get("save_model", False):
        print("[WARN] persistence.save_model is False; model will not be saved.")
        return

    if "timestamp" not in df_train.columns:
        raise ValueError("timestamp column missing for model metadata.")

    train_start = df_train["timestamp"].iloc[0]
    train_end = df_train["timestamp"].iloc[-1]
    model_dir = persistence_cfg.get("model_dir", "outputs/ml_models")
    meta = save_sklearn_model(
        model=model.clf,
        symbol=pipeline_meta.get("symbol", cfg.get("symbol", "BTCUSDT")),
        timeframe=pipeline_meta.get("timeframe", cfg.get("timeframe", "15m")),
        model_cfg=model_cfg,
        label_cfg=label_cfg,
        feature_preset=pipeline_meta.get("feature_preset", "extended"),
        feature_cols=feature_cols,
        train_start=train_start,
        train_end=train_end,
        metrics={"train_size": len(df_train)},
        base_dir=model_dir,
        pipeline_version=pipeline_meta.get("pipeline_version", PIPELINE_VERSION_15M),
    )

    if persistence_cfg.get("use_registry", True):
        register_model(
            meta,
            base_dir=model_dir,
            status="success",
            max_models=persistence_cfg.get("max_models_per_symbol_tf"),
        )

    model_path = Path(meta.meta_path).parent
    if meta.feature_importances:
        sorted_items = sorted(
            meta.feature_importances.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        df_imp = pd.DataFrame(
            [{"feature": name, "importance": float(val)} for name, val in sorted_items]
        )
        out_path = model_path / "feature_importances.csv"
        df_imp.to_csv(out_path, index=False)
        print("Top feature importances:")
        print(df_imp.head(20).to_string(index=False))
    else:
        print("Model has no feature_importances_ attribute; skipping importance export.")

    print(f"[INFO] Saved trained model -> {meta.model_id}")


if __name__ == "__main__":
    main()
