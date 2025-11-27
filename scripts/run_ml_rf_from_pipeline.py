from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from finantradealgo.features.feature_pipeline import (
    build_feature_pipeline_from_system_config,
    get_feature_cols,
)
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig
from finantradealgo.system.config_loader import load_system_config


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str) -> None:
    print(
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )


def main() -> None:
    sys_cfg = load_system_config()
    symbol = sys_cfg.get("symbol", "BTCUSDT")
    timeframe = sys_cfg.get("timeframe", "15m")
    df_feat, pipeline_meta = build_feature_pipeline_from_system_config(sys_cfg)

    preset = pipeline_meta.get("feature_preset") or sys_cfg.get("features", {}).get("feature_preset", "extended")
    feature_cols = pipeline_meta.get("feature_cols") or get_feature_cols(df_feat, preset)
    pipeline_version = pipeline_meta.get("pipeline_version", "unknown")
    log_run_header(symbol, timeframe, preset, pipeline_version)

    label_cfg = LabelConfig.from_dict(sys_cfg.get("ml", {}).get("label"))
    df_lab = add_long_only_labels(df_feat, label_cfg)

    target_col = "label_long"
    if target_col not in df_lab.columns:
        raise ValueError(f"{target_col} column missing after labeling.")

    df_lab = df_lab.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"[INFO] Total labeled rows: {len(df_lab)}")

    split_idx = int(len(df_lab) * 0.7)
    df_train = df_lab.iloc[:split_idx]
    df_test = df_lab.iloc[split_idx:]

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy(dtype=int)
    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test[target_col].to_numpy(dtype=int)

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    model_cfg = SklearnModelConfig.from_dict(sys_cfg.get("ml", {}).get("model"))
    model = SklearnLongModel(model_cfg)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== RandomForest from feature pipeline (15m) ===")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"f1       : {f1:.4f}")
    print(f"accuracy : {accuracy:.4f}")

    out_path = Path("ml_rf_from_pipeline_15m.csv")
    df_metrics = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": len(feature_cols),
            }
        ]
    )
    df_metrics.to_csv(out_path, index=False)
    print(f"[INFO] Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
