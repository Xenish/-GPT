from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from finantradealgo.features.feature_pipeline_15m import build_feature_pipeline_15m
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig


def main() -> None:
    symbol = "BTCUSDT"
    base_dir = Path("data")
    ohlcv_path = base_dir / "ohlcv" / f"{symbol}_15m.csv"
    funding_path = base_dir / "external" / "funding" / f"{symbol}_funding_15m.csv"
    oi_path = base_dir / "external" / "open_interest" / f"{symbol}_oi_15m.csv"

    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Missing OHLCV CSV at {ohlcv_path}")

    df_feat, feature_cols = build_feature_pipeline_15m(
        csv_ohlcv_path=str(ohlcv_path),
        csv_funding_path=str(funding_path) if funding_path.exists() else None,
        csv_oi_path=str(oi_path) if oi_path.exists() else None,
    )

    label_cfg = LabelConfig(
        horizon=8,
        pos_threshold=0.003,
        fee_slippage=0.001,
    )
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

    model_cfg = SklearnModelConfig(
        model_type="random_forest",
        params={
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        random_state=42,
    )
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
