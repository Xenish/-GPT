"""
Simple ML model comparison script using the standard research profile paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels

FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "trend_score",
]


def prepare_ml_dataset_15m(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X, y dataset from a 15m OHLCV CSV using the standard label pipeline.
    """
    df = load_ohlcv_csv(path)
    feat_config = FeatureConfig()
    df_feat = add_basic_features(df, feat_config)
    label_config = LabelConfig(horizon=5, pos_threshold=0.003, fee_slippage=0.001)
    df_lab = add_long_only_labels(df_feat, label_config)
    df_ml = df_lab.dropna(subset=["label_long"] + FEATURE_COLS).copy()
    X = df_ml[FEATURE_COLS]
    y = df_ml["label_long"].astype(int)
    return X, y


def main() -> None:
    X, y = prepare_ml_dataset_15m("data/ohlcv/BTCUSDT_15m.csv")

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = [
        (
            "LogReg",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "GradBoost",
            GradientBoostingClassifier(
                random_state=42,
            ),
        ),
    ]

    rows = []
    for name, clf in models:
        print(f"\n=== Training model: {name} ===")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rows.append(
            {
                "model_name": name,
                "precision": prec,
                "recall": rec,
                "accuracy": acc,
                "f1": f1,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )
        print(f"  precision: {prec:.4f}")
        print(f"  recall   : {rec:.4f}")
        print(f"  accuracy : {acc:.4f}")
        print(f"  f1       : {f1:.4f}")

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv("ml_model_comparison_15m.csv", index=False)
    print("\n=== Model comparison ===")
    print(df_cmp)


if __name__ == "__main__":
    main()
