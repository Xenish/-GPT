from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels


def main() -> None:
    # 1) 15 dakikalık veriyi yükle
    df = load_ohlcv_csv("data/ohlcv/BTCUSDT_15m.csv")

    # 2) Feature'ları ekle
    feat_config = FeatureConfig()
    df_feat = add_basic_features(df, feat_config)

    # 3) Label'ları ekle (horizon/thresh fee ayarları run_ml_backtest_15m ile aynı olsun)
    label_config = LabelConfig(
        horizon=5,
        pos_threshold=0.003,
        fee_slippage=0.001,
    )
    df_lab = add_long_only_labels(df_feat, label_config)

    # 4) ML'de kullanacağımız feature kolonları
    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "vol_10",
        "vol_20",
        "trend_score",
    ]

    # Label + feature kolonlarında NaN olan satırları at
    df_ml = df_lab.dropna(subset=["label_long"] + feature_cols).copy()

    X = df_ml[feature_cols]
    y = df_ml["label_long"].astype(int)

    # 70/30 train-test split
    split_idx = int(len(df_ml) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5) RandomForest modeli (model_comparison dosyasına benzer olsun)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Test setinde pozitif sınıf olasılıkları
    y_proba = clf.predict_proba(X_test)[:, 1]

    # 6) Farklı threshold'lar için metrikleri hesapla
    thresholds = np.linspace(0.40, 0.80, 9)  # 0.40, 0.45, ..., 0.80
    rows: list[dict] = []

    for th in thresholds:
        y_hat = (y_proba >= th).astype(int)

        prec = precision_score(y_test, y_hat, zero_division=0)
        rec = recall_score(y_test, y_hat, zero_division=0)
        f1 = f1_score(y_test, y_hat, zero_division=0)
        acc = accuracy_score(y_test, y_hat)

        rows.append(
            {
                "threshold": th,
                "precision": prec,
                "recall": rec,
                "accuracy": acc,
                "f1": f1,
                "pred_long_rate": float(y_hat.mean()),
            }
        )

    df_th = pd.DataFrame(rows)
    print("\n=== Threshold sweep (RandomForest, 15m) ===")
    print(df_th)

    # En iyi F1'e göre seçilen threshold
    best = df_th.sort_values("f1", ascending=False).iloc[0]
    print("\n=== Best by F1 ===")
    print(
        f"threshold={best['threshold']:.2f}, "
        f"precision={best['precision']:.3f}, "
        f"recall={best['recall']:.3f}, "
        f"accuracy={best['accuracy']:.3f}, "
        f"f1={best['f1']:.3f}, "
        f"pred_long_rate={best['pred_long_rate']:.3f}"
    )

    # CSV'ye kaydet
    out_path = "ml_rf_threshold_sweep_15m.csv"
    df_th.to_csv(out_path, index=False)
    print(f"\nSaved threshold sweep results to: {out_path}")


if __name__ == "__main__":
    main()
