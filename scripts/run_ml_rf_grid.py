from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finantradealgo.features.feature_pipeline import (
    build_feature_pipeline_from_system_config,
    get_feature_cols,
)
from finantradealgo.ml.hyperparameter_search import run_rf_grid_search
from finantradealgo.ml.labels import build_labels_from_config
from finantradealgo.system.config_loader import load_config


def main() -> None:
    cfg = load_config("research")
    symbol = cfg.get("symbol", "AIAUSDT")
    timeframe = cfg.get("timeframe", "15m")

    df_features, meta = build_feature_pipeline_from_system_config(
        cfg,
        symbol=symbol,
        timeframe=timeframe,
    )

    feature_preset = cfg.get("ml", {}).get("feature_preset", "extended")
    feature_cols = meta.get("feature_cols") or get_feature_cols(df_features, feature_preset)
    df_features = df_features.dropna(subset=feature_cols, how="any")
    X = df_features[feature_cols].values

    df_labels, label_cfg = build_labels_from_config(df_features, cfg)
    target_col = label_cfg.target_col
    if target_col not in df_labels.columns:
        raise ValueError(f"Label column {target_col} not found in labels dataframe")
    y = df_labels[target_col].values

    mask = ~np.isnan(X).any(axis=1) & ~pd.isna(y)
    X = X[mask]
    y = y[mask]

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 3],
    }

    results = run_rf_grid_search(X, y, param_grid)

    out_dir = Path("outputs/ml_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rf_grid_results_{symbol}_{timeframe}.csv"

    rows = []
    for r in results:
        rows.append(
            {
                "params_json": json.dumps(r["params"], sort_keys=True),
                "mean_accuracy": r["mean_accuracy"],
                "std_accuracy": r["std_accuracy"],
                "mean_f1": r["mean_f1"],
                "std_f1": r["std_f1"],
                "n_splits": r["n_splits"],
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} RF grid results to {out_path}")


if __name__ == "__main__":
    main()
