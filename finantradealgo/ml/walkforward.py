from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig


@dataclass
class WalkForwardConfig:
    initial_train_size: int = 3000
    train_window: Optional[int] = 3000
    retrain_every: int = 50
    proba_entry: float = 0.55
    model_config: SklearnModelConfig = field(default_factory=SklearnModelConfig)
    min_class_samples: int = 20


def add_walkforward_ml_signals(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    config: WalkForwardConfig,
    log_metrics: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = df.copy()
    n = len(df)
    if n <= config.initial_train_size:
        raise ValueError(
            f"Veri çok kısa: len={n}, initial_train_size={config.initial_train_size}"
        )

    model = SklearnLongModel(config.model_config)
    proba_arr = np.full(n, np.nan, dtype=float)
    signal_arr = np.full(n, np.nan, dtype=float)

    metrics_blocks: List[Dict[str, Any]] = []
    block_id = 0

    i = config.initial_train_size
    while i < n:
        if config.train_window is None:
            train_start = 0
        else:
            train_start = max(0, i - config.train_window)
        train_end = i

        train_slice = df.iloc[train_start:train_end]

        X_train = train_slice[feature_cols]
        y_train = train_slice[label_col].astype(int)

        if y_train.nunique() < 2 or len(y_train) < config.min_class_samples:
            i += config.retrain_every
            continue

        model.fit(X_train, y_train)

        j_end = min(i + config.retrain_every, n)
        pred_slice = df.iloc[i:j_end]
        X_pred = pred_slice[feature_cols]

        proba_chunk = model.predict_proba(X_pred)[:, 1]
        proba_arr[i:j_end] = proba_chunk

        y_pred_chunk = (proba_chunk >= config.proba_entry).astype(int)
        signal_arr[i:j_end] = y_pred_chunk

        if log_metrics:
            y_true_chunk = pred_slice[label_col].astype(int).values
            metrics_blocks.append(
                {
                    "block_id": block_id,
                    "train_start_idx": train_start,
                    "train_end_idx": train_end,
                    "pred_start_idx": i,
                    "pred_end_idx": j_end,
                    "train_size": int(len(y_train)),
                    "pred_size": int(len(y_true_chunk)),
                    "train_pos_rate": float(y_train.mean()),
                    "pred_pos_rate": float(y_true_chunk.mean()),
                    "precision": float(
                        precision_score(y_true_chunk, y_pred_chunk, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true_chunk, y_pred_chunk, zero_division=0)
                    ),
                    "accuracy": float(accuracy_score(y_true_chunk, y_pred_chunk)),
                    "f1": float(f1_score(y_true_chunk, y_pred_chunk, zero_division=0)),
                }
            )

        block_id += 1
        i = j_end

    df["ml_proba_long"] = proba_arr
    df["ml_signal_long"] = signal_arr

    wf_metrics_df: Optional[pd.DataFrame]
    if log_metrics and metrics_blocks:
        wf_metrics_df = pd.DataFrame(metrics_blocks)
    else:
        wf_metrics_df = None

    return df, wf_metrics_df
