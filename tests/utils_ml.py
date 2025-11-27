from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from finantradealgo.features.feature_pipeline import get_feature_cols
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import SklearnLongModel, SklearnModelConfig


def prepare_ml_eval_df(
    df: pd.DataFrame,
    meta: dict,
    cfg: dict,
    split_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, str]:
    ml_section = cfg.get("ml", {}) or {}
    label_cfg = LabelConfig.from_dict(ml_section.get("label"))
    df_lab = add_long_only_labels(df, label_cfg)
    df_lab = df_lab.dropna(subset=["label_long"]).reset_index(drop=True)
    if len(df_lab) < 20:
        raise AssertionError("Not enough labeled rows to run ML workflow.")

    split_idx = max(int(len(df_lab) * split_ratio), 1)
    if split_idx >= len(df_lab):
        split_idx = len(df_lab) - 1

    df_train = df_lab.iloc[:split_idx].copy()
    df_eval = df_lab.iloc[split_idx:].copy().reset_index(drop=True)

    feature_cols: List[str] = meta.get("feature_cols") or []
    if not feature_cols:
        preset = meta.get("feature_preset", cfg.get("features", {}).get("feature_preset", "extended"))
        feature_cols = get_feature_cols(df_lab, preset=preset)

    missing = [col for col in feature_cols if col not in df_train.columns]
    if missing:
        raise AssertionError(f"Missing feature columns for ML dataset: {missing}")

    ml_model_cfg = SklearnModelConfig.from_dict(ml_section.get("model"))
    model = SklearnLongModel(ml_model_cfg)

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["label_long"].astype(int).to_numpy()
    model.fit(X_train, y_train)

    X_eval = df_eval[feature_cols].to_numpy()
    proba = model.predict_proba(X_eval)[:, 1]

    backtest_cfg = ml_section.get("backtest", {}) or {}
    proba_col = backtest_cfg.get("proba_column", "ml_proba_long")
    df_eval[proba_col] = proba
    return df_eval, proba_col
