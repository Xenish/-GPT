import numpy as np
import pandas as pd

from finantradealgo.ml.walkforward import WalkForwardConfig, add_walkforward_ml_signals
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels


def test_walkforward_adds_signals():
    ts = pd.date_range("2025-01-01", periods=200, freq="15min")
    # Oscillating price to produce both positive/negative fwd returns
    prices = np.sin(np.linspace(0, 6.28, len(ts))) + np.linspace(1.0, 1.5, len(ts))
    df = pd.DataFrame({"timestamp": ts, "close": prices})
    # Add simple labels
    df = add_long_only_labels(df, LabelConfig(horizon=5, pos_threshold=0.001, fee_slippage=0.0))

    cfg = WalkForwardConfig(
        initial_train_size=50,
        train_window=50,
        retrain_every=25,
        proba_entry=0.5,
    )
    df_out, metrics = add_walkforward_ml_signals(
        df,
        feature_cols=["close"],
        label_col="label_long",
        config=cfg,
        log_metrics=False,
    )
    assert len(df_out) == len(df)
    assert "ml_long_proba" in df_out.columns
    assert "ml_long_signal" in df_out.columns
    # Warmup should yield NaNs
    assert df_out["ml_long_proba"].isna().sum() >= cfg.initial_train_size
    # Some non-NaN probabilities after warmup
    assert df_out["ml_long_proba"].notna().sum() > 0

    # Ensure ordering preserved (timestamp sorted)
    assert df_out["timestamp"].is_monotonic_increasing
