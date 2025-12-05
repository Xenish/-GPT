import pandas as pd

from finantradealgo.ml.labels import LabelConfig, add_long_only_labels


def test_add_long_only_labels_basic():
    ts = pd.date_range("2025-01-01", periods=50, freq="15min")
    prices = pd.Series(range(1, 51), index=ts)
    df = pd.DataFrame({"timestamp": ts, "close": prices.values})

    cfg = LabelConfig(horizon=5, pos_threshold=0.01, fee_slippage=0.0)
    out = add_long_only_labels(df, cfg)

    assert len(out) == len(df)
    assert "fwd_return" in out.columns
    assert "label_long" in out.columns
    assert out["label_long"].isna().sum() == cfg.horizon
    ratio = out["label_long"].dropna().mean()
    # Positive forward return should imply labels mostly 1
    assert ratio > 0.5
