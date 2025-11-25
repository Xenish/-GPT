from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.features.feature_pipeline_15m import (
    FeaturePipelineConfig,
    build_feature_pipeline_15m,
)


def main() -> None:
    symbol = "BTCUSDT"
    csv_path = "data/ohlcv/BTCUSDT_15m.csv"

    df_ohlcv = load_ohlcv_csv(csv_path)
    print(f"[INFO] Raw OHLCV shape: {df_ohlcv.shape}")

    cfg = FeaturePipelineConfig(
        rule_allowed_hours=list(range(8, 18)),
        rule_allowed_weekdays=[0, 1, 2, 3, 4],
    )

    df_feat, _ = build_feature_pipeline_15m(df_ohlcv, symbol=symbol, cfg=cfg)

    print(f"[INFO] Feature DF shape: {df_feat.shape}")
    print("[INFO] First 40 columns:")
    print(list(df_feat.columns)[:40])
    print("[INFO] Last 5 rows:")
    print(df_feat.tail())

    out_path = "data/features/BTCUSDT_features_15m.csv"
    df_feat.to_csv(out_path, index=False)
    print(f"[INFO] Saved features -> {out_path}")


if __name__ == "__main__":
    main()
