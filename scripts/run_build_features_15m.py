from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from finantradealgo.features.feature_pipeline_15m import (
    FeaturePipelineConfig,
    build_feature_pipeline_15m,
)


def main() -> None:
    symbol = "BTCUSDT"
    ohlcv_path = Path("data/ohlcv") / f"{symbol}_15m.csv"
    funding_path = Path("data/external/funding") / f"{symbol}_funding_15m.csv"
    oi_path = Path("data/external/open_interest") / f"{symbol}_oi_15m.csv"

    cfg = FeaturePipelineConfig(
        rule_allowed_hours=list(range(8, 18)),
        rule_allowed_weekdays=[0, 1, 2, 3, 4],
    )

    df_feat, feature_cols = build_feature_pipeline_15m(
        csv_ohlcv_path=str(ohlcv_path),
        pipeline_cfg=cfg,
        csv_funding_path=str(funding_path) if funding_path.exists() else None,
        csv_oi_path=str(oi_path) if oi_path.exists() else None,
    )

    print(f"[INFO] Feature DF shape: {df_feat.shape}")
    print("[INFO] First 40 columns:")
    print(list(df_feat.columns)[:40])
    print("[INFO] Last 5 rows:")
    print(df_feat.tail())

    out_path = Path("data/features") / f"{symbol}_features_15m.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path, index=False)
    print(f"[INFO] Saved features -> {out_path}")


if __name__ == "__main__":
    main()
