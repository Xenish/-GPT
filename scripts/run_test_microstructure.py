from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finantradealgo.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline,
)


def main() -> None:
    symbol = "AIAUSDT"
    ohlcv_path = PROJECT_ROOT / "data" / "ohlcv" / f"{symbol}_15m.csv"

    cfg = FeaturePipelineConfig(
        use_base=True,
        use_ta=True,
        use_candles=True,
        use_osc=True,
        use_htf=True,
        use_external=False,
        use_rule_signals=False,
        use_microstructure=True,
        use_market_structure=False,
        drop_na=False,
    )

    df, meta = build_feature_pipeline(
        csv_ohlcv_path=str(ohlcv_path),
        pipeline_cfg=cfg,
    )
    feat_cols = meta.get("feature_cols", [])

    ms_cols = [c for c in df.columns if c.startswith("ms_")]
    print("Microstructure columns:", ms_cols)
    print(df[["timestamp", "close"] + ms_cols].tail(20))


if __name__ == "__main__":
    main()
