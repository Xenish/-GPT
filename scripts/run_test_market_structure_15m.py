from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finantradealgo.features.feature_pipeline_15m import (
    FeaturePipelineConfig,
    build_feature_pipeline_15m,
)


def main() -> None:
    symbol = "AIAUSDT"
    ohlcv_path = PROJECT_ROOT / "data" / "ohlcv" / f"{symbol}_15m.csv"

    cfg = FeaturePipelineConfig(
        use_base=True,
        use_ta=False,
        use_candles=True,
        use_osc=False,
        use_htf=False,
        use_external=False,
        use_rule_signals=False,
        use_microstructure=False,
        use_market_structure=True,
        drop_na=False,
    )

    df, meta = build_feature_pipeline_15m(
        csv_ohlcv_path=str(ohlcv_path),
        pipeline_cfg=cfg,
    )
    feat_cols = meta.get("feature_cols", [])

    ms_cols = [c for c in df.columns if c.startswith("ms_")]
    print("Market structure columns:", ms_cols)
    print(df[["timestamp", "high", "low"] + ms_cols].tail(30))


if __name__ == "__main__":
    main()
