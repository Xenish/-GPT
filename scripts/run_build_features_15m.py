from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from finantradealgo.features.feature_pipeline_15m import (
    build_feature_pipeline_from_system_config,
)
from finantradealgo.system.config_loader import load_system_config


def log_run_header(symbol: str, timeframe: str, preset: str, pipeline_version: str) -> None:
    print(
        f"[RUN] symbol={symbol} timeframe={timeframe} "
        f"feature_preset={preset} pipeline_version={pipeline_version}"
    )


def main() -> None:
    sys_cfg = load_system_config()
    data_cfg = sys_cfg.get("data", {})
    symbol = sys_cfg.get("symbol", "BTCUSDT")
    timeframe = sys_cfg.get("timeframe", "15m")

    df_feat, pipeline_meta = build_feature_pipeline_from_system_config(sys_cfg)
    feature_cols = pipeline_meta.get("feature_cols", [])
    preset = pipeline_meta.get("feature_preset", sys_cfg.get("features", {}).get("feature_preset", "extended"))
    pipeline_version = pipeline_meta.get("pipeline_version", "unknown")
    log_run_header(symbol, timeframe, preset, pipeline_version)

    print(f"[INFO] Feature DF shape: {df_feat.shape}")
    print("[INFO] First 40 columns:")
    print(list(df_feat.columns)[:40])
    print("[INFO] Last 5 rows:")
    print(df_feat.tail())

    features_dir = Path(data_cfg.get("features_dir", "data/features"))
    out_path = features_dir / f"{symbol}_features_{timeframe}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path, index=False)
    print(f"[INFO] Saved features -> {out_path}")


if __name__ == "__main__":
    main()
