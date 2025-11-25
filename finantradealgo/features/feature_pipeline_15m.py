from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.core.external_features import add_external_features
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.features.candle_features import (
    CandleFeatureConfig,
    add_candlestick_features,
)
from finantradealgo.features.multi_tf_features import (
    MultiTFConfig,
    add_multitf_1h_features,
)
from finantradealgo.features.market_structure_features import (
    MarketStructureConfig,
    add_market_structure_features_15m,
)
from finantradealgo.features.microstructure_features import (
    MicrostructureFeatureConfig,
    add_microstructure_features_15m,
)
from finantradealgo.features.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.features.rule_signals import RuleSignalConfig, add_rule_signals_v1
from finantradealgo.features.ta_features import TAFeatureConfig, add_ta_features
from finantradealgo.system.config_loader import load_system_config

PIPELINE_VERSION_15M = "v1.0.0"


@dataclass
class FeaturePipelineConfig:
    use_base: bool = True
    use_ta: bool = True
    use_candles: bool = True
    use_osc: bool = True
    use_htf: bool = True
    use_microstructure: bool = True
    use_market_structure: bool = True
    use_external: bool = True
    use_rule_signals: bool = True
    drop_na: bool = True
    feature_preset: str = "extended"

    rule_allowed_hours: Optional[List[int]] = None
    rule_allowed_weekdays: Optional[List[int]] = None

    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    ta_cfg: TAFeatureConfig = field(default_factory=TAFeatureConfig)
    candle_cfg: CandleFeatureConfig = field(default_factory=CandleFeatureConfig)
    osc_cfg: OscFeatureConfig = field(default_factory=OscFeatureConfig)
    mtf_cfg: MultiTFConfig = field(default_factory=MultiTFConfig)
    micro_cfg: MicrostructureFeatureConfig = field(default_factory=MicrostructureFeatureConfig)
    market_cfg: MarketStructureConfig = field(default_factory=MarketStructureConfig)
    rule_cfg: RuleSignalConfig = field(default_factory=RuleSignalConfig)


def build_feature_pipeline_15m(
    csv_ohlcv_path: str,
    pipeline_cfg: Optional[FeaturePipelineConfig] = None,
    csv_funding_path: Optional[str] = None,
    csv_oi_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = pipeline_cfg or FeaturePipelineConfig()

    df = load_ohlcv_csv(csv_ohlcv_path)

    if cfg.use_base:
        df = add_basic_features(df, cfg.feature_cfg)

    if cfg.use_ta:
        df = add_ta_features(df, cfg.ta_cfg)

    if cfg.use_candles:
        df = add_candlestick_features(df, cfg.candle_cfg)

    if cfg.use_osc:
        df = add_osc_features(df, cfg.osc_cfg)

    if cfg.use_htf:
        df = add_multitf_1h_features(df, cfg.mtf_cfg)

    if cfg.use_microstructure:
        df = add_microstructure_features_15m(df, cfg.micro_cfg)

    if cfg.use_market_structure:
        df = add_market_structure_features_15m(df, cfg.market_cfg)

    if cfg.use_external:
        funding_path = csv_funding_path if csv_funding_path else None
        if funding_path and not Path(funding_path).exists():
            print(f"[WARN] Funding CSV not found at {funding_path}; skipping funding merge.")
            funding_path = None

        oi_path = csv_oi_path if csv_oi_path else None
        if oi_path and not Path(oi_path).exists():
            print(f"[WARN] OI CSV not found at {oi_path}; skipping OI merge.")
            oi_path = None

        df = add_external_features(
            df,
            csv_funding_path=funding_path,
            csv_oi_path=oi_path,
        )

    if cfg.use_rule_signals:
        rule_cfg = cfg.rule_cfg
        if cfg.rule_allowed_hours is not None:
            rule_cfg.allowed_hours = cfg.rule_allowed_hours
        if cfg.rule_allowed_weekdays is not None:
            rule_cfg.allowed_weekdays = cfg.rule_allowed_weekdays
        df = add_rule_signals_v1(df, rule_cfg)

    if cfg.drop_na:
        df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_cols_15m(df, preset=cfg.feature_preset)
    pipeline_meta: Dict[str, Any] = {
        "feature_cols": feature_cols,
        "feature_preset": cfg.feature_preset,
        "pipeline_version": PIPELINE_VERSION_15M,
    }
    return df, pipeline_meta


def get_feature_cols_15m(df: pd.DataFrame, preset: str = "extended") -> List[str]:
    blacklist_exact = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rule_long_entry",
        "rule_long_exit",
        "signal",
    }

    blacklist_prefixes = (
        "label_",
        "target_",
    )

    numeric_cols: List[str] = []

    for col in df.columns:
        if col in blacklist_exact:
            continue
        if any(col.startswith(prefix) for prefix in blacklist_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    ms_cols = [c for c in numeric_cols if c.startswith("ms_")]
    base_cols = [c for c in numeric_cols if c not in ms_cols]

    preset_lower = preset.lower()
    if preset_lower == "core":
        return base_cols

    if preset_lower == "extended":
        return list(dict.fromkeys(base_cols + ms_cols))

    return base_cols


def build_feature_pipeline_from_system_config(
    sys_cfg: Optional[Dict[str, Any]] = None,
    pipeline_cfg: Optional[FeaturePipelineConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience helper that wires system.yml -> FeaturePipelineConfig and
    resolves CSV paths for OHLCV + external data.
    """
    cfg = sys_cfg or load_system_config()
    feature_section = cfg.get("features", {})

    rule_section = cfg.get("rule", {})

    rule_cfg = RuleSignalConfig.from_dict(rule_section)

    if pipeline_cfg is None:
        fp_cfg = FeaturePipelineConfig(
            use_base=feature_section.get("use_base", True),
            use_ta=feature_section.get("use_ta", True),
            use_candles=feature_section.get("use_candles", True),
            use_osc=feature_section.get("use_osc", True),
            use_htf=feature_section.get("use_htf", True),
            use_microstructure=feature_section.get("use_microstructure", False),
            use_market_structure=feature_section.get("use_market_structure", False),
            use_external=feature_section.get("use_external", True),
            use_rule_signals=feature_section.get("use_rule_signals", True),
            drop_na=feature_section.get("drop_na", True),
            feature_preset=feature_section.get("feature_preset", "extended"),
            rule_allowed_hours=rule_section.get("allowed_hours"),
            rule_allowed_weekdays=rule_section.get("allowed_weekdays"),
            rule_cfg=rule_cfg,
        )
    else:
        fp_cfg = pipeline_cfg
        fp_cfg.rule_cfg = rule_cfg
        if fp_cfg.rule_allowed_hours is None:
            fp_cfg.rule_allowed_hours = rule_section.get("allowed_hours")
        if fp_cfg.rule_allowed_weekdays is None:
            fp_cfg.rule_allowed_weekdays = rule_section.get("allowed_weekdays")

    symbol = cfg.get("symbol", "BTCUSDT")
    timeframe = cfg.get("timeframe", "15m")

    data_section = cfg.get("data", {})
    ohlcv_dir = Path(data_section.get("ohlcv_dir", "data/ohlcv"))
    external_dir = Path(data_section.get("external_dir", "data/external"))

    csv_ohlcv = ohlcv_dir / f"{symbol}_{timeframe}.csv"
    csv_funding = external_dir / "funding" / f"{symbol}_funding_{timeframe}.csv"
    csv_oi = external_dir / "open_interest" / f"{symbol}_oi_{timeframe}.csv"

    df, pipeline_meta = build_feature_pipeline_15m(
        csv_ohlcv_path=str(csv_ohlcv),
        pipeline_cfg=fp_cfg,
        csv_funding_path=str(csv_funding) if csv_funding.exists() else None,
        csv_oi_path=str(csv_oi) if csv_oi.exists() else None,
    )

    pipeline_meta["symbol"] = symbol
    pipeline_meta["timeframe"] = timeframe
    return df, pipeline_meta
