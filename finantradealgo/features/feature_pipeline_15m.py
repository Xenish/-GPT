from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
) -> Tuple[pd.DataFrame, List[str]]:
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

    feature_cols = get_default_feature_cols(df)
    return df, feature_cols


def get_default_feature_cols(df: pd.DataFrame) -> List[str]:
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

    feature_cols: List[str] = []

    for col in df.columns:
        if col in blacklist_exact:
            continue
        if any(col.startswith(prefix) for prefix in blacklist_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols
