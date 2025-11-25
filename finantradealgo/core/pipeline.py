from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from finantradealgo.data_engine.loader import load_ohlcv_csv
from finantradealgo.core.external_features import (
    ExternalFeatureConfig,
    add_external_features_15m,
)
from finantradealgo.features.base_features import FeatureConfig, add_basic_features
from finantradealgo.features.candle_features import (
    CandleFeatureConfig,
    add_candlestick_features,
)
from finantradealgo.features.multi_tf_features import (
    MultiTFConfig,
    add_multitf_1h_features,
)
from finantradealgo.features.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.features.rule_signals import RuleSignalConfig, add_rule_signals_v1
from finantradealgo.features.ta_features import TAFeatureConfig, add_ta_features


@dataclass
class FeaturePipelineConfig:
    """
    15m feature pipeline config.

    use_* flagleri ile hangi blokların dahil olacağını kontrol ediyorsun.
    Mesela saf ML feature seti için use_rule_signals=False bırakmak mantıklı.
    """

    use_ta: bool = True
    use_candles: bool = True
    use_osc: bool = True
    use_mtf: bool = True
    use_external: bool = True
    use_rule_signals: bool = False

    rule_allowed_hours: Optional[List[int]] = None
    rule_allowed_weekdays: Optional[List[int]] = None

    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    ta_cfg: TAFeatureConfig = field(default_factory=TAFeatureConfig)
    candle_cfg: CandleFeatureConfig = field(default_factory=CandleFeatureConfig)
    osc_cfg: OscFeatureConfig = field(default_factory=OscFeatureConfig)
    mtf_cfg: MultiTFConfig = field(default_factory=MultiTFConfig)
    rule_cfg: RuleSignalConfig = field(default_factory=RuleSignalConfig)


def build_feature_pipeline_15m(
    data_source,
    symbol: Optional[str] = None,
    cfg: Optional[FeaturePipelineConfig] = None,
    external_cfg: Optional[ExternalFeatureConfig] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    data_source: CSV path veya hali hazırda DataFrame.
    """
    pipeline_cfg = cfg or FeaturePipelineConfig()

    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        df = load_ohlcv_csv(str(data_source))
    df = add_basic_features(df, pipeline_cfg.feature_cfg)

    if pipeline_cfg.use_ta:
        df = add_ta_features(df, pipeline_cfg.ta_cfg)

    if pipeline_cfg.use_candles:
        df = add_candlestick_features(df, pipeline_cfg.candle_cfg)

    if pipeline_cfg.use_osc:
        df = add_osc_features(df, pipeline_cfg.osc_cfg)

    if pipeline_cfg.use_mtf:
        df = add_multitf_1h_features(df, pipeline_cfg.mtf_cfg)

    if pipeline_cfg.use_external and external_cfg is not None:
        df = add_external_features_15m(df, external_cfg)

    if pipeline_cfg.use_rule_signals:
        rule_cfg = pipeline_cfg.rule_cfg
        rule_cfg.allowed_hours = pipeline_cfg.rule_allowed_hours
        rule_cfg.allowed_weekdays = pipeline_cfg.rule_allowed_weekdays
        df = add_rule_signals_v1(df, rule_cfg)

    non_feature_exact = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    non_feature_prefixes = [
        "rule_",
        "label_",
    ]

    feature_cols: List[str] = []
    for col in df.columns:
        if col in non_feature_exact:
            continue
        if any(col.startswith(pref) for pref in non_feature_prefixes):
            continue
        if col == "signal":
            continue
        feature_cols.append(col)

    return df, feature_cols
