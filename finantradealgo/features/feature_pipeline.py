from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pandas as pd

from finantradealgo.data_engine.loader import (
    load_ohlcv_csv,
    load_flow_features,
    load_sentiment_features,
)
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
    add_market_structure_features,
    compute_market_structure_with_zones,
)
from finantradealgo.market_structure.config import MarketStructureConfig
from finantradealgo.features.microstructure_features import add_microstructure_features
from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.features.osc_features import OscFeatureConfig, add_osc_features
from finantradealgo.features.rule_signals import RuleSignalConfig, add_rule_signals_v1
from finantradealgo.features.ta_features import TAFeatureConfig, add_ta_features
from finantradealgo.features.flow_features import add_flow_features
from finantradealgo.features.sentiment_features import add_sentiment_features
from finantradealgo.system.config_loader import load_system_config, DataConfig

PIPELINE_VERSION = "v1.0.0"
logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineResult:
    """
    Result from build_feature_pipeline containing DataFrame and metadata.

    Attributes:
        df: DataFrame with all features
        meta: Metadata dictionary containing:
            - feature_cols: List of feature column names
            - feature_preset: Preset name used
            - pipeline_version: Version string
            - market_structure_zones: (optional) List of Zone objects if requested
    """
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Allow tuple unpacking for backward compatibility: df, meta = result"""
        return iter((self.df, self.meta))


@dataclass
class FeaturePipelineConfig:
    use_base: bool = True
    use_ta: bool = True
    use_candles: bool = True
    use_osc: bool = True
    use_htf: bool = True
    use_microstructure: bool = False
    use_market_structure: bool = False
    market_structure_return_zones: bool = False  # If True, zones are added to meta
    use_external: bool = True
    use_rule_signals: bool = True
    use_flow_features: bool = False
    use_sentiment_features: bool = False
    drop_na: bool = True
    feature_preset: str = "extended"
    bar_mode: str = "time" # For UI/info purposes, not directly used in pipeline logic


    rule_allowed_hours: Optional[List[int]] = None
    rule_allowed_weekdays: Optional[List[int]] = None

    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    ta_cfg: TAFeatureConfig = field(default_factory=TAFeatureConfig)
    candle_cfg: CandleFeatureConfig = field(default_factory=CandleFeatureConfig)
    osc_cfg: OscFeatureConfig = field(default_factory=OscFeatureConfig)
    mtf_cfg: MultiTFConfig = field(default_factory=MultiTFConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    market_structure: MarketStructureConfig = field(default_factory=MarketStructureConfig)
    rule_cfg: RuleSignalConfig = field(default_factory=RuleSignalConfig)


def build_feature_pipeline(
    csv_ohlcv_path: str,
    pipeline_cfg: Optional[FeaturePipelineConfig] = None,
    csv_funding_path: Optional[str] = None,
    csv_oi_path: Optional[str] = None,
    flow_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    data_cfg: Optional[DataConfig] = None,
) -> FeaturePipelineResult:
    cfg = pipeline_cfg or FeaturePipelineConfig()

    df = load_ohlcv_csv(csv_ohlcv_path, config=data_cfg)

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
        df = add_microstructure_features(df, cfg.microstructure)

    meta: Dict[str, Any] = {}

    if cfg.use_market_structure:
        if cfg.market_structure_return_zones:
            result = compute_market_structure_with_zones(df, cfg.market_structure)
            df = pd.concat([df, result.features], axis=1)
            meta["market_structure_zones"] = result.zones
        else:
            df = add_market_structure_features(df, cfg.market_structure)

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

    if cfg.use_flow_features and flow_df is not None and not flow_df.empty:
        df = add_flow_features(df, flow_df)

    if cfg.use_sentiment_features and sentiment_df is not None and not sentiment_df.empty:
        df = add_sentiment_features(df, sentiment_df)

    if cfg.drop_na:
        df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_cols(df, preset=cfg.feature_preset)
    meta.update({
        "feature_cols": feature_cols,
        "feature_preset": cfg.feature_preset,
        "pipeline_version": PIPELINE_VERSION,
    })

    return FeaturePipelineResult(df=df, meta=meta)


def get_feature_cols(df: pd.DataFrame, preset: str = "extended") -> List[str]:
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
    flow_cols = [c for c in numeric_cols if c.startswith("flow_")]
    sent_cols = [c for c in numeric_cols if c.startswith("sentiment_")]
    base_cols = [
        c for c in numeric_cols if c not in set(ms_cols + flow_cols + sent_cols)
    ]

    preset_lower = preset.lower()
    if preset_lower == "core":
        return base_cols

    if preset_lower == "extended":
        return list(dict.fromkeys(base_cols + ms_cols + flow_cols + sent_cols))

    if preset_lower == "flow":
        return list(dict.fromkeys(base_cols + flow_cols))

    if preset_lower in {"flow_sentiment", "flow-sentiment"}:
        return list(dict.fromkeys(base_cols + flow_cols + sent_cols))

    return base_cols


def build_feature_pipeline_from_system_config(
    sys_cfg: Optional[Dict[str, Any]] = None,
    pipeline_cfg: Optional[FeaturePipelineConfig] = None,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
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
            use_flow_features=feature_section.get("use_flow_features", False),
            use_sentiment_features=feature_section.get("use_sentiment_features", False),
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

    live_cfg = cfg.get("live_cfg")
    data_section = cfg.get("data", {}) or {}
    data_symbols = data_section.get("symbols") or []
    resolved_symbol = symbol or cfg.get("symbol") or getattr(live_cfg, "symbol", None)
    if not resolved_symbol and data_symbols:
        resolved_symbol = data_symbols[0]
    resolved_timeframe = (
        timeframe
        or data_section.get("timeframe")
        or cfg.get("timeframe")
        or getattr(live_cfg, "timeframe", None)
    )
    if not resolved_symbol:
        raise ValueError("Symbol must be provided via args or system config.")
    if not resolved_timeframe:
        raise ValueError("Timeframe must be provided via args or system config.")

    ohlcv_dir = Path(data_section.get("ohlcv_dir", "data/ohlcv"))
    external_dir = Path(data_section.get("external_dir", "data/external"))
    ohlcv_template = data_section.get("ohlcv_path_template")
    if ohlcv_template:
        csv_ohlcv = Path(ohlcv_template.format(symbol=resolved_symbol, timeframe=resolved_timeframe))
    else:
        csv_ohlcv = ohlcv_dir / f"{resolved_symbol}_{resolved_timeframe}.csv"
    csv_funding = external_dir / "funding" / f"{resolved_symbol}_{resolved_timeframe}_funding.csv"
    csv_oi = external_dir / "open_interest" / f"{resolved_symbol}_{resolved_timeframe}_oi.csv"
    flow_dir = data_section.get("flow_dir")
    sentiment_dir = data_section.get("sentiment_dir")
    base_data_dir = data_section.get("base_dir", "data")
    data_cfg = DataConfig.from_dict(data_section) # Extract DataConfig from the loaded system config

    flow_df = None
    if fp_cfg.use_flow_features:
        flow_df = load_flow_features(
            resolved_symbol,
            resolved_timeframe,
            flow_dir=flow_dir,
            base_dir=base_data_dir,
            data_cfg=data_cfg, # Pass data_cfg to load_flow_features
        )
        if flow_df is None or flow_df.empty:
            logger.warning(
                "[PIPELINE] Flow features requested but no flow data loaded; disabling flow features."
            )
            fp_cfg.use_flow_features = False

    sentiment_df = None
    if fp_cfg.use_sentiment_features:
        sentiment_df = load_sentiment_features(
            resolved_symbol,
            resolved_timeframe,
            sentiment_dir=sentiment_dir,
            base_dir=base_data_dir,
            data_cfg=data_cfg, # Pass data_cfg to load_sentiment_features
        )
        if sentiment_df is None or sentiment_df.empty:
            logger.warning(
                "[PIPELINE] Sentiment features requested but no sentiment data loaded; disabling sentiment features."
            )
            fp_cfg.use_sentiment_features = False

    df, pipeline_meta = build_feature_pipeline(
        csv_ohlcv_path=str(csv_ohlcv),
        pipeline_cfg=fp_cfg,
        csv_funding_path=str(csv_funding) if csv_funding.exists() else None,
        csv_oi_path=str(csv_oi) if csv_oi.exists() else None,
        flow_df=flow_df,
        sentiment_df=sentiment_df,
        data_cfg=data_cfg, # Pass data_cfg here
    )

    pipeline_meta["symbol"] = resolved_symbol
    pipeline_meta["timeframe"] = resolved_timeframe
    return df, pipeline_meta
