"""
Batch ML training script for multiple symbol/timeframe combinations.

Uses ml.targets configuration from system.research.yml to train models only
for specified combinations instead of all possible combinations.

Usage:
    python scripts/run_ml_train_batch.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.feature_pipeline import (
    PIPELINE_VERSION,
    build_feature_pipeline_from_system_config,
)
from finantradealgo.ml.labels import LabelConfig, add_long_only_labels
from finantradealgo.ml.model import (
    SklearnLongModel,
    SklearnModelConfig,
    save_sklearn_model,
    set_global_seed,
)
from finantradealgo.ml.model_registry import register_model
from finantradealgo.ml.ml_utils import get_ml_targets, is_ml_enabled
from finantradealgo.system.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_model_for_target(
    cfg: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    """
    Train ML model for a single symbol/timeframe combination.

    Args:
        cfg: System configuration
        symbol: Trading symbol
        timeframe: Timeframe string

    Returns:
        Dictionary with training results and metadata
    """
    logger.info(f"Training model for {symbol} {timeframe}...")
    start_time = time.time()

    try:
        # Build features
        df, pipeline_meta = build_feature_pipeline_from_system_config(
            cfg,
            symbol=symbol,
            timeframe=timeframe,
        )

        logger.info(f"Loaded {len(df)} bars with {len(df.columns)} features")

        # Extract configuration
        ml_cfg_obj = cfg.get("ml_cfg") or {}
        ml_cfg = cfg.get("ml", {}) or {}
        persistence_cfg = ml_cfg.get("persistence", {}) or {}
        feature_cols = pipeline_meta.get("feature_cols")

        if not feature_cols:
            raise ValueError("Pipeline metadata missing feature_cols for training")

        # Add labels
        label_cfg = LabelConfig.from_dict(ml_cfg.get("label"))
        df_lab = add_long_only_labels(df, label_cfg)
        target_col = "label_long"

        if target_col not in df_lab.columns:
            raise ValueError(f"{target_col} column missing after labeling")

        # Prepare training data
        df_train = df_lab.dropna(subset=[target_col]).reset_index(drop=True)

        if df_train.empty:
            raise ValueError("No rows available after labeling for training")

        X_train = df_train[feature_cols].to_numpy()
        y_train = df_train[target_col].to_numpy(dtype=int)

        logger.info(
            f"Training data: {len(df_train)} samples, "
            f"{len(feature_cols)} features, "
            f"class distribution: {y_train.sum()}/{len(y_train)}"
        )

        # Train model
        model_cfg = SklearnModelConfig.from_dict(ml_cfg.get("model"))
        if model_cfg.random_state is not None:
            set_global_seed(model_cfg.random_state)
        model = SklearnLongModel(model_cfg)
        model.fit(X_train, y_train)

        logger.info(f"Model trained successfully for {symbol} {timeframe}")

        # Save model
        model_dir = Path(getattr(ml_cfg_obj, "model_dir", None) or persistence_cfg.get("model_dir", "outputs/ml_models"))
        model_dir.mkdir(parents=True, exist_ok=True)

        if "timestamp" not in df_train.columns:
            raise ValueError("timestamp column missing for model metadata.")

        config_snapshot = {
            "profile": cfg.get("profile"),
            "symbol": symbol,
            "timeframe": timeframe,
            "label": label_cfg.__dict__,
            "model": model_cfg.__dict__,
            "feature_preset": pipeline_meta.get("feature_preset", "extended"),
            "feature_cols": feature_cols,
        }

        meta = save_sklearn_model(
            model=model.clf,
            symbol=symbol,
            timeframe=timeframe,
            model_cfg=model_cfg,
            label_cfg=label_cfg,
            feature_preset=pipeline_meta.get("feature_preset", "extended"),
            feature_cols=feature_cols,
            train_start=pd.to_datetime(df_train["timestamp"].iloc[0]),
            train_end=pd.to_datetime(df_train["timestamp"].iloc[-1]),
            metrics={"train_size": len(df_train)},
            base_dir=str(model_dir),
            pipeline_version=pipeline_meta.get("pipeline_version", PIPELINE_VERSION),
            seed=model_cfg.random_state,
            config_snapshot=config_snapshot,
        )

        logger.info(f"Model saved with ID: {meta.model_id}")

        # Register model
        if persistence_cfg.get("use_registry", True):
            register_model(
                meta,
                base_dir=str(model_dir),
                status="success",
                max_models=persistence_cfg.get("max_models_per_symbol_tf"),
            )
            logger.info(f"Model registered in registry")

        elapsed = time.time() - start_time

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_id": meta.model_id,
            "train_samples": len(df_train),
            "features": len(feature_cols),
            "elapsed": elapsed,
            "status": "success",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed to train model for {symbol} {timeframe}: {e}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "elapsed": elapsed,
            "status": "failed",
            "error": str(e),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch ML model training (research profile only)")
    parser.add_argument("--symbol", type=str, help="Override single symbol (optional)")
    parser.add_argument("--timeframe", type=str, help="Override single timeframe (optional)")
    parser.add_argument("--seed", type=int, help="Random seed propagated to model config")
    args = parser.parse_args()

    # Set dummy FCM key for config loading
    if not os.getenv("FCM_SERVER_KEY"):
        os.environ["FCM_SERVER_KEY"] = "dummy_ml_train_key"

    # Load config
    logger.info(f"Loading config profile '{args.profile}'")
    cfg = load_config("research")
    if args.symbol:
        cfg["symbol"] = args.symbol
    if args.timeframe:
        cfg["timeframe"] = args.timeframe
    if args.seed is not None:
        cfg.setdefault("ml", {}).setdefault("model", {})["random_state"] = args.seed
        set_global_seed(args.seed)

    # Check if ML is enabled
    if not is_ml_enabled(cfg):
        logger.warning("ML is disabled in config (ml.enabled=false)")
        return

    # Get ML targets
    targets = get_ml_targets(cfg)

    if not targets:
        logger.error("No ML targets found in configuration")
        return

    logger.info(f"Training {len(targets)} models...")

    # Train models for each target
    results = []
    overall_start = time.time()

    for idx, (symbol, timeframe) in enumerate(targets, 1):
        logger.info(f"[{idx}/{len(targets)}] Processing {symbol} {timeframe}")

        result = train_model_for_target(cfg, symbol, timeframe)
        results.append(result)

    overall_elapsed = time.time() - overall_start

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Total targets: {len(targets)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total time: {overall_elapsed:.2f}s")

    if success_count > 0:
        avg_time = sum(r["elapsed"] for r in results if r["status"] == "success") / success_count
        logger.info(f"Average time per model: {avg_time:.2f}s")

    # Show successful models
    if success_count > 0:
        logger.info("")
        logger.info("Successful models:")
        for r in results:
            if r["status"] == "success":
                logger.info(
                    f"  {r['symbol']:10s} {r['timeframe']:5s} -> "
                    f"{r['train_samples']:6d} samples, {r['features']:3d} features "
                    f"({r['elapsed']:.2f}s) [ID: {r['model_id'][:12]}...]"
                )

    # Show failed models
    if failed_count > 0:
        logger.info("")
        logger.error("Failed models:")
        for r in results:
            if r["status"] == "failed":
                logger.error(
                    f"  {r['symbol']:10s} {r['timeframe']:5s} -> "
                    f"{r.get('error', 'Unknown error')}"
                )


if __name__ == "__main__":
    main()
