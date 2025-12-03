"""
Batch ML training script for multiple symbol/timeframe combinations.

Uses ml.targets configuration from system.research.yml to train models only
for specified combinations instead of all possible combinations.

Usage:
    python scripts/run_ml_train_batch.py
    python scripts/run_ml_train_batch.py --config config/system.research.yml
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
        ml_cfg = cfg.get("ml", {})
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
        model = SklearnLongModel(model_cfg)
        model.fit(X_train, y_train)

        logger.info(f"Model trained successfully for {symbol} {timeframe}")

        # Save model
        model_dir = Path(persistence_cfg.get("model_dir", "outputs/ml_models"))
        model_dir.mkdir(parents=True, exist_ok=True)

        model_id = save_sklearn_model(
            model=model.model,
            model_dir=str(model_dir),
            symbol=symbol,
            timeframe=timeframe,
            feature_cols=feature_cols,
            label_cfg=label_cfg,
            model_cfg=model_cfg,
            pipeline_meta=pipeline_meta,
        )

        logger.info(f"Model saved with ID: {model_id}")

        # Register model
        if persistence_cfg.get("use_registry", True):
            register_model(
                model_dir=str(model_dir),
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_cfg.type,
                metrics={},  # Add metrics if available
            )
            logger.info(f"Model registered in registry")

        elapsed = time.time() - start_time

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_id": model_id,
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

    parser = argparse.ArgumentParser(description="Batch ML model training")
    parser.add_argument(
        "--profile",
        choices=["research", "live"],
        default="research",
        help="Config profile to load (default: research)",
    )
    args = parser.parse_args()

    # Set dummy FCM key for config loading
    if not os.getenv("FCM_SERVER_KEY"):
        os.environ["FCM_SERVER_KEY"] = "dummy_ml_train_key"

    # Load config
    logger.info(f"Loading config profile '{args.profile}'")
    cfg = load_config(args.profile)

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
