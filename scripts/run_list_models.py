from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.ml.model_registry import load_registry, validate_registry_entry
from finantradealgo.system.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="List ML models from the registry.")
    parser.add_argument("--model-dir", help="Override model directory (defaults to config).")
    parser.add_argument("--only-valid", action="store_true", help="Show only entries with artifacts present.")
    parser.add_argument("--only-symbol", help="Filter by symbol, e.g., BTCUSDT.")
    parser.add_argument("--only-timeframe", help="Filter by timeframe, e.g., 15m.")
    parser.add_argument("--only-model-type", help="Filter by model type, e.g., RandomForest.")
    args = parser.parse_args()

    cfg = load_config("research")
    ml_cfg = cfg.get("ml", {})
    persistence_cfg = ml_cfg.get("persistence", {})
    default_model_dir = persistence_cfg.get("model_dir", "outputs/ml_models")
    model_dir = Path(args.model_dir or default_model_dir)

    registry = load_registry(
        str(model_dir),
        symbol=args.only_symbol,
        timeframe=args.only_timeframe,
        model_type=args.only_model_type,
    )

    if not registry.entries:
        print("No models in registry.")
        return

    print(f"Model registry in {model_dir}:\n")

    for entry in registry.entries:
        artifacts_ok = validate_registry_entry(str(model_dir), entry)
        if args.only_valid and not artifacts_ok:
            continue
        status = entry.status
        print(
            f"- {entry.model_id} | {entry.symbol}/{entry.timeframe} | "
            f"type={entry.model_type} | status={status} | "
            f"created_at={entry.created_at} | artifacts_ok={artifacts_ok}"
        )


if __name__ == "__main__":
    main()
