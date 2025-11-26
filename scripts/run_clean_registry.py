from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.ml.model_registry import (
    load_registry,
    validate_registry_entry,
    _registry_index_path,  # type: ignore
)
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean ML registry by removing broken entries.")
    parser.add_argument("--model-dir", default="outputs/ml_models", help="Model directory (registry lives here).")
    parser.add_argument("--prune-dirs", action="store_true", help="Also delete missing/broken model directories.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    registry = load_registry(str(model_dir))
    if not registry.entries:
        print("No registry entries found.")
        return

    index_path = Path(_registry_index_path(str(model_dir)))
    df = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    if df.empty:
        print("Registry index is empty.")
        return

    keep_rows = []
    removed = []
    for _, row in df.iterrows():
        dummy_entry = registry.entries[0].__class__(
            model_id=row["model_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            model_type=row["model_type"],
            created_at=row["created_at"],
            path=row["path"],
            status=row.get("status", "success"),
        )
        if validate_registry_entry(str(model_dir), dummy_entry):
            keep_rows.append(row)
        else:
            removed.append(row["model_id"])
            if args.prune_dirs:
                target_dir = model_dir / row["model_id"]
                if target_dir.exists():
                    for child in target_dir.iterdir():
                        if child.is_file():
                            child.unlink()
                        elif child.is_dir():
                            for sub in child.rglob("*"):
                                if sub.is_file():
                                    sub.unlink()
                                elif sub.is_dir():
                                    sub.rmdir()
                            child.rmdir()
                    target_dir.rmdir()

    if not removed:
        print("No broken entries found.")
        return

    pd.DataFrame(keep_rows).to_csv(index_path, index=False)
    print(f"Removed broken entries: {removed}")
    print(f"Registry index updated at {index_path}")


if __name__ == "__main__":
    main()
