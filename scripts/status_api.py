"""
Simple status reporter for data/warehouse backends.

Usage:
    FT_TIMESCALE_DSN=postgresql://... python -m scripts.status_api --profile research
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import psycopg2  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore

# Ensure project root on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from finantradealgo.system.config_loader import load_config
from finantradealgo.data_engine.data_backend import build_backend


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report backend status.")
    parser.add_argument("--profile", default="research", choices=["research", "live"])
    args = parser.parse_args(argv)

    cfg = load_config(args.profile)
    data_cfg = cfg["data_cfg"]
    warehouse_cfg = cfg.get("warehouse_cfg")

    print(f"Profile: {args.profile}")
    print(f"Data backend: {data_cfg.backend}")
    if warehouse_cfg:
        print(f"Warehouse backend: {warehouse_cfg.backend}")
        print(f"Warehouse dsn_env: {warehouse_cfg.dsn_env}")

    # Try building data backend
    try:
        backend = build_backend(data_cfg)
        print(f"Data backend instantiated: {backend.__class__.__name__}")
    except Exception as exc:
        print(f"Data backend init failed: {exc}")

    # DB connectivity check if applicable
    if warehouse_cfg and warehouse_cfg.backend.lower() in ("timescale", "postgres"):
        if psycopg2 is None:
            print("psycopg2 not installed; cannot check DB connectivity.")
        else:
            dsn = warehouse_cfg.get_dsn(allow_missing=True)
            if not dsn:
                print("DSN not set; cannot check DB connectivity.")
            else:
                try:
                    conn = psycopg2.connect(dsn)
                    conn.close()
                    print("DB connectivity: OK")
                except Exception as exc:
                    print(f"DB connectivity failed: {exc}")


if __name__ == "__main__":
    main()
