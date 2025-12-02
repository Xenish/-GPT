"""
Run Research Service API.

This script starts the Research Service FastAPI application, which provides
REST endpoints for strategy search, backtesting, and scenario analysis.

Usage:
    python scripts/run_research_api.py

    # With custom config
    FT_CONFIG_PATH=config/system.research.yml python scripts/run_research_api.py

    # With custom host/port
    python scripts/run_research_api.py --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run research API server."""
    parser = argparse.ArgumentParser(description="Run Research Service API")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/system.research.yml",
        help="Config file path (default: config/system.research.yml)",
    )
    args = parser.parse_args()

    # Set config path environment variable
    os.environ["FT_CONFIG_PATH"] = args.config

    # Verify config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        print(f"        Please ensure {config_path} exists")
        sys.exit(1)

    print("=" * 60)
    print("Starting Research Service API")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Config: {args.config}")
    print(f"  Reload: {args.reload}")
    print("=" * 60)
    print()

    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed. Install with:")
        print("        pip install uvicorn[standard]")
        sys.exit(1)

    # Run uvicorn
    uvicorn.run(
        "services.research_service.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
