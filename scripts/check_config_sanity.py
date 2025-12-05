"""
Simple config sanity checks to catch profile mixups in CI.

Usage:
    python scripts/check_config_sanity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_CFG = ROOT / "config/system.research.yml"
LIVE_CFG = ROOT / "config/system.live.yml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def check_research_config(cfg: dict) -> list[str]:
    errors: list[str] = []
    exch_type = cfg.get("exchange", {}).get("type", "").lower()
    if exch_type == "live":
        errors.append("research config exchange.type must not be 'live'")
    profile = cfg.get("profile")
    if profile and profile != "research":
        errors.append("research config profile must be 'research'")
    return errors


def check_live_config(cfg: dict) -> list[str]:
    errors: list[str] = []
    exch_type = cfg.get("exchange", {}).get("type", "").lower()
    if exch_type != "live":
        errors.append("live config exchange.type must be 'live'")
    profile = cfg.get("profile")
    if profile and profile != "live":
        errors.append("live config profile must be 'live'")
    kill = cfg.get("kill_switch", {})
    if not kill:
        errors.append("live config must define kill_switch block")
    risk = cfg.get("risk", {})
    if not risk:
        errors.append("live config must define risk block")
    return errors


def main() -> None:
    errors: list[str] = []
    research_cfg = load_yaml(RESEARCH_CFG)
    live_cfg = load_yaml(LIVE_CFG)
    errors.extend(check_research_config(research_cfg))
    errors.extend(check_live_config(live_cfg))

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        sys.exit(1)
    print("Config sanity checks passed.")


if __name__ == "__main__":
    main()
