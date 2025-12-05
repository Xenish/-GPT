from __future__ import annotations

from pathlib import Path


def test_no_legacy_config_flag_left():
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "scripts",
        root / "finantradealgo",
    ]

    for base in targets:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "--config" not in text, f"Legacy --config found in {path}"
