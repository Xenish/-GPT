from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/check_config_sanity.py")


def test_check_config_sanity_cli_pass():
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
