from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


def init_logger(run_id: str, log_dir: str | Path, level: str = "INFO") -> logging.Logger:
    """
    Initialize a console + file logger for live or backtest runs. Each run_id
    gets its own file under log_dir/run_id.log. Returns the configured logger.
    """
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{run_id}.log"

    logger = logging.getLogger(run_id)
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    # Avoid duplicating handlers when re-initializing the same run_id
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    setattr(logger, "log_path", log_path)
    return logger


__all__ = ["init_logger"]
