"""
utils/logger.py
────────────────
Loguru setup with console + rotating file sinks.
Call `setup_logging()` once at application startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config import settings


def setup_logging() -> None:
    logger.remove()  # Remove default handler

    # Console — colourful, human-readable
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,
    )

    # File — JSON-structured for log aggregation
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        compression="gz",
        serialize=True,  # JSON format
        enqueue=True,
    )

    logger.info("Logging configured. Level={}", settings.log_level)
