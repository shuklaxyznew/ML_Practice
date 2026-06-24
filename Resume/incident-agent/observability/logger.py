import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from config.settings import settings


import json
import time


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for machine parsing.
    Production log aggregators parse this format automatically.
    """
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
            "module":     record.module,
            "function":   record.funcName,
            "line":       record.lineno,
        })

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with Rich formatting for terminal
    and plain text for the log file.
    Call this at the top of every module:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level.upper())

    # Terminal handler — rich formatted, human-readable
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(settings.log_level.upper())
    logger.addHandler(console_handler)

    # File handler — plain text, machine-parseable
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger