"""Centralized Structured Logging System for Project Jane.

Provides a unified, asynchronous-safe logger that streams diagnostic telemetry,
tracebacks, and warnings to `logs/jane.log` without cluttering the interactive
terminal CLI session.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

_INITIALIZED = False
DEFAULT_LOG_PATH = "logs/jane.log"


def setup_logger(
    log_path: str = DEFAULT_LOG_PATH,
    level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3
) -> logging.Logger:
    """Configures the root 'Jane' logger with rotating file persistence."""
    global _INITIALIZED
    logger = logging.getLogger("Jane")

    if not _INITIALIZED:
        logger.setLevel(level)
        logger.propagate = False

        # Ensure target directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # File Handler with detailed timestamp and traceback formatting
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        _INITIALIZED = True

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a namespaced child logger under 'Jane'."""
    if not _INITIALIZED:
        setup_logger()
    base = logging.getLogger("Jane")
    if name:
        return base.getChild(name)
    return base
