"""Centralized logging configuration."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / f"log_{datetime.now():%Y-%m-%d}.log"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""
    root_logger = logging.getLogger()

    # Prevent duplicate handlers
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # File handler
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return configured logger instance."""
    return logging.getLogger(name)
