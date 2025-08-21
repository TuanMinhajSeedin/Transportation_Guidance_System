#!/usr/bin/env python3
"""
Centralized logging setup for the Transport Query Application.
Provides a rotating file handler and console output.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a configured logger with file and console handlers."""
    logger = logging.getLogger(name)

    if getattr(logger, "_configured", False):
        return logger

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
    try:
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # Fallback to current directory if path invalid
        log_dir = os.getcwd()

    log_path = os.path.join(log_dir, "app.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation (1 MB, keep 5 backups)
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logger
    logger.setLevel(getattr(logging, log_level_str, logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger._configured = True  # type: ignore[attr-defined]
    logger.debug(f"Logger initialized. Level={log_level_str}, File={log_path}")
    return logger


