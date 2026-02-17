"""Utility helpers for the project (logging helpers)."""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the application.

    The logger prints to stderr with a simple format. Multiple calls
    return the same logger instance (handlers are added only once).
    """
    logger_name = name or "ghsummarizer"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["get_logger"]
