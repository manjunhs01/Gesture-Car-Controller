"""Logging configuration: detailed log file for developers, quiet console."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from gesture_car.utils.paths import logs_dir

_CONFIGURED = False
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", console: bool = True) -> None:
    """Configure the root logger once. Safe to call repeatedly."""
    global _CONFIGURED
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    if _CONFIGURED:
        return
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir() / "gesture_car.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError:  # unwritable home: keep going with console only
        console = True
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        root.addHandler(stream)
    _CONFIGURED = True
