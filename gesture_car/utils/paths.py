"""Locations of user data (settings, calibration, models, logs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_HOME = "GESTURE_CAR_HOME"


def bundle_dir() -> Path | None:
    """Folder holding files shipped inside a PyInstaller build, or None when running from source."""
    root = getattr(sys, "_MEIPASS", None)
    return Path(root) if root else None


def user_data_dir() -> Path:
    """Return (and create) the per-user data directory.

    Override with the ``GESTURE_CAR_HOME`` environment variable.
    """
    override = os.environ.get(ENV_HOME)
    if override:
        base = Path(override)
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home())) / "GestureCar"
    else:
        base = Path.home() / ".gesture_car"
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    path = user_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def calibration_path() -> Path:
    return user_data_dir() / "calibration.json"
