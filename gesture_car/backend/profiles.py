"""Game profiles: small JSON files in ``profiles/`` that override parts of ``config.json`` for one game or situation.

A profile has the same sections as ``config.json`` (only the keys it wants to change) plus an optional ``description``.
Adding a game means adding a JSON file; no code changes are needed."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gesture_car.config import Config, _apply_section, apply_feel
from gesture_car.utils.errors import GestureCarError

log = logging.getLogger(__name__)


def profiles_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "profiles"


def list_profiles(directory: Path | None = None) -> dict[str, str]:
    """Profile name -> description."""
    found: dict[str, str] = {}
    for path in sorted((directory or profiles_dir()).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            found[path.stem] = str(data.get("description", "")) if isinstance(data, dict) else ""
        except (OSError, ValueError):
            found[path.stem] = "(unreadable profile)"
    return found


def apply_profile(cfg: Config, name: str, directory: Path | None = None) -> None:
    """Apply profile ``name`` on top of ``cfg``. Raises :class:`GestureCarError` for an unknown or broken profile."""
    path = (directory or profiles_dir()) / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        names = ", ".join(list_profiles(directory)) or "none"
        raise GestureCarError(f"Unknown profile '{name}'", user_message=f"Unknown profile '{name}'. Available profiles: {names}") from None
    except (OSError, ValueError) as exc:
        raise GestureCarError(f"Profile '{name}' unreadable: {exc}", user_message=f"Profile '{name}' could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise GestureCarError(f"Profile '{name}' must contain a JSON object")
    sections = cfg.to_dict()
    for section, raw in data.items():
        if section == "description":
            continue
        if section not in sections or not isinstance(raw, dict):
            log.warning("Profile %s: unknown section %r ignored", name, section)
            continue
        _apply_section(getattr(cfg, section), raw, section)
    apply_feel(cfg.steering)
    log.info("Profile '%s' applied", name)
