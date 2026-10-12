"""Windows console tweaks."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

_ENABLE_QUICK_EDIT = 0x0040
_ENABLE_EXTENDED_FLAGS = 0x0080
_STD_INPUT_HANDLE = -10


def disable_quick_edit() -> bool:
    """Turn off the Windows console 'QuickEdit' mode.

    With QuickEdit on, a single click inside the console window starts a text selection and Windows FREEZES every program that
    writes to that console until you press Enter or Esc. For a program that logs continuously this looks exactly like
    "VisionDrive got stuck". Returns True if the mode was changed."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False  # not attached to a console (pythonw, redirected input)
        new_mode = (mode.value & ~_ENABLE_QUICK_EDIT) | _ENABLE_EXTENDED_FLAGS
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:
        log.debug("Could not change the console mode", exc_info=True)
        return False
