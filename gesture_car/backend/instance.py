"""Single-instance guard and remote stop (``python main.py --stop``) for the windowless controller."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from gesture_car.utils.paths import user_data_dir


def pid_file() -> Path:
    return user_data_dir() / "controller.pid"


def stop_file() -> Path:
    return user_data_dir() / "controller.stop"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid() -> int | None:
    try:
        pid = int(pid_file().read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def request_stop(wait_s: float = 8.0) -> bool:
    """Ask a running controller to quit. Returns True if none is running afterwards."""
    if running_pid() is None:
        return True
    stop_file().write_text("stop", encoding="utf-8")
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        if running_pid() is None:
            return True
        time.sleep(0.1)
    return False
