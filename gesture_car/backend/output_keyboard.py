"""Keyboard backend: SendInput scan codes on Windows (works with DirectInput games), pynput elsewhere.

By default both WASD and the arrow keys are pressed together (``output.key_preset``).

Steering is analog-by-PWM: the left/right key is pulsed fast and its duty cycle equals the steering amount."""

from __future__ import annotations

import logging
import sys
import threading
from typing import Protocol

from gesture_car.backend.output import ControlState
from gesture_car.config import OutputConfig

log = logging.getLogger(__name__)

# US-layout set-1 scan codes. Keys in EXTENDED need the extended-key flag.
SCANCODES: dict[str, int] = {
    **dict(zip("qwertyuiop", range(16, 26), strict=True)),
    **dict(zip("asdfghjkl", range(30, 39), strict=True)),
    **dict(zip("zxcvbnm", range(44, 51), strict=True)),
    **dict(zip("1234567890", range(2, 12), strict=True)),
    "space": 57, "enter": 28, "esc": 1, "tab": 15, "shift": 42, "ctrl": 29, "alt": 56,
    "up": 72, "down": 80, "left": 75, "right": 77,
    **{f"f{i}": 58 + i for i in range(1, 11)}, "f11": 87, "f12": 88,
}  # fmt: skip
EXTENDED = {"up", "down", "left", "right"}


def normalise_key(name: str) -> str:
    key = name.strip().lower()
    if key not in SCANCODES:
        raise ValueError(f"unsupported key {name!r}; use a-z, 0-9, arrows, space, enter, esc, tab, shift, ctrl, alt, f1-f12")
    return key


class KeySink(Protocol):
    def apply(self, desired: set[str]) -> None: ...
    def release_all(self) -> None: ...


class _HeldTracker:
    """Common press/release bookkeeping: only changes are sent, and release_all is always safe."""

    def __init__(self) -> None:
        self.held: set[str] = set()
        self._release_failed: set[str] = set()  # keys whose release is being retried (logged once)
        self._lock = threading.Lock()

    def _press(self, key: str) -> None:
        raise NotImplementedError

    def _release(self, key: str) -> None:
        raise NotImplementedError

    def apply(self, desired: set[str]) -> None:
        with self._lock:
            for key in sorted(self.held - desired):  # releases first, so a key never sticks
                try:
                    self._release(key)
                except Exception:  # still down: keep it held so the release is retried on every update, never forgotten
                    if key not in self._release_failed:
                        log.exception("Could not release key %s; retrying", key)
                        self._release_failed.add(key)
                    continue
                self.held.discard(key)
                self._release_failed.discard(key)
            for key in sorted(desired - self.held):
                self._press(key)
                self.held.add(key)

    def release_all(self) -> None:
        with self._lock:
            for key in sorted(self.held):
                try:
                    self._release(key)
                except Exception:  # never let one failure leave other keys held
                    log.exception("Could not release key %s", key)
            self.held.clear()
            self._release_failed.clear()


class RecordingSink(_HeldTracker):
    """Sends nothing. Records events; used for dry runs (``--background-dry-run``) and tests."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def _press(self, key: str) -> None:
        self.events.append(("down", key))
        log.info("[dry-run] key down: %s", key)

    def _release(self, key: str) -> None:
        self.events.append(("up", key))
        log.info("[dry-run] key up: %s", key)


class WindowsSendInputSink(_HeldTracker):
    """Hardware-scan-code key events through ``SendInput`` (no extra packages, no window)."""

    def __init__(self) -> None:
        super().__init__()
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        ulong_ptr = ctypes.c_size_t

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr))  # fmt: skip

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr))  # fmt: skip

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = (("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD))

        class _Union(ctypes.Union):
            _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT))

        class INPUT(ctypes.Structure):
            _fields_ = (("type", wintypes.DWORD), ("u", _Union))

        self._INPUT, self._KEYBDINPUT = INPUT, KEYBDINPUT
        self._send = ctypes.windll.user32.SendInput  # type: ignore[attr-defined]

    def _event(self, key: str, up: bool) -> bool:
        flags = 0x0008 | (0x0002 if up else 0) | (0x0001 if key in EXTENDED else 0)  # SCANCODE | KEYUP | EXTENDEDKEY
        ki = self._KEYBDINPUT(0, SCANCODES[key], flags, 0, 0)
        inp = self._INPUT(1, self._INPUT._fields_[1][1](ki=ki))  # type 1 = keyboard
        return bool(self._send(1, self._ctypes.byref(inp), self._ctypes.sizeof(inp)) == 1)

    def _press(self, key: str) -> None:
        if not self._event(key, False):
            log.warning("SendInput refused key %s (down); another app may be elevated", key)

    def _release(self, key: str) -> None:
        if not self._event(key, True):  # raising keeps the key in ``held``, so the key-up is sent again next update
            raise OSError(f"SendInput refused key-up for {key}; another app may be elevated")


class PynputSink(_HeldTracker):
    """Cross-platform fallback (needs ``pip install pynput``; on Linux an X11 session)."""

    def __init__(self) -> None:
        super().__init__()
        from pynput.keyboard import Controller, Key

        self._controller = Controller()
        special = {"up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right, "space": Key.space, "enter": Key.enter,
                   "esc": Key.esc, "tab": Key.tab, "shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt}  # fmt: skip
        special.update({f"f{i}": getattr(Key, f"f{i}") for i in range(1, 13)})
        self._special = special

    def _resolve(self, key: str):  # type: ignore[no-untyped-def]
        return self._special.get(key, key)

    def _press(self, key: str) -> None:
        self._controller.press(self._resolve(key))

    def _release(self, key: str) -> None:
        self._controller.release(self._resolve(key))


def create_sink(dry_run: bool = False) -> KeySink:
    if dry_run:
        return RecordingSink()
    if sys.platform.startswith("win"):
        return WindowsSendInputSink()
    try:
        return PynputSink()
    except Exception as exc:
        raise RuntimeError(
            "No keyboard output is available on this system. Install pynput (pip install pynput) or run with --background-dry-run."
        ) from exc


def pwm_on(duty: float, now: float, period_s: float, min_duty: float, full_duty: float) -> bool:
    """Is a PWM key down at time ``now``? Below ``min_duty`` never, above ``full_duty`` always, else for ``duty`` of each period."""
    if duty < min_duty:
        return False
    if duty >= full_duty:
        return True
    return (now % period_s) / period_s < duty


class PwmPulser:
    """Steering PWM that keys can only follow at the loop's ticks: each period starts with a pulse of ``duty * period``
    plus what earlier periods were owed (or minus what they got too much), held until that much was actually held.

    Plain phase sampling (``pwm_on``) drops pulses shorter than one tick depending on tick timing, so small steering
    values came out as nothing or as irregular bursts. Here a pulse is at least one tick long, small values give one short
    pulse every few periods at a steady rhythm, and the average duty follows the value exactly.
    """

    def __init__(self, period_s: float) -> None:
        self.period_s = period_s
        self.reset()

    def reset(self) -> None:
        self.on = False
        self._last: float | None = None
        self._carry = 0.0  # on-time owed by (+) or overpaid in (-) earlier periods, seconds
        self._target = 0.0  # on-time earned in this period (integral of duty)
        self._held = 0.0  # on-time actually held in this period

    def update(self, duty: float, now: float) -> bool:
        p = self.period_s
        if self._last is not None:
            dt = min(max(now - self._last, 0.0), p)
            self._target += duty * dt
            self._held += dt if self.on else 0.0
            if now % p < self._last % p or now - self._last >= p:  # a new period began: settle the old one
                self._carry = max(-p, min(p, self._carry + self._target - self._held))
                self._target = self._held = 0.0
        self._last = now
        self.on = self._held < duty * p + self._carry
        return self.on


PRESETS: dict[str, tuple[str, str, str, str]] = {  # accelerate, brake, left, right
    "wasd": ("w", "s", "a", "d"),
    "arrows": ("up", "down", "left", "right"),
}


def resolve_keys(cfg: OutputConfig) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """Key sets for accelerate / brake / left / right. ``both`` presses the WASD key and the arrow key together, so the
    controller works in games and browser tabs that use either."""
    if cfg.key_preset == "custom":
        custom = (cfg.key_accelerate, cfg.key_brake, cfg.key_left, cfg.key_right)
        return tuple(frozenset({normalise_key(k)}) for k in custom)  # type: ignore[return-value]
    names = ("wasd", "arrows") if cfg.key_preset == "both" else (cfg.key_preset,)
    return tuple(frozenset(PRESETS[n][i] for n in names) for i in range(4))  # type: ignore[return-value]


class KeyboardOutput:
    """Acceleration = hold W; brake = S; nitro = Shift; steering = A/D pulse-width modulated (all configurable)."""

    name = "keyboard"

    def __init__(self, cfg: OutputConfig, sink: KeySink | None = None, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.sink = sink or create_sink(dry_run)
        self.accelerate, self.brake, self.left, self.right = resolve_keys(cfg)
        self.nitro = normalise_key(cfg.key_nitro)
        self.reverse = normalise_key(cfg.key_reverse)
        self._steer_pwm = PwmPulser(cfg.pwm_period_s)
        self._steer_side = 0  # -1 left, +1 right, 0 none: a new side starts a fresh pulse train

    def update(self, state: ControlState, now: float) -> None:
        c = self.cfg
        keys: set[str] = set()
        if state.throttle > 0.0:
            keys |= self.accelerate
        if pwm_on(state.brake, now, c.pwm_period_s, 0.05, 0.95):
            keys |= self.brake
        if state.nitro:
            keys.add(self.nitro)
        if state.reverse:
            keys.add(self.reverse)
        duty = min(abs(state.steer), 1.0)
        side = (state.steer > 0) - (state.steer < 0) if duty > 0.0 and duty >= c.pwm_min_duty else 0
        if side != self._steer_side:
            self._steer_pwm.reset()
            self._steer_side = side
        if side and (self._steer_pwm.update(duty, now) or duty >= c.pwm_full_duty):
            keys |= self.right if side > 0 else self.left
        self.sink.apply(keys)

    def release_all(self) -> None:
        self._steer_pwm.reset()
        self._steer_side = 0
        self.sink.release_all()
