"""Virtual Xbox 360 gamepad through ViGEmBus (``pip install vgamepad``): true analog steering.

Left stick X = steering, right trigger = acceleration, left trigger = brake, one face/shoulder button = nitro, another =
reverse (only with ``output.reverse_mode: button``; by default reversing just keeps the left trigger held).
"""

from __future__ import annotations

import logging
from typing import Any

from gesture_car.backend.output import ControlState
from gesture_car.config import OutputConfig

log = logging.getLogger(__name__)

KEEPALIVE_S = 0.25
EPSILON = 0.004
BUTTONS = {
    "a": "XUSB_GAMEPAD_A",
    "b": "XUSB_GAMEPAD_B",
    "x": "XUSB_GAMEPAD_X",
    "y": "XUSB_GAMEPAD_Y",
    "lb": "XUSB_GAMEPAD_LEFT_SHOULDER",
    "rb": "XUSB_GAMEPAD_RIGHT_SHOULDER",
}


class GamepadOutput:
    name = "gamepad"

    def __init__(self, cfg: OutputConfig) -> None:
        import vgamepad as vg  # raises ImportError if not installed, or an error if ViGEmBus is missing

        self.cfg = cfg
        self._vg: Any = vg
        self._button = getattr(vg.XUSB_BUTTON, BUTTONS[cfg.nitro_button])
        self._reverse_button = getattr(vg.XUSB_BUTTON, BUTTONS[cfg.reverse_button])
        self._reverse_down = False
        self.pad = vg.VX360Gamepad()
        self._last: tuple[float, float, float, bool, bool] | None = None
        self._last_sent = 0.0
        self._nitro_down = False
        log.info("Virtual Xbox 360 gamepad created (nitro = %s button)", cfg.nitro_button)

    def update(self, state: ControlState, now: float) -> None:
        values = (
            max(-1.0, min(1.0, state.steer)),
            max(0.0, min(1.0, state.throttle * self.cfg.auto_accel_level)),
            max(0.0, min(1.0, state.brake)),
            state.nitro,
            state.reverse,
        )
        last = self._last
        changed = last is None or values[3:] != last[3:] or any(abs(a - b) > EPSILON for a, b in zip(values[:3], last[:3], strict=True))
        if not changed and now - self._last_sent < KEEPALIVE_S:
            return
        steer, throttle, brake, nitro, reverse = values
        self.pad.left_joystick_float(steer, 0.0)
        self.pad.right_trigger_float(throttle)
        self.pad.left_trigger_float(brake)
        if nitro and not self._nitro_down:
            self.pad.press_button(self._button)
        elif not nitro and self._nitro_down:
            self.pad.release_button(self._button)
        self._nitro_down = nitro
        if reverse and not self._reverse_down:
            self.pad.press_button(self._reverse_button)
        elif not reverse and self._reverse_down:
            self.pad.release_button(self._reverse_button)
        self._reverse_down = reverse
        self.pad.update()
        self._last, self._last_sent = values, now

    def release_all(self) -> None:
        try:
            self.pad.reset()
            self.pad.update()
        except Exception:
            log.exception("Could not reset the virtual gamepad")
        self._nitro_down = False
        self._reverse_down = False
        self._last = None
