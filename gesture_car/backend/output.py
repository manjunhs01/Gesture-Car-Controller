"""Output interface shared by the gamepad and keyboard backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from gesture_car.config import OutputConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlState:
    """What the car should do right now. Steering -1 (full left) .. +1 (full right); throttle and brake 0..1."""

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    nitro: bool = False
    reverse: bool = False  # press the reverse key / button (output.reverse_mode "button"; brake_hold mode keeps brake instead)


NEUTRAL = ControlState()


class ControlOutput(Protocol):
    name: str

    def update(self, state: ControlState, now: float) -> None: ...
    def release_all(self) -> None: ...


def create_output(cfg: OutputConfig, dry_run: bool = False) -> ControlOutput:
    """Create the configured backend; falls back to the keyboard when the virtual gamepad is unavailable (if allowed)."""
    from gesture_car.backend.output_keyboard import KeyboardOutput

    if dry_run:
        return KeyboardOutput(cfg, dry_run=True)
    if cfg.backend == "gamepad":
        from gesture_car.backend.output_gamepad import GamepadOutput

        try:
            return GamepadOutput(cfg)
        except Exception as exc:
            if not cfg.fallback_to_keyboard:
                raise RuntimeError(f"Virtual gamepad unavailable: {exc}") from exc
            log.warning(
                "Virtual gamepad unavailable (%s). Install ViGEmBus and 'pip install vgamepad', or set output.backend to "
                "'keyboard'. Falling back to the keyboard.",
                exc,
            )
    return KeyboardOutput(cfg)
