"""Record hand-landmark frames to a file and replay them through the control logic without a camera.

Replaying makes testing deterministic: the same recording always produces the same steering / brake / nitro output.
File format: JSON lines, one frame per line: ``{"t": seconds, "hands": [{"points": [[x, y, z] * 21], "side": "Left", "score": 0.9, "aspect": 1.33}]}``
followed (for the recorded session) by the control state in ``"state"``."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np

from gesture_car.backend.controller import ControlLogic
from gesture_car.backend.landmarks import HandLandmarks
from gesture_car.backend.output import ControlState
from gesture_car.config import Config


class Recorder:
    """Appends frames to a JSON-lines file. Use as a context manager or call :meth:`close`."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = path.open("w", encoding="utf-8")
        self._t0: float | None = None
        self.frames = 0

    def write(self, t: float, hands: tuple[HandLandmarks, ...], state: ControlState) -> None:
        if self._fh is None:
            return
        if self._t0 is None:
            self._t0 = t
        record = {
            "t": round(t - self._t0, 4),
            "hands": [
                {"points": np.round(h.points, 5).tolist(), "side": h.handedness, "score": round(float(h.score), 3), "aspect": round(h.aspect, 4)}
                for h in hands
            ],
            "state": {"steer": round(state.steer, 4), "throttle": state.throttle, "brake": state.brake, "nitro": state.nitro, "reverse": state.reverse},
        }
        self._fh.write(json.dumps(record) + "\n")
        self.frames += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_frames(path: Path) -> Iterator[tuple[float, tuple[HandLandmarks, ...]]]:
    """Yield ``(time, hands)`` for each recorded frame."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hands = tuple(
                HandLandmarks(np.asarray(h["points"], dtype=float), h.get("side", "Unknown"), float(h.get("score", 1.0)), float(h.get("aspect", 4 / 3)))
                for h in rec["hands"]
            )
            yield float(rec["t"]), hands


def replay(path: Path, cfg: Config) -> list[tuple[float, ControlState, str]]:
    """Run a recording through a fresh :class:`ControlLogic`; returns ``(time, state, mode)`` per frame."""
    logic = ControlLogic(cfg)
    out: list[tuple[float, ControlState, str]] = []
    for t, hands in read_frames(path):
        state = logic.process(hands, t)
        out.append((t, state, logic.mode.value))
    return out


def summarize(frames: list[tuple[float, ControlState, str]]) -> str:
    if not frames:
        return "Recording is empty."
    n = len(frames)
    duration = frames[-1][0] - frames[0][0]
    steer = [s.steer for _, s, _ in frames]
    modes: dict[str, int] = {}
    for _, _, mode in frames:
        modes[mode] = modes.get(mode, 0) + 1
    brake = sum(1 for _, s, _ in frames if s.brake > 0.5)
    nitro = sum(1 for _, s, _ in frames if s.nitro)
    mode_text = ", ".join(f"{m} {100 * c // n}%" for m, c in sorted(modes.items(), key=lambda kv: -kv[1]))
    return (
        f"{n} frames over {duration:.1f} s | modes: {mode_text}\n"
        f"steering min {min(steer):+.2f} max {max(steer):+.2f} | brake on {100 * brake // n}% of frames | nitro on {100 * nitro // n}% of frames"
    )
