"""Procedural 21-point hand model used ONLY by automated tests and dev tools.

It produces landmark geometry (what MediaPipe would output) so the geometry,
classification, smoothing, control, physics and state logic can be tested
deterministically without a camera. The application itself never uses it.
"""

from __future__ import annotations

import math

import numpy as np

from gesture_car.backend.landmarks import HandLandmarks
from gesture_car.backend.pipeline import TrackingResult

# Hand-local frame: x right, y up, wrist at the origin, wrist -> middle knuckle = 1.0
MCP = {
    "index": (-0.28, 0.98),
    "middle": (0.0, 1.0),
    "ring": (0.26, 0.95),
    "pinky": (0.50, 0.85),
}
SPREAD_DEG = {"index": 6.0, "middle": 0.0, "ring": -5.0, "pinky": -12.0}
LENGTHS = {"index": 1.0, "middle": 1.05, "ring": 0.98, "pinky": 0.78}
SEGMENTS = (0.55, 0.33, 0.27)
FINGERS = ("index", "middle", "ring", "pinky")

THUMB_OPEN = np.array([[-0.25, 0.20, 0.0], [-0.50, 0.45, 0.0], [-0.70, 0.62, 0.0], [-0.88, 0.78, 0.0]])
THUMB_UP = np.array([[-0.25, 0.20, 0.0], [-0.38, 0.55, 0.0], [-0.50, 0.95, 0.0], [-0.60, 1.30, 0.0]])
THUMB_CURLED = np.array([[-0.25, 0.20, 0.0], [-0.42, 0.45, -0.05], [-0.32, 0.62, -0.12], [-0.12, 0.70, -0.15]])


SPREAD_FAN_DEG = {"index": -14.0, "middle": -4.0, "ring": 6.0, "pinky": 16.0}  # added when fingers are fanned out


def _finger_points(name: str, extension: float, fan: float = 0.0) -> list[tuple[float, float, float]]:
    """MCP, PIP, DIP, TIP for a finger with extension 0 (curled) .. 1 (straight)."""
    mx, my = MCP[name]
    spread = math.radians(SPREAD_DEG[name] + SPREAD_FAN_DEG[name] * fan)
    dx, dy = math.sin(spread), math.cos(spread)
    flex = 1.0 - extension
    angles = [math.radians(85 * flex)]
    angles.append(angles[0] + math.radians(95 * flex))
    angles.append(angles[1] + math.radians(60 * flex))
    points = [(mx, my, 0.0)]
    x, y, z = mx, my, 0.0
    for seg, alpha in zip(SEGMENTS, angles, strict=True):
        length = seg * LENGTHS[name]
        x += dx * math.cos(alpha) * length
        y += dy * math.cos(alpha) * length
        z += -math.sin(alpha) * length
        points.append((x, y, z))
    return points


def make_hand(
    extension: float | tuple[float, float, float, float] = 1.0,
    thumb: float | None = None,
    thumb_up: bool = False,
    fan: float = 0.0,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.22,
    roll_deg: float = 0.0,
    aspect: float = 4.0 / 3.0,
    handedness: str = "Right",
    score: float = 0.95,
) -> HandLandmarks:
    """Build a hand.

    ``extension`` is one value for all four fingers (1 = open palm, 0 = fist) or
    a tuple (index, middle, ring, pinky). ``center`` is where the *palm centre*
    appears in the normalised image; ``scale`` is the wrist-to-knuckle length as
    a fraction of image height.
    """
    ext = (extension,) * 4 if isinstance(extension, (int, float)) else tuple(extension)
    thumb_ext = ext[1] if thumb is None else thumb
    local = np.zeros((21, 3))
    local[1:5] = THUMB_UP if thumb_up else THUMB_CURLED + (THUMB_OPEN - THUMB_CURLED) * thumb_ext
    for i, name in enumerate(FINGERS):
        local[5 + 4 * i : 9 + 4 * i] = _finger_points(name, ext[i], fan)

    palm_local = local[[0, 5, 9, 13, 17], :2].mean(axis=0)
    roll = math.radians(roll_deg)
    c, s = math.cos(roll), math.sin(roll)
    rotated = local.copy()
    rotated[:, 0] = local[:, 0] * c + local[:, 1] * s
    rotated[:, 1] = -local[:, 0] * s + local[:, 1] * c
    palm_rot = np.array(
        [palm_local[0] * c + palm_local[1] * s, -palm_local[0] * s + palm_local[1] * c]
    )
    rotated[:, :2] -= palm_rot  # palm centre at the origin

    norm = np.zeros((21, 3), dtype=np.float32)
    # lengths in image-height units: scale * local; x divides by aspect to get width fractions
    norm[:, 0] = center[0] + rotated[:, 0] * scale / aspect
    norm[:, 1] = center[1] - rotated[:, 1] * scale
    norm[:, 2] = rotated[:, 2] * scale / aspect
    return HandLandmarks(points=norm, handedness=handedness, score=score, aspect=aspect)


def fist(**kwargs: object) -> HandLandmarks:
    return make_hand(extension=0.0, thumb=0.0, **kwargs)  # type: ignore[arg-type]


def open_palm(**kwargs: object) -> HandLandmarks:
    """Five extended, fanned-out fingers."""
    return make_hand(extension=1.0, thumb=1.0, fan=1.0, **kwargs)  # type: ignore[arg-type]


def thumbs_up(**kwargs: object) -> HandLandmarks:
    """Fist with the thumb pointing up."""
    return make_hand(extension=0.0, thumb_up=True, **kwargs)  # type: ignore[arg-type]


def wheel_pair(
    angle_deg: float = 0.0,
    left: HandLandmarks | None = None,
    right: HandLandmarks | None = None,
    separation: float = 0.5,
    aspect: float = 4.0 / 3.0,
    center: tuple[float, float] = (0.5, 0.55),
    **kwargs: object,
) -> tuple[HandLandmarks, HandLandmarks]:
    """Two hands on an imaginary wheel rotated ``angle_deg`` clockwise (positive = right turn).

    ``separation`` is the distance between the palms in image-height units. ``left``/``right`` can be given to use other
    hand shapes (their position is then overwritten).
    """
    half = separation / 2.0
    dx, dy = half * math.cos(math.radians(angle_deg)), half * math.sin(math.radians(angle_deg))
    out = []
    for sign, hand in ((-1.0, left), (1.0, right)):
        pos = (center[0] + sign * dx / aspect, center[1] + sign * dy)
        out.append(fist(center=pos, aspect=aspect, **kwargs) if hand is None else hand)
    return out[0], out[1]


def place(hand_factory: object, angle_deg: float, side: str, separation: float = 0.5, aspect: float = 4.0 / 3.0, center: tuple[float, float] = (0.5, 0.55)) -> HandLandmarks:
    """A hand made by ``hand_factory`` (e.g. ``open_palm``) at the left/right end of the wheel line."""
    half = separation / 2.0
    sign = -1.0 if side == "left" else 1.0
    pos = (center[0] + sign * half * math.cos(math.radians(angle_deg)) / aspect, center[1] + sign * half * math.sin(math.radians(angle_deg)))
    return hand_factory(center=pos, aspect=aspect)  # type: ignore[operator]


def make_result(
    hands: list[HandLandmarks], sequence: int, timestamp: float, size: tuple[int, int] = (640, 480)
) -> TrackingResult:
    return TrackingResult(
        sequence=sequence,
        timestamp=timestamp,
        hands=tuple(hands),
        inference_ms=5.0,
        frame_size=size,
        preview_rgb=None,
        pipeline_fps=30.0,
    )
