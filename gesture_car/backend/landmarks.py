"""Hand-landmark geometry: distances, joint angles, finger extension, orientation.

All measurements are normalised by hand size and computed in an isotropic
coordinate frame (x scaled by the image aspect ratio), so results do not depend
on camera resolution, aspect ratio or distance from the camera.

Landmark numbering follows MediaPipe Hands (21 points).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gesture_car.backend import constants as C

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
# (mcp, pip, dip, tip) of the four non-thumb fingers
FINGER_CHAINS = (
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
# Skeleton connections used by the preview overlay
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)  # fmt: skip

PALM_WIDTH_TO_LENGTH = 0.75  # index-MCP..pinky-MCP width relative to wrist..middle-MCP


@dataclass(frozen=True)
class HandLandmarks:
    """One detected hand.

    ``points`` is a (21, 3) float array: x, y in [0, 1] of the (mirrored) image
    and z as MediaPipe's relative depth. ``aspect`` is image width / height.
    """

    points: np.ndarray
    handedness: str = "Unknown"
    score: float = 1.0
    aspect: float = 4.0 / 3.0

    def __post_init__(self) -> None:
        if self.points.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {self.points.shape}")


@dataclass(frozen=True)
class FingerThresholds:
    """Calibrated ranges used to turn angles / ratios into a 0..1 extension score."""

    angle_curled_deg: float = C.FINGER_ANGLE_CURLED_DEG
    angle_straight_deg: float = C.FINGER_ANGLE_STRAIGHT_DEG
    ratio_curled: float = C.FINGER_RATIO_CURLED
    ratio_straight: float = C.FINGER_RATIO_STRAIGHT
    thumb_dist_curled: float = C.THUMB_DIST_CURLED
    thumb_dist_straight: float = C.THUMB_DIST_STRAIGHT


@dataclass(frozen=True)
class HandFeatures:
    """Everything the gesture engine needs to know about a hand in one frame."""

    palm_center: tuple[float, float]  # normalised image coordinates
    wrist: tuple[float, float]
    fingertips: tuple[tuple[float, float], ...]
    palm_size: float  # isotropic units (fractions of image height)
    finger_extension: tuple[float, ...]  # thumb, index, middle, ring, pinky in 0..1
    rotation_deg: float  # 0 = fingers up, positive = clockwise on screen
    palm_normal_z: float  # sign tells which side of the palm faces the camera
    bbox: tuple[float, float, float, float] = field(default=(0.0, 0.0, 0.0, 0.0))

    @property
    def four_finger_extension(self) -> tuple[float, ...]:
        """Index..pinky extension (the thumb is too variable to classify with)."""
        return self.finger_extension[1:]


# ----------------------------------------------------------------------- maths
def clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def ramp(value: float, low: float, high: float) -> float:
    """Linear 0..1 ramp: 0 at/below ``low``, 1 at/above ``high``."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return clip01((value - low) / (high - low))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` (degrees, 0..180) between segments b->a and b->c.

    A fully straight joint is 180 degrees. Degenerate segments return 180.
    """
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def to_isotropic(hand: HandLandmarks) -> np.ndarray:
    """Landmarks as a (21, 3) array with x and z scaled by the aspect ratio."""
    scale = np.array([hand.aspect, 1.0, hand.aspect], dtype=np.float64)
    return hand.points.astype(np.float64) * scale


def palm_size(pts: np.ndarray) -> float:
    """Hand-size reference: robust to foreshortening of either palm dimension."""
    length = euclidean(pts[WRIST, :2], pts[MIDDLE_MCP, :2])
    width = euclidean(pts[INDEX_MCP, :2], pts[PINKY_MCP, :2]) / PALM_WIDTH_TO_LENGTH
    return max(length, width, 1e-6)


def finger_extension(
    pts: np.ndarray, chain: tuple[int, int, int, int], thresholds: FingerThresholds
) -> float:
    """0 (fully curled) .. 1 (fully straight) for index/middle/ring/pinky."""
    mcp, pip, dip, tip = chain
    pip_angle = joint_angle_deg(pts[mcp], pts[pip], pts[dip])
    dip_angle = joint_angle_deg(pts[pip], pts[dip], pts[tip])
    angle = 0.65 * pip_angle + 0.35 * dip_angle
    reach_pip = max(euclidean(pts[WRIST], pts[pip]), 1e-6)
    ratio = euclidean(pts[WRIST], pts[tip]) / reach_pip
    angle_score = ramp(angle, thresholds.angle_curled_deg, thresholds.angle_straight_deg)
    ratio_score = ramp(ratio, thresholds.ratio_curled, thresholds.ratio_straight)
    return 0.5 * angle_score + 0.5 * ratio_score


def thumb_extension(pts: np.ndarray, size: float, thresholds: FingerThresholds) -> float:
    """Thumb extension from its IP angle and tip distance to the pinky knuckle."""
    angle = joint_angle_deg(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP])
    dist = euclidean(pts[THUMB_TIP], pts[PINKY_MCP]) / size
    angle_score = ramp(angle, 110.0, 160.0)
    dist_score = ramp(dist, thresholds.thumb_dist_curled, thresholds.thumb_dist_straight)
    return 0.5 * angle_score + 0.5 * dist_score


def compute_features(
    hand: HandLandmarks, thresholds: FingerThresholds | None = None
) -> HandFeatures:
    """Derive position, size, finger states and orientation from raw landmarks."""
    thresholds = thresholds or FingerThresholds()
    pts = to_isotropic(hand)
    size = palm_size(pts)
    extension = [thumb_extension(pts, size, thresholds)]
    extension += [finger_extension(pts, chain, thresholds) for chain in FINGER_CHAINS]

    norm = hand.points
    palm_center = norm[list(PALM_POINTS), :2].mean(axis=0)
    xy_min = norm[:, :2].min(axis=0)
    xy_max = norm[:, :2].max(axis=0)

    axis = pts[MIDDLE_MCP, :2] - pts[WRIST, :2]
    rotation = math.degrees(math.atan2(axis[0], -axis[1]))

    a = pts[INDEX_MCP, :2] - pts[WRIST, :2]
    b = pts[PINKY_MCP, :2] - pts[WRIST, :2]
    cross = float(a[0] * b[1] - a[1] * b[0])
    mag = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    normal_z = cross / mag
    if hand.handedness.lower() == "left":
        normal_z = -normal_z

    return HandFeatures(
        palm_center=(float(palm_center[0]), float(palm_center[1])),
        wrist=(float(norm[WRIST, 0]), float(norm[WRIST, 1])),
        fingertips=tuple((float(norm[i, 0]), float(norm[i, 1])) for i in FINGERTIPS),
        palm_size=size,
        finger_extension=tuple(float(e) for e in extension),
        rotation_deg=rotation,
        palm_normal_z=normal_z,
        bbox=(float(xy_min[0]), float(xy_min[1]), float(xy_max[0]), float(xy_max[1])),
    )
