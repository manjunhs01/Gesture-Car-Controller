"""Brake (open palm), reverse (two open palms held) and nitro (thumbs-up on one or both hands), with frame confirmation and release hysteresis.

An open palm is decided from finger states only (all five fingers extended), never from the wheel angle. One palm on either
hand brakes; both hands open brakes and, held for ``reverse_after_s``, reverses for as long as it is held. Brake (and so
reverse) always wins over nitro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gesture_car.backend.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandFeatures,
    HandLandmarks,
    compute_features,
    euclidean,
    to_isotropic,
)
from gesture_car.backend.steering import LEVEL_DEG, wheel_angle_deg
from gesture_car.config import GestureConfig

WHEEL_GRIP_MAX_FINGERS = 2  # a hand with at most this many extended fingers is gripping the wheel, not showing a palm


@dataclass(frozen=True)
class HandPose:
    """Per-frame, per-hand result before any temporal filtering."""

    open_palm: bool
    thumbs_up: bool
    fingers: int  # how many fingers (thumb included) are extended, 0..5
    spread: float  # mean fingertip-gap / knuckle-gap (1.0 = parallel fingers)
    thumb_tilt_deg: float  # angle of the thumb from straight up (0 = up, 180 = down)


def finger_spread(pts: np.ndarray) -> float:
    """How far the fingers fan out: mean over adjacent finger pairs of (tip distance / knuckle distance)."""
    pairs = ((INDEX_TIP, MIDDLE_TIP, INDEX_MCP, MIDDLE_MCP), (MIDDLE_TIP, RING_TIP, MIDDLE_MCP, RING_MCP), (RING_TIP, PINKY_TIP, RING_MCP, PINKY_MCP))
    ratios = [euclidean(pts[t1, :2], pts[t2, :2]) / max(euclidean(pts[m1, :2], pts[m2, :2]), 1e-6) for t1, t2, m1, m2 in pairs]
    return float(np.mean(ratios))


def unroll(pts: np.ndarray, roll_deg: float) -> np.ndarray:
    """Points turned back by ``roll_deg`` (clockwise on screen = positive) around the wrist, so "up" means the wheel's up."""
    if roll_deg == 0.0:
        return pts
    a = math.radians(-roll_deg)
    c, s = math.cos(a), math.sin(a)
    out = pts.copy()
    x, y = pts[:, 0] - pts[WRIST, 0], pts[:, 1] - pts[WRIST, 1]
    out[:, 0] = pts[WRIST, 0] + x * c - y * s  # image y grows downwards, so this turns clockwise for a > 0
    out[:, 1] = pts[WRIST, 1] + x * s + y * c
    return out


def thumb_tilt_deg(pts: np.ndarray) -> float:
    """Angle between the thumb (MCP -> tip) and straight up on the screen, in degrees."""
    v = pts[THUMB_TIP, :2] - pts[THUMB_MCP, :2]
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return 180.0
    cos_up = float(-v[1] / norm)  # image y grows downwards
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_up))))


def classify_hand(
    hand: HandLandmarks, cfg: GestureConfig, features: HandFeatures | None = None, roll_deg: float = 0.0, min_thumb: float | None = None
) -> HandPose:
    """Open palm = all five fingers extended (optionally also spread). Thumbs-up = thumb up, other four curled.

    ``roll_deg`` measures the thumbs-up "up" relative to a wheel rolled that far clockwise (used whenever both hands are
    seen); ``min_thumb`` overrides ``thumbs_up_min_thumb`` (only for the two-hand thumbs-up, where the hands confirm each other).
    """
    feats = features or compute_features(hand)
    pts = unroll(to_isotropic(hand), roll_deg)
    ext = feats.finger_extension  # thumb, index, middle, ring, pinky
    spread = finger_spread(pts)
    tilt = thumb_tilt_deg(pts)
    fingers = int(ext[0] >= cfg.thumb_extended_min) + sum(e >= cfg.finger_extended_min for e in ext[1:])
    open_palm = fingers == 5 and spread >= cfg.palm_min_spread
    others_curled = max(ext[1:]) <= cfg.thumbs_up_max_finger
    # the thumb tip must also be the highest point of the hand, so a sideways fist with a relaxed thumb is not a "thumbs-up"
    highest = pts[THUMB_TIP, 1] <= min(pts[i, 1] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST)) - 0.1 * feats.palm_size
    thumb_min = cfg.thumbs_up_min_thumb if min_thumb is None else min_thumb
    thumbs_up = ext[0] >= thumb_min and others_curled and tilt <= cfg.thumbs_up_max_tilt_deg and bool(highest)
    return HandPose(open_palm=open_palm and not thumbs_up, thumbs_up=thumbs_up, fingers=fingers, spread=spread, thumb_tilt_deg=tilt)


class Latch:
    """On after ``on_frames`` consecutive positive frames; off only after ``release_s`` without a positive frame."""

    def __init__(self, on_frames: int, release_s: float) -> None:
        self.on_frames, self.release_s = on_frames, release_s
        self.reset()

    def reset(self) -> None:
        self.active = False
        self._streak = 0
        self._last_seen = -1e9

    def update(self, present: bool, t: float) -> bool:
        if present:
            self._streak += 1
            self._last_seen = t
            if self._streak >= self.on_frames:
                self.active = True
        else:
            self._streak = 0
            if self.active and t - self._last_seen > self.release_s:
                self.active = False
        return self.active


@dataclass(frozen=True)
class GestureState:
    brake: bool = False  # confirmed: one or two open palms
    nitro: bool = False
    raw_brake: bool = False  # this frame, before confirmation
    raw_nitro: bool = False
    raw_palms: bool = False  # this frame: both hands are open palms
    palms: bool = False  # confirmed two open palms (with release hysteresis)
    reverse: bool = False  # two palms held for at least reverse_after_s
    hold_s: float = 0.0  # how long the two palms have been held
    thumbs: tuple[bool | None, bool | None] = (None, None)  # thumbs-up this frame, left / right hand in the image (--debug)
    fingers: tuple[int | None, int | None] = (None, None)  # extended fingers of the left / right hand in the image


class GestureEngine:
    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._brake = Latch(cfg.confirm_frames, cfg.release_s)
        self._palms = Latch(cfg.confirm_frames, cfg.release_s)
        self._palms_since: float | None = None
        self._nitro = Latch(cfg.confirm_frames, cfg.release_s)
        self._nitro_was_on = False
        self._nitro_toggled = False
        self._pulse_until = 0.0

    def reset(self) -> None:
        self._brake.reset()
        self._palms.reset()
        self._palms_since = None
        self._nitro.reset()
        self._nitro_was_on = False
        self._nitro_toggled = False
        self._pulse_until = 0.0

    def _nitro_output(self, held: bool, t: float) -> bool:
        """Apply the configured nitro mode to the debounced thumbs-up."""
        rising = held and not self._nitro_was_on
        self._nitro_was_on = held
        mode = self.cfg.nitro_mode
        if mode == "toggle":
            if rising:
                self._nitro_toggled = not self._nitro_toggled
            return self._nitro_toggled
        if mode == "pulse":
            if rising:
                self._pulse_until = t + self.cfg.nitro_pulse_s
            return t < self._pulse_until
        return held

    def update(self, hands: tuple[HandLandmarks, ...], t: float) -> GestureState:
        usable = sorted((h for h in hands if h.score >= self.cfg.min_hand_score), key=lambda h: float(h.points[WRIST, 0]))
        poses = [classify_hand(h, self.cfg) for h in usable]  # left to right in the image, whatever MediaPipe calls them
        raw_brake = any(p.open_palm for p in poses)
        raw_palms = len(poses) == 2 and all(p.open_palm for p in poses)
        thumbs = [p.thumbs_up for p in poses]
        if len(poses) == 2 and not all(thumbs):
            # the hands roll with the wheel, so a thumbs-up made while steering leans off the screen's up: each hand is
            # also judged against the wheel's up (normal thresholds), which makes one thumbs-up work on a turned wheel
            angle = wheel_angle_deg(tuple(usable))
            roll = 0.0 if angle is None else angle - LEVEL_DEG
            thumbs = [t or classify_hand(h, self.cfg, roll_deg=roll).thumbs_up for t, h in zip(thumbs, usable, strict=True)]
            # two thumbs-up: each thumb only needs to count as extended; the two hands agreeing is the confirmation
            relaxed = [classify_hand(h, self.cfg, roll_deg=roll, min_thumb=self.cfg.thumb_extended_min).thumbs_up for h in usable]
            if all(relaxed):
                thumbs = relaxed
        raw_nitro = any(thumbs)
        if len(poses) == 2 and all(p.fingers <= WHEEL_GRIP_MAX_FINGERS for p in poses):
            # both hands clearly back on the wheel: that is a positive "released", not a missed palm frame, so brake and
            # reverse stop on this frame instead of waiting out the anti-flicker hysteresis
            self._palms.reset()
            self._brake.reset()
        palms = self._palms.update(raw_palms, t)
        if palms:
            self._palms_since = t if self._palms_since is None else self._palms_since
        else:
            self._palms_since = None  # released (past the hysteresis): the hold starts again from zero
        hold = t - self._palms_since if self._palms_since is not None else 0.0
        brake = self._brake.update(raw_brake, t) or palms
        nitro = self._nitro.update(raw_nitro, t)
        fingers = tuple(p.fingers for p in poses) if len(poses) == 2 else (poses[0].fingers, None) if poses else (None, None)
        side_thumbs = tuple(thumbs) if len(poses) == 2 else (thumbs[0], None) if poses else (None, None)
        return GestureState(
            brake=brake,
            nitro=self._nitro_output(nitro, t) and not brake,  # brake (and reverse) beat nitro
            raw_brake=raw_brake,
            raw_nitro=raw_nitro,
            raw_palms=raw_palms,
            palms=palms,
            reverse=palms and hold >= self.cfg.reverse_after_s,
            hold_s=hold,
            thumbs=side_thumbs,  # type: ignore[arg-type]
            fingers=fingers,  # type: ignore[arg-type]
        )
