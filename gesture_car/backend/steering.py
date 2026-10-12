"""Two-hand imaginary steering wheel: hand positions -> wheel angle -> steering value in [-1, 1].

The wheel angle comes from the line between the two hands, in degrees where a level wheel (hands side by side) is 90.
Clockwise rotation as seen by the driver (right hand goes down, left hand goes up) increases it: 120 steers right, 60 left.
Hands are ordered by their x position in the (mirrored) image, so the result does not depend on MediaPipe's left/right labels.

Neutral is calibrated: the average wheel angle over a short steady hold at start-up (and on F9), then slowly re-centred
while the wheel is held near straight. Steering is purely proportional and stateless: ``offset`` = the shortest signed
difference between the angle and neutral, and steering = ``clamp(offset / full_lock_deg, -1, 1)``, exactly 0 inside the
tiny dead zone (``deadzone_deg``). Nothing is latched: the same hand pose always gives the same steering.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from gesture_car.backend.filters import EmaFilter, OneEuroFilter, Smoother
from gesture_car.backend.landmarks import PALM_POINTS, WRIST, HandLandmarks
from gesture_car.config import SteeringConfig


def hand_anchor(hand: HandLandmarks, anchor: str) -> tuple[float, float]:
    """Hand position in isotropic image units (x scaled by the aspect ratio so angles are true angles)."""
    pts = hand.points
    xy = pts[WRIST, :2] if anchor == "wrist" else pts[list(PALM_POINTS), :2].mean(axis=0)
    return float(xy[0] * hand.aspect), float(xy[1])


LEVEL_DEG = 90.0  # wheel angle of a level wheel; clockwise (right) rotation increases it
LINE_PERIOD_DEG = 180.0  # hands are sorted by x, so the wheel line has no direction: angles repeat every 180 deg
RECENTER_TAU_S = 2.0  # auto-recenter: neutral closes the remaining gap with this time constant
MAX_PREDICT_DT_S = 0.1  # prediction never extrapolates further than this (after a dropped frame)


def wheel_angle_deg(hands: tuple[HandLandmarks, ...], anchor: str = "palm", min_separation: float = 0.0) -> float | None:
    """Wheel angle in degrees (90 = level, larger = clockwise = right, 0..180) or None if it cannot be measured."""
    if len(hands) != 2:
        return None
    a, b = sorted((hand_anchor(h, anchor) for h in hands), key=lambda p: p[0])
    dx, dy = b[0] - a[0], b[1] - a[1]  # image y grows downwards: right hand lower -> dy > 0 -> clockwise
    if math.hypot(dx, dy) < max(min_separation, 1e-6):
        return None
    return LEVEL_DEG + math.degrees(math.atan2(dy, dx))


def shortest_offset(angle_deg: float, neutral_deg: float, period_deg: float = LINE_PERIOD_DEG) -> float:
    """Signed difference ``angle - neutral`` wrapped into [-period/2, period/2), so wrapping never causes a jump.

    The default period is 180 because the wheel line has no direction: angle 2 with neutral 178 is +4, not -176.
    """
    half = period_deg / 2.0
    return (angle_deg - neutral_deg + half) % period_deg - half


def steering_from_offset(offset_deg: float, deadzone_deg: float, full_lock_deg: float) -> float:
    """Purely proportional: ``clamp(offset / full_lock_deg, -1, 1)``; exactly 0 inside the dead zone.

    With full lock at 35 deg: 5 deg -> 14%, 17.5 deg -> 50%, 35 deg or more -> 100%.
    """
    if abs(offset_deg) < deadzone_deg:
        return 0.0
    return min(1.0, max(-1.0, offset_deg / max(full_lock_deg, 1e-6)))


def make_smoother(cfg: SteeringConfig) -> Smoother:
    if cfg.smoothing == "ema":
        return EmaFilter(cfg.ema_alpha)
    return OneEuroFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)


@dataclass
class CalibrationProgress:
    done: bool
    elapsed: float


GAP_RESET_S = 0.5  # a hand pair missing longer than this restarts the hold; shorter dropouts are ignored
MIN_SAMPLES = 12  # a steady window needs at least this many measured frames
SPREAD_LOW, SPREAD_HIGH = 10, 90  # percentiles used for the spread (ignores a few outlier frames)
NEUTRAL_INLIER_DEG = 3.0  # frames this close to the median are averaged into the neutral


class NeutralCalibrator:
    """Finds the neutral wheel angle: a window of ``seconds`` in which the angle stays within ``max_jitter_deg``.

    Real hands shake and MediaPipe drops a hand for a frame now and then, so this is deliberately tolerant: short dropouts
    (< ``GAP_RESET_S``) are ignored, the spread is measured between the 10th and 90th percentile (a few bad frames do not
    matter) and the window slides, so it finishes at the first steady ``seconds``-long stretch. The neutral is the average
    angle over that stretch, leaving out frames more than ``NEUTRAL_INLIER_DEG`` from its median (hands still settling).
    """

    def __init__(self, seconds: float, max_jitter_deg: float) -> None:
        self.seconds, self.max_jitter = seconds, max_jitter_deg
        self.neutral: float | None = None
        self.status = "waiting for both hands"
        self._samples: deque[tuple[float, float]] = deque()
        self._last_t: float | None = None

    def reset(self) -> None:
        self.neutral = None
        self._samples.clear()
        self._last_t = None
        self.status = "waiting for both hands"

    def update(self, angle: float | None, t: float) -> CalibrationProgress:
        if self.neutral is not None:
            return CalibrationProgress(True, self.seconds)
        if angle is None:
            if self._last_t is not None and t - self._last_t > GAP_RESET_S:
                self._samples.clear()
                self._last_t = None
            return CalibrationProgress(False, self._elapsed(t))
        if self._last_t is not None and t - self._last_t > GAP_RESET_S:
            self._samples.clear()
        self._last_t = t
        self._samples.append((t, angle))
        while self._samples and t - self._samples[0][0] > self.seconds:
            self._samples.popleft()
        elapsed = self._elapsed(t)
        if elapsed >= self.seconds * 0.95 and len(self._samples) >= MIN_SAMPLES:
            first = self._samples[0][1]
            values = np.array([first + shortest_offset(a, first) for _, a in self._samples])  # continuous across the wrap
            spread = float(np.percentile(values, SPREAD_HIGH) - np.percentile(values, SPREAD_LOW))
            if spread <= self.max_jitter:
                steady = values[np.abs(values - np.median(values)) <= NEUTRAL_INLIER_DEG]
                self.neutral = float(np.mean(steady))
                self.status = f"calibrated (neutral {self.neutral:.1f} deg)"
                return CalibrationProgress(True, elapsed)
            self.status = f"hold steadier (moving {spread:.0f} deg, need under {self.max_jitter:.0f})"
        else:
            self.status = f"holding {elapsed:.1f}/{self.seconds:.1f} s"
        return CalibrationProgress(False, elapsed)

    def _elapsed(self, t: float) -> float:
        return t - self._samples[0][0] if self._samples else 0.0


def describe_pair(hands: tuple[HandLandmarks, ...], anchor: str, min_separation: float) -> str:
    """Human-readable reason why the wheel angle can or cannot be measured right now."""
    if len(hands) != 2:
        return f"need both hands in view (seeing {len(hands)})"
    a, b = sorted((hand_anchor(h, anchor) for h in hands), key=lambda p: p[0])
    if math.hypot(b[0] - a[0], b[1] - a[1]) < max(min_separation, 1e-6):
        return "hands too close together - hold them wider apart"
    return "ok"


class SteeringEstimator:
    """Calibrated, lightly filtered steering from successive hand pairs.

    No latch, rate limit or "last direction" memory: every frame the steering is recomputed from the current hands. The
    only state is the neutral angle, the One Euro filter and a short output history used to hold the wheel while a hand
    is off it.
    """

    def __init__(self, cfg: SteeringConfig) -> None:
        self.cfg = cfg
        self.calibrator = NeutralCalibrator(cfg.calibration_seconds, cfg.calibration_max_jitter_deg)
        self._smoother = make_smoother(cfg)
        self._history: deque[tuple[float, float]] = deque()  # (t, steering) for the hand-off-the-wheel look-back
        self.value = 0.0
        self.wheel_angle: float | None = None  # last measured wheel angle (90 = level), unfiltered
        self.raw_offset: float | None = None  # last unfiltered offset from neutral (degrees, + = right)
        self.angle: float | None = None  # last filtered + predicted offset from neutral (degrees, + = right)
        self._last_t: float | None = None  # previous measured frame, for the prediction step
        self._near_since: float | None = None  # auto-recenter: since when the wheel has been held near straight

    @property
    def calibrated(self) -> bool:
        return self.calibrator.neutral is not None

    def recalibrate(self) -> None:
        self.calibrator.reset()
        self.reset_filter()

    def reset_filter(self) -> None:
        """Forget filter history (used after the hands were lost so stale motion does not leak in)."""
        self._smoother.reset()
        self._history.clear()
        self.value = 0.0
        self.wheel_angle = None
        self.raw_offset = None
        self.angle = None
        self._last_t = None
        self._near_since = None

    def set_neutral(self, angle: float) -> None:
        self.calibrator.neutral = angle

    def update(self, hands: tuple[HandLandmarks, ...], t: float) -> float | None:
        """Returns the steering value, or None while uncalibrated / when the hand pair is not measurable."""
        raw = wheel_angle_deg(hands, self.cfg.anchor, self.cfg.min_hand_separation)
        self.wheel_angle = raw
        if not self.calibrated:
            self.calibrator.update(raw, t)
            self.value = 0.0
            return None
        if raw is None:
            self.raw_offset = None
            return None
        neutral = self.calibrator.neutral
        assert neutral is not None
        offset = shortest_offset(raw, neutral)
        self.raw_offset = offset
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), MAX_PREDICT_DT_S)
        self._last_t = t
        self._auto_recenter(offset, t, dt)
        filtered = self._smoother.update(offset, t)
        predicted = filtered + self._smoother.velocity * dt * self.cfg.predict_frames  # one frame ahead
        self.angle = predicted
        if abs(offset) < self.cfg.deadzone_deg or abs(filtered) < self.cfg.deadzone_deg:
            # hands straight right now: exactly 0, immediately, whatever the filter still remembers; and a single noisy
            # frame cannot leave the dead zone on its own (the filtered angle must agree)
            value = 0.0
        else:
            value = steering_from_offset(predicted, self.cfg.deadzone_deg, self.cfg.full_lock_deg)
        value = -value if self.cfg.invert_steering else value
        self.value = value
        self._history.append((t, value))
        while self._history and t - self._history[0][0] > 1.0:
            self._history.popleft()
        return value

    def _auto_recenter(self, offset: float, t: float, dt: float) -> None:
        """Slow drift correction: held within ``auto_recenter_deg`` of neutral for ``auto_recenter_after_s``, the neutral
        is nudged toward the current angle (time constant ``RECENTER_TAU_S``). A real turn never moves it."""
        window = self.cfg.auto_recenter_deg
        if window <= 0.0 or abs(offset) > window:
            self._near_since = None
            return
        if self._near_since is None:
            self._near_since = t
        if t - self._near_since >= self.cfg.auto_recenter_after_s and self.calibrator.neutral is not None:
            self.calibrator.neutral += offset * min(1.0, dt / RECENTER_TAU_S)

    def value_before(self, t: float, lookback_s: float) -> float:
        """Steering from ``lookback_s`` ago: used to hold the wheel while a hand is off it (out of view) for brake/nitro.

        With no history that old (just after the hands came back) the oldest known value is held, never a forced 0.
        """
        target = t - lookback_s
        chosen = self._history[0][1] if self._history else self.value
        for ts, v in self._history:
            if ts > target:
                break
            chosen = v
        return chosen
