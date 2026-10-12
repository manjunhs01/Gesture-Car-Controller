import dataclasses
import math

import pytest

from gesture_car.backend.steering import (
    NeutralCalibrator,
    SteeringEstimator,
    shortest_offset,
    steering_from_offset,
    wheel_angle_deg,
)
from gesture_car.config import SteeringConfig
from tests.support.synthetic_hand import wheel_pair

DZ, FULL = 1.0, 30.0  # the defaults: steering = clamp(offset / 30), exactly 0 inside 1 deg
DT = 1 / 30


def s(offset):
    return steering_from_offset(offset, DZ, FULL)


def test_defaults_match_the_spec():
    cfg = SteeringConfig()
    assert (cfg.full_lock_deg, cfg.deadzone_deg, cfg.invert_steering) == (FULL, DZ, False)
    assert (cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff) == (0.9, 0.08, 1.0)
    assert (cfg.predict_frames, cfg.auto_recenter_deg, cfg.auto_recenter_after_s) == (0.5, 6.0, 3.0)
    assert not hasattr(cfg, "turn_rate_per_s") and not hasattr(cfg, "return_rate_per_s")  # no rate limit / step


# ------------------------------------------------------------------ mapping
@pytest.mark.parametrize(
    ("offset", "expected"),
    [(0, 0.0), (0.5, 0.0), (5, 5 / 30), (10, 10 / 30), (15, 0.5), (30, 1.0), (50, 1.0), (89, 1.0)],
)
def test_purely_proportional(offset, expected):
    assert s(offset) == pytest.approx(expected)
    assert s(-offset) == pytest.approx(-expected)  # left is the exact mirror of right


def test_spec_values():
    assert s(0) == 0.0
    assert s(0.5) == 0.0 and s(-0.5) == 0.0
    assert s(1) == pytest.approx(0.033, abs=0.001) and s(-1) == pytest.approx(-0.033, abs=0.001)  # a slight tilt counts
    assert s(5) == pytest.approx(0.167, abs=0.005) and s(-5) == pytest.approx(-0.167, abs=0.005)
    assert s(15) == pytest.approx(0.5) and s(-15) == pytest.approx(-0.5)
    assert s(30) == 1.0 and s(-30) == -1.0 and s(35) == 1.0 and s(70) == 1.0 and s(-70) == -1.0


def test_only_a_tiny_dead_zone():
    assert s(0.9) == 0.0 and s(-0.9) == 0.0
    assert s(1.0) == pytest.approx(1.0 / 30)  # no step to bridge: offset / 30 straight from the dead-zone edge
    assert s(-1.1) == pytest.approx(-1.1 / 30)


def test_monotonic_and_clamped():
    values = [s(a / 10) for a in range(-900, 901)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    assert min(values) == -1.0 and max(values) == 1.0


# ------------------------------------------------------------------ geometry
def test_wheel_angle_matches_geometry_and_direction():
    for rotation in (-40, -10, 0, 10, 40):
        measured = wheel_angle_deg(wheel_pair(rotation))
        assert measured == pytest.approx(90 + rotation, abs=0.5)  # level wheel = 90
    assert wheel_angle_deg(wheel_pair(20)) > 90  # clockwise = right


@pytest.mark.parametrize("rotation", [-60, -25, -5, 0, 5, 25, 60])
def test_hand_order_and_handedness_labels_never_change_the_angle(rotation):
    lh, rh = wheel_pair(rotation)
    lh, rh = dataclasses.replace(lh, handedness="Left"), dataclasses.replace(rh, handedness="Right")
    expected = wheel_angle_deg((lh, rh))
    swapped_labels = (dataclasses.replace(lh, handedness="Right"), dataclasses.replace(rh, handedness="Left"))
    same_labels = (dataclasses.replace(lh, handedness="Right"), rh)
    assert wheel_angle_deg((rh, lh)) == pytest.approx(expected)  # order MediaPipe reports them in
    assert wheel_angle_deg(swapped_labels) == pytest.approx(expected)  # labels flipped (MediaPipe does this)
    assert wheel_angle_deg(swapped_labels[::-1]) == pytest.approx(expected)
    assert wheel_angle_deg(same_labels) == pytest.approx(expected)  # both called "Right"


def test_wheel_angle_needs_exactly_two_separated_hands():
    lh, rh = wheel_pair(0)
    assert wheel_angle_deg((lh,)) is None
    assert wheel_angle_deg(()) is None
    assert wheel_angle_deg(wheel_pair(0, separation=0.05), min_separation=0.12) is None


@pytest.mark.parametrize(
    ("angle", "neutral", "expected"),
    [
        (90, 90, 0), (100, 90, 10), (80, 90, -10),
        (2, 178, 4), (178, 2, -4), (179, 1, -2), (1, 179, 2), (0, 180, 0), (180, 0, 0),  # wrap-around near 180
        (175, 5, -10), (5, 175, 10),
    ],
)  # fmt: skip
def test_shortest_offset_wraps_without_jumps(angle, neutral, expected):
    assert shortest_offset(angle, neutral) == pytest.approx(expected)


def test_shortest_offset_with_a_full_circle_period():
    assert shortest_offset(-179, 179, period_deg=360) == pytest.approx(2)
    assert shortest_offset(179, -179, period_deg=360) == pytest.approx(-2)
    assert shortest_offset(10, 350, period_deg=360) == pytest.approx(20)


def test_shortest_offset_is_continuous_across_the_wrap():
    """Sweeping the raw angle through 180/0 never makes the offset jump."""
    neutral = 178.0
    offsets = [shortest_offset((neutral + d / 10) % 180, neutral) for d in range(-300, 301)]
    assert max(abs(b - a) for a, b in zip(offsets, offsets[1:], strict=False)) < 0.2


# ------------------------------------------------------------------ estimator
def _calibrated(cfg=None, neutral_angle=0.0):
    est = SteeringEstimator(cfg or SteeringConfig())
    est.set_neutral(wheel_angle_deg(wheel_pair(neutral_angle)))
    return est


def _settle(est, rotation, t, frames=20):
    for _ in range(frames):
        t += DT
        est.update(wheel_pair(rotation), t)
    return t


@pytest.mark.parametrize("offset", [0, 1, 5, 10, 17.5, 35, 60, -1, -5, -10, -17.5, -35, -60])
def test_estimator_steers_relative_to_the_calibrated_neutral(offset):
    """Real hand geometry, a driver whose 'straight' is tilted 7 deg: steering follows offset / 30 from that pose."""
    est = _calibrated(neutral_angle=7.0)
    _settle(est, 7.0 + offset, 0.0, frames=60)
    assert est.value == pytest.approx(s(offset), abs=0.01)


def test_estimator_near_the_180_wrap():
    """Neutral just below 180 and hands just past it (raw angle near 0): +4 deg, not -176."""
    est = SteeringEstimator(SteeringConfig())
    est.set_neutral(178.0)
    _settle(est, 2.0 - 90.0, 0.0, frames=30)  # wheel angle 2
    assert est.raw_offset == pytest.approx(4.0, abs=0.5)
    assert est.value == pytest.approx(4.0 / 30, abs=0.015)


def test_straight_hands_give_exactly_zero_immediately():
    est = _calibrated()
    t = _settle(est, 30, 0.0, frames=30)
    assert est.value > 0.8
    t += DT
    assert est.update(wheel_pair(0.5), t) == 0.0  # the very first straight frame, no filter tail
    t += DT
    assert est.update(wheel_pair(-0.5), t) == 0.0


def test_no_latch_after_turning_past_vertical_or_a_glitch():
    """The old unwrap latch stuck at full lock after a turn past vertical or one bad frame. Now nothing is remembered."""
    est = _calibrated()
    t = 0.0
    for rotation in range(0, 101, 5):  # big right turn, past vertical
        t = _settle(est, rotation, t, frames=1)
    t = _settle(est, 0, t, frames=1)
    assert est.value == 0.0
    t = _settle(est, 70, t, frames=15)  # hard right
    t = _settle(est, -80, t, frames=1)  # one bad frame (hand order flips)
    _settle(est, 0, t, frames=1)
    assert est.value == 0.0


def test_same_pose_always_gives_the_same_steering():
    """Stateless: approaching 10 deg from the left or from the right ends at the same value."""
    a, b = _calibrated(), _calibrated()
    _settle(a, _settle(a, -40, 0.0) and 10, 2.0, frames=30)
    _settle(b, _settle(b, 40, 0.0) and 10, 2.0, frames=30)
    assert a.value == pytest.approx(b.value, abs=1e-3) == pytest.approx(10 / 30, abs=0.01)


def test_invert_steering_swaps_directions():
    est = _calibrated(SteeringConfig(invert_steering=True))
    _settle(est, 15, 0.0, frames=60)
    assert est.value == pytest.approx(-0.5, abs=0.01)


def test_estimator_left_turn_is_negative_and_invert_flips():
    left = _calibrated()
    inv = _calibrated(SteeringConfig(invert_steering=True))
    t = 0.0
    for _ in range(60):
        t += DT
        left.update(wheel_pair(-35), t)
        inv.update(wheel_pair(-35), t)
    assert left.value < -0.9
    assert inv.value > 0.9


def test_turns_arrive_fast():
    """A step to 17.5 deg reaches 90% of its 50% within 3 frames (100 ms at 30 fps); a small 5 deg turn within 4."""
    for target, frames in ((17.5, 3), (5.0, 4)):
        est = _calibrated()
        t = _settle(est, 0, 0.0)
        _settle(est, target, t, frames=frames)
        assert est.value >= 0.9 * s(target), f"{target} deg: {est.value:.3f} after {frames} frames"


def test_prediction_reduces_lag_during_a_turn():
    def lag(predict):
        est = _calibrated(SteeringConfig(predict_frames=predict))
        t = _settle(est, 0, 0.0)
        for i in range(1, 11):  # turning at 60 deg/s
            t += DT
            est.update(wheel_pair(2.0 * i), t)
        return 20.0 - est.angle

    assert lag(1.0) < lag(0.0)


def test_smoothing_removes_jitter():
    est = _calibrated()
    t, outs = 0.0, []
    for i in range(120):
        t += DT
        jitter = 1.5 if i % 2 else -1.5  # +-1.5 degrees of landmark noise around a 20 degree turn
        outs.append(est.update(wheel_pair(20 + jitter), t))
    tail = outs[60:]
    assert max(tail) - min(tail) < 0.06


# ------------------------------------------------------------------ auto-recenter
def test_auto_recenter_absorbs_a_small_held_offset():
    est = _calibrated()
    neutral = est.calibrator.neutral
    t = _settle(est, 4.0, 0.0, frames=85)  # 2.8 s: not yet
    assert est.calibrator.neutral == pytest.approx(neutral)
    assert est.value == pytest.approx(4 / 30, abs=0.01)
    _settle(est, 4.0, t, frames=8 * 30)  # held near straight: neutral follows
    assert est.calibrator.neutral == pytest.approx(neutral + 4.0, abs=0.5)
    assert est.value == 0.0


def test_auto_recenter_never_moves_neutral_during_a_real_turn():
    est = _calibrated()
    neutral = est.calibrator.neutral
    _settle(est, 10.0, 0.0, frames=10 * 30)
    assert est.calibrator.neutral == pytest.approx(neutral)
    assert est.value == pytest.approx(10 / 30, abs=0.01)


def test_auto_recenter_can_be_switched_off():
    est = _calibrated(SteeringConfig(auto_recenter_deg=0.0))
    neutral = est.calibrator.neutral
    _settle(est, 4.0, 0.0, frames=10 * 30)
    assert est.calibrator.neutral == pytest.approx(neutral)


# ------------------------------------------------------------------ calibration
def test_calibration_averages_the_held_angle():
    cal = NeutralCalibrator(2.0, 10.0)
    t = 0.0
    for i in range(70):
        t += DT
        cal.update(93.0 + (2.0 if i % 2 else -1.0), t)  # alternating 92 / 95: average 93.5, median would differ
    assert cal.neutral == pytest.approx(93.5, abs=0.1)


def test_calibration_across_the_180_wrap():
    cal = NeutralCalibrator(2.0, 10.0)
    t = 0.0
    for i in range(70):
        t += DT
        cal.update(179.0 if i % 2 else 1.0, t)  # the same line, either side of the wrap
    assert shortest_offset(cal.neutral, 180.0) == pytest.approx(0.0, abs=0.1)


def test_calibrator_needs_a_steady_hold():
    cal = NeutralCalibrator(2.0, 10.0)
    t = 0.0
    for _ in range(40):  # 1.33 s: not enough
        t += DT
        cal.update(5.0, t)
    assert cal.neutral is None
    for _ in range(40):
        t += DT
        cal.update(5.0, t)
    assert cal.neutral == pytest.approx(5.0)


def test_calibrator_tolerates_hand_shake_and_single_frame_dropouts():
    import random

    rng = random.Random(3)
    cal = NeutralCalibrator(2.0, 10.0)
    t = 0.0
    for i in range(150):
        t += DT
        if i % 7 == 3:
            cal.update(None, t)  # MediaPipe loses a hand for one frame now and then
        else:
            cal.update(8.0 + rng.uniform(-2.5, 2.5), t)  # +-2.5 deg of natural hand shake
        if cal.neutral is not None:
            break
    assert cal.neutral == pytest.approx(8.0, abs=1.5)
    assert t < 3.0  # finished soon after the 2 s hold


def test_calibrator_ignores_early_movement_and_uses_the_steady_part():
    cal = NeutralCalibrator(1.0, 10.0)
    t = 0.0
    for angle in [30.0, 25.0, 20.0, 10.0, 4.0] * 3:  # hands still moving into position
        t += DT
        cal.update(angle, t)
    assert cal.neutral is None
    for _ in range(60):
        t += DT
        cal.update(2.0, t)
    assert cal.neutral == pytest.approx(2.0, abs=0.5)


def test_calibrator_restarts_after_a_long_gap_and_reports_status():
    cal = NeutralCalibrator(1.0, 10.0)
    t = 0.0
    for _ in range(20):
        t += DT
        cal.update(0.0, t)
    t += 0.7  # hands out of view for 0.7 s
    cal.update(None, t)
    t += DT
    cal.update(1.0, t)
    assert cal.neutral is None and "holding" in cal.status
    for _ in range(40):
        t += DT
        cal.update(1.0, t)
    assert cal.neutral == pytest.approx(1.0, abs=0.3)


def test_describe_pair_explains_why():
    from gesture_car.backend.steering import describe_pair

    lh, rh = wheel_pair(0)
    assert "need both hands" in describe_pair((lh,), "palm", 0.08)
    assert describe_pair((lh, rh), "palm", 0.08) == "ok"
    assert "too close" in describe_pair(wheel_pair(0, separation=0.04), "palm", 0.08)


def test_estimator_calibrates_itself_from_frames_then_steers():
    est = SteeringEstimator(SteeringConfig(calibration_seconds=1.0))
    t = 0.0
    for _ in range(25):  # 0.83 s < 1 s hold
        t += DT
        assert est.update(wheel_pair(4.0), t) is None
    assert not est.calibrated
    for _ in range(15):
        t += DT
        est.update(wheel_pair(4.0), t)
    assert est.calibrated
    for _ in range(40):
        t += DT
        est.update(wheel_pair(4.0 + 15), t)
    assert est.value == pytest.approx(0.5, abs=0.03)


def test_value_before_returns_older_value():
    est = _calibrated()
    t = 0.0
    for _ in range(30):
        t += DT
        est.update(wheel_pair(0.0), t)
    early = est.value
    for _ in range(10):
        t += DT
        est.update(wheel_pair(40.0), t)
    assert abs(est.value_before(t, 0.5) - early) < 0.1
    assert est.value > 0.1
    assert math.isfinite(est.value_before(t, 0.0))


def test_feel_presets_only_change_smoothing():
    from gesture_car.config import apply_feel

    for feel in ("arcade", "normal", "realistic"):
        cfg = SteeringConfig(feel=feel)
        apply_feel(cfg)
        assert (cfg.full_lock_deg, cfg.deadzone_deg) == (FULL, DZ), feel
    normal = SteeringConfig(feel="normal")
    apply_feel(normal)
    assert (normal.one_euro_min_cutoff, normal.one_euro_beta) == (1.5, 0.05)


# ------------------------------------------------------------------ noisy tracking
def _jittered(hands, rng, sigma):
    from gesture_car.backend.landmarks import HandLandmarks

    out = []
    for h in hands:
        pts = h.points.copy()
        pts[:, :2] += rng.normal(0, sigma, 2) + rng.normal(0, sigma, (pts.shape[0], 2))  # whole-hand + per-point jitter
        out.append(HandLandmarks(points=pts, handedness=h.handedness, score=h.score, aspect=h.aspect))
    return tuple(out)


def test_default_steering_is_sensitive_to_small_turns():
    from gesture_car.config import OutputConfig

    assert OutputConfig().pwm_min_duty == 0.0  # the keyboard backend adds no threshold of its own
    est = _calibrated(SteeringConfig(auto_recenter_deg=0.0))
    for angle in (2, 5, 10):
        est.reset_filter()
        value = 0.0
        for i in range(30):
            value = est.update(wheel_pair(angle), i / 15) or 0.0
        assert value == pytest.approx(s(angle), abs=0.005), f"a {angle} deg turn gave {value:.3f}"


def test_holding_still_with_jittery_tracking_stays_near_zero():
    import numpy as np

    def run(sigma):
        rng = np.random.default_rng(7)
        est = _calibrated()
        return np.abs([est.update(_jittered(wheel_pair(0), rng, sigma), i / 30) or 0.0 for i in range(600)])  # 20 s

    typical = run(0.005)  # typical tracking noise (~0.9 deg): a rare tiny twitch at most
    assert np.mean(typical > 0) < 0.01 and typical.max() < 0.06
    heavy = run(0.01)  # heavy noise (~1.8 deg): rare, small twitches only (the price of a fast filter + prediction)
    assert heavy.max() < 0.15 and np.mean(heavy > 0) < 0.20
