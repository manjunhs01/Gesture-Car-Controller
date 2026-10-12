"""Brake and reverse from open palms, with synthetic landmarks: brake, reverse after the hold, release, flicker."""

import math

import pytest

from gesture_car.backend.controller import ControlLogic, debug_line
from gesture_car.backend.gestures import GestureEngine
from gesture_car.config import Config
from tests.support.synthetic_hand import fist, make_hand, place, thumbs_up, wheel_pair

DT = 1 / 30


def palm(x, y=0.42, thumb=0.8, fan=0.25):
    """A natural open palm: fingers straight, not fanned wide, thumb relaxed."""
    return make_hand(extension=1.0, thumb=thumb, fan=fan, center=(x, y))


TWO_PALMS = (palm(0.32), palm(0.68, 0.46))  # slightly uneven heights: the wheel angle would read about +7 deg


class Driver:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.logic = ControlLogic(self.cfg)
        self.t = 0.0
        self.feed(wheel_pair(0), 3.0)
        assert self.logic.steering.calibrated

    def feed(self, hands, seconds):
        states = []
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            states.append(self.logic.process(tuple(hands), self.t))
        return states


def test_two_palms_brake_within_two_frames_and_stop_acceleration():
    d = Driver()
    first, second, *_ = d.feed(TWO_PALMS, 0.3)
    assert first.brake == 0.0 and first.throttle == 1.0  # one frame is not enough (anti-flicker)
    assert second.brake == 1.0 and second.throttle == 0.0 and not second.nitro
    assert d.logic.last_gestures.palms and not d.logic.last_gestures.reverse


def test_holding_two_palms_longer_than_reverse_after_s_reverses_and_keeps_reversing():
    d = Driver()
    states = d.feed(TWO_PALMS, 0.5)
    assert not d.logic.last_gestures.reverse  # 0.5 s < 0.6 s
    states = d.feed(TWO_PALMS, 0.2)
    assert d.logic.last_gestures.reverse
    states = d.feed(TWO_PALMS, 3.0)  # keeps reversing as long as the pose is held
    assert all(d_.brake == 1.0 and d_.throttle == 0.0 for d_ in states)  # brake_hold: reverse = brake kept held
    assert d.logic.last_gestures.hold_s > 3.0


def test_reverse_starts_at_the_configured_hold_time():
    cfg = Config()
    cfg.gestures.reverse_after_s = 1.0
    d = Driver(cfg)
    d.feed(TWO_PALMS, 0.9)
    assert not d.logic.last_gestures.reverse
    d.feed(TWO_PALMS, 0.2)
    assert d.logic.last_gestures.reverse


def test_back_on_the_wheel_stops_reverse_and_brake_at_once_and_acceleration_resumes():
    d = Driver()
    d.feed(TWO_PALMS, 1.5)
    assert d.logic.last_gestures.reverse
    state = d.feed(wheel_pair(0), DT)[0]  # the very first frame with both fists on the wheel
    assert state.brake == 0.0 and state.throttle == 1.0 and not state.reverse
    assert not d.logic.last_gestures.reverse and d.logic.last_gestures.hold_s == 0.0


def test_dropping_the_hands_out_of_view_releases_after_the_short_hysteresis():
    d = Driver()
    d.feed(TWO_PALMS, 1.5)
    d.feed((), 0.15)  # hands gone (no positive wheel pose): released once the 100 ms hysteresis is over
    assert not d.logic.last_gestures.reverse and not d.logic.last_gestures.palms
    state = d.feed(wheel_pair(0), 0.2)[-1]
    assert state.brake == 0.0 and state.throttle == 1.0


def test_flicker_does_not_toggle_reverse():
    """A palm missed for a single frame (MediaPipe flicker) neither resets the hold nor drops reverse."""
    d = Driver()
    half_palm = (palm(0.32), make_hand(extension=(1.0, 1.0, 1.0, 0.4), thumb=0.8, center=(0.68, 0.46)))  # 4 fingers
    seen = []
    for i in range(60):  # 2 s, every 6th frame one hand is misread
        hands = half_palm if i % 6 == 5 else TWO_PALMS
        state = d.feed(hands, DT)[0]
        seen.append((d.logic.last_gestures.reverse, state.brake))
    first_reverse = next(i for i, (r, _) in enumerate(seen) if r)
    assert first_reverse <= 21  # 0.6 s after confirmation, not restarted by the misread frames
    assert all(r for r, _ in seen[first_reverse:])  # never toggles off
    assert all(b == 1.0 for _, b in seen[1:])


def test_a_single_palm_still_brakes_but_never_reverses():
    d = Driver()
    states = d.feed((place(lambda **k: palm(k["center"][0], k["center"][1]), 0, "left"), fist(center=(0.7, 0.55))), 2.0)
    assert states[-1].brake == 1.0 and states[-1].throttle == 0.0
    assert not d.logic.last_gestures.reverse and not d.logic.last_gestures.palms


def palms_tilted(deg, separation=0.5, aspect=4 / 3, center=(0.5, 0.45)):
    """Two natural palms whose centres lie on a wheel line rotated ``deg`` clockwise (same geometry as wheel_pair)."""
    half = separation / 2
    dx, dy = half * math.cos(math.radians(deg)) / aspect, half * math.sin(math.radians(deg))
    return (palm(center[0] - dx, center[1] - dy), palm(center[0] + dx, center[1] + dy))


def test_palms_steer_proportionally_by_default_while_braking_and_reversing():
    d = Driver()
    assert d.cfg.gestures.palms_steering == "full"
    d.feed(palms_tilted(20), 1.5)
    g, state = d.logic.last_gestures, d.logic.last_state
    assert state.steer == pytest.approx(20 / 30, abs=0.02)  # about 67%: offset / full lock, as on the wheel
    assert g.brake and g.reverse and state.brake == 1.0 and state.throttle == 0.0
    d.feed(palms_tilted(-20), 1.0)
    assert d.logic.last_state.steer == pytest.approx(-20 / 30, abs=0.02)
    assert d.logic.last_gestures.brake and d.logic.last_gestures.reverse  # steering never interrupts brake / reverse


def test_level_palms_give_exactly_zero_steering():
    d = Driver()
    d.feed(palms_tilted(20), 1.0)
    states = d.feed(palms_tilted(0), 1.0)
    assert states[0].steer == 0.0  # the first level frame: inside the dead zone at once
    assert all(s.steer == 0.0 for s in states)
    assert d.logic.last_gestures.reverse


def test_palm_steering_follows_the_configured_full_lock():
    cfg = Config()
    cfg.steering.full_lock_deg = 40.0
    d = Driver(cfg)
    d.feed(palms_tilted(20), 1.5)
    assert d.logic.last_state.steer == pytest.approx(0.5, abs=0.02)


def test_steering_is_frozen_at_zero_while_two_palms_are_held_in_frozen_mode():
    cfg = Config()
    cfg.gestures.palms_steering = "frozen"
    d = Driver(cfg)
    assert d.feed(wheel_pair(20), 1.0)[-1].steer > 0.5
    tilted = (palm(0.32, 0.36), palm(0.68, 0.52))  # palms at clearly different heights: wheel angle ~ +24 deg
    states = d.feed(tilted, 1.0)
    assert all(s.steer == 0.0 for s in states)


def test_intentional_mode_steers_only_past_the_threshold():
    cfg = Config()
    cfg.gestures.palms_steering = "intentional"
    d = Driver(cfg)
    assert d.feed(TWO_PALMS, 0.5)[-1].steer == 0.0  # ~7 deg: not clearly intended
    steep = (palm(0.32, 0.30), palm(0.68, 0.60))  # ~ +40 deg
    assert d.feed(steep, 0.5)[-1].steer > 0.5


def test_frozen_palms_do_not_count_as_hands_lost_or_corrupt_the_neutral():
    cfg = Config()
    cfg.gestures.palms_steering = "frozen"
    d = Driver(cfg)
    neutral = d.logic.steering.calibrator.neutral
    d.feed(TWO_PALMS, 5.0)
    assert d.logic.mode.value == "DRIVING"
    assert d.logic.steering.calibrator.neutral == neutral  # raised palms never feed auto-recenter / calibration


def test_brake_beats_nitro():
    d = Driver()
    states = d.feed((place(thumbs_up, 0, "left"), place(lambda **k: palm(k["center"][0], k["center"][1]), 0, "right")), 1.0)
    assert all(not s.nitro for s in states[2:]) and states[-1].brake == 1.0
    d.feed(TWO_PALMS, 1.0)
    assert not d.logic.last_state.nitro


def test_button_reverse_mode_presses_reverse_instead_of_brake():
    cfg = Config()
    cfg.output.reverse_mode = "button"
    d = Driver(cfg)
    early = d.feed(TWO_PALMS, 0.3)[-1]
    assert early.brake == 1.0 and not early.reverse  # braking first
    late = d.feed(TWO_PALMS, 0.6)[-1]
    assert late.reverse and late.brake == 0.0 and late.throttle == 0.0
    back = d.feed(wheel_pair(0), DT)[0]
    assert not back.reverse and back.throttle == 1.0


def test_keyboard_and_gamepad_send_reverse():
    from gesture_car.backend.output import ControlState
    from gesture_car.backend.output_keyboard import KeyboardOutput, RecordingSink

    cfg = Config().output
    sink = RecordingSink()
    out = KeyboardOutput(cfg, sink)
    out.update(ControlState(brake=1.0), 0.0)
    assert {"s", "down"} <= sink.held and "w" not in sink.held
    out.update(ControlState(reverse=True), 0.01)
    assert sink.held == {"r"}
    out.update(ControlState(throttle=1.0), 0.02)
    assert "r" not in sink.held and "s" not in sink.held


def test_engine_reports_left_and_right_finger_counts_by_image_position():
    e = GestureEngine(Config().gestures)
    state = e.update((fist(center=(0.7, 0.5)), palm(0.3)), 0.0)  # reported right-first: still ordered left to right
    assert state.fingers == (5, 0)
    assert e.update((palm(0.3),), DT).fingers == (5, None)
    assert e.update((), 2 * DT).fingers == (None, None)


def test_debug_line_shows_palm_brake_reverse_and_hold():
    d = Driver()
    d.feed(TWO_PALMS, 1.0)
    line = debug_line(d.logic)
    assert "fingers L5 R5" in line and "palm_confirmed 1" in line and "brake_active 1" in line and "reverse_active 1" in line
    assert "hold 0.9" in line


@pytest.mark.parametrize("thumb", [0.6, 1.0])
def test_engine_alone_two_palms_reverse(thumb):
    e = GestureEngine(Config().gestures)
    hands = (palm(0.3, thumb=thumb), palm(0.7, thumb=thumb))
    t, state = 0.0, None
    for _ in range(25):  # 0.83 s
        t += DT
        state = e.update(hands, t)
    assert state.brake and state.palms and state.reverse


def test_older_freeze_setting_still_freezes(tmp_path):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"gestures": {"palms_steering": "freeze"}}), encoding="utf-8")
    d = Driver(Config.load(path))
    assert all(s.steer == 0.0 for s in d.feed(palms_tilted(20), 1.0))
