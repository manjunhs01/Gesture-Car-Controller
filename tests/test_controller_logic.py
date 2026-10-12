import pytest

from gesture_car.backend.controller import ControlLogic, Mode
from gesture_car.config import Config
from tests.support.synthetic_hand import fist, make_hand, open_palm, place, thumbs_up, wheel_pair

DT = 1 / 30


class Driver:
    """Feeds synthetic frames to ControlLogic with a running clock."""

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.cfg.steering.calibration_seconds = 1.0
        self.logic = ControlLogic(self.cfg)
        self.t = 100.0

    def feed(self, hands, seconds=0.5):
        state = None
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            state = self.logic.process(tuple(hands), self.t)
        return state

    def calibrate(self, angle=0.0):
        self.feed(wheel_pair(angle), 1.5)
        assert self.logic.steering.calibrated
        return self.feed(wheel_pair(angle), 0.3)


def test_not_driving_until_calibrated():
    d = Driver()
    state = d.feed(wheel_pair(0), 0.5)
    assert d.logic.mode is Mode.CALIBRATING
    assert state.throttle == 0 and state.steer == 0


def test_automatic_acceleration_after_calibration():
    d = Driver()
    state = d.calibrate()
    assert d.logic.mode is Mode.DRIVING
    assert state.throttle == 1.0 and state.brake == 0 and not state.nitro


def test_steering_left_and_right_proportional():
    d = Driver()
    d.calibrate()
    assert d.feed(wheel_pair(10), 1.0).steer < d.feed(wheel_pair(35), 1.0).steer
    assert d.feed(wheel_pair(-35), 1.0).steer < -0.5
    assert abs(d.feed(wheel_pair(0), 2.0).steer) < 0.02


def test_open_palm_brakes_and_overrides_acceleration():
    d = Driver()
    d.calibrate()
    hands = (place(open_palm, 0.0, "left"), fist(center=(0.75, 0.55)))
    state = d.feed(hands, 0.5)
    assert state.brake == 1.0 and state.throttle == 0.0 and not state.nitro
    state = d.feed(wheel_pair(0), 0.6)  # palm lowered
    assert state.brake == 0.0 and state.throttle == 1.0


def test_thumbs_up_is_nitro_and_keeps_accelerating():
    d = Driver()
    d.calibrate()
    state = d.feed((place(thumbs_up, 0.0, "right"), fist(center=(0.25, 0.55))), 0.5)
    assert state.nitro and state.throttle == 1.0 and state.brake == 0.0


def test_brake_wins_over_nitro():
    d = Driver()
    d.calibrate()
    state = d.feed((place(open_palm, 0.0, "left"), place(thumbs_up, 0.0, "right")), 0.5)
    assert state.brake == 1.0 and not state.nitro


def both_thumbs_up(angle):
    return (place(thumbs_up, angle, "left"), place(thumbs_up, angle, "right"))


def rolled_thumbs_up(angle, side):
    """A thumbs-up fist rolled with a wheel turned ``angle`` degrees clockwise."""
    return place(lambda **k: make_hand(extension=0.0, thumb_up=True, roll_deg=angle, **k), angle, side)


def test_both_thumbs_up_is_nitro_after_two_frames_and_released_after_hysteresis():
    d = Driver()
    d.calibrate()
    states = [d.feed(both_thumbs_up(0), DT) for _ in range(3)]
    assert not states[0].nitro and states[1].nitro and states[2].nitro  # 2-frame confirmation
    assert d.logic.last_gestures.thumbs == (True, True)
    assert d.feed(both_thumbs_up(0), 1.0).nitro  # stays on while held
    assert d.feed(wheel_pair(0), DT).nitro  # one missed frame is bridged by the release hysteresis
    assert not d.feed(wheel_pair(0), 0.2).nitro


def test_both_thumbs_up_on_a_turned_wheel_is_nitro():
    d = Driver()
    d.calibrate()
    hands = (rolled_thumbs_up(-35, "left"), rolled_thumbs_up(-35, "right"))  # each alone leans more than 50 deg off screen-up
    state = d.feed(hands, 0.5)
    assert state.nitro and state.throttle == 1.0 and state.steer < -0.5


def test_one_thumbs_up_is_still_nitro():
    d = Driver()
    d.calibrate()
    state = d.feed((place(fist, 0, "left"), place(thumbs_up, 0, "right")), 0.5)
    assert state.nitro and d.logic.last_gestures.thumbs == (False, True)


def rolled_fist(angle, side):
    """A fist gripping a wheel turned ``angle`` degrees clockwise."""
    return place(lambda **k: make_hand(extension=0.0, thumb=0.0, roll_deg=angle, **k), angle, side)


@pytest.mark.parametrize("angle", [30, -30])
@pytest.mark.parametrize("side", ["left", "right"])
def test_one_thumbs_up_on_a_wheel_tilted_30_degrees_is_nitro(angle, side):
    """Only one hand shows the thumb; it is rolled with the wheel, so it is judged against the wheel's up."""
    d = Driver()
    d.calibrate()
    other = "right" if side == "left" else "left"
    thumb, grip = rolled_thumbs_up(angle, side), rolled_fist(angle, other)
    assert not d.feed((rolled_fist(angle, "left"), rolled_fist(angle, "right")), 0.5).nitro  # plain fists: no nitro
    state = d.feed((thumb, grip) if side == "left" else (grip, thumb), 0.5)
    assert state.nitro and state.throttle == 1.0 and abs(state.steer) > 0.9
    assert d.logic.last_gestures.thumbs == ((True, False) if side == "left" else (False, True))
    assert not d.feed((rolled_fist(angle, "left"), rolled_fist(angle, "right")), 0.2).nitro  # thumb down: released


@pytest.mark.parametrize("angle", [20, -20])
def test_both_thumbs_up_while_steering_is_nitro_and_keeps_steering(angle):
    d = Driver()
    d.calibrate()
    steer = d.feed((rolled_fist(angle, "left"), rolled_fist(angle, "right")), 1.0).steer
    assert steer == pytest.approx(angle / 30, abs=0.03)
    state = d.feed((rolled_thumbs_up(angle, "left"), rolled_thumbs_up(angle, "right")), 0.5)
    assert state.nitro and state.steer == pytest.approx(steer, abs=0.03)
    assert not d.feed((rolled_fist(angle, "left"), rolled_fist(angle, "right")), 0.2).nitro


def test_nitro_with_both_thumbs_up_keeps_steering_at_plus_20_degrees():
    d = Driver()
    d.calibrate()
    steer = d.feed(wheel_pair(20), 1.0).steer
    assert steer == pytest.approx(20 / 30, abs=0.02)
    state = d.feed(both_thumbs_up(20), 0.6)
    assert state.nitro and state.steer == pytest.approx(steer, abs=0.02)


def test_brake_beats_nitro_from_both_thumbs_up():
    d = Driver()
    d.calibrate()
    assert d.feed(both_thumbs_up(0), 0.3).nitro
    state = d.feed((place(open_palm, 0, "left"), place(thumbs_up, 0, "right")), 0.3)
    assert state.brake == 1.0 and not state.nitro


def test_steering_is_held_while_a_hand_is_off_the_wheel_out_of_view_for_a_gesture():
    d = Driver()
    d.calibrate()
    steady = d.feed(wheel_pair(25), 1.5).steer
    assert steady > 0.3
    state = d.feed((open_palm(center=(0.5, 0.4)),), 0.2)  # one hand left the picture, the other shows a palm
    assert state.brake == 1.0
    assert state.steer == pytest.approx(steady, abs=0.05)  # held until the hands-lost timeout


def test_with_both_hands_visible_the_wheel_always_follows_the_hands():
    """No gesture may freeze steering while both hands are measured, whatever the freeze setting."""
    for freeze in (True, False):
        cfg = Config()
        cfg.safety.freeze_steering_during_gesture = freeze
        d = Driver(cfg)
        d.calibrate()
        d.feed(wheel_pair(25), 1.5)
        state = d.feed((fist(center=(0.38, 0.55)), open_palm(center=(0.62, 0.20))), 1.0)
        assert state.steer < 0, freeze  # wheel line now tilts the other way


@pytest.mark.parametrize("hand", [thumbs_up, open_palm])
def test_gesture_shaped_hands_on_the_wheel_return_to_exactly_zero(hand):
    """Fists with thumbs on top look like thumbs-up and open hands look like a palm. They used to freeze the steering at
    the last turn, so the car kept turning with the hands straight."""
    d = Driver()
    d.calibrate()
    assert d.feed(wheel_pair(30), 1.0).steer > 0.8
    state = d.feed((place(hand, 0, "left"), place(hand, 0, "right")), 2 / 30)
    assert state.steer == 0.0


def test_a_duplicated_hand_never_holds_a_turn():
    """MediaPipe sometimes reports one hand twice. Both 'hands' are visible but unmeasurable: that must not keep the
    last turn forever; it is treated like a missing pair (held briefly, then centred)."""
    d = Driver()
    d.calibrate()
    assert d.feed(wheel_pair(30), 1.0).steer > 0.8
    duplicate = fist(center=(0.6, 0.55))
    state = d.feed((duplicate, duplicate), 0.5)
    assert d.logic.mode is Mode.HANDS_LOST
    assert state.steer == 0.0


def test_hands_lost_centres_steering_releases_nitro_and_stops_acceleration():
    d = Driver()
    d.calibrate()
    d.feed(wheel_pair(30), 1.0)
    d.feed((place(thumbs_up, 30, "right"), fist(center=(0.25, 0.55))), 0.4)  # nitro on
    state = d.feed((), 0.5)  # hands gone > 300 ms
    assert d.logic.mode is Mode.HANDS_LOST
    assert state.steer == 0 and not state.nitro and state.throttle == 0 and state.brake == 0  # neutral by default


def test_hands_lost_gentle_brake_is_configurable():
    cfg = Config()
    cfg.safety.lost_behavior = "gentle_brake"
    d = Driver(cfg)
    d.calibrate()
    state = d.feed((), 0.6)
    assert state.brake == pytest.approx(cfg.safety.gentle_brake_level) and state.throttle == 0


def test_short_dropout_under_300ms_is_ignored():
    d = Driver()
    d.calibrate()
    before = d.feed(wheel_pair(20), 1.5)
    during = d.feed((), 0.2)
    assert d.logic.mode is Mode.DRIVING
    assert during.steer == pytest.approx(before.steer, abs=0.01)
    assert during.throttle == 1.0


def test_one_hand_only_counts_as_lost_but_palm_still_brakes():
    d = Driver()
    d.calibrate()
    state = d.feed((open_palm(center=(0.5, 0.5)),), 0.6)
    assert d.logic.mode is Mode.HANDS_LOST
    assert state.brake == 1.0 and state.steer == 0
    state = d.feed((thumbs_up(center=(0.5, 0.5)),), 0.6)
    assert not state.nitro  # safe state: nitro released


def test_recovers_after_hands_return():
    d = Driver()
    d.calibrate()
    d.feed((), 0.6)
    state = d.feed(wheel_pair(0), 0.5)
    assert d.logic.mode is Mode.DRIVING and state.throttle == 1.0


def test_stalled_camera_triggers_safe_state_via_tick():
    d = Driver()
    d.calibrate()
    last = d.t
    state = d.logic.tick(last + 0.1)
    assert state.throttle == 1.0
    state = d.logic.tick(last + 0.5)
    assert state.throttle == 0 and state.steer == 0 and not state.nitro


def test_recalibrate_resets_to_calibration_mode():
    d = Driver()
    d.calibrate()
    d.logic.recalibrate()
    state = d.feed(wheel_pair(0), 0.3)
    assert d.logic.mode is Mode.CALIBRATING and state.throttle == 0
    d.feed(wheel_pair(12), 1.5)
    assert d.logic.steering.calibrated
    assert abs(d.feed(wheel_pair(12), 0.5).steer) < 0.01  # the held pose (F9) is the new straight


def test_hands_too_close_to_measure_hold_briefly_then_centre():
    d = Driver()
    d.calibrate()
    steer = d.feed(wheel_pair(10), 2.0).steer
    assert steer == pytest.approx(10 / 30, abs=0.01)
    state = d.feed(wheel_pair(10, separation=0.03), 0.2)  # unmeasurable: a short dropout keeps the wheel
    assert d.logic.mode is Mode.DRIVING and state.steer == pytest.approx(steer, abs=0.01)
    state = d.feed(wheel_pair(10, separation=0.03), 0.5)  # but never longer than hands_lost_s
    assert d.logic.mode is Mode.HANDS_LOST and state.steer == 0.0


def test_brake_and_nitro_gestures_never_zero_the_steering():
    d = Driver()
    d.calibrate()
    steer = d.feed(wheel_pair(8), 2.0).steer
    assert steer == pytest.approx(8 / 30, abs=0.01)
    lh = place(fist, 8, "left")
    for hand in (open_palm, thumbs_up):
        state = d.feed((lh, place(hand, 8, "right")), 0.6)
        assert state.steer == pytest.approx(steer, abs=0.02), hand.__name__


def test_debug_line_shows_raw_angle_neutral_offset_and_percent():
    from gesture_car.backend.controller import debug_line

    d = Driver()
    d.calibrate()
    d.feed(wheel_pair(4), 1.0)  # shorter than the 3 s auto-recenter delay
    line = debug_line(d.logic)
    assert "raw angle   94.0" in line and "neutral   90.0" in line and "offset   +4.0" in line and "+13.4%" in line
    stages = dict(wait=1.0, detect=17.0, handoff=0.1, steer=0.05, send=0.02, total=19.0)
    line = debug_line(d.logic, 30.0, stages)
    assert "thumb_up L0 R0 nitro_active 0" in line
    assert "cam 30.0 fps" in line and "detect  17.0" in line and "total  19.0 ms" in line


def test_pause_outputs_nothing_and_resume_recalibrates():
    d = Driver()
    d.calibrate()
    d.logic.set_paused(True)
    assert d.feed(wheel_pair(0), 0.3).throttle == 0
    d.logic.set_paused(False)
    assert d.logic.mode is Mode.CALIBRATING


def test_calibration_completes_with_shaky_hands_and_dropped_frames():
    """Regression: with real-world jitter and a hand dropped now and then, calibration must still finish in a few seconds."""
    import random

    rng = random.Random(7)
    cfg = Config()  # default 2 s hold
    logic = ControlLogic(cfg)
    t = 100.0
    for i in range(300):  # 10 s at 30 fps
        t += DT
        hands = (wheel_pair(6.0 + rng.uniform(-3, 3))[0],) if i % 6 == 2 else wheel_pair(6.0 + rng.uniform(-3, 3))
        logic.process(hands, t)
        if logic.steering.calibrated:
            break
    assert logic.steering.calibrated, logic.calibration_message()
    assert t - 100.0 < 4.0


def test_preview_renders_wheel_panel_and_rotates_with_hands():
    import numpy as np

    from gesture_car.frontend.debug_view import PANEL_H, PIPE_H, RENDER_W, render

    RENDER_H = 480  # a 320x240 test frame is scaled to RENDER_W (640) x 480
    from gesture_car.backend.pipeline import TrackingResult

    d = Driver()
    d.calibrate()
    frames = []
    for angle in (0.0, 40.0, -40.0):
        hands = wheel_pair(angle)
        d.feed(hands, 1.5)
        res = TrackingResult(1, 0.0, hands, 5.0, (320, 240), np.zeros((240, 320, 3), np.uint8), 30.0)
        frames.append(render(res, d.logic))
    assert all(f.shape == (RENDER_H + PIPE_H + PANEL_H, RENDER_W, 3) for f in frames)
    panel = [f[240:] for f in frames]
    assert not np.array_equal(panel[0], panel[1]) and not np.array_equal(panel[1], panel[2])  # wheel/bars change with the hands


def test_feel_presets_change_smoothing_but_never_the_proportional_mapping():
    from gesture_car.config import SteeringConfig, apply_feel

    for feel in ("arcade", "normal", "realistic"):
        cfg = SteeringConfig(feel=feel, full_lock_deg=30.0, deadzone_deg=1.0)
        apply_feel(cfg)
        assert (cfg.full_lock_deg, cfg.deadzone_deg) == (30.0, 1.0), feel
    arcade = SteeringConfig(feel="arcade")
    apply_feel(arcade)
    assert arcade.one_euro_min_cutoff > SteeringConfig().one_euro_min_cutoff


def test_preview_shows_calibration_guidance():
    import numpy as np

    from gesture_car.frontend.debug_view import _guide_text

    d = Driver()
    d.feed(wheel_pair(0), 0.2)
    assert "wheel" in _guide_text(d.logic)
    d.calibrate()
    assert _guide_text(d.logic) == ""
    d.feed((), 0.6)
    assert "BOTH" in _guide_text(d.logic)
    assert np is not None


def test_preview_fits_on_small_screens():
    from gesture_car.frontend.debug_view import fit_scale

    # 640x680 preview on a 1536x720 work area (125% scaled laptop): must shrink so the bottom is not cut off
    assert fit_scale(1.6, 640, 680, (1536, 720)) < 1.0
    assert 680 * fit_scale(1.6, 640, 680, (1536, 720)) + 64 <= 720
    assert fit_scale(1.0, 640, 680, (3840, 2000)) == 1.0


def test_flip_steering_hotkey_inverts_and_saves(tmp_path, monkeypatch):
    import json

    import gesture_car.config as config_mod
    from gesture_car.backend.controller import GestureController
    from gesture_car.config import Config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config_mod, "default_path", lambda: cfg_file)
    ctrl = GestureController(Config(), dry_run=True)
    assert ctrl.cfg.steering.invert_steering is False
    ctrl.flip_steering()
    assert ctrl.cfg.steering.invert_steering is True
    assert json.loads(cfg_file.read_text())["steering"]["invert_steering"] is True
    ctrl.flip_steering()
    assert ctrl.cfg.steering.invert_steering is False


def test_direction_words():
    from gesture_car.frontend.debug_view import _direction

    assert _direction(0.5) == "RIGHT >>"
    assert _direction(-0.5) == "<< LEFT"
    assert _direction(0.0) == "straight"


def test_emergency_stop_releases_and_pauses():
    from gesture_car.backend.controller import GestureController
    from gesture_car.backend.output import ControlState
    from gesture_car.config import Config

    ctrl = GestureController(Config(), dry_run=True)
    released = []

    class Out:
        def release_all(self) -> None:
            released.append(True)

        def update(self, state: ControlState, now: float) -> None:
            pass

    ctrl.output = Out()  # type: ignore[assignment]
    ctrl.emergency_stop()
    assert released and ctrl.logic.paused
