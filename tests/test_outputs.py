import sys
import types

import pytest

from gesture_car.backend.output import ControlState, create_output
from gesture_car.backend.output_keyboard import KeyboardOutput, RecordingSink, normalise_key, pwm_on
from gesture_car.config import OutputConfig


def make_kb(**kw):
    kw.setdefault("key_preset", "wasd")
    sink = RecordingSink()
    return KeyboardOutput(OutputConfig(**kw), sink), sink


def test_pwm_duty_matches_steering_amount():
    period = 0.05
    for duty in (0.1, 0.25, 0.5, 0.75, 0.9):
        samples = 2000
        on = sum(pwm_on(duty, i * period / samples, period, 0.08, 0.95) for i in range(samples))
        assert on / samples == pytest.approx(duty, abs=0.01)
    assert not pwm_on(0.05, 0.0, period, 0.08, 0.95)
    assert pwm_on(0.97, 0.049, period, 0.08, 0.95)


def test_keyboard_acceleration_brake_nitro_keys():
    out, sink = make_kb()
    out.update(ControlState(throttle=1.0), 1.0)
    assert sink.held == {"w"}
    out.update(ControlState(throttle=0.0, brake=1.0), 1.1)
    assert sink.held == {"s"}  # brake overrides acceleration
    out.update(ControlState(throttle=1.0, nitro=True), 1.2)
    assert sink.held == {"w", "shift"}


def test_keyboard_steering_keys_and_direction():
    out, sink = make_kb()
    out.update(ControlState(steer=1.0, throttle=1.0), 1.0)
    assert sink.held == {"w", "d"}
    out.update(ControlState(steer=-1.0, throttle=1.0), 1.0)
    assert sink.held == {"w", "a"}
    out.update(ControlState(steer=0.0, throttle=1.0), 1.0)
    assert sink.held == {"w"}


def test_keyboard_half_steering_pulses_about_half_the_time():
    out, sink = make_kb()
    on = 0
    n = 1000
    for i in range(n):
        out.update(ControlState(steer=0.5), i * 0.002)
        on += "d" in sink.held
    assert 0.45 < on / n < 0.55
    assert "a" not in sink.held


@pytest.mark.parametrize("steer", [0.027, 0.04, 0.06, 0.1])
def test_keyboard_small_steering_gives_regular_pulses_at_the_loop_rate(steer):
    """At 250 Hz a 50 ms period is 12.5 ticks: tiny duties used to fall between ticks and send nothing at all."""
    out, sink = make_kb()
    on = [None] * 1000
    for i in range(1000):
        out.update(ControlState(steer=steer), 1000.0123 + i / 250)
        on[i] = "d" in sink.held
    assert sum(on) / len(on) == pytest.approx(steer, abs=0.005)  # average follows the value exactly
    starts = [i for i in range(1, len(on)) if on[i] and not on[i - 1]]
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert len(starts) >= 10 and max(gaps) <= 40  # a pulse at least every 160 ms, never a dead stretch


def test_keyboard_refused_release_is_retried_until_the_key_is_up():
    class FlakySink(RecordingSink):
        refuse = 2

        def _release(self, key):
            if self.refuse:
                self.refuse -= 1
                raise OSError("refused")
            super()._release(key)

    sink = FlakySink()
    out = KeyboardOutput(OutputConfig(key_preset="wasd"), sink)
    out.update(ControlState(throttle=1.0, nitro=True), 0.0)
    out.update(ControlState(throttle=1.0), 0.01)
    out.update(ControlState(throttle=1.0), 0.02)
    assert "shift" in sink.held  # not forgotten while the key-up is refused
    out.update(ControlState(throttle=1.0), 0.03)
    assert sink.held == {"w"} and sink.events[-1] == ("up", "shift")


def test_keyboard_release_all_and_configurable_keys():
    out, sink = make_kb(key_preset="custom", key_accelerate="up", key_nitro="space", key_left="left", key_right="right")
    out.update(ControlState(steer=1.0, throttle=1.0, nitro=True), 0.0)
    assert sink.held == {"up", "space", "right"}
    out.release_all()
    assert sink.held == set()


def test_unknown_key_is_rejected_clearly():
    with pytest.raises(ValueError):
        normalise_key("banana")


class FakePad:
    def __init__(self):
        self.calls = []

    def left_joystick_float(self, x, y):
        self.calls.append(("stick", x, y))

    def right_trigger_float(self, v):
        self.calls.append(("rt", v))

    def left_trigger_float(self, v):
        self.calls.append(("lt", v))

    def press_button(self, b):
        self.calls.append(("press", b))

    def release_button(self, b):
        self.calls.append(("release", b))

    def reset(self):
        self.calls.append(("reset",))

    def update(self):
        self.calls.append(("update",))


@pytest.fixture
def fake_vgamepad(monkeypatch):
    mod = types.ModuleType("vgamepad")
    mod.XUSB_BUTTON = types.SimpleNamespace(
        XUSB_GAMEPAD_A="A", XUSB_GAMEPAD_B="B", XUSB_GAMEPAD_X="X", XUSB_GAMEPAD_Y="Y",
        XUSB_GAMEPAD_LEFT_SHOULDER="LB", XUSB_GAMEPAD_RIGHT_SHOULDER="RB",
    )  # fmt: skip
    mod.VX360Gamepad = FakePad
    monkeypatch.setitem(sys.modules, "vgamepad", mod)
    return mod


def test_gamepad_maps_analog_steering_triggers_and_nitro(fake_vgamepad):
    out = create_output(OutputConfig(backend="gamepad"))
    assert out.name == "gamepad"
    out.update(ControlState(steer=0.37, throttle=1.0, brake=0.0, nitro=True), 1.0)
    calls = out.pad.calls
    assert ("stick", 0.37, 0.0) in calls and ("rt", 1.0) in calls and ("lt", 0.0) in calls and ("press", "A") in calls
    out.pad.calls.clear()
    out.update(ControlState(steer=-0.5, throttle=0.0, brake=1.0, nitro=False), 1.1)
    calls = out.pad.calls
    assert ("stick", -0.5, 0.0) in calls and ("rt", 0.0) in calls and ("lt", 1.0) in calls and ("release", "A") in calls


def test_gamepad_skips_unchanged_frames_but_keeps_alive(fake_vgamepad):
    out = create_output(OutputConfig(backend="gamepad"))
    out.update(ControlState(steer=0.2, throttle=1.0), 1.0)
    n = len(out.pad.calls)
    out.update(ControlState(steer=0.2, throttle=1.0), 1.01)
    assert len(out.pad.calls) == n
    out.update(ControlState(steer=0.2, throttle=1.0), 1.5)  # keep-alive
    assert len(out.pad.calls) > n


def test_gamepad_release_all_resets_pad(fake_vgamepad):
    out = create_output(OutputConfig(backend="gamepad"))
    out.update(ControlState(steer=1.0, throttle=1.0, nitro=True), 1.0)
    out.release_all()
    assert out.pad.calls[-2:] == [("reset",), ("update",)]


def test_gamepad_nitro_button_configurable(fake_vgamepad):
    out = create_output(OutputConfig(backend="gamepad", nitro_button="rb"))
    out.update(ControlState(nitro=True), 1.0)
    assert ("press", "RB") in out.pad.calls


def test_falls_back_to_keyboard_when_gamepad_driver_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "vgamepad", None)  # import raises ImportError
    out = create_output(OutputConfig(backend="gamepad"), dry_run=False) if False else None
    cfg = OutputConfig(backend="gamepad", fallback_to_keyboard=True)
    monkeypatch.setattr("gesture_car.backend.output_keyboard.create_sink", lambda dry_run=False: RecordingSink())
    out = create_output(cfg)
    assert out.name == "keyboard"
    with pytest.raises(RuntimeError):
        create_output(OutputConfig(backend="gamepad", fallback_to_keyboard=False))


def test_dry_run_never_sends_real_input():
    out = create_output(OutputConfig(backend="gamepad"), dry_run=True)
    assert out.name == "keyboard" and isinstance(out.sink, RecordingSink)


def test_default_backend_is_keyboard_pressing_wasd_and_arrows_together():
    cfg = OutputConfig()
    assert cfg.backend == "keyboard" and cfg.key_preset == "both"
    out, sink = make_kb(key_preset="both")
    out.update(ControlState(steer=1.0, throttle=1.0, nitro=True), 0.0)
    assert sink.held == {"w", "up", "d", "right", "shift"}
    out.update(ControlState(steer=-1.0, brake=1.0), 0.0)
    assert sink.held == {"s", "down", "a", "left"}


def test_arrows_preset_only_presses_arrows():
    out, sink = make_kb(key_preset="arrows")
    out.update(ControlState(steer=-1.0, throttle=1.0), 0.0)
    assert sink.held == {"up", "left"}


def test_selftest_taps_every_output_without_real_input():
    from gesture_car.backend.selftest import run_output_test
    from gesture_car.config import Config

    assert run_output_test(Config(), dry_run=True, sleep=False) == 0
