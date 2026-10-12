"""Runs the real GestureController loop with a fake camera pipeline and a recording key sink (no real input is sent)."""

import threading
import time

from gesture_car.backend import instance
from gesture_car.backend.controller import GestureController
from gesture_car.backend.output_keyboard import KeyboardOutput, RecordingSink
from gesture_car.backend.pipeline import TrackingResult
from gesture_car.config import Config
from tests.support.synthetic_hand import wheel_pair


class FakeCamera:
    is_ok = True

    def stop(self):
        pass


class FakePipeline:
    def __init__(self):
        self.hands = ()
        self.seq = 0
        self.stopped = False
        self._lock = threading.Lock()

    def set_hands(self, hands):
        with self._lock:
            self.hands = hands

    def latest(self):
        with self._lock:
            self.seq += 1
            return TrackingResult(self.seq, time.monotonic(), tuple(self.hands), 5.0, (640, 480), None, 30.0)

    def wait_for_new(self, last_sequence, timeout):
        time.sleep(min(max(timeout, 0.0), 1 / 30))  # a 30 fps camera
        return self.latest()

    def stop(self):
        self.stopped = True


def make(monkeypatch, sink):
    cfg = Config()
    cfg.output.backend = "keyboard"
    cfg.steering.calibration_seconds = 0.5
    cfg.app.loop_hz = 100
    ctrl = GestureController(cfg)
    pipe = FakePipeline()

    def fake_start():
        ctrl.camera, ctrl.pipeline = FakeCamera(), pipe

    monkeypatch.setattr(ctrl, "_start_vision", fake_start)
    monkeypatch.setattr(ctrl, "_start_hotkeys", lambda: None)
    monkeypatch.setattr("gesture_car.backend.controller.create_output", lambda c, d=False: KeyboardOutput(c, sink))
    return ctrl, pipe


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_loop_calibrates_accelerates_steers_and_releases_everything_on_quit(monkeypatch):
    sink = RecordingSink()
    ctrl, pipe = make(monkeypatch, sink)
    pipe.set_hands(wheel_pair(0))
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("rc", ctrl.run()))
    thread.start()
    try:
        assert wait_for(lambda: "w" in sink.held), "automatic acceleration never started"
        assert instance.running_pid() is not None
        pipe.set_hands(wheel_pair(35))
        assert wait_for(lambda: "d" in sink.held), "right turn never produced the right key"
        pipe.set_hands(())
        assert wait_for(lambda: "w" not in sink.held and "d" not in sink.held), "hands lost did not enter the safe state"
    finally:
        ctrl.request_quit()
        thread.join(5)
    assert result["rc"] == 0
    assert sink.held == set()
    assert pipe.stopped
    assert instance.running_pid() is None  # pid file removed


def test_stop_file_quits_the_loop(monkeypatch):
    sink = RecordingSink()
    ctrl, pipe = make(monkeypatch, sink)
    pipe.set_hands(wheel_pair(0))
    thread = threading.Thread(target=ctrl.run)
    thread.start()
    assert wait_for(lambda: instance.running_pid() is not None)
    instance.stop_file().write_text("stop")
    thread.join(5)
    assert not thread.is_alive()
    assert sink.held == set()


def test_pause_releases_keys(monkeypatch):
    sink = RecordingSink()
    ctrl, pipe = make(monkeypatch, sink)
    pipe.set_hands(wheel_pair(0))
    thread = threading.Thread(target=ctrl.run)
    thread.start()
    try:
        assert wait_for(lambda: "w" in sink.held)
        ctrl.toggle_pause()
        assert wait_for(lambda: sink.held == set())
        ctrl.toggle_pause()  # resume -> recalibrates -> accelerates again
        assert wait_for(lambda: "w" in sink.held)
    finally:
        ctrl.request_quit()
        thread.join(5)


def test_no_camera_means_no_input_and_clean_quit(monkeypatch):
    sink = RecordingSink()
    ctrl, _ = make(monkeypatch, sink)
    monkeypatch.setattr(ctrl, "_start_vision", lambda: (_ for _ in ()).throw(RuntimeError("no camera")))
    thread = threading.Thread(target=ctrl.run)
    thread.start()
    time.sleep(0.3)
    ctrl.request_quit()
    thread.join(5)
    assert not thread.is_alive() and sink.events == []


def test_stop_file_works_even_while_camera_is_missing(monkeypatch):
    sink = RecordingSink()
    ctrl, _ = make(monkeypatch, sink)
    monkeypatch.setattr(ctrl, "_start_vision", lambda: (_ for _ in ()).throw(RuntimeError("no camera")))
    thread = threading.Thread(target=ctrl.run)
    thread.start()
    assert wait_for(lambda: instance.running_pid() is not None)
    instance.stop_file().write_text("stop")
    thread.join(5)
    assert not thread.is_alive()


class LosableCamera:
    """Fake camera whose connection the test can cut and restore."""

    def __init__(self):
        self.is_ok = True

    def stop(self):
        pass


def test_camera_loss_releases_every_key_at_once_and_driving_resumes_after_reconnect(monkeypatch):
    from gesture_car.backend.controller import Mode

    sink = RecordingSink()
    ctrl, pipe = make(monkeypatch, sink)
    camera = LosableCamera()
    monkeypatch.setattr(ctrl, "_start_vision", lambda: setattr(ctrl, "camera", camera) or setattr(ctrl, "pipeline", pipe))
    pipe.set_hands(wheel_pair(0))
    thread = threading.Thread(target=ctrl.run)
    thread.start()
    try:
        assert wait_for(lambda: "w" in sink.held), "never started driving"
        pipe.set_hands(wheel_pair(35))
        assert wait_for(lambda: {"w", "d"} <= sink.held), "never steered"
        camera.is_ok = False  # cable pulled
        assert wait_for(lambda: sink.held == set()), "keys stayed pressed after the camera was lost"
        assert wait_for(lambda: ctrl.logic.mode is Mode.CAMERA_LOST)  # set right after the release, same loop pass
        time.sleep(0.2)  # the fake pipeline keeps returning (stale) hands: nothing may be pressed again
        assert sink.held == set() and ctrl.logic.mode is Mode.CAMERA_LOST
        camera.is_ok = True  # reconnected
        assert wait_for(lambda: "w" in sink.held), "driving did not resume after the camera came back"
        assert ctrl.logic.mode is not Mode.CAMERA_LOST
    finally:
        ctrl.request_quit()
        thread.join(5)
    assert sink.held == set()


def test_camera_lost_preview_shows_banner_even_without_any_frame():
    import numpy as np

    from gesture_car.backend.controller import ControlLogic, Mode, _blank_result
    from gesture_car.frontend.debug_view import _guide_text, render

    logic = ControlLogic(Config())
    logic.mode = Mode.CAMERA_LOST
    img = render(_blank_result(), logic)
    assert img is not None and img.ndim == 3
    assert "camera" in _guide_text(logic).lower()
    logic.mode = Mode.HANDS_LOST
    other = render(_blank_result(), logic)
    assert other is not None and not np.array_equal(img, other)  # a different banner is drawn
