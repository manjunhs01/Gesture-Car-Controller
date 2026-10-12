"""Profiles, nitro modes, recording / replay and presentation rendering."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gesture_car.backend.controller import ControlLogic
from gesture_car.backend.gestures import GestureEngine
from gesture_car.backend.profiles import apply_profile, list_profiles, profiles_dir
from gesture_car.backend.recording import Recorder, read_frames, replay, summarize
from gesture_car.config import Config
from gesture_car.utils.errors import GestureCarError
from tests.support.synthetic_hand import thumbs_up, wheel_pair


def test_all_shipped_profiles_load_and_apply():
    names = list_profiles()
    assert {"browser-racing", "forza-gamepad", "gta-v", "asphalt-pc", "presentation"} <= set(names)
    for name in names:
        cfg = Config()
        apply_profile(cfg, name)  # must not raise
        assert names[name]  # has a description


def test_profile_changes_settings():
    cfg = Config()
    apply_profile(cfg, "forza-gamepad")
    assert cfg.output.backend == "gamepad"
    assert cfg.steering.feel == "realistic"
    assert cfg.gestures.nitro_mode == "pulse"


def test_unknown_profile_raises_helpful_error():
    with pytest.raises(GestureCarError) as exc:
        apply_profile(Config(), "does-not-exist")
    assert "browser-racing" in exc.value.user_message


def test_bad_profile_values_are_ignored(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"output": {"backend": "toaster"}, "nope": {}}))
    cfg = Config()
    apply_profile(cfg, "bad", tmp_path)
    assert cfg.output.backend == Config().output.backend
    assert profiles_dir().exists()


def _run_nitro(mode: str) -> list[bool]:
    cfg = Config()
    cfg.gestures.nitro_mode = mode
    cfg.gestures.nitro_pulse_s = 0.3
    engine = GestureEngine(cfg.gestures)
    out = []
    t = 0.0
    # thumbs-up shown for 0.5 s, hidden for 0.5 s, shown again for 0.5 s, hidden
    for shown in (True, False, True, False):
        for _ in range(15):
            t += 1 / 30
            hands = (thumbs_up(),) if shown else ()
            out.append(engine.update(hands, t).nitro)
    return out


def test_nitro_hold_follows_gesture():
    out = _run_nitro("hold")
    assert out[10] and not out[29] and out[40] and not out[59]


def test_nitro_toggle_latches_until_next_gesture():
    out = _run_nitro("toggle")
    assert out[10]  # on after the first thumbs-up
    assert out[29]  # stays on after the thumb is gone
    assert not out[40]  # the second thumbs-up switches it off
    assert not out[59]


def test_nitro_pulse_is_a_short_burst():
    out = _run_nitro("pulse")
    assert out[5]  # burst started by the first thumbs-up
    assert not out[29]  # and is over (0.3 s) long before 1 s has passed
    assert out[35]  # a new thumbs-up starts a new burst


def test_record_and_replay_are_identical(tmp_path):
    cfg = Config()
    logic = ControlLogic(cfg)
    path = tmp_path / "run.jsonl"
    live = []
    with Recorder(path) as rec:
        t = 0.0
        for i in range(150):
            t += 1 / 30
            hands = wheel_pair(0.0) if i < 90 else wheel_pair(25.0)
            state = logic.process(hands, t)
            rec.write(t, hands, state)
            live.append(state.steer)
    frames = list(read_frames(path))
    assert len(frames) == 150
    replayed = replay(path, Config())
    assert np.allclose([s.steer for _, s, _ in replayed], live, atol=0.01)  # landmarks are stored with 5 decimals
    assert "frames over" in summarize(replayed)
    assert max(s.steer for _, s, _ in replayed) > 0.2  # the right turn really was replayed


def test_replay_of_empty_recording(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert summarize(replay(path, Config())) == "Recording is empty."


def test_presentation_render_adds_header_and_safe_banner():
    from gesture_car.backend.pipeline import TrackingResult
    from gesture_car.frontend.debug_view import HEADER_H, PANEL_H, PIPE_H, RENDER_W, render

    logic = ControlLogic(Config())
    rgb = np.zeros((480, 640, 3), np.uint8)
    result = TrackingResult(1, 0.0, (), 1.0, (640, 480), rgb, 30.0)
    plain = render(result, logic)
    pres = render(result, logic, presentation=True)
    assert plain is not None and pres is not None
    assert plain.shape == (480 + PIPE_H + PANEL_H, RENDER_W, 3)
    assert pres.shape[0] == plain.shape[0] + HEADER_H


def test_preview_runs_in_its_own_thread_and_never_blocks(monkeypatch):
    import threading
    import time

    import gesture_car.frontend.debug_view as dv
    from gesture_car.backend.pipeline import TrackingResult

    shown: list[tuple[str, tuple[int, ...] | None]] = []

    class SlowWindow:
        name = "fake"

        def show(self, img):
            shown.append((threading.current_thread().name, None if img is None else img.shape))
            if img is not None:
                time.sleep(0.3)  # a window that blocks (e.g. the user is dragging it)
            return False

        def close(self) -> None:
            pass

    monkeypatch.setattr(dv, "_CvWindow", SlowWindow)
    view = dv.DebugView(1.0)
    logic = ControlLogic(Config())
    rgb = np.zeros((480, 640, 3), np.uint8)
    result = TrackingResult(1, 0.0, (), 1.0, (640, 480), rgb, 30.0)
    start = time.monotonic()
    for _ in range(5):
        view.show(result, logic)
        time.sleep(0.06)
    assert time.monotonic() - start < 0.6  # the caller never waited for the 0.3 s window
    time.sleep(0.8)
    view.close()
    frames = [s for s in shown if s[1] is not None]
    assert frames and all(name == "preview-window" for name, _ in shown)


def test_quick_edit_helper_is_harmless_off_windows():
    from gesture_car.utils.console import disable_quick_edit

    assert disable_quick_edit() in (True, False)
