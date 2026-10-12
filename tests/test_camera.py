"""Webcam support with a fake camera (no hardware): selection, fallback, backends, resolution, URLs, disconnects."""

import logging
import sys
import time

import cv2
import pytest

from gesture_car.backend import camera as cam
from gesture_car.backend import constants as C
from gesture_car.backend.camera import (
    MJPG_FOURCC,
    CameraManager,
    CameraStatus,
    backend_order,
    closest_modes,
    find_camera,
    open_source,
    parse_source,
    probe_cameras,
)
from gesture_car.config import CameraConfig
from tests.support.fake_camera import FakeDevice, FakeSystem

URL = "http://192.168.1.5:8080/video"


@pytest.fixture(autouse=True)
def fast_timing(monkeypatch):
    monkeypatch.setattr(C, "CAMERA_RECONNECT_INTERVAL_S", 0.05)
    monkeypatch.setattr(C, "CAMERA_LOST_AFTER_S", 0.3)
    monkeypatch.setattr(C, "CAMERA_LOST_AFTER_URL_S", 0.3)
    monkeypatch.setattr(C, "CAMERA_FPS_PROBE_S", 0.1)
    monkeypatch.setattr(C, "CAMERA_SPEED_CHECK_S", 0.2)
    monkeypatch.setattr(cam, "FPS_WARMUP_FRAMES", 2)
    monkeypatch.setattr(sys, "platform", "win32")  # exercise the Windows backend order on every OS


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------- backend order
def test_windows_auto_tries_directshow_then_media_foundation_then_default():
    assert [api for _, api in backend_order("auto", 0, "win32")] == [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]


@pytest.mark.parametrize(("choice", "api"), [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)])
def test_forced_backend_is_the_only_one_tried(choice, api):
    assert [a for _, a in backend_order(choice, 0, "win32")] == [api]


def test_other_platforms_and_urls_use_their_own_backends():
    assert [a for _, a in backend_order("dshow", 0, "linux")] == [cv2.CAP_ANY]
    assert [a for _, a in backend_order("auto", URL, "win32")] == [cv2.CAP_FFMPEG, cv2.CAP_ANY]


def test_falls_back_from_directshow_to_media_foundation():
    system = FakeSystem({0: FakeDevice(apis={cv2.CAP_MSMF})})
    opened = open_source(0, CameraConfig(), system)
    assert opened is not None and opened.backend == "Media Foundation"
    assert [api for _, api, _ in system.calls] == [cv2.CAP_DSHOW, cv2.CAP_MSMF]


def test_forced_backend_does_not_silently_use_another():
    system = FakeSystem({0: FakeDevice(apis={cv2.CAP_DSHOW})})
    assert open_source(0, CameraConfig(backend="msmf"), system) is None
    assert [api for _, api, _ in system.calls] == [cv2.CAP_MSMF]


# ------------------------------------------------------------------------ MJPG
def test_mjpg_is_requested_when_supported():
    system = FakeSystem({0: FakeDevice()})
    opened = open_source(0, CameraConfig(), system)
    assert opened is not None and opened.fourcc == "MJPG"
    assert (cv2.CAP_PROP_FOURCC, MJPG_FOURCC) in system.captures[0].sets


def test_camera_that_breaks_with_mjpg_is_reopened_without_it():
    system = FakeSystem({0: FakeDevice(mjpg=False)})
    opened = open_source(0, CameraConfig(), system)
    assert opened is not None and opened.fourcc == ""
    assert len(system.calls) == 2 and all(p != cv2.CAP_PROP_FOURCC for p, _ in system.captures[1].sets)


def test_mjpg_can_be_switched_off():
    system = FakeSystem({0: FakeDevice()})
    assert open_source(0, CameraConfig(mjpg=False), system) is not None
    assert all(p != cv2.CAP_PROP_FOURCC for p, _ in system.captures[0].sets)


# ------------------------------------------------------------------ resolution
def test_closest_modes_orders_by_distance():
    assert closest_modes((1920, 1080))[0] == (1920, 1080)
    assert closest_modes((1280, 720))[:2] == [(1280, 720), (1280, 960)]


@pytest.mark.parametrize(
    ("requested", "modes", "expected"),
    [
        ((640, 480), ((640, 480), (1280, 720)), (640, 480)),
        ((1280, 720), ((640, 480), (1280, 720), (1920, 1080)), (1280, 720)),
        ((1920, 1080), ((640, 480), (1280, 720)), (1280, 720)),  # 1080p refused -> closest supported is 720p
        ((1280, 720), ((640, 480), (800, 600)), (800, 600)),
    ],
)
def test_uses_the_closest_mode_the_camera_supports(requested, modes, expected):
    system = FakeSystem({0: FakeDevice(modes=modes)})
    opened = open_source(0, CameraConfig(width=requested[0], height=requested[1]), system)
    assert opened is not None and (opened.width, opened.height) == expected


# ------------------------------------------------------------ selection/fallback
def test_chosen_camera_is_used_when_it_works():
    system = FakeSystem({0: FakeDevice(), 1: FakeDevice()})
    opened = find_camera(CameraConfig(index=1), system)
    assert opened is not None and opened.source == 1
    assert system.sources_tried() == [1]


def test_falls_back_to_another_camera_and_logs_it(caplog):
    system = FakeSystem({2: FakeDevice()})
    with caplog.at_level(logging.WARNING, logger="gesture_car.backend.camera"):
        opened = find_camera(CameraConfig(index=0), system)
    assert opened is not None and opened.source == 2
    assert "Using camera 2 instead of camera 0" in caplog.text


def test_no_camera_gives_a_clear_message():
    manager = CameraManager(CameraConfig(), FakeSystem())
    assert manager.start() is False
    assert manager.status is CameraStatus.FAILED
    for hint in ("other app", "Privacy & security > Camera", "USB cable", "--list-cameras"):
        assert hint in manager.error


def test_manager_reports_what_it_really_uses():
    manager = CameraManager(CameraConfig(index=0, width=1920, height=1080), FakeSystem({0: FakeDevice()}))
    try:
        assert manager.start()
        assert (manager.source, manager.actual_width, manager.actual_height) == (0, 1280, 720)
        assert manager.backend_name == "DirectShow" and manager.fourcc == "MJPG"
        assert "camera 0 via DirectShow, 1280x720 MJPG" in manager.description
    finally:
        manager.stop()


# -------------------------------------------------------------------- URL source
def test_source_setting_accepts_numbers_and_urls():
    assert parse_source(CameraConfig(index=3)) == 3
    assert parse_source(CameraConfig(index=3, source="1")) == 1
    assert parse_source(CameraConfig(source=f"  {URL} ")) == URL


def test_url_source_opens_the_stream_with_timeouts_and_no_mode_changes():
    system = FakeSystem({URL: FakeDevice(modes=((1280, 720),))})
    opened = open_source(URL, CameraConfig(), system)
    assert opened is not None and opened.source == URL and opened.backend == "FFMPEG"
    source, api, params = system.calls[0]
    assert (source, api) == (URL, cv2.CAP_FFMPEG)
    assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params and cv2.CAP_PROP_READ_TIMEOUT_MSEC in params
    assert [p for p, _ in system.captures[0].sets] == [cv2.CAP_PROP_BUFFERSIZE]  # the phone app decides the size


def test_failed_url_is_not_swapped_for_a_webcam():
    system = FakeSystem({0: FakeDevice()})
    manager = CameraManager(CameraConfig(source=URL), system)
    assert manager.start() is False
    assert system.sources_tried() == [URL]
    assert "SAME Wi-Fi" in manager.error


def test_lagging_stream_keeps_only_the_newest_frame():
    device = FakeDevice(fps=300)
    manager = CameraManager(CameraConfig(source=URL), FakeSystem({URL: device}))
    try:
        assert manager.start()
        seen = []
        last = 0
        for _ in range(5):
            time.sleep(0.05)  # a slow consumer (hand tracking)
            frame = manager.wait_for_new(last, 1.0)
            assert frame is not None
            seen.append(frame.sequence)
            last = frame.sequence
        assert all(b - a > 1 for a, b in zip(seen, seen[1:], strict=False)), "older frames should have been dropped"
        newest = manager.latest()
        assert newest is not None and newest.sequence >= seen[-1]
    finally:
        manager.stop()


# ---------------------------------------------------------- disconnect/reconnect
def test_unplugged_camera_is_reported_lost_then_reconnects():
    device = FakeDevice()
    manager = CameraManager(CameraConfig(), FakeSystem({0: device}))
    try:
        assert manager.start()
        assert wait_for(lambda: manager.latest() is not None) and manager.is_ok
        device.plugged.clear()
        assert wait_for(lambda: manager.status is CameraStatus.RECONNECTING)
        assert not manager.is_ok
        time.sleep(0.2)  # several reconnect attempts fail while unplugged: no crash
        assert manager.status is CameraStatus.RECONNECTING
        before = manager.latest()
        device.plugged.set()
        assert wait_for(lambda: manager.is_ok), "camera did not come back"
        assert wait_for(lambda: (f := manager.latest()) is not None and before is not None and f.sequence > before.sequence)
    finally:
        manager.stop()


def test_camera_that_stops_sending_counts_as_lost():
    device = FakeDevice()
    manager = CameraManager(CameraConfig(), FakeSystem({0: device}))
    try:
        assert manager.start()
        assert wait_for(lambda: manager.is_ok)
        device.hang.set()  # driver stuck inside read()
        assert wait_for(lambda: not manager.is_ok, timeout=2.0)
    finally:
        device.hang.clear()
        manager.stop()


# --------------------------------------------------------------------- probing
def test_list_cameras_reports_size_fps_and_names():
    system = FakeSystem({0: FakeDevice(modes=((640, 480),)), 2: FakeDevice(modes=((1280, 720),), apis={cv2.CAP_MSMF})})
    found = probe_cameras(CameraConfig(), system, max_index=3, names=["Integrated Camera", "USB Cam (busy)", "Logitech"])
    assert [(c.index, c.works) for c in found] == [(0, True), (1, False), (2, True)]
    assert (found[0].width, found[0].height, found[0].name, found[0].backend) == (640, 480, "Integrated Camera", "DirectShow")
    assert found[0].fps > 0
    assert found[1].name == "USB Cam (busy)"
    assert found[2].backend == "Media Foundation" and found[2].name == ""  # DirectShow names only match DirectShow numbers


def test_list_cameras_does_not_hang_on_a_stuck_camera(monkeypatch):
    monkeypatch.setattr(C, "CAMERA_PROBE_TIMEOUT_S", 1.0)
    system = FakeSystem({0: FakeDevice(), 1: FakeDevice()}, open_delay={0: 5.0})
    start = time.monotonic()
    found = probe_cameras(CameraConfig(), system, max_index=2, names=[])
    assert time.monotonic() - start < 3.5
    assert [c.index for c in found] == [1]


def test_probe_uses_quiet_opencv_logging_and_restores_it():
    level = cv2.utils.logging.getLogLevel()
    with cam.quiet_opencv():
        assert cv2.utils.logging.getLogLevel() == cv2.utils.logging.LOG_LEVEL_SILENT
    assert cv2.utils.logging.getLogLevel() == level


# ------------------------------------------------------- fastest backend (auto)
def apis_opened(system):
    return [api for _, api, _ in system.calls]


def test_auto_switches_to_a_faster_backend_when_directshow_is_slow(caplog):
    system = FakeSystem({0: FakeDevice(api_fps={cv2.CAP_DSHOW: 12})})
    with caplog.at_level(logging.INFO, logger="gesture_car.backend.camera"):
        opened = find_camera(CameraConfig(), system)
    assert opened is not None and opened.backend == "Media Foundation"
    assert apis_opened(system) == [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_MSMF]  # measured in a trial, then reopened
    assert "checking whether another backend is faster" in caplog.text


def test_fast_camera_is_not_delayed_by_the_speed_check():
    system = FakeSystem({0: FakeDevice()})
    opened = find_camera(CameraConfig(), system)
    assert opened is not None and opened.backend == "DirectShow"
    assert apis_opened(system) == [cv2.CAP_DSHOW]


def test_slow_everywhere_keeps_the_fastest_backend():
    system = FakeSystem({0: FakeDevice(api_fps={cv2.CAP_DSHOW: 15, cv2.CAP_MSMF: 6, cv2.CAP_ANY: 6})})
    opened = find_camera(CameraConfig(), system)
    assert opened is not None and opened.backend == "DirectShow"
    assert apis_opened(system) == [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]  # measured all, reopened the best
    assert not system.captures[-1].released


def test_forced_backend_skips_the_speed_check():
    system = FakeSystem({0: FakeDevice(api_fps={cv2.CAP_MSMF: 12})})
    opened = find_camera(CameraConfig(backend="msmf"), system)
    assert opened is not None and opened.backend == "Media Foundation"
    assert apis_opened(system) == [cv2.CAP_MSMF]


def test_reconnect_tries_the_backend_that_worked_first():
    device = FakeDevice(api_fps={cv2.CAP_DSHOW: 12})
    system = FakeSystem({0: device})
    manager = CameraManager(CameraConfig(), system)
    try:
        assert manager.start() and manager.backend_name == "Media Foundation"
        opened_before = len(system.calls)
        device.plugged.clear()
        assert wait_for(lambda: manager.status is CameraStatus.RECONNECTING)
        device.plugged.set()
        assert wait_for(lambda: manager.is_ok and len(system.calls) > opened_before)
        assert system.calls[opened_before][1] == cv2.CAP_MSMF
        assert manager.backend_name == "Media Foundation"
    finally:
        manager.stop()


def test_a_hanging_alternative_backend_is_skipped_and_the_first_one_kept(monkeypatch):
    monkeypatch.setattr(C, "CAMERA_SPEED_TRIAL_TIMEOUT_S", 0.5)
    device = FakeDevice(api_fps={cv2.CAP_DSHOW: 15})
    system = FakeSystem({0: device}, open_delay={(0, cv2.CAP_MSMF): 3.0, (0, cv2.CAP_ANY): 3.0})
    start = time.monotonic()
    opened = find_camera(CameraConfig(), system)
    assert time.monotonic() - start < 2.5
    assert opened is not None and opened.backend == "DirectShow"


def test_speed_check_target_ignores_a_driver_that_accepts_60_but_delivers_30():
    from gesture_car.backend.camera import target_fps

    assert target_fps(60, 60.0) == 30.0  # DirectShow reports the 60 it was asked for, then delivers 30
    assert target_fps(60, 0.0) == 30.0
    assert target_fps(15, 0.0) == 15.0
    assert target_fps(30, 25.0) == 25.0
