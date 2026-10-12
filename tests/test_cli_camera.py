"""Webcam command-line flags."""

import pytest

from gesture_car import cli
from gesture_car.backend.camera import CameraInfo
from gesture_car.config import CameraConfig


def apply(*argv, cam=None):
    cam = cam or CameraConfig()
    cli.apply_camera_args(cam, cli.build_parser().parse_args(list(argv)))
    return cam


def test_defaults_leave_config_untouched():
    assert apply() == CameraConfig()


def test_camera_resolution_fps_backend_and_mirror_flags():
    cam = apply("--camera", "2", "--resolution", "1280x720", "--fps", "60", "--camera-backend", "dshow", "--no-mirror")
    assert (cam.index, cam.width, cam.height, cam.fps, cam.backend, cam.mirror) == (2, 1280, 720, 60, "dshow", False)


def test_camera_number_overrides_a_url_from_config():
    cam = apply("--camera", "1", cam=CameraConfig(source="http://phone:8080/video"))
    assert (cam.index, cam.source) == (1, "")


def test_camera_url_flag():
    assert apply("--camera-url", " http://192.168.1.5:8080/video ").source == "http://192.168.1.5:8080/video"


@pytest.mark.parametrize(
    "argv",
    [
        ["--camera", "-1"],
        ["--camera", "x"],
        ["--fps", "500"],
        ["--resolution", "800x600"],
        ["--camera-backend", "v4l2"],
        ["--camera", "1", "--camera-url", "http://x"],
    ],
)
def test_bad_values_are_rejected(argv, capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_list_cameras_prints_size_fps_and_name(monkeypatch, capsys):
    found = [CameraInfo(0, 640, 480, 29.8, "DirectShow", "Integrated Camera", fourcc="MJPG"), CameraInfo(1, name="USB Cam", works=False)]
    monkeypatch.setattr("gesture_car.backend.camera.probe_cameras", lambda cam: found)
    assert cli.list_cameras(CameraConfig()) == 0
    out = capsys.readouterr().out
    assert 'camera 0: 640x480, 29.8 fps measured, DirectShow, MJPG  "Integrated Camera"' in out
    assert "camera 1: Windows lists it, but it gave no picture" in out
    assert "--camera 0" in out


def test_list_cameras_explains_what_to_check_when_none_work(monkeypatch, capsys):
    monkeypatch.setattr("gesture_car.backend.camera.probe_cameras", lambda cam: [])
    assert cli.list_cameras(CameraConfig()) == 1
    assert "Privacy & security > Camera" in capsys.readouterr().out
