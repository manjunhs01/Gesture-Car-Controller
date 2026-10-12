import json

from gesture_car.config import Config


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return p


def test_defaults_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    Config().save(p)
    assert Config.load(p).to_dict() == Config().to_dict()


def test_missing_and_corrupt_files_fall_back_to_defaults(tmp_path):
    assert Config.load(tmp_path / "nope.json").to_dict() == Config().to_dict()
    assert Config.load(write(tmp_path, "{not json")).to_dict() == Config().to_dict()
    assert Config.load(write(tmp_path, "[1,2]")).to_dict() == Config().to_dict()


def test_values_are_applied_clamped_and_validated(tmp_path):
    cfg = Config.load(write(tmp_path, {
        "steering": {"dead_zone_deg": 5, "max_angle_deg": 500, "smoothing": "EMA", "invert": True},
        "output": {"backend": "keyboard", "nitro_button": "z", "key_nitro": "space"},
        "gestures": {"confirm_frames": "three"},
        "bogus": {"x": 1},
        "app": {"loop_hz": 5},
    }))  # fmt: skip
    assert cfg.steering.deadzone_deg == 5.0  # older key names still load
    assert cfg.steering.full_lock_deg == 90.0  # clamped
    assert cfg.steering.smoothing == "ema"
    assert cfg.steering.invert_steering is True
    assert cfg.output.backend == "keyboard" and cfg.output.key_nitro == "space"
    assert cfg.output.nitro_button == "a"  # invalid choice rejected
    assert cfg.gestures.confirm_frames == 2  # wrong type rejected
    assert cfg.app.loop_hz == 30  # clamped up


def test_repo_config_json_matches_defaults():
    from gesture_car.config import default_path

    assert Config.load(default_path()).to_dict() == Config().to_dict()


def test_camera_source_backend_and_mjpg(tmp_path):
    cfg = Config.load(write(tmp_path, {"camera": {"source": 1, "backend": "MSMF", "mjpg": False, "mirror": False}}))
    assert (cfg.camera.source, cfg.camera.backend, cfg.camera.mjpg, cfg.camera.mirror) == ("1", "msmf", False, False)
    cfg = Config.load(write(tmp_path, {"camera": {"source": "http://192.168.1.5:8080/video", "backend": "v4l2"}}))
    assert cfg.camera.source == "http://192.168.1.5:8080/video"
    assert cfg.camera.backend == "auto"  # invalid choice keeps the default
