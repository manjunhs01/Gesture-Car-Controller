from gesture_car.backend.selfcheck import Report, check_gesture, check_wheel
from gesture_car.config import Config
from tests.support.synthetic_hand import fist, open_palm, thumbs_up, wheel_pair

DT = 1 / 30


def frames(hand_sets):
    return [(100.0 + i * DT, h) for i, h in enumerate(hand_sets)]


def test_wheel_check_passes_for_a_steady_hold_then_tilts():
    seq = [wheel_pair(0)] * 70 + [wheel_pair(35)] * 40 + [wheel_pair(-35)] * 40
    step = check_wheel(frames(seq), Config())
    assert step.passed, step.detail


def test_wheel_check_explains_a_missing_hand_and_missing_tilt():
    one = frames([(wheel_pair(0)[0],)] * 150)
    assert not check_wheel(one, Config()).passed and "BOTH hands" in check_wheel(one, Config()).detail
    flat = check_wheel(frames([wheel_pair(0)] * 150), Config())
    assert not flat.passed and "tilt" in flat.detail


def test_gesture_checks():
    palm = frames([(open_palm(), fist(center=(0.7, 0.55)))] * 20)
    assert check_gesture(palm, Config(), "brake").passed
    assert not check_gesture(palm, Config(), "nitro").passed
    thumb = frames([(thumbs_up(), fist(center=(0.7, 0.55)))] * 20)
    assert check_gesture(thumb, Config(), "nitro").passed
    none = frames([(fist(), fist(center=(0.7, 0.55)))] * 20)
    assert "five fingers" in check_gesture(none, Config(), "brake").detail


def test_report_text():
    r = Report()
    r.add("a", True, "x")
    r.add("b", False, "y")
    assert not r.ok and "FAIL  b: y" in r.text() and "SOME CHECKS FAILED" in r.text()
