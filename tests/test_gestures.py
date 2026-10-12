from gesture_car.backend.gestures import GestureEngine, Latch, classify_hand
from gesture_car.config import GestureConfig
from tests.support.synthetic_hand import fist, make_hand, open_palm, place, thumbs_up

CFG = GestureConfig()
DT = 1 / 30


def test_open_palm_detected():
    pose = classify_hand(open_palm(), CFG)
    assert pose.open_palm and not pose.thumbs_up and pose.fingers == 5


def test_a_natural_open_palm_counts_without_spreading_the_fingers():
    """All five fingers extended is a palm: fingers held together and a relaxed thumb used to be rejected."""
    for fan in (0.0, 0.25):
        for thumb in (0.6, 0.8, 1.0):
            pose = classify_hand(make_hand(extension=1.0, thumb=thumb, fan=fan), CFG)
            assert pose.open_palm and pose.fingers == 5, (fan, thumb)


def test_spread_requirement_is_still_available():
    from gesture_car.config import GestureConfig

    strict = GestureConfig(palm_min_spread=1.1)
    assert not classify_hand(make_hand(extension=1.0, thumb=1.0, fan=0.0), strict).open_palm
    assert classify_hand(open_palm(), strict).open_palm


def test_finger_counts():
    assert classify_hand(fist(), CFG).fingers == 0
    assert classify_hand(thumbs_up(), CFG).fingers == 1
    assert classify_hand(make_hand(extension=(1.0, 1.0, 0.0, 0.0), thumb=0.0), CFG).fingers == 2  # peace sign
    assert classify_hand(make_hand(extension=0.35, thumb=0.3), CFG).fingers == 0  # half-curled wheel grip


def test_fist_is_neither_gesture():
    pose = classify_hand(fist(), CFG)
    assert not pose.open_palm and not pose.thumbs_up


def test_thumbs_up_detected():
    pose = classify_hand(thumbs_up(), CFG)
    assert pose.thumbs_up and not pose.open_palm
    assert pose.thumb_tilt_deg < 30


def test_thumb_pointing_down_is_not_thumbs_up():
    assert not classify_hand(thumbs_up(roll_deg=180), CFG).thumbs_up
    assert not classify_hand(thumbs_up(roll_deg=90), CFG).thumbs_up  # thumb pointing sideways


def test_half_open_hand_is_not_a_palm():
    assert not classify_hand(make_hand(extension=(1.0, 1.0, 0.3, 0.3), thumb=1.0, fan=1.0), CFG).open_palm


def test_open_palm_works_at_other_sizes_and_positions():
    for scale in (0.12, 0.3):
        assert classify_hand(open_palm(scale=scale, center=(0.3, 0.4)), CFG).open_palm


def test_latch_needs_confirm_frames_and_holds_through_flicker():
    latch = Latch(3, 0.15)
    t = 0.0
    out = []
    for present in (True, True, False, True, True, True):
        t += DT
        out.append(latch.update(present, t))
    assert out == [False, False, False, False, False, True]  # the gap reset the streak
    t += DT
    assert latch.update(False, t)  # one missed frame: still on (release hysteresis)
    t += 0.2
    assert not latch.update(False, t)  # gone for > release_s


def _engine():
    return GestureEngine(CFG)


def _run(engine, hands, frames):
    t = _run.t
    state = None
    for _ in range(frames):
        t += DT
        state = engine.update(hands, t)
    _run.t = t
    return state


_run.t = 0.0


def test_engine_brake_after_two_frames_either_hand():
    for side in ("left", "right"):
        e = _engine()
        wheel_hand = fist(center=(0.7 if side == "left" else 0.3, 0.55))
        palm = place(open_palm, 0.0, side)
        hands = (palm, wheel_hand) if side == "left" else (wheel_hand, palm)
        assert not _run(e, hands, 1).brake
        assert _run(e, hands, 1).brake


def test_engine_nitro_and_brake_priority():
    e = _engine()
    pair = (place(thumbs_up, 0.0, "left"), fist(center=(0.7, 0.55)))
    assert _run(e, pair, 4).nitro
    both = (place(open_palm, 0.0, "left"), place(thumbs_up, 0.0, "right"))
    state = _run(e, both, 5)
    assert state.brake and not state.nitro  # brake wins


def test_engine_ignores_low_confidence_hands():
    e = _engine()
    weak = open_palm(score=0.2)
    assert not _run(e, (weak, fist()), 6).brake
