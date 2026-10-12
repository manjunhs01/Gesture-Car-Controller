import random

import pytest

from gesture_car.backend.filters import EmaFilter, OneEuroFilter


def test_one_euro_first_sample_passes_through_and_converges():
    f = OneEuroFilter()
    assert f.update(10.0, 0.0) == 10.0
    out = 0.0
    for i in range(1, 200):
        out = f.update(20.0, i / 30)
    assert out == pytest.approx(20.0, abs=0.05)


def test_one_euro_reduces_noise_but_follows_fast_motion():
    rng = random.Random(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    noisy, smooth = [], []
    for i in range(300):
        x = 5.0 + rng.uniform(-1, 1)
        noisy.append(x)
        smooth.append(f.update(x, i / 30))
    assert (max(smooth[100:]) - min(smooth[100:])) < (max(noisy[100:]) - min(noisy[100:])) * 0.5
    g = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    for i in range(30):
        g.update(0.0, i / 30)
    out = [g.update(40.0, (30 + i) / 30) for i in range(6)]
    assert out[-1] > 30.0  # a quick wheel turn is not smeared out


def test_reset_forgets_history():
    f = OneEuroFilter()
    f.update(100.0, 0.0)
    f.reset()
    assert f.update(1.0, 1.0) == 1.0


def test_ema():
    f = EmaFilter(0.5)
    assert f.update(0.0) == 0.0
    assert f.update(10.0) == 5.0


def test_light_correction_leaves_a_normal_bright_room_alone_and_never_flickers():
    import numpy as np

    from gesture_car.backend.preprocessing import LightCorrector

    corrector = LightCorrector()
    for level in (185, 192, 188, 194, 186, 190):  # a real webcam in a lit room hovers around 190
        frame = np.full((48, 64, 3), level, np.uint8)
        assert np.array_equal(corrector(frame), frame)
    assert corrector.mode == "none"


def test_light_correction_has_a_dead_band():
    import numpy as np

    from gesture_car.backend.preprocessing import LightCorrector

    corrector = LightCorrector()
    darkened = []
    for level in (60, 78, 82, 78, 90, 80):  # dark: on below 70, stays on until above 85
        frame = np.full((48, 64, 3), level, np.uint8)
        darkened.append(corrector(frame).mean() > level)  # brightened?
    assert darkened == [True, True, True, True, False, False]
    states = []
    for level in (220, 205, 210, 195, 210):  # washed out: on above 215, stays on until below 200
        corrector(np.full((48, 64, 3), level, np.uint8))
        states.append(corrector.mode)
    assert states == ["bright", "bright", "bright", "none", "none"]
