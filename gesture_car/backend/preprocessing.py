"""Frame preprocessing: mirror, downscale, light correction, BGR -> RGB."""

from __future__ import annotations

import cv2
import numpy as np

# Brightness correction only for clearly bad exposure, with a dead band (hysteresis): a normal picture sitting near a
# threshold must not flicker between corrected and uncorrected frames, because the sudden brightness jump between two
# frames makes the hand tracker lose hands and adds jitter. A well-lit room is around 120-195 mean brightness.
DARK_ON, DARK_OFF = 70.0, 85.0  # brighten below DARK_ON; keep brightening until above DARK_OFF
BRIGHT_ON, BRIGHT_OFF = 215.0, 200.0  # darken above BRIGHT_ON (washed out); keep darkening until below BRIGHT_OFF
DARK_GAMMA = 0.6  # < 1 brightens
BRIGHT_GAMMA = 1.5  # > 1 darkens
_SAMPLE_STEP = 8


def _gamma_lut(gamma: float) -> np.ndarray:
    table = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    return np.clip(table, 0, 255).astype(np.uint8)


_DARK_LUT = _gamma_lut(DARK_GAMMA)
_BRIGHT_LUT = _gamma_lut(BRIGHT_GAMMA)


def mean_brightness(bgr: np.ndarray) -> float:
    """Cheap brightness estimate from a sparse sample of pixels."""
    return float(bgr[::_SAMPLE_STEP, ::_SAMPLE_STEP].mean())


class LightCorrector:
    """Gamma correction for clearly too dark / too bright scenes that stays on or off steadily (hysteresis)."""

    def __init__(self) -> None:
        self.mode = "none"  # none | dark | bright

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        brightness = mean_brightness(bgr)
        if (self.mode == "dark" and brightness > DARK_OFF) or (self.mode == "bright" and brightness < BRIGHT_OFF):
            self.mode = "none"
        if self.mode == "none":
            if brightness < DARK_ON:
                self.mode = "dark"
            elif brightness > BRIGHT_ON:
                self.mode = "bright"
        if self.mode == "dark":
            return cv2.LUT(bgr, _DARK_LUT)
        if self.mode == "bright":
            return cv2.LUT(bgr, _BRIGHT_LUT)
        return bgr


def correct_lighting(bgr: np.ndarray) -> np.ndarray:
    """One-off correction (no memory between frames); the video pipeline uses a :class:`LightCorrector`."""
    return LightCorrector()(bgr)


def prepare_frame(bgr: np.ndarray, mirror: bool, max_width: int, corrector: LightCorrector | None = None) -> np.ndarray:
    """Return a contiguous RGB frame ready for hand detection.

    Mirroring makes moving the hand to the user's right move it right on screen. Pass the same ``corrector`` for every
    frame of a video so the light correction does not flicker.
    """
    height, width = bgr.shape[:2]
    if width > max_width:
        scale = max_width / width
        bgr = cv2.resize(bgr, (max_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    if mirror:
        bgr = cv2.flip(bgr, 1)
    bgr = (corrector or LightCorrector())(bgr)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
