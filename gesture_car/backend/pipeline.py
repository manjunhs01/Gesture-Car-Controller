"""Vision worker thread: camera frame -> preprocessing -> hand landmarks.

Inference runs off the render thread so Pygame keeps drawing smoothly even when
MediaPipe takes longer than a frame. Results are published as immutable
:class:`TrackingResult` objects.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from gesture_car.backend import constants as C
from gesture_car.backend.camera import CameraManager
from gesture_car.backend.hand_tracker import HandTracker
from gesture_car.backend.landmarks import HandLandmarks
from gesture_car.backend.preprocessing import LightCorrector, prepare_frame
from gesture_car.config import VisionConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingResult:
    """Hand detections for one camera frame."""

    sequence: int
    timestamp: float  # capture time, time.monotonic()
    hands: tuple[HandLandmarks, ...]
    inference_ms: float
    frame_size: tuple[int, int]  # (width, height) of the processed frame
    preview_rgb: np.ndarray | None = None
    pipeline_fps: float = 0.0
    wait_ms: float = 0.0  # capture -> hand detection started (frame waiting for the vision thread)
    published: float = 0.0  # time.monotonic() when this result was handed to the output thread

    @property
    def num_hands(self) -> int:
        return len(self.hands)


class VisionPipeline:
    """Runs hand tracking continuously on the newest camera frame."""

    def __init__(
        self,
        camera: CameraManager,
        tracker: HandTracker,
        settings: VisionConfig,
        mirror: bool,
        preview_enabled: Callable[[], bool] = lambda: False,
    ) -> None:
        self._camera = camera
        self._tracker = tracker
        self._settings = settings
        self._mirror = mirror
        self._preview_enabled = preview_enabled
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()  # wakes the output thread the moment a result exists (no polling delay)
        self._result: TrackingResult | None = None
        self._fps = 0.0
        self._errors = 0
        self._light = LightCorrector()  # one per video stream: steady correction, no frame-to-frame flicker

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        try:
            self._tracker.close()
        except Exception:  # closing must never raise into shutdown
            log.exception("Error closing hand tracker")

    def latest(self) -> TrackingResult | None:
        with self._cond:
            return self._result

    def wait_for_new(self, last_sequence: int, timeout: float) -> TrackingResult | None:
        """The newest result if it is newer than ``last_sequence``; otherwise block until one is published (or timeout)."""
        with self._cond:
            if (self._result is None or self._result.sequence <= last_sequence) and timeout > 0:
                self._cond.wait(timeout)
            result = self._result
        return result if result is not None and result.sequence > last_sequence else None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return self._tracker.backend_name

    @property
    def consecutive_errors(self) -> int:
        return self._errors

    # ----------------------------------------------------------------- thread
    def _run(self) -> None:
        last_sequence = 0
        last_done = time.monotonic()
        while not self._stop.is_set():
            frame = self._camera.wait_for_new(last_sequence, C.VISION_WAIT_TIMEOUT_S)
            if frame is None:
                continue
            last_sequence = frame.sequence
            wait_ms = max(0.0, (time.monotonic() - frame.timestamp) * 1000.0)
            started = time.perf_counter()
            try:
                rgb = prepare_frame(frame.image, self._mirror, self._settings.inference_width, self._light)
                hands = self._tracker.detect(rgb, int(frame.timestamp * 1000))
                self._errors = 0
            except Exception:  # MediaPipe/OpenCV failures must not kill the thread
                self._errors += 1
                if self._errors <= 3 or self._errors % 100 == 0:
                    log.exception("Hand tracking failed on frame %d", frame.sequence)
                time.sleep(0.01)
                continue
            inference_ms = (time.perf_counter() - started) * 1000.0
            now = time.monotonic()
            dt = now - last_done
            last_done = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1.0 / dt
            preview = self._make_preview(rgb) if self._preview_enabled() else None
            height, width = rgb.shape[:2]
            result = TrackingResult(
                sequence=frame.sequence,
                timestamp=frame.timestamp,
                hands=tuple(hands),
                inference_ms=inference_ms,
                frame_size=(width, height),
                preview_rgb=preview,
                pipeline_fps=self._fps,
                wait_ms=wait_ms,
                published=time.monotonic(),
            )
            with self._cond:
                self._result = result
                self._cond.notify_all()

    @staticmethod
    def _make_preview(rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        if width <= C.PREVIEW_WIDTH:
            return rgb.copy()
        scale = C.PREVIEW_WIDTH / width
        small = cv2.resize(rgb, (C.PREVIEW_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(small)
