"""MediaPipe hand detection behind a small backend-neutral interface.

Two backends are supported because MediaPipe changed its API:

* ``tasks``  - the HandLandmarker task API (needs ``hand_landmarker.task``, which
  is downloaded once to the user data directory).
* ``legacy`` - ``mediapipe.solutions.hands`` (model bundled in the wheel; removed
  from the newest MediaPipe releases).

``auto`` prefers ``tasks`` and falls back to ``legacy``. With ``vision.use_gpu`` the tasks backend also tries MediaPipe's
GPU delegate, benchmarks it against the CPU and keeps the faster one (falling back to the CPU, with a log line, when the
GPU delegate is not supported - as on Windows).
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from gesture_car.backend import constants as C
from gesture_car.backend.landmarks import NUM_LANDMARKS, HandLandmarks
from gesture_car.config import VisionConfig
from gesture_car.utils.errors import TrackerError
from gesture_car.utils.paths import bundle_dir, models_dir

log = logging.getLogger(__name__)


class TrackerBackend(Protocol):
    name: str

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandLandmarks]: ...

    def close(self) -> None: ...


# ------------------------------------------------------------------ model file
def model_file_path(settings: VisionConfig) -> Path:
    if settings.model_path:
        return Path(settings.model_path).expanduser()
    bundled = bundle_dir()  # the installer ships the model so the game works offline
    if bundled is not None and (bundled / "models" / C.MODEL_FILENAME).is_file():
        return bundled / "models" / C.MODEL_FILENAME
    return models_dir() / C.MODEL_FILENAME


def ensure_model(path: Path, url: str = C.MODEL_URL) -> Path:
    """Return ``path`` if it holds a valid model, downloading it first if needed."""
    if path.is_file() and path.stat().st_size >= C.MODEL_MIN_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    log.info("Downloading hand model from %s", url)
    try:
        with urllib.request.urlopen(url, timeout=C.MODEL_DOWNLOAD_TIMEOUT_S) as response, open(
            tmp, "wb"
        ) as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise TrackerError(
            f"Could not download the hand model: {exc}",
            user_message=(
                "The hand-tracking model is missing and could not be downloaded. "
                f"Download it from {url} and save it as {path}."
            ),
        ) from exc
    if tmp.stat().st_size < C.MODEL_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise TrackerError("Downloaded hand model is too small (corrupt download)")
    tmp.replace(path)
    return path


# -------------------------------------------------------------------- helpers
def _to_hand(
    xyz: np.ndarray, handedness: str, score: float, width: int, height: int
) -> HandLandmarks | None:
    if xyz.shape != (NUM_LANDMARKS, 3) or not np.all(np.isfinite(xyz)):
        return None
    return HandLandmarks(
        points=xyz.astype(np.float32),
        handedness=handedness,
        score=float(score),
        aspect=width / max(1, height),
    )


class TasksBackend:
    """MediaPipe Tasks HandLandmarker (VIDEO running mode: tracks across frames, the palm detector only re-runs when tracking
    is lost)."""

    name = "tasks"

    def __init__(self, settings: VisionConfig, delegate: str = "CPU") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = ensure_model(model_file_path(settings))
        self.delegate = delegate
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model), delegate=getattr(mp_python.BaseOptions.Delegate, delegate)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=settings.min_detection_confidence,
            min_hand_presence_confidence=settings.min_presence_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandLandmarks]:
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)  # must strictly increase
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        height, width = rgb.shape[:2]
        hands: list[HandLandmarks] = []
        for i, landmarks in enumerate(result.hand_landmarks):
            xyz = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            label, score = "Unknown", 1.0
            if i < len(result.handedness) and result.handedness[i]:
                top = result.handedness[i][0]
                label, score = top.category_name, top.score
            hand = _to_hand(xyz, label, score, width, height)
            if hand is not None:
                hands.append(hand)
        return hands

    def close(self) -> None:
        self._landmarker.close()


class LegacyBackend:
    """``mediapipe.solutions.hands`` (available in older MediaPipe releases)."""

    name = "legacy"

    def __init__(self, settings: VisionConfig) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise TrackerError("This MediaPipe version has no legacy 'solutions' API")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=settings.max_hands,
            model_complexity=settings.model_complexity,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandLandmarks]:
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True
        height, width = rgb.shape[:2]
        hands: list[HandLandmarks] = []
        if not results.multi_hand_landmarks:
            return hands
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            xyz = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
            label, score = "Unknown", 1.0
            if results.multi_handedness and i < len(results.multi_handedness):
                top = results.multi_handedness[i].classification[0]
                label, score = top.label, top.score
            hand = _to_hand(xyz, label, score, width, height)
            if hand is not None:
                hands.append(hand)
        return hands

    def close(self) -> None:
        self._hands.close()


class HandTracker:
    """Backend-neutral facade used by the vision pipeline."""

    def __init__(self, backend: TrackerBackend) -> None:
        self._backend = backend

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandLandmarks]:
        return self._backend.detect(rgb, timestamp_ms)

    def close(self) -> None:
        self._backend.close()


GPU_BENCH_FRAMES = 20  # timed detections per delegate (after a few warm-up runs)


def benchmark_ms(backend: TrackerBackend, width: int = 640, height: int = 480, frames: int = GPU_BENCH_FRAMES) -> float:
    """Median detection time on a blank frame (no hands: the palm detector runs every time, the worst case)."""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    times = []
    for i in range(frames + 3):
        started = time.perf_counter()
        backend.detect(rgb, i + 1)
        if i >= 3:
            times.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(times))


def create_tasks_backend(settings: VisionConfig) -> TrackerBackend:
    """CPU tasks backend, or the GPU one if ``use_gpu`` is set, the GPU delegate works and it is really faster."""
    if not settings.use_gpu:
        return TasksBackend(settings)
    try:
        gpu = TasksBackend(settings, delegate="GPU")
    except Exception as exc:  # MediaPipe raises assorted errors when the GPU delegate is unsupported
        log.warning("GPU hand tracking unavailable (%s); using the CPU", str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
        return TasksBackend(settings)
    cpu = TasksBackend(settings)
    try:
        gpu_ms, cpu_ms = benchmark_ms(gpu), benchmark_ms(cpu)
    except Exception as exc:
        gpu.close()
        log.warning("GPU hand tracking failed during the speed test (%s); using the CPU", exc)
        return cpu
    if gpu_ms < cpu_ms:
        cpu.close()
        log.info("Hand tracking on the GPU: %.1f ms per frame (CPU %.1f ms)", gpu_ms, cpu_ms)
        return gpu
    gpu.close()
    log.info("Hand tracking on the CPU: %.1f ms per frame (GPU was slower: %.1f ms)", cpu_ms, gpu_ms)
    return cpu


def create_hand_tracker(settings: VisionConfig) -> HandTracker:
    """Create the configured tracker, falling back between backends in ``auto`` mode.

    Raises :class:`TrackerError` (with a user-safe message) if none can start.
    """
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise TrackerError(
            f"MediaPipe is not installed: {exc}",
            user_message="MediaPipe is not installed. Run: pip install -r requirements.txt",
        ) from exc

    order = {"auto": ("tasks", "legacy"), "tasks": ("tasks",), "legacy": ("legacy",)}[settings.backend]
    errors: list[str] = []
    last_user_message = TrackerError.user_message
    for name in order:
        try:
            backend: TrackerBackend = create_tasks_backend(settings) if name == "tasks" else LegacyBackend(settings)
        except TrackerError as exc:
            errors.append(f"{name}: {exc}")
            log.warning("Hand tracker backend %s unavailable: %s", name, exc)
            last_user_message = exc.user_message
            continue
        except Exception as exc:  # MediaPipe raises assorted RuntimeError/ValueError types
            errors.append(f"{name}: {exc}")
            log.exception("Hand tracker backend %s failed to initialise", name)
            last_user_message = TrackerError.user_message
            continue
        log.info("Hand tracker backend: %s", name)
        return HandTracker(backend)
    raise TrackerError("; ".join(errors), user_message=last_user_message)
