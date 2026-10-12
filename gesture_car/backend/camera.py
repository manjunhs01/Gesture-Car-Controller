"""Threaded webcam capture with validation, discovery, fallback and reconnection.

A background thread keeps reading frames so the render loop never blocks on the
camera. Only the newest frame is kept (no queue growth, no lag build-up); this also
keeps a lagging phone stream current, because older frames are simply overwritten.

Sources: a webcam number (``camera.index`` / ``--camera``) or a network stream URL
(``camera.source`` / ``--camera-url``, e.g. a phone running IP Webcam or DroidCam).
On Windows the capture backend order is DirectShow, Media Foundation, then OpenCV's
default (``camera.backend``). USB webcams are asked for MJPG, and the closest
supported resolution to the requested one is used.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import cv2
import numpy as np

from gesture_car.backend import constants as C
from gesture_car.config import CameraConfig

log = logging.getLogger(__name__)

T = TypeVar("T")
Source = int | str  # webcam number or stream URL
CaptureFactory = Callable[[Source, int, list[int]], Any]  # (source, api, params) -> cv2.VideoCapture-like object

# Common webcam modes, used to find the closest supported one when the requested mode is refused.
STANDARD_MODES: tuple[tuple[int, int], ...] = (
    (320, 240), (424, 240), (640, 360), (640, 480), (800, 600), (848, 480), (960, 540),
    (1024, 768), (1280, 720), (1280, 960), (1600, 900), (1920, 1080),
)  # fmt: skip
MAX_MODE_TRIES = 6
OPEN_TEST_BUDGET_S = 6.0  # a phone stream whose reads keep timing out is given up after this long
FPS_LOG_AFTER_S = 3.0
FPS_WARMUP_FRAMES = 10  # frames thrown away before measuring the real FPS
FPS_WARMUP_S = 2.0
MJPG_FOURCC = sum(ord(ch) << 8 * i for i, ch in enumerate("MJPG"))  # same as cv2.VideoWriter_fourcc(*"MJPG")
BACKEND_NAMES = {"dshow": "DirectShow", "msmf": "Media Foundation", "ffmpeg": "FFMPEG", "default": "default"}


class CameraStatus(Enum):
    CLOSED = "CLOSED"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CameraFrame:
    """One captured BGR frame. ``timestamp`` is ``time.monotonic()`` at capture."""

    image: np.ndarray
    timestamp: float
    sequence: int


@dataclass(frozen=True)
class CameraInfo:
    """One camera found by :func:`probe_cameras`. ``works`` is False for a camera Windows lists but that gave no picture."""

    index: int
    width: int = 0
    height: int = 0
    fps: float = 0.0  # measured, not what the driver claims
    backend: str = ""
    name: str = ""
    works: bool = True
    fourcc: str = ""


@dataclass
class OpenedCamera:
    """A capture that has already delivered a valid frame, plus what it really runs at."""

    cap: Any
    source: Source
    backend: str
    width: int
    height: int
    fps: float  # as reported by the driver (0 if unknown)
    fourcc: str
    backend_key: str = ""  # dshow | msmf | ffmpeg | default
    measured_fps: float = 0.0  # set when a speed check ran


def is_valid_frame(frame: object) -> bool:
    """A usable frame is a non-empty HxWx3 uint8 array."""
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
        and frame.dtype == np.uint8
    )


# ------------------------------------------------------------------ choosing
def parse_source(settings: CameraConfig) -> Source:
    """``camera.source`` if set (a number means that webcam, anything else is a URL), otherwise ``camera.index``."""
    text = settings.source.strip()
    if not text:
        return settings.index
    if text.isdigit():
        return int(text)
    return text


def describe_source(source: Source | None) -> str:
    if source is None:
        return "no camera"
    return f"camera {source}" if isinstance(source, int) else f"stream {source}"


def backend_order(choice: str, source: Source, platform: str | None = None) -> list[tuple[str, int]]:
    """Capture backends to try, in order, as (name, OpenCV API id)."""
    if isinstance(source, str):
        return [("ffmpeg", cv2.CAP_FFMPEG), ("default", cv2.CAP_ANY)]
    if not (platform or sys.platform).startswith("win"):
        return [("default", cv2.CAP_ANY)]  # dshow / msmf only exist on Windows
    if choice == "dshow":
        return [("dshow", cv2.CAP_DSHOW)]
    if choice == "msmf":
        return [("msmf", cv2.CAP_MSMF)]
    return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)]


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def closest_modes(requested: tuple[int, int], modes: tuple[tuple[int, int], ...] = STANDARD_MODES) -> list[tuple[int, int]]:
    """``modes`` sorted from closest to furthest from ``requested`` (ties: the larger mode first)."""
    return sorted(modes, key=lambda m: (_distance(m, requested), -m[0]))


def default_factory(source: Source, api: int, params: list[int]) -> Any:
    return cv2.VideoCapture(source, api, params) if params else cv2.VideoCapture(source, api)


@contextlib.contextmanager
def quiet_opencv() -> Iterator[None]:
    """Hide OpenCV's own "can't open camera by index" messages while searching; we log clear messages ourselves."""
    cvlog = getattr(getattr(cv2, "utils", None), "logging", None)
    previous = None
    try:
        previous = cvlog.getLogLevel()  # type: ignore[union-attr]
        cvlog.setLogLevel(getattr(cvlog, "LOG_LEVEL_SILENT", 0))  # type: ignore[union-attr]
    except Exception:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            with contextlib.suppress(Exception):
                cvlog.setLogLevel(previous)  # type: ignore[union-attr]


# ------------------------------------------------------------------- opening
def _size(cap: Any) -> tuple[int, int]:
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)


def _set_size(cap: Any, mode: tuple[int, int]) -> tuple[int, int]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode[1])
    return _size(cap)


def _fourcc(cap: Any) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    text = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    return text if code and text.isprintable() and text.strip() else ""


# CAP_PROP_AUTO_EXPOSURE values differ per backend: DirectShow uses 0.75 (auto) / 0.25 (manual), Media Foundation 1 / 0.
AUTO_EXPOSURE_VALUES = {"dshow": (0.75, 0.25)}
DEFAULT_AUTO_EXPOSURE_VALUES = (1.0, 0.0)


def apply_exposure(cap: Any, settings: CameraConfig, backend_key: str) -> None:
    """Set auto or fixed exposure explicitly. Webcam drivers remember a manual exposure left behind by another app (a dark
    picture), and auto-exposure in a dim room can halve the frame rate; a fixed exposure keeps the full rate."""
    auto_value, manual_value = AUTO_EXPOSURE_VALUES.get(backend_key, DEFAULT_AUTO_EXPOSURE_VALUES)
    with contextlib.suppress(cv2.error):
        if settings.auto_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_value)
            cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)


def _apply_mode(cap: Any, width: int, height: int, fps: int, mjpg: bool) -> None:
    """Ask for MJPG, the requested size (or the closest standard mode the camera accepts) and the frame rate."""
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
    requested = (width, height)
    got = _set_size(cap, requested)
    if got != requested and got[0] > 0:
        offered = got
        tries = 0
        for mode in closest_modes(requested):
            if mode == requested:
                continue
            if _distance(mode, requested) >= _distance(offered, requested) or tries >= MAX_MODE_TRIES:
                break  # what the driver offered is already at least as close
            tries += 1
            if _set_size(cap, mode) == mode:
                got = mode
                break
        if got != _size(cap):
            _set_size(cap, got)
        log.info("Camera refused %dx%d; using the closest mode it supports: %dx%d", width, height, *got)
    cap.set(cv2.CAP_PROP_FPS, fps)


def _first_frame(cap: Any) -> np.ndarray | None:
    """Prove the camera really delivers pictures (some open fine but only return errors)."""
    deadline = time.monotonic() + OPEN_TEST_BUDGET_S
    for _ in range(C.CAMERA_OPEN_TEST_READS):
        ok, frame = cap.read()
        if ok and is_valid_frame(frame):
            return np.asarray(frame)
        if time.monotonic() > deadline:
            break
        time.sleep(0.03)
    return None


def _open_with(source: Source, settings: CameraConfig, factory: CaptureFactory, name: str, api: int, params: list[int]) -> OpenedCamera | None:
    """Open ``source`` with one backend (MJPG first, then without); None if it delivers no frame."""
    url = isinstance(source, str)
    for mjpg in (True, False) if settings.mjpg and not url else (False,):
        cap = None
        try:
            cap = factory(source, api, params)
            if not cap.isOpened():
                return None  # the device did not open at all; retrying without MJPG would not help
            if not url:
                _apply_mode(cap, settings.width, settings.height, settings.fps, mjpg)
                apply_exposure(cap, settings, name)
            with contextlib.suppress(cv2.error):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = _first_frame(cap)
            if frame is not None:
                height, width = frame.shape[:2]  # the picture itself is the truth, not what the driver reports
                opened = OpenedCamera(
                    cap, source, BACKEND_NAMES[name], width, height, float(cap.get(cv2.CAP_PROP_FPS) or 0.0), _fourcc(cap), name
                )
                cap = None
                return opened
            log.debug("%s opened via %s%s but gave no picture", describe_source(source), name, " (MJPG)" if mjpg else "")
        except cv2.error:
            log.debug("OpenCV error opening %s via %s", describe_source(source), name, exc_info=True)
        finally:
            if cap is not None:
                cap.release()
    return None


def target_fps(requested: int, driver_fps: float) -> float:
    """The rate a backend must reach to be kept without trying the others: what was asked for, capped at what the driver
    reports and at a normal webcam's 30 fps. Drivers often accept 60 but deliver 30; that must not make every backend
    look slow (and cost seconds of start-up). The search is for backends that drop to 15-20 fps."""
    target = min(float(requested), float(C.CAMERA_TYPICAL_FPS))
    return min(target, driver_fps) if driver_fps > 0 else target


def _trial_fps(source: Source, settings: CameraConfig, factory: CaptureFactory, name: str, api: int, params: list[int]) -> float | None:
    """Open with one backend, measure its real FPS, release it. None if it gives no picture."""
    opened = _open_with(source, settings, factory, name, api, params)
    if opened is None:
        return None
    try:
        return measure_fps(opened.cap, C.CAMERA_SPEED_CHECK_S)
    finally:
        opened.cap.release()


def open_source(
    source: Source,
    settings: CameraConfig,
    factory: CaptureFactory = default_factory,
    speed_check: bool = False,
    first: str = "",
) -> OpenedCamera | None:
    """Open ``source`` with each backend in turn; None if nothing delivers a frame.

    ``speed_check`` (webcams, ``camera.backend: auto``): if the first working backend delivers clearly fewer frames than
    asked for, the other backends are measured too and the fastest is used. Some webcams give 20 fps through DirectShow but
    30 through Media Foundation. A camera that is already fast is not delayed. ``first`` moves that backend to the front
    (used when reconnecting, so the backend that worked before is tried first).
    """
    url = isinstance(source, str)
    params: list[int] = []
    if url:
        # Low-latency network streaming: no extra buffering, give up instead of hanging when the phone goes away.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, C.CAMERA_URL_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, C.CAMERA_URL_TIMEOUT_MS]
    order = backend_order(settings.backend, source)
    order.sort(key=lambda item: item[0] != first)  # stable: only moves ``first`` to the front
    check = speed_check and not url and settings.backend == "auto" and len(order) > 1
    best: tuple[float, str, int] | None = None  # (measured fps, name, api)
    target = float(settings.fps)
    for name, api in order:
        if best is not None:
            # An alternative backend: measured in a helper thread with a time limit, because some drivers take very
            # long to open (or never answer); a stuck one is skipped and the earlier backend is kept.
            trial = _run_with_timeout(functools.partial(_trial_fps, source, settings, factory, name, api, params), C.CAMERA_SPEED_TRIAL_TIMEOUT_S)
            if trial is not None and trial > best[0] * C.CAMERA_SPEED_BETTER_RATIO:
                best = (trial, name, api)
                if trial >= C.CAMERA_SPEED_OK_RATIO * target:
                    break  # fast enough: no need to try more backends
            continue
        opened = _open_with(source, settings, factory, name, api, params)
        if opened is None:
            continue
        if not check:
            return opened
        fps = measure_fps(opened.cap, C.CAMERA_SPEED_CHECK_S)
        opened.measured_fps = fps
        target = target_fps(settings.fps, opened.fps)
        if fps >= C.CAMERA_SPEED_OK_RATIO * target:
            return opened  # fast enough: keep it
        log.info("%s via %s delivers only %.0f fps; checking whether another backend is faster...", describe_source(source), opened.backend, fps)
        opened.cap.release()  # webcams are exclusive: free it before trying the next backend
        best = (fps, name, api)
    if best is None:
        return None
    fps, name, api = best
    opened = _open_with(source, settings, factory, name, api, params)
    if opened is not None:
        opened.measured_fps = fps
        log.info("Picked %s: the fastest backend for this camera (%.0f fps)", opened.backend, fps)
    return opened


def find_camera(
    settings: CameraConfig, factory: CaptureFactory = default_factory, preferred: Source | None = None
) -> OpenedCamera | None:
    """Open the chosen source; if a webcam number fails, try the other webcams (0..9) and use the first that works."""
    primary = preferred if preferred is not None else parse_source(settings)
    with quiet_opencv():
        opened = open_source(primary, settings, factory, speed_check=True)
        if opened is not None or isinstance(primary, str):
            return opened  # a URL is an explicit choice: never silently swap it for a webcam
        log.warning("Camera %d could not be opened; looking for another camera...", primary)
        for index in range(C.CAMERA_PROBE_MAX_INDEX + 1):
            if index == primary:
                continue
            opened = open_source(index, settings, factory, speed_check=True)
            if opened is not None:
                log.warning("Using camera %d instead of camera %d (pick one with --camera N; see --list-cameras)", index, primary)
                return opened
    return None


def no_camera_message(source: Source) -> str:
    """What to check when no camera works, in plain words."""
    if isinstance(source, str):
        return (
            f"Could not open the phone/network camera {source}.\n"
            "Please check:\n"
            "  1. The camera app (IP Webcam / DroidCam) is running on the phone and the screen shows the address.\n"
            "  2. The phone and this PC are on the SAME Wi-Fi network.\n"
            "  3. The address is typed exactly (including http:// and the port) and opens in a web browser on this PC.\n"
            "  4. Windows Firewall or a VPN is not blocking the connection."
        )
    return (
        "No working camera was found.\n"
        "Please check:\n"
        "  1. No other app is using the camera (Zoom, Teams, Skype, the Windows Camera app, a browser tab). Close them.\n"
        "  2. Windows camera privacy: Settings > Privacy & security > Camera > turn on 'Camera access' and\n"
        "     'Let desktop apps access your camera'.\n"
        "  3. The USB cable is firmly plugged in (try another USB port), and any camera privacy shutter or\n"
        "     camera on/off key is open.\n"
        "Then run 'python main.py --list-cameras' to see which cameras work, and pick one with --camera N.\n"
        "If an external webcam only fails here, try '--camera-backend dshow' or '--camera-backend msmf'."
    )


# ------------------------------------------------------------------- probing
def measure_fps(cap: Any, seconds: float) -> float:
    """Frames per second the camera really delivers, timed over ``seconds`` after a short warm-up.

    The first frames after opening are slow (auto-exposure settling), so they are read and thrown away first."""
    warm_end = time.monotonic() + FPS_WARMUP_S
    good = 0
    while good < FPS_WARMUP_FRAMES and time.monotonic() < warm_end:
        ok, frame = cap.read()
        good += bool(ok and is_valid_frame(frame))
    frames = 0
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if ok and is_valid_frame(frame):
            frames += 1
    elapsed = time.monotonic() - start
    return frames / elapsed if elapsed > 0 else 0.0


def _run_with_timeout(fn: Callable[[], T], timeout: float) -> T | None:
    """Run ``fn`` in a helper thread; None if it raises or does not finish in time (a stuck driver cannot hang us)."""
    box: list[T] = []

    def work() -> None:
        try:
            box.append(fn())
        except Exception:
            log.debug("Camera probe failed", exc_info=True)

    thread = threading.Thread(target=work, name="camera-probe", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        log.warning("A camera took longer than %.0f s to answer; skipped", timeout)
        return None
    return box[0] if box else None


def probe_cameras(
    settings: CameraConfig | None = None,
    factory: CaptureFactory = default_factory,
    max_index: int = C.CAMERA_PROBE_MAX_INDEX,
    names: list[str] | None = None,
) -> list[CameraInfo]:
    """Cameras 0..``max_index`` with their real resolution and measured FPS (Windows: also their names)."""
    settings = settings or CameraConfig()
    if names is None:
        from gesture_car.backend.camera_names import directshow_camera_names

        names = directshow_camera_names()
    found: list[CameraInfo] = []

    def probe(index: int) -> CameraInfo | None:
        opened = open_source(index, settings, factory, speed_check=True)
        if opened is None:
            return None
        try:
            fps = measure_fps(opened.cap, C.CAMERA_FPS_PROBE_S)
        finally:
            opened.cap.release()
        name = names[index] if opened.backend == "DirectShow" and index < len(names) else ""
        return CameraInfo(index, opened.width, opened.height, fps, opened.backend, name, fourcc=opened.fourcc)

    with quiet_opencv():
        for index in range(max_index + 1):
            info = _run_with_timeout(functools.partial(probe, index), C.CAMERA_PROBE_TIMEOUT_S)
            if info is None and index < len(names):
                info = CameraInfo(index, name=names[index], works=False)  # Windows sees it, but it gave no picture
            if info is not None:
                found.append(info)
    return found


# ------------------------------------------------------------------- manager
class CameraManager:
    """Owns the capture device and the capture thread."""

    def __init__(self, settings: CameraConfig, factory: CaptureFactory = default_factory) -> None:
        self._settings = settings
        self._factory = factory
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: CameraFrame | None = None
        self._sequence = 0
        self._status = CameraStatus.CLOSED
        self._error = ""
        self._fps = 0.0
        self._last_good = 0.0
        self._lost_after = C.CAMERA_LOST_AFTER_S
        self._attempts = 0
        self.source: Source | None = None
        self.backend_name = ""
        self._backend_key = ""
        self.fourcc = ""
        self.actual_width = 0
        self.actual_height = 0

    # ------------------------------------------------------------- properties
    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def error(self) -> str:
        """Human-readable reason for the last failure ('' if none)."""
        return self._error

    @property
    def measured_fps(self) -> float:
        return self._fps

    @property
    def is_ok(self) -> bool:
        """Running AND recently delivered a frame (a camera that silently stops sending counts as lost)."""
        return self._status is CameraStatus.RUNNING and time.monotonic() - self._last_good < self._lost_after

    @property
    def description(self) -> str:
        fmt = f" {self.fourcc}" if self.fourcc else ""
        return f"{describe_source(self.source)} via {self.backend_name}, {self.actual_width}x{self.actual_height}{fmt}"

    def frame_age(self, now: float | None = None) -> float:
        """Seconds since the newest frame; infinity if none yet."""
        frame = self._frame
        if frame is None:
            return float("inf")
        return (now if now is not None else time.monotonic()) - frame.timestamp

    # -------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Open the camera (falling back to other webcams) and start capturing. Returns False on failure.

        On failure ``error`` explains what to check; nothing is raised so callers can show a friendly message.
        """
        self.stop()
        s = self._settings
        try:
            opened = find_camera(s, self._factory)
        except cv2.error:  # pragma: no cover - driver specific
            log.exception("OpenCV error opening %s", describe_source(parse_source(s)))
            opened = None
        if opened is None:
            self._status = CameraStatus.FAILED
            self._error = no_camera_message(parse_source(s))
            log.error("%s unavailable", describe_source(parse_source(s)))
            return False
        self._use(opened)
        self._status = CameraStatus.RUNNING
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the thread and release the device (idempotent)."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None
        self._status = CameraStatus.CLOSED

    def _use(self, opened: OpenedCamera) -> None:
        s = self._settings
        self._cap = opened.cap
        self.source = opened.source
        self.backend_name = opened.backend
        self._backend_key = opened.backend_key
        self.fourcc = opened.fourcc
        self.actual_width, self.actual_height = opened.width, opened.height
        self._lost_after = C.CAMERA_LOST_AFTER_URL_S if isinstance(opened.source, str) else C.CAMERA_LOST_AFTER_S
        self._last_good = time.monotonic()
        self._error = ""
        fps = f"{opened.fps:.0f}" if opened.fps > 0 else "?"
        if isinstance(opened.source, str):
            log.info("Using %s: %dx%d @ %s fps (the phone app sets the size)", self.description, opened.width, opened.height, fps)
        else:
            log.info("Using %s @ %s fps (asked for %dx%d @ %d)", self.description, fps, s.width, s.height, s.fps)

    # ------------------------------------------------------------------ reads
    def latest(self) -> CameraFrame | None:
        return self._frame

    def wait_for_new(self, last_sequence: int, timeout: float) -> CameraFrame | None:
        """Block until a frame newer than ``last_sequence`` exists (or timeout)."""
        with self._cond:
            if self._frame is None or self._frame.sequence <= last_sequence:
                self._cond.wait(timeout)
            frame = self._frame
        if frame is not None and frame.sequence > last_sequence:
            return frame
        return None

    # ----------------------------------------------------------------- thread
    def _run(self) -> None:
        failures = 0
        last_time = time.monotonic()
        counted = 0
        count_start = last_time
        fps_logged = False
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                if not self._reconnect():
                    continue
                cap = self._cap
                failures = 0
                fps_logged, counted, count_start = False, 0, time.monotonic()
            try:
                ok, image = cap.read()
            except cv2.error:  # pragma: no cover - driver specific
                log.exception("Camera read raised")
                ok, image = False, None
            now = time.monotonic()
            if ok and is_valid_frame(image):
                failures = 0
                self._last_good = now
                dt = now - last_time
                last_time = now
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1.0 / dt
                self._sequence += 1
                with self._cond:  # newest frame only: an older, unread frame is dropped here
                    self._frame = CameraFrame(np.asarray(image), now, self._sequence)
                    self._cond.notify_all()
                if self._status is CameraStatus.RECONNECTING:
                    self._status = CameraStatus.RUNNING
                counted += 1
                if not fps_logged and now - count_start >= FPS_LOG_AFTER_S:
                    fps_logged = True
                    log.info("Camera delivers %.1f fps", counted / (now - count_start))
            else:
                failures += 1
                if failures >= C.CAMERA_READ_FAIL_LIMIT or now - self._last_good > self._lost_after:
                    self._lose(cap)
                else:
                    self._stop.wait(0.005)

    def _lose(self, cap: Any) -> None:
        log.warning(
            "CAMERA LOST (%s). Trying to reconnect every %.0f s - check the cable / phone app.",
            describe_source(self.source), C.CAMERA_RECONNECT_INTERVAL_S,
        )  # fmt: skip
        self._status = CameraStatus.RECONNECTING
        self._error = "Camera lost. Trying to reconnect..."
        self._attempts = 0
        with contextlib.suppress(cv2.error):
            cap.release()
        self._cap = None
        self._fps = 0.0

    def _reconnect(self) -> bool:
        """Try to reopen the same source; wait between attempts. Returns True on success."""
        if self._stop.wait(C.CAMERA_RECONNECT_INTERVAL_S):
            return False
        if self.source is None:
            return False
        self._attempts += 1
        try:
            with quiet_opencv():
                opened = open_source(self.source, self._settings, self._factory, first=self._backend_key)
        except cv2.error:  # pragma: no cover - driver specific
            opened = None
        if opened is None:
            if self._attempts == 1 or self._attempts % 5 == 0:
                log.warning("Camera reconnect attempt %d failed; still trying every %.0f s", self._attempts, C.CAMERA_RECONNECT_INTERVAL_S)
            return False
        self._use(opened)
        self._status = CameraStatus.RUNNING
        log.info("Camera reconnected after %d attempt(s)", self._attempts)
        return True

