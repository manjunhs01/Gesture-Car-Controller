"""A fake ``cv2.VideoCapture`` world for camera tests: no hardware, fully deterministic.

``FakeSystem`` is passed as the capture factory. It holds ``FakeDevice`` objects keyed by source (webcam number or URL) and
records every open call, so tests can check which cameras and backends were tried and in what order.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from gesture_car.backend.camera import MJPG_FOURCC


class FakeDevice:
    def __init__(
        self,
        modes: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720)),
        apis: set[int] | None = None,
        fps: float = 200.0,
        mjpg: bool = True,
        api_fps: dict[int, float] | None = None,
    ) -> None:
        self.modes = modes  # supported (width, height); the first one is the power-on default
        self.apis = apis  # backends that can open it (None = all)
        self.fps = fps
        self.api_fps = api_fps or {}  # per-backend frame rate (e.g. slow DirectShow, fast Media Foundation)
        self.mjpg = mjpg  # False: asking for MJPG makes every read fail (like some cheap webcams)
        self.plugged = threading.Event()
        self.plugged.set()
        self.hang = threading.Event()  # set: read() blocks (a stuck driver)
        self.reads = 0


class FakeCapture:
    def __init__(self, device: FakeDevice | None, api: int) -> None:
        self.device = device
        self.api = api
        self.size = device.modes[0] if device else (0, 0)
        self.want = list(self.size)
        self.fourcc = 0
        self.sets: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming
        d = self.device
        return d is not None and d.plugged.is_set() and (d.apis is None or self.api in d.apis)

    def set(self, prop: int, value: float) -> bool:
        self.sets.append((prop, value))
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = int(value)
        elif prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self.want[0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 1] = int(value)
            if self.device and tuple(self.want) in self.device.modes:
                self.size = (self.want[0], self.want[1])
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.size[1])
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FOURCC:
            return float(self.fourcc)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        d = self.device
        if d is None or self.released:
            return False, None
        while d.hang.is_set():
            time.sleep(0.01)
        time.sleep(1.0 / d.api_fps.get(self.api, d.fps))
        if not d.plugged.is_set() or (self.fourcc == MJPG_FOURCC and not d.mjpg):
            return False, None
        d.reads += 1
        return True, np.full((self.size[1], self.size[0], 3), d.reads % 256, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeSystem:
    """Capture factory: ``FakeSystem({0: FakeDevice()})(source, api, params)`` -> FakeCapture."""

    def __init__(self, devices: dict[int | str, FakeDevice] | None = None, open_delay: dict[object, float] | None = None) -> None:
        self.devices = devices or {}
        self.open_delay = open_delay or {}
        self.calls: list[tuple[int | str, int, list[int]]] = []
        self.captures: list[FakeCapture] = []

    def __call__(self, source: int | str, api: int, params: list[int]) -> FakeCapture:
        self.calls.append((source, api, list(params)))
        delay = self.open_delay.get((source, api), self.open_delay.get(source, 0.0))  # key: source, or (source, api)
        if delay:
            time.sleep(delay)
        cap = FakeCapture(self.devices.get(source), api)
        self.captures.append(cap)
        return cap

    def sources_tried(self) -> list[int | str]:
        seen: list[int | str] = []
        for source, _, _ in self.calls:
            if source not in seen:
                seen.append(source)
        return seen
