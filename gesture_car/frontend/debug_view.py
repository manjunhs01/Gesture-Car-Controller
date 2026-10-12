"""Camera preview window (on by default; ``--no-preview`` hides it).

A small always-on-top window docked at the top-right corner of the screen that does not take keyboard focus (Windows), so the
game keeps focus. Shows the camera, hand skeletons, the wheel line and the current outputs."""

from __future__ import annotations

import base64
import contextlib
import logging
import math
import sys
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from gesture_car.backend.landmarks import HAND_CONNECTIONS
from gesture_car.backend.pipeline import TrackingResult
from gesture_car.backend.steering import hand_anchor

if TYPE_CHECKING:
    from gesture_car.backend.controller import ControlLogic

log = logging.getLogger(__name__)

WINDOW = "VisionDrive"
MARGIN = 12


def _work_area() -> tuple[int, int]:
    """Usable screen size (width, height) without the taskbar, in the same pixel units OpenCV windows use."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # type: ignore[attr-defined]
                return int(rect.right - rect.left), int(rect.bottom - rect.top)
        except Exception:
            log.debug("Could not read the work area", exc_info=True)
    return 1920, 1040


def _screen_width() -> int:
    return _work_area()[0]


def fit_scale(requested: float, img_w: int, img_h: int, area: tuple[int, int]) -> float:
    """Largest scale <= ``requested`` at which the whole preview (incl. title bar) fits on screen, so nothing is cut off."""
    max_h = area[1] - 2 * MARGIN - 40
    max_w = area[0] * 0.75
    return max(0.3, min(requested, max_h / img_h, max_w / img_w))


RENDER_W = 640  # the preview is always drawn this wide (camera frames are upscaled), so it is big and sharp
PANEL_H = 200
HEADER_H = 66  # title band (presentation mode only)
PIPE_H = 46  # pipeline strip: hands -> points -> angle -> gesture -> keys -> game
SS = 3  # supersampling factor for the dashboard panel (anti-aliased, "HD" wheel)
WHEEL_COLOR = (235, 235, 235)
BG = (26, 24, 22)
MODE_COLORS = {"DRIVING": (90, 220, 120), "CALIBRATING": (60, 190, 255), "HANDS LOST": (70, 70, 255), "CAMERA LOST": (60, 60, 255), "PAUSED": (180, 180, 180)}
FONT = cv2.FONT_HERSHEY_DUPLEX


def _poly(img: np.ndarray, pts: list[tuple[float, float]], color: tuple[int, ...]) -> None:
    cv2.fillPoly(img, [np.array([(int(x), int(y)) for x, y in pts], dtype=np.int32)], color, cv2.LINE_AA)


def draw_wheel(img: np.ndarray, cx: int, cy: int, r: int, rotation_deg: float, color: tuple[int, int, int]) -> None:
    """A realistic car steering wheel (padded rim with highlight, tapered T-spokes, metal hub, red 12 o'clock marker),
    rotated clockwise by ``rotation_deg``. Drawn on a supersampled canvas and scaled down, so edges are smooth."""
    pad = 16
    size = (2 * (r + pad)) * SS
    c = size // 2
    big = np.full((size, size, 3), BG, dtype=np.uint8)
    R = r * SS
    rim_w = max(3, int(r * 0.16)) * SS

    def pt(radius: float, deg: float, dx: float = 0.0) -> tuple[float, float]:
        a = math.radians(deg + rotation_deg)
        return c + radius * math.cos(a) - dx * math.sin(a), c + radius * math.sin(a) + dx * math.cos(a)

    dark = tuple(int(v * 0.35) for v in color)
    mid = tuple(int(v * 0.7) for v in color)
    # spokes: tapered, from the hub to the rim (left, right, bottom = T shape)
    for deg in (180.0, 0.0, 90.0):
        w_hub, w_rim = 0.16 * R, 0.07 * R
        _poly(big, [pt(0.15 * R, deg, -w_hub), pt(R, deg, -w_rim), pt(R, deg, w_rim), pt(0.15 * R, deg, w_hub)], mid)
        _poly(big, [pt(0.15 * R, deg, -w_hub * 0.35), pt(R, deg, -w_rim * 0.35), pt(R, deg, w_rim * 0.35), pt(0.15 * R, deg, w_hub * 0.35)], color)
    # rim: dark outer ring, body, bright highlight on the upper-left (does not rotate: it is the "light")
    cv2.circle(big, (c, c), R + rim_w // 2, dark, rim_w + 4 * SS, cv2.LINE_AA)
    cv2.circle(big, (c, c), R, mid, rim_w, cv2.LINE_AA)
    cv2.ellipse(big, (c, c), (R, R), 0, 200, 260, color, max(2, rim_w // 3), cv2.LINE_AA)
    # grip stitches on the rim
    for k in range(0, 360, 12):
        p0, p1 = pt(R - rim_w * 0.42, k), pt(R + rim_w * 0.42, k)
        cv2.line(big, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), dark, max(1, SS // 2), cv2.LINE_AA)
    # hub: dark disc, metal ring, centre cap
    hub = int(0.27 * R)
    cv2.circle(big, (c, c), hub + 2 * SS, dark, -1, cv2.LINE_AA)
    cv2.circle(big, (c, c), hub, mid, -1, cv2.LINE_AA)
    cv2.circle(big, (c, c), int(hub * 0.72), color, 2 * SS, cv2.LINE_AA)
    cv2.circle(big, (c, c), int(hub * 0.42), dark, -1, cv2.LINE_AA)
    # red 12 o'clock marker (shows how far the wheel is turned)
    m0, m1 = pt(R - rim_w * 0.5, -90.0), pt(R + rim_w * 0.5, -90.0)
    cv2.line(big, (int(m0[0]), int(m0[1])), (int(m1[0]), int(m1[1])), (60, 60, 255), max(3, int(rim_w * 0.45)), cv2.LINE_AA)
    small = cv2.resize(big, (size // SS, size // SS), interpolation=cv2.INTER_AREA)
    x0, y0 = cx - small.shape[1] // 2, cy - small.shape[0] // 2
    h, w = img.shape[:2]
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(w, x0 + small.shape[1]), min(h, y0 + small.shape[0])
    if xe > xs and ye > ys:
        img[ys:ye, xs:xe] = small[ys - y0 : ye - y0, xs - x0 : xe - x0]


def draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, color: tuple[int, int, int], centred: bool) -> None:
    """Rounded progress bar (value 0..1, or -1..1 when ``centred``)."""
    rad = h // 2
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 58, 56), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (110, 108, 104), max(1, SS // 2), cv2.LINE_AA)
    value = max(-1.0, min(1.0, value))
    if centred:
        mid = x + w // 2
        end = mid + int(value * w / 2)
        if end != mid:
            cv2.rectangle(img, (min(mid, end), y + 2), (max(mid, end), y + h - 2), color, -1, cv2.LINE_AA)
        cv2.line(img, (mid, y - rad // 2), (mid, y + h + rad // 2), (230, 230, 230), max(1, SS // 2), cv2.LINE_AA)
    elif value > 0:
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + int(value * (w - 4)), y + h - 2), color, -1, cv2.LINE_AA)


def render(result: TrackingResult, logic: ControlLogic, presentation: bool = False) -> np.ndarray | None:
    """Draw the preview image (BGR): camera, hand skeletons, wheel line and current outputs. Pure; no window needed."""
    if result.preview_rgb is None:
        return None
    src = cv2.cvtColor(result.preview_rgb, cv2.COLOR_RGB2BGR)
    sh, sw = src.shape[:2]
    w = RENDER_W
    h = int(round(sh * w / sw))
    img = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC if w > sw else cv2.INTER_AREA)
    anchors: list[tuple[int, int]] = []
    for hand in result.hands:
        pts = [(int(x * w), int(y * h)) for x, y in hand.points[:, :2]]  # plain ints: OpenCV rejects numpy scalars in some builds
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 220, 120), 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(img, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
        ax, ay = hand_anchor(hand, logic.cfg.steering.anchor)
        anchors.append((int(ax / hand.aspect * w), int(ay * h)))
    if len(anchors) == 2:
        cv2.line(img, anchors[0], anchors[1], (0, 160, 255), 4, cv2.LINE_AA)
        for anc in anchors:
            cv2.circle(img, anc, 8, (0, 160, 255), -1, cv2.LINE_AA)
    s = logic.last_state
    mode_color = MODE_COLORS.get(logic.mode.value, (200, 200, 200))
    lines = [
        (f"{logic.mode.value}", mode_color),
        (f"steer {s.steer:+.2f}  {_direction(s.steer)}", (255, 255, 255)),
        (f"throttle {s.throttle:.1f}  brake {s.brake:.1f}  nitro {'ON' if s.nitro else '-'}", (255, 255, 255)),
        (f"{logic.fps:.0f} fps   {logic.latency_ms:.0f} ms   tracking {_confidence(result):.0%}", (200, 200, 200)),
    ]
    for i, (text, tcolor) in enumerate(lines):
        pos = (12, 28 + 28 * i)
        cv2.putText(img, text, pos, FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)  # outline so it reads on any background
        cv2.putText(img, text, pos, FONT, 0.7, tcolor, 2, cv2.LINE_AA)
    guide = _guide_text(logic)
    if guide:
        cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(img, guide[: w // 12], (12, h - 13), FONT, 0.7, (60, 220, 255), 1, cv2.LINE_AA)
    if logic.mode.value == "HANDS LOST":
        _safe_banner(img)
    elif logic.mode.value == "CAMERA LOST":
        img = cv2.convertScaleAbs(img, alpha=0.35)  # dim the last picture: it is frozen
        _safe_banner(img, "CAMERA LOST", "Reconnecting... all keys released")
    parts = [img, _pipeline(w, result, logic), _panel(w, logic)]
    if presentation:
        parts.insert(0, _header(w, logic))
    return np.vstack(parts)


def _safe_banner(img: np.ndarray, title: str = "SAFE MODE", subtitle: str = "Hands not detected - controls released") -> None:
    """Red banner shown while the failsafe is active (hands lost or camera lost)."""
    h, w = img.shape[:2]
    y0 = h // 2 - 44
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + 88), (30, 30, 190), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    for i, (text, scale) in enumerate(((title, 1.0), (subtitle, 0.6))):
        size = cv2.getTextSize(text, FONT, scale, 2)[0]
        cv2.putText(img, text, ((w - size[0]) // 2, y0 + 38 + 36 * i), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _header(width: int, logic: ControlLogic) -> np.ndarray:
    """Title band for presentations."""
    band = np.full((HEADER_H, width, 3), (40, 30, 20), dtype=np.uint8)
    cv2.putText(band, "VISIONDRIVE", (14, 34), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(band, "HAND-GESTURE-CONTROLLED CAR GAME", (14, 56), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    mode = logic.mode.value
    color = MODE_COLORS.get(mode, (200, 200, 200))
    size = cv2.getTextSize(mode, FONT, 0.7, 2)[0]
    cv2.putText(band, mode, (width - size[0] - 14, 40), FONT, 0.7, color, 2, cv2.LINE_AA)
    cv2.line(band, (0, HEADER_H - 1), (width, HEADER_H - 1), color, 2, cv2.LINE_AA)
    return band


def _confidence(result: TrackingResult) -> float:
    """Average hand-detection confidence of the current frame (0 when no hand is visible)."""
    return float(np.mean([h.score for h in result.hands])) if result.hands else 0.0


def _keys_text(logic: ControlLogic) -> str:
    s = logic.last_state
    keys = []
    if s.steer > 0.05:
        keys.append("D")
    elif s.steer < -0.05:
        keys.append("A")
    if s.throttle > 0.05:
        keys.append("W")
    if s.brake > 0.05:
        keys.append("S")
    if s.nitro:
        keys.append("SHIFT")
    return "+".join(keys) if keys else "none"


def _gesture_text(logic: ControlLogic) -> str:
    s = logic.last_state
    if s.brake > 0.5:
        return "OPEN PALM"
    if s.nitro:
        return "THUMBS UP"
    return "none"


def _pipeline(width: int, result: TrackingResult, logic: ControlLogic) -> np.ndarray:
    """Strip showing the processing pipeline live: AI vision -> gesture engine -> control engine -> game."""
    strip = np.full((PIPE_H, width, 3), (40, 36, 32), dtype=np.uint8)
    angle = logic.steering.angle
    angle_txt = "-" if angle is None or math.isnan(angle) else f"{angle:+.0f} deg"
    chips = [
        (f"{result.num_hands} HANDS", "camera + AI"),
        (f"{result.num_hands * 21} POINTS", "landmarks"),
        (angle_txt, "wheel angle"),
        (_gesture_text(logic), "gesture"),
        (_keys_text(logic), "input"),
        ("GAME", "control"),
    ]
    n = len(chips)
    gap = 16
    cw = (width - gap * (n - 1) - 12) // n
    for i, (top, bottom) in enumerate(chips):
        x = 6 + i * (cw + gap)
        active = not (top in ("-", "none") or top.startswith("0 "))
        color = (90, 220, 120) if active else (110, 110, 110)
        cv2.rectangle(strip, (x, 4), (x + cw, PIPE_H - 4), color, 1, cv2.LINE_AA)
        cv2.putText(strip, top[:12], (x + 5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(strip, bottom, (x + 5, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
        if i < n - 1:
            cv2.arrowedLine(strip, (x + cw + 2, PIPE_H // 2), (x + cw + gap - 2, PIPE_H // 2), (200, 200, 200), 1, cv2.LINE_AA, tipLength=0.5)
    return strip


def _direction(steer: float) -> str:
    """Words for the steering direction the GAME receives (so you can see at a glance if left/right is reversed)."""
    if steer > 0.05:
        return "RIGHT >>"
    if steer < -0.05:
        return "<< LEFT"
    return "straight"


def _guide_text(logic: ControlLogic) -> str:
    """One line telling the driver what to do right now."""
    mode = logic.mode.value
    if mode == "CALIBRATING":
        return "Hold both hands like a wheel: " + logic.calibration_message()
    if mode == "HANDS LOST":
        return "Show BOTH hands to the camera"
    if mode == "CAMERA LOST":
        return "Check the camera cable / phone app"
    if mode == "PAUSED":
        return "Paused (F8 to resume)"
    return "Turns the wrong way? Press F7 to flip left/right" if abs(logic.last_state.steer) > 0.2 else ""


def _panel(width: int, logic: ControlLogic) -> np.ndarray:
    """Dashboard strip under the camera: HD rotating steering wheel plus steering / throttle / brake / nitro bars."""
    s = logic.last_state
    angle = logic.steering.angle
    rotation = 0.0 if angle is None or math.isnan(angle) else max(-90.0, min(90.0, angle))  # the wheel you are turning
    mode_color = MODE_COLORS.get(logic.mode.value, (200, 200, 200))
    panel = np.full((PANEL_H, width, 3), BG, dtype=np.uint8)
    r = 78
    draw_wheel(panel, r + 24, PANEL_H // 2, r, rotation, mode_color if logic.mode.value == "DRIVING" else WHEEL_COLOR)
    # bars + text on a supersampled canvas (smooth edges), then scaled down
    x = 2 * r + 70
    big = np.full((PANEL_H * SS, (width - x) * SS, 3), BG, dtype=np.uint8)
    bw = (width - x - 24) * SS
    cv2.putText(big, f"{logic.mode.value}    {rotation:+.0f} deg", (0, 28 * SS), FONT, 0.9 * SS, mode_color, SS, cv2.LINE_AA)
    rows = (("STEER", s.steer, (255, 170, 60), True), ("GAS", s.throttle, (90, 220, 120), False), ("BRAKE", s.brake, (70, 70, 255), False))
    lx = 92 * SS
    for i, (label, value, color, centred) in enumerate(rows):
        y = (46 + i * 38) * SS
        cv2.putText(big, label, (0, y + 17 * SS), FONT, 0.6 * SS, (210, 210, 210), SS, cv2.LINE_AA)
        draw_bar(big, lx, y, bw - lx, 22 * SS, value, color, centred)
    nitro_color = (0, 215, 255) if s.nitro else (95, 95, 95)
    cv2.putText(big, "NITRO", (0, 180 * SS), FONT, 1.0 * SS, nitro_color, 2 * SS if s.nitro else SS, cv2.LINE_AA)
    if s.nitro:
        cv2.rectangle(big, (lx, 160 * SS), (bw, 186 * SS), nitro_color, -1, cv2.LINE_AA)
    right = cv2.resize(big, (width - x, PANEL_H), interpolation=cv2.INTER_AREA)
    panel[:, x:] = right
    return panel


class _CvWindow:
    """OpenCV window (default)."""

    name = "opencv"

    def __init__(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        with contextlib.suppress(Exception):
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        self._docked = False

    def show(self, img: np.ndarray | None) -> bool:
        """Draw one frame (``None`` = just keep the window responsive); returns True when the user closed it (q)."""
        if img is None:
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        cv2.imshow(WINDOW, img)
        if not self._docked:
            self._docked = True
            with contextlib.suppress(Exception):
                cv2.moveWindow(WINDOW, max(0, _screen_width() - img.shape[1] - MARGIN), MARGIN)
            with contextlib.suppress(Exception):
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()


class _TkWindow:
    """Fallback if OpenCV has no GUI support: a Tk window (tkinter ships with python.org Python on Windows)."""

    name = "tkinter"

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title(WINDOW)
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.label = tk.Label(self.root, bd=0)
        self.label.pack()
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._placed = False
        self._photo: object = None

    def _on_close(self) -> None:
        self.closed = True

    def show(self, img: np.ndarray | None) -> bool:
        if img is None:
            self.root.update()
            return self.closed
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return self.closed
        photo = self._tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        self.label.configure(image=photo)
        self._photo = photo  # keep a reference
        if not self._placed:
            self._placed = True
            self.root.update_idletasks()
            self.root.geometry(f"+{max(0, self.root.winfo_screenwidth() - img.shape[1] - MARGIN)}+{MARGIN}")
        self.root.update()
        return self.closed

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.root.destroy()


class DebugView:
    """Small always-on-top camera window at the top-right corner.

    The window runs in its OWN thread: drawing, ``imshow`` and the Windows message loop (dragging or clicking the window
    can block it for a while) can never stall the control loop. Any failure falls back or disables the preview, never the
    controller. Frames are drawn at most ``MAX_FPS`` times a second."""

    MAX_FPS = 20.0

    def __init__(self, scale: float = 1.0, presentation: bool = False) -> None:
        self._scale = scale
        self._presentation = presentation
        self._area = _work_area()
        self._closed = False
        self._last_offer = 0.0
        self._lock = threading.Lock()
        self._pending: tuple[TrackingResult, ControlLogic, np.ndarray | None] | None = None
        self._wake = threading.Event()
        self._opened = threading.Event()
        self._thread = threading.Thread(target=self._run, name="preview-window", daemon=True)
        self._thread.start()
        self._opened.wait(5.0)  # so the "opened" / error message appears before the controller starts

    # -- control-loop side (must stay cheap and never block) ------------------------------------
    def show(self, result: TrackingResult, logic: ControlLogic) -> None:
        if self._closed:
            return
        now = time.monotonic()
        if now - self._last_offer < 1.0 / self.MAX_FPS:
            return
        self._last_offer = now
        rgb = None if result.preview_rgb is None else result.preview_rgb.copy()
        with self._lock:
            self._pending = (result, logic, rgb)
        self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(2.0)

    # -- window thread ----------------------------------------------------------------------------
    def _run(self) -> None:
        win: _CvWindow | _TkWindow | None = None
        for factory in (_CvWindow, _TkWindow):
            try:
                win = factory()
                break
            except Exception as exc:
                log.warning("Preview window via %s unavailable: %s", factory.__name__.strip("_").replace("Window", ""), exc)
        if win is None:
            self._closed = True
            log.error("Could not open a preview window (the controller keeps running without it)")
            self._opened.set()
            return
        log.info("Camera preview opened (%s) at the top-right corner. Click it and press q (or close it) to hide it.", win.name)
        self._opened.set()
        try:
            while not self._closed:
                self._wake.wait(0.25)
                self._wake.clear()
                with self._lock:
                    item, self._pending = self._pending, None
                if item is None:
                    if win.show(None):  # keep the window responsive while no new frame arrives
                        break
                    continue
                result, logic, rgb = item
                try:
                    if rgb is not None:
                        result = replace(result, preview_rgb=rgb)
                    img = render(result, logic, self._presentation)
                    if img is None:
                        continue
                    scale = fit_scale(self._scale, img.shape[1], img.shape[0], self._area)
                    if abs(scale - 1.0) > 0.01:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                    if win.show(img):
                        break
                except Exception:
                    log.exception("Preview failed; turning the preview off (the controller keeps running)")
                    break
        finally:
            self._closed = True
            win.close()
            log.info("Preview closed (the controller keeps running)")
