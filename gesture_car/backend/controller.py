"""Brain and runtime of the silent controller.

* :class:`ControlLogic` is pure (landmarks in, :class:`ControlState` out) and fully unit-tested.
* :class:`GestureController` wires camera -> hand tracking -> logic -> output, global hotkeys, optional tray and debug view.

Threads: camera capture (newest frame only) -> vision (hand detection) -> output (logic + keys/gamepad, woken the moment a
result is published, otherwise ticking at ``app.loop_hz`` for keyboard PWM). The main thread only does housekeeping: camera
reconnects, preview, ``--debug`` output and stats, none of which sits in the steering path.

Safety: every exit path (normal quit, F10, Ctrl+C, exception, ``atexit``) releases all keys / resets the virtual gamepad, and
a watchdog thread does the same if the output thread ever stalls.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from gesture_car.backend import instance
from gesture_car.backend.camera import CameraManager
from gesture_car.backend.gestures import GestureEngine, GestureState
from gesture_car.backend.hand_tracker import create_hand_tracker
from gesture_car.backend.landmarks import HandLandmarks
from gesture_car.backend.output import NEUTRAL, ControlOutput, ControlState, create_output
from gesture_car.backend.pipeline import TrackingResult, VisionPipeline
from gesture_car.backend.steering import SteeringEstimator, describe_pair
from gesture_car.config import Config
from gesture_car.utils.errors import CameraError, GestureCarError

log = logging.getLogger(__name__)

WATCHDOG_TIMEOUT_S = 0.5
STATS_INTERVAL_S = 10.0
STOP_POLL_S = 0.25
CALIB_MSG_S = 2.0
CAMERA_RETRY_S = 5.0


class Mode(Enum):
    CALIBRATING = "CALIBRATING"
    DRIVING = "DRIVING"
    HANDS_LOST = "HANDS LOST"
    CAMERA_LOST = "CAMERA LOST"
    PAUSED = "PAUSED"


class ControlLogic:
    """Hands -> steering / brake / reverse / nitro / automatic acceleration, including the hands-lost safe state."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.steering = SteeringEstimator(cfg.steering)
        self.gestures = GestureEngine(cfg.gestures)
        self.mode = Mode.CALIBRATING
        self.paused = False
        self.last_state = NEUTRAL
        self._last_pair_t: float | None = None
        self._last_frame_t: float | None = None
        self._steer = 0.0
        self._frozen: float | None = None
        self._latest: tuple[HandLandmarks, ...] = ()
        self.last_gestures = GestureState()  # for --debug and the preview
        self.fps = 0.0  # shown in the preview (set by the controller loop)
        self.latency_ms = 0.0

    # ------------------------------------------------------------- commands
    def recalibrate(self) -> None:
        self.steering.recalibrate()
        self.gestures.reset()
        self._last_pair_t = None
        self._frozen = None
        self._steer = 0.0
        self.mode = Mode.CALIBRATING
        log.info("Recalibrating: hold both hands in the steering-wheel pose for %.1f s", self.cfg.steering.calibration_seconds)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self.mode = Mode.PAUSED
            self.last_state = NEUTRAL
        else:
            self.recalibrate()  # the pose may have changed while paused

    def calibration_message(self) -> str:
        """Why calibration is (still) waiting, in plain words."""
        pair = describe_pair(self._latest, self.cfg.steering.anchor, self.cfg.steering.min_hand_separation)
        return pair if pair != "ok" else self.steering.calibrator.status

    # ------------------------------------------------------------- per frame
    def process(self, hands: tuple[HandLandmarks, ...], t: float) -> ControlState:
        """Handle one tracked camera frame captured at monotonic time ``t``."""
        self._latest = hands
        self._last_frame_t = t
        if self.paused:
            return self.last_state
        # Gestures first, from finger states only: the wheel logic below can never swallow a palm.
        gestures = self.gestures.update(hands, t)
        self.last_gestures = gestures
        palms = gestures.palms or gestures.raw_palms  # two open palms (raw too, so steering cannot twitch before confirmation)
        palms_mode = self.cfg.gestures.palms_steering
        was_calibrated = self.steering.calibrated
        steer: float | None = None
        if not palms or (palms_mode in ("full", "intentional") and was_calibrated):  # palms never feed the calibration
            steer = self.steering.update(hands, t)  # the same model: line between the palm centres, neutral, dead zone, filter
        if self.steering.calibrated and not was_calibrated:
            log.info("✓ Calibration complete (%s)\nSYSTEM READY - you can steer now", self.steering.calibrator.status)
        if not self.steering.calibrated:
            if self.mode is not Mode.CALIBRATING:
                self.mode = Mode.CALIBRATING
            self._last_pair_t = None
            self.last_state = ControlState(brake=1.0 if gestures.brake else 0.0)  # brake is always honoured (never reverse)
            return self.last_state
        if palms:
            # both hands are deliberately in view (not "hands lost"). full: normal proportional steering; intentional: only a
            # clearly tilted pair steers; frozen: 0. An unmeasurable pair gives 0, never a held turn.
            offset = self.steering.raw_offset
            clear = offset is not None and abs(offset) >= self.cfg.gestures.palms_intentional_deg
            allowed = palms_mode == "full" or (palms_mode == "intentional" and clear)
            self._steer = steer if allowed and steer is not None else 0.0
            self._last_pair_t = t
            return self._compose(t, gestures.brake, gestures.nitro, True, measured=True, reverse=gestures.reverse)
        measured = steer is not None
        if steer is not None:  # only a measured pair counts: an unmeasurable one (e.g. one hand seen twice) never holds a turn
            self._steer = steer
            self._last_pair_t = t
        gesture_active = gestures.raw_brake or gestures.raw_nitro or gestures.brake or gestures.nitro
        return self._compose(t, gestures.brake, gestures.nitro, gesture_active, measured=measured)

    def tick(self, now: float) -> ControlState:
        """Called every loop iteration: applies time-based rules (hands lost / camera stalled) even with no new frame."""
        if self.paused or not self.steering.calibrated:
            return self.last_state
        if self._last_frame_t is not None and now - self._last_frame_t > self.cfg.safety.hands_lost_s:
            return self._compose(now, False, False, False, force_lost=True)
        return self.last_state

    def _compose(
        self,
        t: float,
        brake: bool,
        nitro: bool,
        gesture_active: bool,
        force_lost: bool = False,
        measured: bool = False,
        reverse: bool = False,
    ) -> ControlState:
        s, safety = self.cfg.output, self.cfg.safety
        lost = force_lost or self._last_pair_t is None or t - self._last_pair_t > safety.hands_lost_s
        if lost:
            if self.mode is not Mode.HANDS_LOST:
                log.warning("Steering hands lost: centring, releasing nitro (%s)", safety.lost_behavior)
                self.steering.reset_filter()
                self._steer = 0.0
                self._frozen = None
            self.mode = Mode.HANDS_LOST
            level = 1.0 if brake else (safety.gentle_brake_level if safety.lost_behavior == "gentle_brake" else 0.0)
            self.last_state = ControlState(steer=0.0, throttle=0.0, brake=level, nitro=False)
            return self.last_state
        if self.mode is not Mode.DRIVING:
            log.info("Driving")
        self.mode = Mode.DRIVING
        steer = self._steer
        if not measured and gesture_active and safety.freeze_steering_during_gesture:
            # a hand is off the wheel (out of view) making a gesture: hold the wheel from just before it left, until the
            # hands-lost timeout. With both hands measured this never applies: the wheel always follows the hands.
            if self._frozen is None:
                self._frozen = self.steering.value_before(t, safety.freeze_lookback_s)
            steer = self._frozen
        else:
            self._frozen = None
        reverse_button = reverse and s.reverse_mode == "button"
        self.last_state = ControlState(
            steer=steer,
            throttle=0.0 if brake or reverse else s.auto_accel_level,  # automatic acceleration; brake / reverse stop it
            # brake_hold: reversing = keep braking (racing games reverse when brake is held at a standstill);
            # button: the reverse key/button replaces the brake so the two do not fight
            brake=1.0 if brake and not reverse_button else 0.0,
            nitro=nitro and not brake and not reverse,
            reverse=reverse_button,
        )
        return self.last_state


# =============================================================== runtime
DEBUG_PRINT_S = 0.1  # --debug prints at most 10 lines per second
HOUSEKEEPING_HZ = 60  # main thread: preview, --debug, stats, camera reconnect (never in the steering path)


class StageTimes:
    """Per-stage latency sums for the frames sent since the last ``take()`` (written by the output thread)."""

    STAGES = ("wait", "detect", "handoff", "steer", "send", "total")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sums = dict.fromkeys(self.STAGES, 0.0)
        self._count = 0

    def add(self, **ms: float) -> None:
        with self._lock:
            for key, value in ms.items():
                self._sums[key] += value
            self._count += 1

    def take(self) -> tuple[int, dict[str, float]]:
        """(frames, mean ms per stage) since the previous call, and start a new window."""
        with self._lock:
            count, sums = self._count, self._sums
            self._sums, self._count = dict.fromkeys(self.STAGES, 0.0), 0
        return count, {k: v / count for k, v in sums.items()} if count else dict.fromkeys(self.STAGES, 0.0)


def debug_line(logic: ControlLogic, camera_fps: float = 0.0, stages: dict[str, float] | None = None) -> str:
    """One console line for --debug: wheel angle, neutral, offset and steering sent, then FPS and per-stage latency."""
    est = logic.steering
    neutral = est.calibrator.neutral

    def num(value: float | None, fmt: str) -> str:
        return "   -  " if value is None else format(value, fmt)

    g = logic.last_gestures

    def count(n: int | None) -> str:
        return "-" if n is None else str(n)

    def flag(b: bool | None) -> str:
        return "-" if b is None else str(int(b))

    line = (
        f"[{logic.mode.value:<11}] raw angle {num(est.wheel_angle, '6.1f')}  neutral {num(neutral, '6.1f')}  "
        f"offset {num(est.raw_offset, '+6.1f')}  steering {logic.last_state.steer * 100:+6.1f}% | "
        f"fingers L{count(g.fingers[0])} R{count(g.fingers[1])} palm_confirmed {int(g.palms)} brake_active {int(g.brake)} "
        f"reverse_active {int(g.reverse)} hold {g.hold_s:4.2f}s | thumb_up L{flag(g.thumbs[0])} R{flag(g.thumbs[1])} "
        f"nitro_active {int(g.nitro)}"
    )
    if stages is None:
        return line
    return (
        f"{line} | cam {camera_fps:4.1f} fps track {logic.fps:4.1f} fps | wait {stages['wait']:4.1f} detect {stages['detect']:5.1f} "
        f"handoff {stages['handoff']:4.1f} steer {stages['steer']:4.2f} send {stages['send']:4.2f} | total {stages['total']:5.1f} ms"
    )


def _blank_result() -> TrackingResult:
    """Black frame for the preview when the camera is lost before it ever delivered a picture."""
    return TrackingResult(0, time.monotonic(), (), 0.0, (320, 240), np.zeros((240, 320, 3), dtype=np.uint8))


PRIORITY_CLASSES = {"high": 0x80, "above_normal": 0x8000, "normal": 0x20, "below_normal": 0x4000}  # Windows priority classes
PRIORITY_NICE = {"high": -10, "above_normal": -5, "normal": 0, "below_normal": 5}


def set_process_priority(level: str) -> None:
    """High priority keeps camera frames and key events on time while a game is using the CPU."""
    if level == "normal":
        return
    try:
        import psutil

        proc = psutil.Process()
        if sys.platform.startswith("win"):
            proc.nice(PRIORITY_CLASSES[level])
        else:
            proc.nice(PRIORITY_NICE[level])
        log.info("Process priority: %s", level)
        return
    except ImportError:
        pass
    except Exception as exc:  # e.g. raising priority on Linux needs root
        log.warning("Could not set process priority to %s (%s)", level, exc)
        return
    try:
        if sys.platform.startswith("win"):
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), PRIORITY_CLASSES[level]):
                raise OSError("SetPriorityClass failed")
        else:
            os.nice(PRIORITY_NICE[level])
        log.info("Process priority: %s", level)
    except (OSError, AttributeError) as exc:
        log.warning("Could not set process priority to %s (%s)", level, exc)


def _never_throttle_timers() -> None:
    """Windows 11 may ignore a background process's 1 ms timer request (power throttling); opt this process out."""
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = (("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG))

    execution_speed, ignore_timer_resolution, process_power_throttling = 0x1, 0x4, 4
    state = PowerThrottlingState(1, execution_speed | ignore_timer_resolution, 0)  # controlled and off = never throttled
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state), ctypes.sizeof(state))


@contextlib.contextmanager
def fine_timer() -> Any:
    """Windows waits in 15.6 ms steps by default, so a 4 ms output tick would really be 16 ms. Ask for 1 ms steps."""
    winmm = None
    if sys.platform.startswith("win"):
        with contextlib.suppress(OSError, AttributeError):
            import ctypes

            _never_throttle_timers()  # before Windows 10 1709 the call fails harmlessly
            winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
            winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(1)


class GestureController:
    def __init__(
        self,
        cfg: Config,
        dry_run: bool = False,
        debug: bool = False,
        record_path: Path | None = None,
        presentation: bool = False,
        console_debug: bool = False,
    ) -> None:
        self.cfg = cfg
        self.presentation = presentation
        self.record_path = record_path
        self._recorder: Any = None
        self.dry_run = dry_run
        self.debug = debug
        self.console_debug = console_debug  # print angle / offset / steering / stage timings to the console (--debug)
        self.logic = ControlLogic(cfg)
        self.output: ControlOutput | None = None
        self.camera: CameraManager | None = None
        self.pipeline: VisionPipeline | None = None
        self._stop = threading.Event()
        self._last_beat = time.monotonic()
        self._hotkeys: Any = None
        self._tray: Any = None
        self.timing = StageTimes()  # read by --debug
        self._stats = StageTimes()  # read by the 10 s stats line
        self._two_hands = 0  # frames with both hands in the current stats window

    # ------------------------------------------------------------ lifecycle
    def request_quit(self) -> None:
        self._stop.set()

    def toggle_pause(self) -> None:
        self.logic.set_paused(not self.logic.paused)
        if self.output is not None and self.logic.paused:
            self.output.release_all()
        log.info("Paused" if self.logic.paused else "Resumed")

    def request_recalibrate(self) -> None:
        self.logic.recalibrate()

    def emergency_stop(self) -> None:
        """Release every key / stick / trigger immediately and pause until F8 is pressed."""
        if self.output is not None:
            self.output.release_all()
        if not self.logic.paused:
            self.logic.set_paused(True)
        log.warning("EMERGENCY STOP: all controls released, paused (press F8 to resume)")

    def flip_steering(self) -> None:
        """Swap left/right live (hotkey) and remember it in config.json, for cameras/drivers that mirror the picture."""
        sc = self.cfg.steering
        sc.invert_steering = not sc.invert_steering
        log.info("Steering direction flipped: turning your wheel RIGHT now steers %s", "LEFT (inverted)" if sc.invert_steering else "RIGHT (normal)")
        try:
            import json

            from gesture_car.config import default_path

            path = default_path()
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            steering = data.setdefault("steering", {})
            steering.pop("invert", None)  # older name of the same setting
            steering["invert_steering"] = sc.invert_steering
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            log.warning("Could not save the steering direction to config.json (%s); it applies until you quit", exc)

    def _start_vision(self) -> None:
        cam_cfg = self.cfg.camera
        camera = CameraManager(cam_cfg)
        if not camera.start():  # start() already falls back to the other webcams
            raise CameraError(camera.error, user_message=camera.error or None)
        try:
            tracker = create_hand_tracker(self.cfg.vision)
            pipeline = VisionPipeline(
                camera, tracker, self.cfg.vision, mirror=cam_cfg.mirror, preview_enabled=lambda: self.debug
            )
            pipeline.start()
        except Exception:
            camera.stop()
            raise
        self.camera, self.pipeline = camera, pipeline

    def _stop_vision(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.camera is not None:
            self.camera.stop()
        self.pipeline = self.camera = None

    def _start_hotkeys(self) -> None:
        try:
            from pynput import keyboard

            hk = self.cfg.hotkeys
            self._hotkeys = keyboard.GlobalHotKeys(
                {
                    hk.pause: self.toggle_pause,
                    hk.recalibrate: self.request_recalibrate,
                    hk.emergency_stop: self.emergency_stop,
                    hk.flip_steering: self.flip_steering,
                    hk.quit: self.request_quit,
                }
            )
            self._hotkeys.start()
            log.info("Hotkeys: %s pause/resume, %s recalibrate, %s flip steering, %s EMERGENCY STOP, %s quit", hk.pause, hk.recalibrate, hk.flip_steering, hk.emergency_stop, hk.quit)
        except Exception as exc:
            log.warning("Global hotkeys unavailable (%s). Quit with Ctrl+C or 'python main.py --stop'.", exc)

    def _start_tray(self) -> None:
        if not self.cfg.app.tray:
            return
        try:
            from gesture_car.frontend.tray import start_tray

            self._tray = start_tray(self)
        except Exception as exc:
            log.warning("Tray icon unavailable (%s); continuing without it", exc)

    # ----------------------------------------------------------------- run
    def run(self) -> int:
        running = instance.running_pid()
        if running is not None and running != os.getpid():
            log.error("The controller is already running (pid %s). Use 'python main.py --stop' first.", running)
            return 4
        try:
            self.output = create_output(self.cfg.output, self.dry_run)
        except Exception as exc:
            log.error("%s", exc)
            return 5
        output = self.output
        instance.pid_file().write_text(str(os.getpid()), encoding="utf-8")
        instance.stop_file().unlink(missing_ok=True)
        atexit.register(output.release_all)
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, lambda *_: self._stop.set())
        set_process_priority(self.cfg.app.priority)
        watchdog = threading.Thread(target=self._watchdog, name="output-watchdog", daemon=True)
        watchdog.start()
        self._start_hotkeys()
        self._start_tray()
        debug_view = None
        if self.debug:
            from gesture_car.frontend.debug_view import DebugView

            debug_view = DebugView(self.cfg.app.preview_scale, presentation=self.presentation)
        if self.record_path is not None:
            from gesture_car.backend.recording import Recorder

            self._recorder = Recorder(self.record_path)
            log.info("Recording tracked frames to %s", self.record_path)
        log.info("✓ %s controller ready", output.name)
        log.info(
            "Controller started (backend=%s, dry_run=%s). Hold both hands in the steering-wheel pose to calibrate, then CLICK YOUR "
            "GAME WINDOW: keys go to whichever window has focus (not this console).",
            output.name, self.dry_run,
        )  # fmt: skip
        sender = threading.Thread(target=self._output_loop, args=(output,), name="output", daemon=True)
        try:
            with fine_timer():
                sender.start()
                self._loop(debug_view)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            if sender.is_alive():
                sender.join(timeout=1.0)
            output.release_all()
            self._stop_vision()
            if self._hotkeys is not None:
                with contextlib.suppress(Exception):
                    self._hotkeys.stop()
            if self._tray is not None:
                with contextlib.suppress(Exception):
                    self._tray.stop()
            if debug_view is not None:
                debug_view.close()
            if self._recorder is not None:
                self._recorder.close()
                log.info("Recording saved: %d frames in %s (replay with --replay)", self._recorder.frames, self.record_path)
            instance.pid_file().unlink(missing_ok=True)
            instance.stop_file().unlink(missing_ok=True)
            log.info("Controller stopped; all inputs released")
        return 0

    def _watchdog(self) -> None:
        while not self._stop.wait(0.1):
            if time.monotonic() - self._last_beat > WATCHDOG_TIMEOUT_S and self.output is not None:
                self.output.release_all()

    def _output_loop(self, output: ControlOutput) -> None:
        """The steering path: wakes the instant a tracked frame is published, turns it into controls and sends them.

        Between frames it still ticks at ``app.loop_hz`` (keyboard PWM, hands-lost timeout). It never sleeps while a new
        frame is waiting, and nothing slow (preview, logging, camera reconnects) runs on this thread.
        """
        errors = 0
        while not self._stop.is_set():
            try:
                self._output_passes(output)
            except Exception:  # a bug must not leave a key held or silently stop steering: release, log, carry on
                errors += 1
                output.release_all()
                if errors <= 3 or errors % 100 == 0:
                    log.exception("Output thread error (%d); all inputs released", errors)
                self._stop.wait(0.05)

    def _output_passes(self, output: ControlOutput) -> None:
        period = 1.0 / self.cfg.app.loop_hz
        sequence = 0
        released = True
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            self._last_beat = now
            pipeline, camera = self.pipeline, self.camera
            camera_ok = pipeline is not None and camera is not None and camera.is_ok
            if not camera_ok or self.logic.paused:
                if not released:
                    output.release_all()  # nothing may stay pressed while the camera is gone or paused
                    released = True
                if pipeline is not None and not camera_ok:
                    self.logic.last_state = NEUTRAL
                    if self.logic.mode is not Mode.PAUSED:
                        self.logic.mode = Mode.CAMERA_LOST
                self._stop.wait(period)
                continue
            assert pipeline is not None
            result = pipeline.wait_for_new(sequence, next_tick - now)
            picked = time.monotonic()
            if camera is not None and not camera.is_ok:
                continue  # lost while waiting: released on the next pass, this frame is never sent
            released = False
            started = time.perf_counter()
            if result is not None:
                sequence = result.sequence
                state = self.logic.process(result.hands, result.timestamp)
            else:
                state = self.logic.tick(picked)
            steered = time.perf_counter()
            output.update(state, time.monotonic())
            sent = time.perf_counter()
            if result is not None:
                total = max(0.0, (time.monotonic() - result.timestamp) * 1000.0)
                stages = {
                    "wait": result.wait_ms,
                    "detect": result.inference_ms,
                    "handoff": max(0.0, (picked - result.published) * 1000.0) if result.published else 0.0,
                    "steer": (steered - started) * 1000.0,
                    "send": (sent - steered) * 1000.0,
                    "total": total,
                }
                self.timing.add(**stages)
                self._stats.add(**stages)
                self._two_hands += len(result.hands) == 2
                self.logic.fps = result.pipeline_fps
                self.logic.latency_ms = total
                if self._recorder is not None:
                    self._recorder.write(result.timestamp, result.hands, state)
            if picked >= next_tick:
                next_tick += period
                if next_tick <= picked:  # a late wake-up: skip the missed ticks instead of firing them back to back
                    next_tick = picked + period

    def _loop(self, debug_view: object | None) -> None:
        """Housekeeping on the main thread: stop requests, camera (re)connection, preview, --debug output and stats."""
        period = 1.0 / HOUSEKEEPING_HZ
        next_retry = 0.0
        camera_lost = False
        vision_failures = 0
        last_stats = last_stop_check = time.monotonic()
        last_calib_msg = last_debug = 0.0
        shown = 0
        while not self._stop.is_set():
            start = time.monotonic()
            if start - last_stop_check >= STOP_POLL_S:
                last_stop_check = start
                if instance.stop_file().exists():
                    log.info("Stop requested")
                    self._stop.set()
            if self._stop.is_set():
                break
            if self.pipeline is None:
                if start >= next_retry:
                    next_retry = start + CAMERA_RETRY_S
                    try:
                        self._start_vision()
                        vision_failures = 0
                        log.info("✓ Camera connected\n✓ AI hand tracking initialized (camera and hand tracker running)")
                    except (GestureCarError, Exception) as exc:
                        vision_failures += 1
                        message = exc.user_message if isinstance(exc, GestureCarError) else str(exc)
                        if vision_failures > 1:  # the full checklist once; then a short line per retry
                            message = message.splitlines()[0] if message else "unknown error"
                        log.error("Camera/tracker unavailable: %s (retrying in %.0f s)", message, CAMERA_RETRY_S)
                self._stop.wait(0.05)
                continue
            lost_now = self.camera is not None and not self.camera.is_ok
            if lost_now != camera_lost:
                camera_lost = lost_now
                if lost_now:
                    log.warning("CAMERA LOST: all keys released. Reconnecting every few seconds (check the cable / phone app)...")
                else:
                    log.info("Camera back: show both hands to continue")
            result = self.pipeline.latest()
            if debug_view is not None and (camera_lost or (result is not None and result.sequence != shown)):
                shown = result.sequence if result is not None else shown
                debug_view.show(result or _blank_result(), self.logic)  # type: ignore[attr-defined]
            if self.console_debug and start - last_debug >= DEBUG_PRINT_S:
                last_debug = start
                count, stages = self.timing.take()
                if count:
                    camera_fps = float(getattr(self.camera, "measured_fps", 0.0))
                    print(debug_line(self.logic, camera_fps, stages), flush=True)
            if self.logic.mode is Mode.CALIBRATING and not self.logic.paused and start - last_calib_msg >= CALIB_MSG_S:
                last_calib_msg = start
                log.info("Calibrating: %s", self.logic.calibration_message())
            if start - last_stats >= STATS_INTERVAL_S:
                frames, stages = self._stats.take()
                two_hands, self._two_hands = self._two_hands, 0
                state = self.logic.last_state
                log.info(
                    "mode=%s tracking=%.0f fps, both hands seen in %d%% of frames, latency %.1f ms "
                    "(wait %.1f, detect %.1f, handoff %.1f, steer %.2f, send %.2f), steer=%+.2f brake=%.1f nitro=%s",
                    self.logic.mode.value, frames / (start - last_stats), 100 * two_hands // max(1, frames), stages["total"],
                    stages["wait"], stages["detect"], stages["handoff"], stages["steer"], stages["send"],
                    state.steer, state.brake, state.nitro,
                )  # fmt: skip
                last_stats = start
            spare = period - (time.monotonic() - start)
            if spare > 0:
                self._stop.wait(spare)
