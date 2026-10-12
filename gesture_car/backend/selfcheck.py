"""``python main.py --check``: guided self-check with the REAL camera. Tells you which part works (camera, wheel, brake, nitro, output).

Each step prints PASS/FAIL with the numbers behind it; the report is also saved to ``check.txt`` in the data folder
(``%APPDATA%\\GestureCar``) so it can be pasted into a bug report.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field

from gesture_car.backend.gestures import GestureEngine
from gesture_car.backend.landmarks import HandLandmarks
from gesture_car.backend.steering import SteeringEstimator
from gesture_car.config import Config
from gesture_car.utils.paths import user_data_dir

PHASE_S = 7.0
LEAD_IN_S = 3
MIN_TWO_HAND_RATIO = 0.5
MIN_STEER = 0.3
MIN_GESTURE_FRAMES = 5

Frame = tuple[float, tuple[HandLandmarks, ...]]


@dataclass
class Step:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    steps: list[Step] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.steps.append(Step(name, passed, detail))

    @property
    def ok(self) -> bool:
        return all(s.passed for s in self.steps)

    def text(self) -> str:
        lines = [f"{'PASS' if s.passed else 'FAIL'}  {s.name}: {s.detail}" for s in self.steps]
        lines.append("ALL CHECKS PASSED" if self.ok else "SOME CHECKS FAILED")
        return "\n".join(lines)


# ---- pure analysis (unit-tested with synthetic landmarks) --------------------------------------------------------------
def check_wheel(frames: list[Frame], cfg: Config) -> Step:
    est = SteeringEstimator(cfg.steering)
    two = 0
    lo = hi = 0.0
    for t, hands in frames:
        two += len(hands) == 2
        value = est.update(hands, t)
        if value is not None:
            lo, hi = min(lo, value), max(hi, value)
    ratio = two / max(1, len(frames))
    detail = f"both hands in {ratio * 100:.0f}% of frames; calibrated={est.calibrated}; steering range {lo:+.2f}..{hi:+.2f}"
    if ratio < MIN_TWO_HAND_RATIO:
        detail += "  -> keep BOTH hands in view, shoulder-width apart, better light"
    elif not est.calibrated:
        detail += f"  -> hold the pose steadier ({est.calibrator.status})"
    elif lo > -MIN_STEER or hi < MIN_STEER:
        detail += "  -> tilt the wheel further, both left AND right (right hand down = right turn)"
    return Step("two-hand steering wheel", ratio >= MIN_TWO_HAND_RATIO and est.calibrated and lo <= -MIN_STEER and hi >= MIN_STEER, detail)


def check_gesture(frames: list[Frame], cfg: Config, which: str) -> Step:
    engine = GestureEngine(cfg.gestures)
    raw = confirmed = 0
    for t, hands in frames:
        state = engine.update(hands, t)
        raw += state.raw_brake if which == "brake" else state.raw_nitro
        confirmed += state.brake if which == "brake" else state.nitro
    label = "open palm -> brake" if which == "brake" else "thumbs-up -> nitro"
    hint = ""
    if confirmed < MIN_GESTURE_FRAMES:
        hint = (
            "  -> straighten ALL five fingers, palm facing the camera (gestures.finger_extended_min / thumb_extended_min)"
            if which == "brake"
            else "  -> thumb straight up, other fingers curled tight (gestures.thumbs_up_max_finger / thumbs_up_max_tilt_deg)"
        )
    return Step(label, confirmed >= MIN_GESTURE_FRAMES, f"seen in {raw} frames, confirmed in {confirmed}{hint}")


# ---- live run -----------------------------------------------------------------------------------------------------------
def _record(pipeline: object, seconds: float) -> list[Frame]:
    frames: list[Frame] = []
    last = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        result = pipeline.latest()  # type: ignore[attr-defined]
        if result is not None and result.sequence != last:
            last = result.sequence
            frames.append((result.timestamp, result.hands))
        time.sleep(0.005)
    return frames


def _countdown(prompt: str) -> None:
    print(f"\n>>> {prompt}", flush=True)
    for n in range(LEAD_IN_S, 0, -1):
        print(f"    starting in {n}...", flush=True)
        time.sleep(1.0)
    print(f"    GO ({PHASE_S:.0f} s)", flush=True)


def run_check(cfg: Config) -> int:
    from gesture_car.backend.controller import GestureController

    report = Report()
    print("VisionDrive self-check. Sit in front of the camera; follow each prompt.\n")
    report.add("system", True, f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    ctrl = GestureController(cfg)
    try:
        ctrl._start_vision()
    except Exception as exc:
        report.add("camera + hand tracker start", False, f"{exc}")
        return _finish(report)
    try:
        assert ctrl.pipeline is not None and ctrl.camera is not None
        report.add("camera + hand tracker start", True, f"{ctrl.camera.description}, tracker {ctrl.pipeline.backend_name}")
        warm = _record(ctrl.pipeline, 2.0)
        fps = len(warm) / 2.0
        report.add("tracking speed", fps >= 15.0, f"{fps:.0f} frames/s (want 30; 15 is the minimum)")
        _countdown("Hold BOTH hands like a steering wheel, then slowly TILT it left, then right")
        report.steps.append(check_wheel(_record(ctrl.pipeline, PHASE_S), cfg))
        _countdown("Show ONE OPEN PALM (all fingers spread) to the camera")
        report.steps.append(check_gesture(_record(ctrl.pipeline, PHASE_S), cfg, "brake"))
        _countdown("Show a THUMBS-UP (thumb up, other fingers curled)")
        report.steps.append(check_gesture(_record(ctrl.pipeline, PHASE_S), cfg, "nitro"))
    finally:
        ctrl._stop_vision()
    try:
        from gesture_car.backend.output import create_output

        out = create_output(cfg.output)
        report.add("output backend", True, f"{out.name} (keys: {cfg.output.key_preset})" if out.name == "keyboard" else out.name)
        out.release_all()
        if cfg.output.backend == "gamepad" and out.name != "gamepad":
            report.add("virtual gamepad", False, "ViGEmBus / vgamepad not available - fell back to the keyboard")
    except Exception as exc:
        report.add("output backend", False, str(exc))
    print("\nTo check that a game receives the keys/gamepad: run  python main.py --test-output")
    return _finish(report)


def _finish(report: Report) -> int:
    text = report.text()
    print("\n=== RESULTS ===\n" + text)
    try:
        path = user_data_dir() / "check.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {path}")
    except OSError:
        pass
    return 0 if report.ok else 1
