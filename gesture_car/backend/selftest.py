"""``python main.py --test-output``: tap every output once so you can check that a game / Notepad / a gamepad tester receives it.

No camera is used. Start it, click the window that should receive the input (a game, a browser tab, Notepad) within the countdown.
"""

from __future__ import annotations

import time

from gesture_car.backend.output import ControlOutput, ControlState, create_output
from gesture_car.config import Config

COUNTDOWN_S = 5
HOLD_S = 0.6
GAP_S = 0.5
STEPS: tuple[tuple[str, ControlState], ...] = (
    ("ACCELERATE", ControlState(throttle=1.0)),
    ("BRAKE", ControlState(brake=1.0)),
    ("STEER FULL LEFT", ControlState(steer=-1.0)),
    ("STEER FULL RIGHT", ControlState(steer=1.0)),
    ("STEER HALF RIGHT (pulsed)", ControlState(steer=0.5)),
    ("NITRO", ControlState(nitro=True)),
)


def _hold(out: ControlOutput, state: ControlState, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        out.update(state, time.monotonic())
        time.sleep(0.005)
    out.release_all()


def run_output_test(cfg: Config, dry_run: bool = False, sleep: bool = True) -> int:
    try:
        out = create_output(cfg.output, dry_run)
    except Exception as exc:
        print(f"Cannot create the output: {exc}")
        return 5
    keys = ""
    if out.name == "keyboard":
        keys = f" (preset '{cfg.output.key_preset}', nitro key '{cfg.output.key_nitro}')"
    print(f"Output test: backend = {out.name}{keys}{' [dry run: nothing is sent]' if dry_run else ''}")
    print(f"Click the game / browser tab / Notepad that should receive the input. Starting in {COUNTDOWN_S} s...")
    try:
        for n in range(COUNTDOWN_S, 0, -1):
            print(f"  {n}...", flush=True)
            if sleep:
                time.sleep(1.0)
        for label, state in STEPS:
            print(f"  sending: {label}", flush=True)
            _hold(out, state, HOLD_S if sleep else 0.02)
            if sleep:
                time.sleep(GAP_S)
    finally:
        out.release_all()
    print("Done. If nothing reached the game: run both as Administrator, make sure the game window was focused,")
    print("and check the game's controls (try --keys arrows, --keys wasd or --backend gamepad).")
    return 0
