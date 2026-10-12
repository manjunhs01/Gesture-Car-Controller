"""Command line entry point. ``--no-preview`` (or ``--debug`` without ``--preview``) runs without any window."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from gesture_car import __version__
from gesture_car.backend import instance
from gesture_car.config import CameraConfig, Config, apply_feel, default_path
from gesture_car.utils.errors import GestureCarError
from gesture_car.utils.logging_setup import setup_logging

RESOLUTIONS = ("640x480", "1280x720", "1920x1080")
CAMERA_EXAMPLES = """webcam examples:
  python main.py --list-cameras                     see which cameras work (number, size, real FPS, name)
  python main.py --camera 1                         use the second webcam (e.g. an external USB webcam)
  python main.py --camera 1 --camera-backend msmf   external webcam that only works with Media Foundation
  python main.py --resolution 1280x720 --fps 30     sharper picture (640x480 is the default: lowest latency)
  python main.py --camera-url http://192.168.1.5:8080/video   phone as webcam (IP Webcam / DroidCam app)
  python main.py --no-mirror                        camera or phone app already mirrors the picture
"""


def _ranged_int(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visiondrive",
        description="VisionDrive: hand-gesture-controlled car game controller (two-hand steering wheel for any game).",
        epilog=CAMERA_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, help="path to config.json (default: next to main.py)")
    p.add_argument("--preview", dest="preview", action="store_true", default=None,
                   help="show the small always-on-top camera window (this is the default)")  # fmt: skip
    p.add_argument("--debug", action="store_true",
                   help="every 100 ms print wheel angle, neutral, offset, steering %%, FPS and per-stage latency (no window unless --preview)")  # fmt: skip
    p.add_argument("--no-preview", dest="preview", action="store_false", help="no window at all (fully silent)")
    p.add_argument("--dry-run", action="store_true", help="process the camera but send no keys / gamepad input")
    p.add_argument("--backend", choices=("keyboard", "gamepad"), help="output backend (overrides config.json)")
    p.add_argument("--keys", choices=("both", "wasd", "arrows"), help="which keys the keyboard backend presses (overrides config.json)")
    p.add_argument("--feel", choices=("arcade", "normal", "realistic"), help="steering feel: arcade = snappy, realistic = heavy wheel, big rotation")
    p.add_argument("--preview-scale", type=float, help="size of the preview window, e.g. 1.6 for a presentation (default 1)")
    p.add_argument("--profile", help="apply a game profile from the profiles folder (see --list-profiles)")
    p.add_argument("--list-profiles", action="store_true", help="list the available game profiles and exit")
    p.add_argument("--record", metavar="FILE", help="save every tracked frame (hand landmarks + controls) to FILE for later replay")
    p.add_argument("--replay", metavar="FILE", help="run a recording through the control logic without a camera and print a summary")
    p.add_argument("--presentation", action="store_true", help="stage/demo look: title header and big preview (does not change what is sent to the game)")
    p.add_argument("--invert", action="store_true", help="swap steering left/right (if the car turns the wrong way; same as pressing F7)")
    p.add_argument("--check", action="store_true", help="guided self-check of camera, wheel, brake, nitro and output; saves a report")
    p.add_argument("--test-output", action="store_true", help="after a 5 s countdown tap every output once, to check a game/Notepad receives it")
    p.add_argument("--stop", action="store_true", help="ask a running controller to quit")
    p.add_argument("--list-cameras", action="store_true", help="list cameras 0-9 with resolution, real FPS and name, then exit")
    cam = p.add_argument_group("webcam (each flag overrides the camera section of config.json)")
    which = cam.add_mutually_exclusive_group()
    which.add_argument("--camera", metavar="N", type=_ranged_int(0, 16), help="webcam number (0 = first; see --list-cameras)")
    which.add_argument("--camera-url", metavar="URL", help="phone or network camera stream, e.g. http://192.168.1.5:8080/video")
    cam.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"),
                     help="Windows capture backend: auto (DirectShow, then Media Foundation), dshow or msmf")  # fmt: skip
    cam.add_argument("--resolution", choices=RESOLUTIONS, help="camera size (default 640x480 = lowest latency; the closest supported size is used)")
    cam.add_argument("--fps", type=_ranged_int(5, 120), help="camera frames per second to ask for (default 60; the closest rate the camera supports is used)")
    cam.add_argument("--no-mirror", dest="mirror", action="store_false", default=None,
                     help="do not mirror the picture (if the camera or phone app already mirrors it)")  # fmt: skip
    p.add_argument("--write-default-config", action="store_true", help="write a default config.json and exit")
    p.add_argument("--version", action="version", version=f"VisionDrive {__version__}")
    return p


def _fix_missing_streams() -> None:
    """``pythonw`` has no stdout/stderr; give them a sink so logging and prints cannot crash."""
    import os
    import sys

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def apply_camera_args(cam: CameraConfig, args: argparse.Namespace) -> None:
    """Copy the webcam flags into the camera settings."""
    if args.camera is not None:
        cam.index = args.camera
        cam.source = ""  # an explicit webcam number wins over a URL in config.json
    if args.camera_url:
        cam.source = args.camera_url.strip()
    if args.camera_backend:
        cam.backend = args.camera_backend
    if args.resolution:
        cam.width, cam.height = (int(v) for v in args.resolution.split("x"))
    if args.fps:
        cam.fps = args.fps
    if args.mirror is not None:
        cam.mirror = args.mirror


def list_cameras(cam: CameraConfig) -> int:
    """Print every camera 0-9 with what it really delivers."""
    from gesture_car.backend.camera import no_camera_message, probe_cameras

    print(f"Looking for cameras 0-9 at {cam.width}x{cam.height} @ {cam.fps} fps (backend: {cam.backend}); this takes a few seconds...")
    found = probe_cameras(cam)
    for c in found:
        name = f'  "{c.name}"' if c.name else ""
        if c.works:
            fmt = f", {c.fourcc}" if c.fourcc else ""
            print(f"  camera {c.index}: {c.width}x{c.height}, {c.fps:.1f} fps measured, {c.backend}{fmt}{name}")
        else:
            print(f"  camera {c.index}: Windows lists it, but it gave no picture (in use by another app, unplugged, or not answering){name}")
    working = [c for c in found if c.works]
    if working:
        print(f"\nUse one with:  python main.py --camera {working[0].index}   (or set camera.index in config.json)")
        return 0
    print("\n" + no_camera_message(0))
    return 1


def main(argv: list[str] | None = None) -> int:
    import os

    os.environ.setdefault("GLOG_minloglevel", "2")  # hide MediaPipe/TensorFlow Lite C++ log noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _fix_missing_streams()
    from gesture_car.utils.console import disable_quick_edit

    disable_quick_edit()  # a click in the console must not freeze the program
    args = build_parser().parse_args(argv)
    if args.stop:
        ok = instance.request_stop()
        print("Controller stopped." if ok else "Controller did not stop in time.")
        return 0 if ok else 1
    path = args.config or default_path()
    if args.write_default_config:
        Config().save(path)
        print(f"Wrote {path}")
        return 0
    setup_logging("INFO", console=True)
    cfg = Config.load(path)
    logging.getLogger().setLevel(cfg.app.log_level)
    if args.list_profiles:
        from gesture_car.backend.profiles import list_profiles

        for name, description in list_profiles().items():
            print(f"{name:18} {description}")
        return 0
    if args.profile:
        from gesture_car.backend.profiles import apply_profile

        try:
            apply_profile(cfg, args.profile)
        except GestureCarError as exc:
            print(exc.user_message)
            return 2
    apply_camera_args(cfg.camera, args)
    if args.list_cameras:
        return list_cameras(cfg.camera)
    if args.backend:
        cfg.output.backend = args.backend
    if args.keys:
        cfg.output.key_preset = args.keys
    if args.preview is not None:
        cfg.app.preview = args.preview
    elif args.debug:
        cfg.app.preview = False
    if args.preview_scale:
        cfg.app.preview_scale = min(3.0, max(0.3, args.preview_scale))
    if args.presentation:
        cfg.app.preview = True
    if args.invert:
        cfg.steering.invert_steering = not cfg.steering.invert_steering
    if args.feel:
        cfg.steering.feel = args.feel
        apply_feel(cfg.steering)
    if args.replay:
        from gesture_car.backend.recording import replay, summarize

        try:
            print(summarize(replay(Path(args.replay), cfg)))
        except (OSError, ValueError, KeyError) as exc:
            print(f"Could not replay {args.replay}: {exc}")
            return 2
        return 0
    if args.check:
        from gesture_car.backend.selfcheck import run_check

        return run_check(cfg)
    if args.test_output:
        from gesture_car.backend.selftest import run_output_test

        return run_output_test(cfg, dry_run=args.dry_run)
    from gesture_car.backend.controller import GestureController

    return GestureController(cfg, dry_run=args.dry_run, debug=cfg.app.preview, record_path=Path(args.record) if args.record else None, presentation=args.presentation, console_debug=args.debug).run()
