"""Typed configuration loaded from ``config.json``.

Every tunable number lives here (no magic numbers in the code). Unknown keys, wrong types and out-of-range values never crash
start-up: they are logged and replaced by the default (or clamped into the allowed range).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

log = logging.getLogger(__name__)

Limits = dict[str, tuple[float, float]]
Choices = dict[str, tuple[str, ...]]


@dataclass
class CameraConfig:
    index: int = 0  # which webcam (0 = first); see --list-cameras
    source: str = ""  # "" = use index; a number ("1") = that webcam; or a phone-stream URL like http://192.168.1.5:8080/video
    backend: str = "auto"  # Windows capture backend: auto (DirectShow, then Media Foundation, then default) | dshow | msmf
    width: int = 640  # 640x480 keeps latency lowest; the closest mode the camera supports is used
    height: int = 480
    fps: int = 60  # asked for; most webcams top out at 30 and the closest rate they support is used
    mjpg: bool = True  # ask USB webcams for MJPG (higher FPS over USB); falls back automatically if unsupported
    mirror: bool = True
    auto_exposure: bool = True  # false = fixed ``exposure``: auto-exposure in a dim room can drop a webcam to 15-20 fps
    exposure: float = -6.0  # fixed exposure when auto_exposure is false (DirectShow units: -6 = 1/64 s; -5 brighter, -7 darker)
    LIMITS: ClassVar[Limits] = {"index": (0, 16), "width": (160, 3840), "height": (120, 2160), "fps": (5, 120), "exposure": (-13.0, 0.0)}
    CHOICES: ClassVar[Choices] = {"backend": ("auto", "dshow", "msmf")}


@dataclass
class VisionConfig:
    backend: str = "auto"  # auto | tasks | legacy
    model_path: str = ""
    max_hands: int = 2
    model_complexity: int = 0  # 0 = fastest; only used by the legacy MediaPipe backend
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.4
    min_tracking_confidence: float = 0.5  # tracking mode: the palm detector only re-runs when tracking drops below this
    inference_width: int = 640
    use_gpu: bool = False  # try MediaPipe's GPU delegate; falls back to the CPU (and logs why) if unsupported or slower
    LIMITS: ClassVar[Limits] = {
        "max_hands": (2, 2),
        "model_complexity": (0, 1),
        "min_detection_confidence": (0.1, 1.0),
        "min_presence_confidence": (0.1, 1.0),
        "min_tracking_confidence": (0.1, 1.0),
        "inference_width": (160, 1920),
    }
    CHOICES: ClassVar[Choices] = {"backend": ("auto", "tasks", "legacy")}


@dataclass
class SteeringConfig:
    feel: str = "custom"  # custom = use the numbers below | arcade (snappy) | normal | realistic (heavy wheel); a preset overrides the smoothing and rate limits, never full_lock_deg / deadzone_deg
    anchor: str = "palm"  # which landmark represents each hand: palm (centre of palm) | wrist
    calibration_seconds: float = 2.0  # hold the wheel straight this long on start-up / F9; the average angle becomes neutral
    calibration_max_jitter_deg: float = 10.0  # the pose must stay within this spread (10th-90th percentile) during the hold
    full_lock_deg: float = 30.0  # steering = clamp(offset / full_lock_deg): 3 deg = 10%, 15 deg = 50%, 30+ = 100%
    deadzone_deg: float = 1.0  # offsets smaller than this give exactly 0, immediately (only absorbs tracking jitter)
    smoothing: str = "one_euro"  # one_euro | ema
    one_euro_min_cutoff: float = 0.9  # Hz; lower = smoother but laggier at rest
    one_euro_beta: float = 0.08  # higher = less lag during turns
    one_euro_d_cutoff: float = 1.0
    ema_alpha: float = 0.4
    predict_frames: float = 0.5  # extrapolate the filtered angle this many frames ahead (hides filter lag; 0 = off)
    auto_recenter_deg: float = 6.0  # held this close to neutral for auto_recenter_after_s -> neutral slowly follows (0 = off)
    auto_recenter_after_s: float = 3.0
    invert_steering: bool = False  # swap left/right (F7 / --invert)
    min_hand_separation: float = 0.08  # hands closer than this (fraction of image height) give no steering
    LIMITS: ClassVar[Limits] = {
        "calibration_seconds": (0.5, 10.0),
        "calibration_max_jitter_deg": (1.0, 30.0),
        "full_lock_deg": (5.0, 90.0),
        "deadzone_deg": (0.0, 15.0),
        "one_euro_min_cutoff": (0.05, 20.0),
        "one_euro_beta": (0.0, 5.0),
        "one_euro_d_cutoff": (0.1, 10.0),
        "ema_alpha": (0.02, 1.0),
        "min_hand_separation": (0.0, 1.0),
        "predict_frames": (0.0, 3.0),
        "auto_recenter_deg": (0.0, 15.0),
        "auto_recenter_after_s": (0.5, 30.0),
    }
    CHOICES: ClassVar[Choices] = {
        "feel": ("custom", "arcade", "normal", "realistic"),
        "anchor": ("palm", "wrist"),
        "smoothing": ("one_euro", "ema"),
    }
    ALIASES: ClassVar[dict[str, str]] = {  # names used by older config.json files
        "dead_zone_deg": "deadzone_deg",
        "max_angle_deg": "full_lock_deg",
        "full_lock_offset_deg": "full_lock_deg",
        "invert": "invert_steering",
    }
    RETIRED: ClassVar[frozenset[str]] = frozenset({"turn_rate_per_s", "return_rate_per_s", "curve_exponent"})  # silently ignored


# Feel presets change only the smoothing: the proportional mapping (full_lock_deg, deadzone_deg) always comes from
# config.json, so a profile can never add a bigger dead zone, a curve or a rate limit.
FEEL_PRESETS: dict[str, dict[str, Any]] = {
    "arcade": {"one_euro_min_cutoff": 2.5, "one_euro_beta": 0.08},
    "normal": {"one_euro_min_cutoff": 1.5, "one_euro_beta": 0.05},
    "realistic": {"one_euro_min_cutoff": 1.0, "one_euro_beta": 0.03},
}


def apply_feel(steering: SteeringConfig) -> None:
    """Overwrite the steering numbers with the chosen ``feel`` preset (``custom`` keeps whatever is in config.json)."""
    for key, value in FEEL_PRESETS.get(steering.feel, {}).items():
        setattr(steering, key, value)


@dataclass
class GestureConfig:
    confirm_frames: int = 2  # consecutive frames needed to switch a gesture on
    release_s: float = 0.1  # a gesture stays on this long after it stops being seen (anti-flicker)
    min_hand_score: float = 0.5
    finger_extended_min: float = 0.6  # index..pinky count as extended from this score (a wheel grip scores under 0.5)
    thumb_extended_min: float = 0.5  # the thumb counts as extended from this score (a fist's thumb scores about 0.3)
    palm_min_spread: float = 0.0  # optional: also require fanned fingers (fingertip gap / knuckle gap, 1.1 = spread); 0 = off
    reverse_after_s: float = 0.6  # two open palms held this long = REVERSE, for as long as they are held
    palms_steering: str = "full"  # while two palms are held: full = normal steering from the line between the palms |
    # intentional = steer only past palms_intentional_deg | frozen = steering 0
    palms_intentional_deg: float = 20.0
    thumbs_up_min_thumb: float = 0.55
    thumbs_up_max_finger: float = 0.40  # the other four fingers must be at most this extended
    thumbs_up_max_tilt_deg: float = 50.0  # thumb must point within this angle of straight up
    nitro_mode: str = "hold"  # hold = nitro while the thumbs-up is shown, toggle = each thumbs-up flips it, pulse = short burst
    nitro_pulse_s: float = 0.5
    LIMITS: ClassVar[Limits] = {
        "confirm_frames": (1, 15),
        "release_s": (0.0, 1.0),
        "min_hand_score": (0.0, 1.0),
        "finger_extended_min": (0.3, 1.0),
        "thumb_extended_min": (0.2, 1.0),
        "palm_min_spread": (0.0, 2.5),
        "reverse_after_s": (0.1, 5.0),
        "palms_intentional_deg": (0.0, 90.0),
        "thumbs_up_min_thumb": (0.2, 1.0),
        "thumbs_up_max_finger": (0.0, 0.9),
        "thumbs_up_max_tilt_deg": (10.0, 90.0),
        "nitro_pulse_s": (0.1, 5.0),
    }
    CHOICES: ClassVar[Choices] = {"nitro_mode": ("hold", "toggle", "pulse"), "palms_steering": ("full", "intentional", "frozen", "freeze")}  # freeze = older name of frozen
    ALIASES: ClassVar[dict[str, str]] = {"palm_min_extension": "finger_extended_min"}  # older config.json name


@dataclass
class SafetyConfig:
    hands_lost_s: float = 0.3  # steering hands missing this long -> safe state
    lost_behavior: str = "neutral"  # neutral | gentle_brake (S/left trigger can mean REVERSE in some games)
    gentle_brake_level: float = 0.35
    freeze_steering_during_gesture: bool = True  # a hand OUT OF VIEW for brake/nitro holds the wheel; never with both hands seen
    freeze_lookback_s: float = 0.15
    LIMITS: ClassVar[Limits] = {
        "hands_lost_s": (0.05, 5.0),
        "gentle_brake_level": (0.0, 1.0),
        "freeze_lookback_s": (0.0, 1.0),
    }
    CHOICES: ClassVar[Choices] = {"lost_behavior": ("neutral", "gentle_brake")}


@dataclass
class OutputConfig:
    backend: str = "keyboard"  # keyboard (works in any game/browser tab) | gamepad (virtual Xbox 360 via ViGEmBus, true analog)
    fallback_to_keyboard: bool = True  # use the keyboard if the virtual gamepad cannot be created
    auto_accel_level: float = 1.0  # right trigger / key duty while accelerating
    nitro_button: str = "a"  # gamepad: a | b | x | y | lb | rb
    key_preset: str = "both"  # both = WASD + arrow keys at once | wasd | arrows | custom (use the key_* fields below)
    key_accelerate: str = "w"
    key_brake: str = "s"
    key_nitro: str = "shift"
    key_left: str = "a"
    key_right: str = "d"
    reverse_mode: str = "brake_hold"  # brake_hold = keep brake held (most racing games reverse that way) | button = press key_reverse / reverse_button
    key_reverse: str = "r"
    reverse_button: str = "b"  # gamepad button for reverse_mode "button": a | b | x | y | lb | rb
    pwm_period_s: float = 0.05  # keyboard steering: one on/off cycle
    pwm_min_duty: float = 0.0  # below this steering amount no key is pressed (0 = every turn outside the dead zone taps the key)
    pwm_full_duty: float = 0.95  # above this the key is simply held
    LIMITS: ClassVar[Limits] = {
        "auto_accel_level": (0.1, 1.0),
        "pwm_period_s": (0.02, 0.5),
        "pwm_min_duty": (0.0, 0.5),
        "pwm_full_duty": (0.5, 1.0),
    }
    CHOICES: ClassVar[Choices] = {
        "backend": ("gamepad", "keyboard"),
        "key_preset": ("both", "wasd", "arrows", "custom"),
        "nitro_button": ("a", "b", "x", "y", "lb", "rb"),
        "reverse_button": ("a", "b", "x", "y", "lb", "rb"),
        "reverse_mode": ("brake_hold", "button"),
    }


@dataclass
class HotkeyConfig:
    pause: str = "<f8>"
    recalibrate: str = "<f9>"
    emergency_stop: str = "<f12>"  # releases every key at once and pauses (F8 resumes)
    flip_steering: str = "<f7>"  # swap left/right if the car turns the wrong way (saved to config.json)
    quit: str = "<f10>"
    LIMITS: ClassVar[Limits] = {}
    CHOICES: ClassVar[Choices] = {}


@dataclass
class AppConfig:
    loop_hz: int = 250  # output thread rate (also drives keyboard PWM); new camera results are sent at once, not on a tick
    priority: str = "high"  # process priority: high keeps frames on time while a game runs | above_normal | normal | below_normal
    log_level: str = "INFO"
    preview: bool = True  # small always-on-top camera window; run_silent.vbs / --no-preview turn it off
    preview_scale: float = 0.5  # window size multiplier (0.5 = about 320 px wide; 0.7 = 450; 1 = 640; 1.5 = projector)
    tray: bool = False  # optional tray icon (Start/Pause/Recalibrate/Quit); needs pystray + pillow
    LIMITS: ClassVar[Limits] = {"loop_hz": (30, 1000), "preview_scale": (0.3, 3.0)}
    CHOICES: ClassVar[Choices] = {"log_level": ("DEBUG", "INFO", "WARNING", "ERROR"), "priority": ("high", "above_normal", "normal", "below_normal")}
    RETIRED: ClassVar[frozenset[str]] = frozenset({"low_priority"})


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: {k.name: getattr(getattr(self, f.name), k.name) for k in fields(getattr(self, f.name))} for f in fields(self)}

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load ``path`` (default ``config.json`` next to ``main.py``); never raises on bad content."""
        path = path or default_path()
        cfg = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            log.info("No %s found; using defaults", path)
            return cfg
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s (%s); using defaults", path, exc)
            return cfg
        if not isinstance(data, dict):
            log.warning("%s must contain a JSON object; using defaults", path)
            return cfg
        for name in data:
            if name not in cfg.to_dict():
                log.warning("Unknown config section %r ignored", name)
        for f in fields(cfg):
            raw = data.get(f.name)
            if isinstance(raw, dict):
                _apply_section(getattr(cfg, f.name), raw, f.name)
            elif raw is not None:
                log.warning("Config section %r must be an object; defaults used", f.name)
        apply_feel(cfg.steering)
        return cfg

    def save(self, path: Path | None = None) -> None:
        (path or default_path()).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.json"


def _coerce(default: Any, value: Any) -> Any:
    """Return ``value`` converted to the type of ``default`` or raise ValueError."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(default, int):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer():
            return int(value)
    elif isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(default, str):
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):  # e.g. "source": 1
            return str(value)
    raise ValueError(f"expected {type(default).__name__}")


def _apply_section(section: Any, raw: dict[str, Any], name: str) -> None:
    known = {f.name for f in fields(section)}
    limits: Limits = section.LIMITS
    choices: Choices = section.CHOICES
    aliases: dict[str, str] = getattr(section, "ALIASES", {})
    retired: frozenset[str] = getattr(section, "RETIRED", frozenset())
    for key, value in raw.items():
        if key in retired:
            continue  # a setting that no longer exists; older config.json files still load without warnings
        if key in aliases:
            if aliases[key] in raw:
                continue  # the new name wins when a file has both
            key = aliases[key]
        if key not in known:
            log.warning("Unknown config key %s.%s ignored", name, key)
            continue
        default = getattr(section, key)
        try:
            coerced = _coerce(default, value)
        except ValueError as exc:
            log.warning("Invalid %s.%s=%r (%s); keeping %r", name, key, value, exc, default)
            continue
        if key in choices:
            text = str(coerced).lower() if key != "log_level" else str(coerced).upper()
            if text not in choices[key]:
                log.warning("Invalid %s.%s=%r; allowed: %s; keeping %r", name, key, value, ", ".join(choices[key]), default)
                continue
            coerced = text
        if key in limits and not isinstance(coerced, (bool, str)):
            low, high = limits[key]
            clamped = min(high, max(low, coerced))
            if clamped != coerced:
                log.warning("%s.%s=%r outside %s..%s; clamped to %r", name, key, coerced, low, high, clamped)
            coerced = type(default)(clamped)
        setattr(section, key, coerced)
