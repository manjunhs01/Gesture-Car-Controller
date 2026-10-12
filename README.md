# VisionDrive: hand-gesture-controlled car game controller

Drive **any game** (browser games and installed PC games) with your webcam and your hands. There is **no built-in game and no
overlay**: a small camera window sits at the top-right (it never takes focus), and the program sends input to whatever game has focus. Use `--no-preview` or `run_silent.vbs` for fully silent mode.

* Both hands form an **imaginary steering wheel** — rotate it left/right for proportional, analog steering.
* **Open palm** = brake. **Both palms held** = brake, then **reverse**. **Thumbs-up** = nitro. **Acceleration is automatic.**
* Output as a **virtual Xbox 360 gamepad** (true analog steering) or as **keyboard keys** (PWM-pulsed A/D for analog feel).


## Controls

| Control | Gesture | Gamepad output | Keyboard output (defaults) |
|---|---|---|---|
| Steering | Rotate the two-hand "wheel" (line between your hands) | Left stick X, analog −1…+1 | `A`/`D` **and** ←/→, pulsed; duty cycle = steering amount |
| Acceleration | none — always on while both hands are tracked | Right trigger | hold `W` **and** ↑ |
| Brake | **Open palm** (all 5 fingers extended; spreading them is not needed) with either hand | Left trigger | `S` **and** ↓ |
| Reverse | **Both hands open palm**, held longer than `gestures.reverse_after_s` (0.6 s); reverses for as long as held | Left trigger kept held | `S` kept held |
| Nitro | **Thumbs-up** (thumb up, other fingers curled) with either hand | `A` button | `Shift` |

Rules: brake and reverse stop acceleration and win over nitro. Palms are recognised from finger states only (confirmed over 2 frames,
100 ms release hysteresis), never from the wheel angle. While both palms are up the steering is held at 0 (`gestures.palms_steering:
intentional` lets a clearly tilted pair past `palms_intentional_deg` steer). Back in the wheel pose (both hands gripping) brake and reverse
stop on that frame and acceleration resumes. With `output.reverse_mode: button` reverse presses `output.key_reverse` (keyboard, default
`R`) / `output.reverse_button` (gamepad) instead of holding the brake. A hand that leaves the picture for a gesture holds the steering from
just before it left. If the steering hands are lost for more than 300 ms the car is centred, nitro is
released and acceleration stops (optionally a gentle brake — see `safety.lost_behavior`). A palm still brakes with only one hand visible.

Hotkeys: **F8** pause/resume · **F9** recalibrate · **F7** flip steering left/right (saved) · **F12** EMERGENCY STOP (releases every key, pauses; F8 resumes) · **F10** quit. Optional tray icon (Pause/Recalibrate/Quit) via `app.tray`.

## Windows setup

1. Install **Python 3.10–3.12** from python.org (tick *Add python.exe to PATH*). Very new Python versions may lack MediaPipe packages.
2. Get the code: `git clone https://github.com/manjunhs01/gesture-car.git`, then `cd gesture-car`.
3. For the **gamepad** mode install the **ViGEmBus** driver (https://github.com/nefarius/ViGEmBus/releases — the project is reported to be
   retired but the last release is still what `vgamepad` uses). Reboot if the installer asks. Skip this for keyboard-only use
   (keyboard is the default).
4. Run `run.bat` — first run creates `.venv` and installs `requirements.txt` (a few minutes; the hand model is downloaded once).
5. For a fully invisible start use `run_silent.vbs` afterwards. Stop with **F10** or `stop.bat`. Logs: `%APPDATA%\GestureCar\logs`.

Commands: `python main.py` · `--dry-run` (camera on, **no input sent**; use this first) · the small camera window is **on by default** (docked top-right, always on top, does not take focus; `q` closes it); `--no-preview` hides it ·
`--test-output` (taps every output once; no camera) · `--keys both|wasd|arrows` · `--backend keyboard|gamepad` · `--stop` · `--list-cameras` · `--camera N` · `--camera-url URL` · `--resolution WxH` · `--fps N` · `--no-mirror` (see *Choosing a webcam*) · `--config path` · `--write-default-config`. `python main.py --help` lists everything.

## Choosing a webcam

```
python main.py --list-cameras
```

lists cameras 0 to 9 with the size and the **frame rate they really deliver** (measured, not what the driver claims) and, on Windows,
their name, for example:

```
  camera 0: 640x480, 29.8 fps measured, DirectShow, YUY2  "Integrated Camera"
  camera 1: 640x480, 30.0 fps measured, DirectShow, MJPG  "HD Pro Webcam C920"
```

Then start with that number: `python main.py --camera 1` (or set `camera.index` in `config.json` to use it every time).

| Flag | `config.json` key | What it does |
|---|---|---|
| `--camera N` | `camera.index` | Webcam number (0 = first). If it cannot be opened, the other cameras are tried automatically and the log says which one is used. |
| `--camera-url URL` | `camera.source` | A phone / network camera stream instead of a webcam (see below). `camera.source` can also hold a webcam number. |
| `--camera-backend auto\|dshow\|msmf` | `camera.backend` | Windows capture system. `auto` tries DirectShow, then Media Foundation, then OpenCV's default. If the first one that works gives clearly fewer frames than asked for, `auto` also measures the others and **uses the fastest** (many laptop cameras give 20 fps through DirectShow but 30 through Media Foundation); this adds a few seconds to the start only for such cameras. Some external USB webcams only work with one backend: then force it. |
| `--resolution 640x480\|1280x720\|1920x1080` | `camera.width`, `camera.height` | Picture size. **640x480 is the default** because low latency matters more than sharpness. If the camera does not offer the size, the closest one it supports is used. |
| `--fps N` | `camera.fps` | Frame rate to ask for (default 60; most webcams deliver 30 at 640x480, and the closest supported rate is used). |
| (none) | `camera.mjpg` | Ask USB webcams for MJPG, which gives higher FPS over USB (default on; if a camera then gives no picture it is reopened without MJPG). |
| `--no-mirror` | `camera.mirror` | Do not mirror the picture (use it if the camera or phone app already mirrors it). |

The log always shows what is really used, e.g. `Using camera 1 via DirectShow, 640x480 MJPG @ 30 fps` and a few seconds later
`Camera delivers 29.9 fps`.

**If the camera disconnects while driving** (cable pulled, phone Wi-Fi drops) every key is released at once, the preview shows a
red **CAMERA LOST** banner, and the program keeps trying to reconnect every 2 seconds. When the picture is back, show both hands
and keep driving. It never leaves a key pressed.

## Phone as a webcam (over Wi-Fi)

No webcam, or want a better one? Use your phone. The phone sends its camera picture over Wi-Fi and VisionDrive reads it.

1. **Put the phone and the PC on the same Wi-Fi network** (the same router; not the phone's mobile data, not a guest network).
2. **Install a camera-streaming app on the phone:**
   * Android: **IP Webcam** (by Pavel Khlebovich) or **DroidCam**, from the Play Store.
   * iPhone: **DroidCam** from the App Store.
3. **Start the stream in the app:**
   * IP Webcam: scroll to the bottom and tap **Start server**. The screen shows an address like `http://192.168.1.5:8080`.
   * DroidCam: open the app. It shows a **WiFi IP** (e.g. `192.168.1.5`) and a **port** (usually `4747`).
4. **Check the address on the PC first:** open it in a web browser (e.g. `http://192.168.1.5:8080` for IP Webcam). If you see the
   phone's camera page, the connection works. If not, see *Troubleshooting* below. **Close the browser tab afterwards**: some apps
   (DroidCam) allow only one viewer at a time.
5. **Start VisionDrive with the video address:**
   * IP Webcam: `python main.py --camera-url http://192.168.1.5:8080/video`
   * DroidCam: `python main.py --camera-url http://192.168.1.5:4747/video`

   Use the numbers your phone shows. With `run.bat` the same flags work: `.\run.bat --camera-url http://192.168.1.5:8080/video`.
6. To use the phone every time, put the address in `config.json`: `"camera": { "source": "http://192.168.1.5:8080/video", ... }`.
   Set `"source": ""` to go back to the webcam.

Tips: stand the phone upright and steady about 1-2 m in front of you, plug in its charger, and keep the app on screen (some phones
stop the camera when the screen turns off). In the app choose a **low resolution such as 640x480**: it lags less. VisionDrive always
uses only the newest picture and drops older ones, so a short Wi-Fi hiccup does not build up a delay. If the car steers the wrong
way, add `--no-mirror` (or press **F7**).

## Practice steering profile (90° straight / 150° left)

Run `run.bat --profile practice-steering` to use the included practice profile. It maps a 90° hand-line angle to straight, 150° to full left, and 30° to full right, with responsive smoothing and auto-recenter disabled so the calibration does not drift while practising. Keep both hands visible and hold your straight pose steadily for about 2 seconds at startup. After calibration, click the game window so it receives the controls. Press **F7** if your camera view makes left/right feel reversed.

## Calibrating

On every start (and after **F9**) hold both hands in your steering-wheel pose, level and steady, for about 2 seconds. That angle becomes
neutral (so you do not need to hold the wheel perfectly level). Calibration restarts if the pose moves more than a few degrees or a hand
disappears. While driving, if you hold the wheel within 6 deg of neutral for 3 s, neutral slowly follows your hands (auto-recenter,
`steering.auto_recenter_deg` / `auto_recenter_after_s`, 0 = off), so a slightly-off calibration cannot leave a constant small turn. Without a window the only feedback is the console / log (`CALIBRATING` → `Driving`); run `run.bat` the first time to watch it.

## Using it in a game

**Default = keyboard mode**, which works in any game or browser tab: it presses WASD *and* the arrow keys together, so games using either
respond. **Keys go to whichever window has focus** — after the console says `Driving`, **click your game window / browser tab** (and click the
game itself inside the page). If the console or Notepad has focus, the keys land there instead.

Check it first with `.\run.bat --test-output`: click Notepad (or your game) during the 5-second countdown and watch the keys appear / the car react.
Try `--keys arrows` or `--keys wasd` if a game dislikes both, and `--backend gamepad` for true analog steering (needs ViGEmBus).

* **Gamepad mode** (`--backend gamepad`): in the game's controls menu pick *Controller / Xbox 360*, then steering = left stick, accelerate = RT, brake = LT, nitro = A
  (change the button with `output.nitro_button`: a, b, x, y, lb, rb). Many browser games (via the Gamepad API) see it after a button press.
* **Keyboard mode:** the game must use W/S/A/D or the arrow keys (+ Shift for nitro; set `output.key_nitro` for others, or `key_preset: custom`). Keys are sent with hardware scan codes so DirectInput
  games receive them. Some games treat `S` as reverse when stopped — keep `safety.lost_behavior` on `neutral`.
* Focus is never taken: the game keeps keyboard focus. **Games running as Administrator ignore input from a normal process — run the
  controller as Administrator too** (right-click `run.bat` → Run as administrator).
* Online games with anti-cheat may flag any synthetic input (keyboard or virtual gamepad). Use at your own risk.

## Game profiles, presentation mode, recording

* **Profiles** (`profiles/*.json`): `.\run.bat --profile forza-gamepad`. `--list-profiles` shows them: `browser-racing`, `asphalt-pc`, `forza-gamepad` (virtual Xbox pad), `gta-v`, `presentation`. A profile only overrides the keys it lists; add a game by adding a small JSON file (same sections as `config.json`, plus a `description`). Command-line flags still win over a profile.
* **Nitro mode** (`gestures.nitro_mode`): `hold` (nitro while the thumbs-up is shown), `toggle` (each thumbs-up flips it) or `pulse` (a short burst, `nitro_pulse_s`).
* **Presentation mode** (`--presentation`, used by `demo.bat`): adds a "VISIONDRIVE - HAND-GESTURE-CONTROLLED CAR GAME" (AI hand tracking) title band. The preview always shows the pipeline strip (hands, landmarks, angle, gesture, keys, game), FPS, latency and tracking confidence. When both hands are lost it shows a red **SAFE MODE** banner. `--dry-run` shows all of this but sends no keys to any game, so it is safe in a classroom.
* **Record and replay**: `--record run.jsonl` saves every tracked frame (landmarks and controls). `--replay run.jsonl` runs it through the control logic with no camera and prints a summary (steering range, brake/nitro time, modes), so tests are repeatable.
* **F12 emergency stop** releases everything and pauses; **F7** flips left/right.

## Steering feel and self-check

* `.\run.bat --feel arcade` (lightest smoothing, snappiest), `--feel normal` (the default smoothing), `--feel realistic` (heavier smoothing). Feels only change the
  One Euro filter: steering is always `offset / full_lock_deg` (35 deg = full lock) with no curve, rate limit or latch. Or set
  `steering.feel` in `config.json` (`custom` = use the individual numbers, which is the default).
* `.\run.bat --check` runs a **guided self-check with your real camera** (camera and speed, two-hand wheel, open palm, thumbs-up, output) and prints
  PASS/FAIL with a hint for each failure. The report is saved as `check.txt` in `%APPDATA%\GestureCar`.
* `demo.bat` starts a presentation setup (compact preview via `--preview-scale 0.55`, realistic feel); see `DEMO.md` for a demo checklist.
* The preview (default) shows a small docked window: camera, hand skeletons, a rotating steering wheel, steering/gas/brake bars and a line of guidance
  ("Hold both hands like a wheel...", "Show BOTH hands").

## Tuning (`config.json`)

All numbers are in `config.json` (run `python main.py --write-default-config` to regenerate it). Invalid values are ignored with a log message.

| Goal | Setting |
|---|---|
| Steering too twitchy / too lazy | `steering.full_lock_deg` (steering = clamp(offset / full_lock_deg), default 35: 5 deg = 14%, 17.5 deg = 50%, 35+ deg = 100%) |
| Steering drifts when hands are idle | raise `steering.deadzone_deg` (default 1.5) |
| What counts as straight | hold the wheel straight for 2 s at start-up (or press F9): the average angle becomes neutral |
| Small turns do nothing with keyboard keys | `output.pwm_min_duty` (default 0: every turn outside the dead zone taps the key) |
| Jittery steering | lower `steering.one_euro_min_cutoff` (smoother, laggier; default 1.5) or set `steering.predict_frames` to 0 (default 1: one frame of prediction hides filter lag but passes a little more noise); raise `one_euro_beta` for less lag in fast turns (default 0.05) |
| Steering lags | `--debug` prints each stage (camera wait, hand detection, steering, send) every 100 ms; hand detection is usually most of it. `app.loop_hz` (default 250) is the output rate between frames; `app.priority` (default high) |
| Webcam slow in a dim room | `camera.auto_exposure: false` and `camera.exposure` (e.g. -5): a fixed exposure keeps 30 fps, but the picture must still be bright enough to track hands |
| Wheel direction reversed | `steering.invert_steering` (older configs' `invert` still works) |
| Palm not detected | run `--dry-run --debug` and read `fingers L. R.`: each hand must count 5. Lower `gestures.finger_extended_min` (default 0.6) or `thumb_extended_min` (0.5) |
| Reverse too early / late | `gestures.reverse_after_s` (default 0.6) |
| Thumbs-up not detected / false positives | `gestures.thumbs_up_max_finger`, `thumbs_up_max_tilt_deg` |
| Gesture flicker | `gestures.confirm_frames`, `gestures.release_s` |
| Wrong camera / more speed | `camera.index` or `camera.source` (see *Choosing a webcam*); `vision.model_complexity` 0 (fastest), `camera` 640×480 |

## Architecture

```
gesture_car/
├── cli.py, config.py        command line, settings (config.json)
├── backend/                 everything that thinks and acts (no windows)
│   ├── camera.py            threaded capture: backends, MJPG, resolution, phone URL, fallback, reconnect
│   ├── camera_names.py      Windows webcam names for --list-cameras
│   ├── pipeline.py, hand_tracker.py, landmarks.py, preprocessing.py   MediaPipe hand tracking (2 hands, own thread)
│   ├── steering.py, filters.py      wheel angle, calibration, auto-recenter, dead zone, proportional mapping, One Euro + prediction
│   ├── gestures.py                  finger counts; open palm = brake, two palms held = reverse, thumbs-up = nitro
│   ├── controller.py                control logic + safety (hands lost) + main loop + hotkeys
│   ├── output.py, output_keyboard.py, output_gamepad.py   keyboard (PWM) or virtual Xbox gamepad
│   └── profiles.py, recording.py, instance.py, selfcheck.py, selftest.py
├── frontend/                what you see
│   ├── debug_view.py        camera window: skeletons, wheel, bars, pipeline strip (own thread)
│   └── tray.py              optional tray icon
└── utils/                   errors, logging, paths, console
```

Flow: camera → hand tracking → control logic (steering + gestures) → output → game. The frontend only reads the backend's state, so the
program also runs without any window (`--no-preview`). Every exit path releases all keys / resets the pad, and a watchdog does the same
if the control loop stalls.

## Testing

`pip install -r requirements-dev.txt`, then `ruff check .`, `mypy gesture_car`, `python -m pytest`. Tests use procedurally generated landmark
geometry (no camera): steering math (dead zone, proportional clamp, wrap-around, centre return, calibration), gesture classification, the hands-lost safe state,
keyboard PWM duty, the gamepad mapping (against a fake `vgamepad`), config validation, webcam handling with a fake camera (selection and fallback,
backend order, MJPG, closest resolution, phone URL, newest-frame-only, disconnect and reconnect, `--list-cameras`) and a full controller
loop with a fake camera and a recording key sink, including that every key is released when the camera is lost (no real input is ever sent
by the tests).

## What has and has not been verified

Verified in a Linux sandbox without a camera: all unit tests, `ruff`, `mypy`, and a controller loop run with synthetic hands.

**Not verified (needs your hardware):** a real webcam, real MediaPipe landmarks (model download was blocked here), whether the gesture
thresholds suit *your* hands/lighting (finger-extension thresholds and thumbs-up are the most likely to need tuning), the **<60 ms latency** target and
**30+ FPS** (the app logs measured FPS and frame→output latency every 10 s so you can check), real `SendInput`/`pynput` delivery,
real `vgamepad`/ViGEmBus behaviour, global hotkeys, the tray icon, and any game's reaction to the input.

Webcam support (later) was also checked on Windows 11 with a built-in laptop camera through DirectShow: opening, 640x480 and 1920x1080,
falling back from a missing camera number to camera 0, the Windows camera name and `--list-cameras`. **Not checked with real hardware:**
an external USB webcam (MJPG, Media Foundation), a phone stream (IP Webcam / DroidCam), and pulling the cable while driving. Those paths are
covered by the fake-camera tests only.

## Troubleshooting

* **`Failed to build pygame` / old install errors:** this version no longer uses pygame. Delete the `.venv` folder and run `run.bat` again.
* **Camera not found / in use:** the program prints what to check. In short:
  1. Close every other app that may use the camera (Zoom, Teams, Skype, the Windows Camera app, browser tabs). Usually only one app can use a webcam at a time.
  2. Windows privacy: **Settings > Privacy & security > Camera**: turn on *Camera access* and *Let desktop apps access your camera*.
  3. Check the USB cable (try another USB port, avoid unpowered hubs) and any privacy shutter / camera on-off key on the laptop.
  4. Run `python main.py --list-cameras` and start with the number that works: `python main.py --camera 1`.
* **`--list-cameras` says "Windows lists it, but it gave no picture":** the camera exists but another app is using it, or the privacy settings above block it.
* **External USB webcam does not work, the built-in one does:** try `--camera-backend dshow` and `--camera-backend msmf`; some webcams work with only one of them. If it gives no picture with MJPG, set `camera.mjpg` to `false`.
* **Low FPS / slow reaction:** the log line every 10 s shows `tracking=.. fps` and `latency .. ms` (camera picture to keys sent). Aim for about 30 fps and under 60 ms. `--list-cameras` shows the real FPS of each camera. With `camera.backend` on `auto` the fastest backend is chosen automatically; you can also compare `--camera-backend dshow` and `--camera-backend msmf` yourself. Webcams slow down in dim light (longer exposure), so add light, and close other apps that use the camera. Keep `--resolution 640x480`. On USB, MJPG (on by default) helps; a busy USB hub can limit FPS.
* **Phone camera does not connect:** phone and PC on the same Wi-Fi; the app is running and on screen; the address (with `http://` and the port) opens in a browser on the PC; close that browser tab before starting (DroidCam allows one viewer); a VPN or Windows Firewall may block it.
* **Phone picture lags:** pick a lower resolution / quality in the phone app, move closer to the router, or use a USB webcam. VisionDrive already drops old frames.
* **CAMERA LOST while driving:** all keys are already released. Plug the camera back in (or restart the phone app); it reconnects by itself within a few seconds.
* **Dark room:** frames are brightened automatically when dark, but good front lighting works far better.
* **"Virtual gamepad unavailable":** install ViGEmBus (and reboot), or use `output.backend: "keyboard"`. With `fallback_to_keyboard` it falls back automatically.
* **Game ignores input:** run both as Administrator; check the game is the focused window; try `--dry-run --debug` to confirm gestures register (`--debug` prints raw angle, neutral, offset, steering %, FPS and per-stage latency to the console every 100 ms, no window).
* **Car turns the wrong way (left goes right):** press **F7** while driving (or run with `--invert`). The preview shows `<< LEFT` / `RIGHT >>` for what the game receives, and it is saved in `config.json`. Some cameras already mirror the picture, which reverses left/right.
* **MediaPipe import errors on very new Python:** install Python 3.12 and recreate `.venv`.
