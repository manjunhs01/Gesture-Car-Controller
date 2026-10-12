# VisionDrive demo guide (10 minutes before)

**What you are showing:** a webcam-only controller. Two hands make an imaginary steering wheel (rotate = steer, proportional), an open palm brakes,
a thumbs-up gives nitro, acceleration is automatic. It reads the camera in the background and sends keyboard keys (or a virtual Xbox gamepad) to any
game or browser tab — no game inside the program.

## Setup checklist
1. Laptop charged and plugged in. Close other apps that use the camera (Zoom, Teams, browser camera tabs).
2. `cd C:\Users\manju\gesture-car`, then `git pull`.
3. Light in FRONT of you (window or lamp facing you). No bright window behind you. Plain background if possible.
4. Sit about an arm's length from the camera so both hands fit in view with room to tilt.
5. Run `.\run.bat --check` and follow the prompts. Fix any FAIL using its hint. Fix this BEFORE the audience arrives.
6. Open your game / browser racing game (one that uses arrow keys or WASD) in a window; for an analog demo open https://hardwaretester.com/gamepad
   (needs the ViGEmBus driver and `--backend gamepad`: the stick will sweep smoothly as you rotate the wheel).
7. Start with `demo.bat` (big preview, realistic feel). Wait for the console/preview to say calibrated, then **click the game window**.

## During the demo
* Hold the wheel pose steady for ~2 s at the start (that is the calibration). F9 recalibrates, F8 pauses, F10 quits.
* Rotate the wheel slowly first so the audience can see the steering wheel in the preview turn with your hands.
* Show brake (open palm, all five fingers straight), reverse (both palms held for over half a second) and nitro (thumbs up, thumb pointing up).
* Take both hands away: the car centres and stops accelerating (safety rule, hands lost > 0.3 s). Mention that as a feature.

## If something goes wrong
* Preview says `Show BOTH hands`: move hands apart / into view, improve light.
* Game does nothing: click the game window (keys go to the focused window); run `.\run.bat --test-output` and click Notepad to prove keys are sent.
* Camera busy: close other camera apps, then `.\run.bat` again. Backup: show `check.txt` (from `--check`) and the test results (`python -m pytest`).

## Honest limits (say them before someone asks)
Lighting and camera quality matter; accuracy depends on the hand shapes being clear; some online games with anti-cheat may block simulated input;
games running as Administrator need the controller started as Administrator too.

## 30-second architecture summary
Webcam -> MediaPipe hand landmarks (2 hands, 30 FPS) -> wheel angle from the line between the hands (dead zone, proportional clamp, One Euro
smoothing with 1-frame prediction) + gesture rules (finger counts: open palm, two palms = reverse, thumbs-up; 2-frame confirmation) -> safety rules (hands lost) -> virtual gamepad or keyboard output.
76+ automated tests cover the steering math, gestures, safety rules and outputs.
