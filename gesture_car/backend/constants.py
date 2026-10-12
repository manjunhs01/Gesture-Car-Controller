"""Fixed technical constants (not user-tunable; tunable values live in config.json)."""

# Camera discovery / recovery
CAMERA_WIDTH = 640  # probe defaults
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_PROBE_MAX_INDEX = 9  # --list-cameras and the automatic fallback look at cameras 0..9
CAMERA_READ_FAIL_LIMIT = 45  # consecutive failed reads before a reconnect attempt
CAMERA_LOST_AFTER_S = 1.5  # no good frame for this long = camera lost (keys released, reconnecting)
CAMERA_LOST_AFTER_URL_S = 3.0  # phone streams over Wi-Fi stall more; allow longer before calling them lost
CAMERA_RECONNECT_INTERVAL_S = 2.0
CAMERA_PROBE_TIMEOUT_S = 20.0  # --list-cameras gives up on one camera after this long (open + speed check + FPS measurement)
CAMERA_URL_TIMEOUT_MS = 5000  # open / read timeout for network (phone) streams
CAMERA_FPS_PROBE_S = 1.0  # --list-cameras measures the real FPS for this long
CAMERA_SPEED_CHECK_S = 1.0  # backend auto: how long the first backend's real FPS is measured at start
CAMERA_TYPICAL_FPS = 30  # what most webcams deliver at 640x480, whatever rate they accept
CAMERA_SPEED_OK_RATIO = 0.85  # at least this share of the requested FPS = fast enough, no other backend is tried
CAMERA_SPEED_BETTER_RATIO = 1.1  # a later backend must be this much faster to replace an earlier one
CAMERA_SPEED_TRIAL_TIMEOUT_S = 8.0  # an alternative backend that takes longer than this to open and measure is skipped
CAMERA_STALE_FRAME_S = 0.5
CAMERA_OPEN_TEST_READS = 10
VISION_WAIT_TIMEOUT_S = 0.1
PREVIEW_WIDTH = 320

# MediaPipe model download
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_MIN_BYTES = 500_000  # a smaller file is a failed/partial download
MODEL_DOWNLOAD_TIMEOUT_S = 30

# Finger-extension geometry ranges (angles in degrees, ratios relative to palm size)
FINGER_ANGLE_CURLED_DEG = 100.0
FINGER_ANGLE_STRAIGHT_DEG = 160.0
FINGER_RATIO_CURLED = 0.75
FINGER_RATIO_STRAIGHT = 1.30
THUMB_DIST_CURLED = 1.00
THUMB_DIST_STRAIGHT = 1.50
