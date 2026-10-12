"""Exception types that carry a message safe to show to normal users."""

from __future__ import annotations


class GestureCarError(Exception):
    """Base class. ``user_message`` is shown in the UI; ``str(exc)`` is logged."""

    user_message = "Something went wrong. Details were written to the log file."

    def __init__(self, detail: str = "", user_message: str | None = None) -> None:
        super().__init__(detail or self.user_message)
        if user_message:
            self.user_message = user_message


class CameraError(GestureCarError):
    user_message = (
        "The camera could not be opened. Make sure it is plugged in, not used by "
        "another app, and that camera access is allowed for desktop apps."
    )


class TrackerError(GestureCarError):
    user_message = (
        "Hand tracking could not start. Check the log file; the MediaPipe "
        "package or its hand model may be missing."
    )


class ConfigError(GestureCarError):
    user_message = "A setting is invalid. Defaults were used instead."
