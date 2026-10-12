"""Optional system-tray icon (off by default): Start/Pause, Recalibrate, Quit. Needs ``pip install pystray pillow``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gesture_car.backend.controller import GestureController


def _icon_image() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (64, 64), (20, 24, 32))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), outline=(240, 240, 240), width=6)  # steering wheel
    draw.line((8, 32, 56, 32), fill=(240, 240, 240), width=6)
    draw.line((32, 32, 32, 56), fill=(240, 240, 240), width=6)
    return image


def start_tray(controller: GestureController) -> Any:
    import pystray

    menu = pystray.Menu(
        pystray.MenuItem(
            lambda item: "Resume" if controller.logic.paused else "Pause", lambda icon, item: controller.toggle_pause(), default=True
        ),
        pystray.MenuItem("Recalibrate", lambda icon, item: controller.request_recalibrate()),
        pystray.MenuItem("Quit", lambda icon, item: controller.request_quit()),
    )
    icon = pystray.Icon("gesture-car", _icon_image(), "VisionDrive", menu)
    icon.run_detached()
    return icon
